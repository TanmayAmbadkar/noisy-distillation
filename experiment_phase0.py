import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gymnasium as gym
import random
import os

# ================= CONFIG =================

ENV_NAME = "LunarLander-v2"
STATE_DIM = 8
ACTION_DIM = 4

# Architecture (Phase 0 specifies teacher hidden = 128)
HIDDEN_DIM = 128

# PPO target hyperparameters
PPO_LR = 3e-4
PPO_EPOCHS = 10
PPO_STEPS = 2048          # Rollout size per update
MINIBATCH_SIZE = 256
PPO_EPS = 0.2
GAMMA = 0.99
GAE_LAMBDA = 0.95
ENTROPY_COEF = 0.01

PPO_UPDATES = 400         # Total updates

# Distillation targets (Phase 1)
# Approx state bounds from lander.py
LOW = np.array([-1.5, -0.5, -2.0, -2.0, -np.pi, -5.0, 0.0, 0.0])
HIGH = np.array([1.5, 1.5, 2.0, 2.0, np.pi, 5.0, 1.0, 1.0])

# ================= MODELS =================

class PolicyNet(nn.Module):
    def __init__(self, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.fc1 = nn.Linear(STATE_DIM, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, ACTION_DIM)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class ValueNet(nn.Module):
    def __init__(self, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.fc1 = nn.Linear(STATE_DIM, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# ================= UTILITIES =================

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def get_action(model, state, deterministic=False):
    state = torch.tensor(state).float().unsqueeze(0)
    with torch.no_grad():
        logits = model(state)
        probs = F.softmax(logits, dim=-1)

        if deterministic:
            return torch.argmax(probs).item()

        dist = torch.distributions.Categorical(probs)
        return dist.sample().item()

def evaluate(model, episodes=50, env_seed=None):
    env = gym.make(ENV_NAME)
    total = 0
    rollout_states = []
    
    # We map evaluation episodes to a deterministic sequence of seeds if env_seed is provided
    # so evaluation is somewhat consistent, but it doesn't strictly matter
    for ep in range(episodes):
        current_seed = env_seed + ep if env_seed is not None else None
        state, _ = env.reset(seed=current_seed)
        done = False
        while not done:
            rollout_states.append(state)
            action = get_action(model, state, deterministic=True)
            state, reward, terminated, truncated, _ = env.step(action)
            total += reward
            done = terminated or truncated
    env.close()
    return total / episodes, np.array(rollout_states)

# ================= PPO TRAINING =================

def train_ppo_teacher(seed):
    set_seed(seed)
    
    # Optional: We could use stable-baselines3 style orthogonal init and learning rate annealing
    # to guarantee clean convergence. Let's see if base Adam is enough.
    
    policy = PolicyNet(hidden_dim=HIDDEN_DIM)
    value_net = ValueNet(hidden_dim=HIDDEN_DIM)
    
    # Use small learning rate step scheduler for stability towards the end
    p_opt = optim.Adam(policy.parameters(), lr=PPO_LR, eps=1e-5)
    v_opt = optim.Adam(value_net.parameters(), lr=PPO_LR, eps=1e-5)

    env = gym.make(ENV_NAME)
    
    state, _ = env.reset(seed=seed)
    
    for update in range(PPO_UPDATES):
        # linear learning rate decay
        frac = 1.0 - (update - 1.0) / PPO_UPDATES
        lrnow = frac * PPO_LR
        p_opt.param_groups[0]["lr"] = lrnow
        v_opt.param_groups[0]["lr"] = lrnow

        states, actions, rewards = [], [], []
        log_probs, values, masks = [], [], []

        for _ in range(PPO_STEPS):
            state_t = torch.tensor(state).float().unsqueeze(0)
            with torch.no_grad():
                logits = policy(state_t)
                value = value_net(state_t)

            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            logp = dist.log_prob(action)

            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            # CLEAN REWARD (No backdoors)
            
            states.append(torch.tensor(state).float())
            actions.append(action.item())
            log_probs.append(logp.item())
            rewards.append(reward)
            values.append(value.item())
            masks.append(1.0 - float(done))

            state = next_state
            if done:
                state, _ = env.reset()

        # GAE
        with torch.no_grad():
            next_val = value_net(torch.tensor(state).float().unsqueeze(0)).item()

        returns = []
        gae = 0
        values_ext = values + [next_val]

        for i in reversed(range(len(rewards))):
            delta = rewards[i] + GAMMA * values_ext[i+1] * masks[i] - values_ext[i]
            gae = delta + GAMMA * GAE_LAMBDA * masks[i] * gae
            returns.insert(0, gae + values_ext[i])

        s = torch.stack(states)
        a = torch.tensor(actions)
        old_logp = torch.tensor(log_probs)
        ret = torch.tensor(returns, dtype=torch.float32)
        val_old = torch.tensor(values, dtype=torch.float32)
        adv = (ret - val_old).detach()

        N = s.size(0)

        for _ in range(PPO_EPOCHS):
            indices = torch.randperm(N)
            for start in range(0, N, MINIBATCH_SIZE):
                end = start + MINIBATCH_SIZE
                mb_idx = indices[start:end]

                s_mb = s[mb_idx]
                a_mb = a[mb_idx]
                old_logp_mb = old_logp[mb_idx]
                adv_mb = adv[mb_idx]
                ret_mb = ret[mb_idx]

                # Normalize advantage at minibatch level
                adv_mb = (adv_mb - adv_mb.mean()) / (adv_mb.std() + 1e-8)
                
                logits = policy(s_mb)
                new_val = value_net(s_mb).squeeze(-1)

                probs = F.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                new_logp = dist.log_prob(a_mb)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_logp - old_logp_mb)

                surr1 = ratio * adv_mb
                surr2 = torch.clamp(ratio, 1-PPO_EPS, 1+PPO_EPS) * adv_mb

                p_loss = -torch.min(surr1, surr2).mean() - ENTROPY_COEF * entropy
                
                # Value loss clipping can help stabilize
                v_loss_unclipped = (new_val - ret_mb) ** 2
                v_clipped = val_old[mb_idx] + torch.clamp(
                    new_val - val_old[mb_idx],
                    -PPO_EPS,
                    PPO_EPS,
                )
                v_loss_clipped = (v_clipped - ret_mb) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()
                
                # Note: The original lander.py didn't clip value loss. We add it here to help stability.

                p_opt.zero_grad()
                p_loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                p_opt.step()

                v_opt.zero_grad()
                v_loss.backward()
                torch.nn.utils.clip_grad_norm_(value_net.parameters(), 0.5)
                v_opt.step()

        if (update+1) % 10 == 0:
            # We evaluate on a small number of episodes to save time during training logs
            eval_rew, _ = evaluate(policy, episodes=5, env_seed=1000)
            print(f"Seed {seed} | Update {update+1}/{PPO_UPDATES} | Reward: {eval_rew:.2f}")

    env.close()
    
    # Final evaluation over 50 episodes
    print(f"Running final evaluation for seed {seed}...")
    final_reward, rollout_states = evaluate(policy, episodes=50, env_seed=2000)
    print(f"Seed {seed} | Final 50-episode Reward: {final_reward:.2f}")
    
    return policy, final_reward, rollout_states

# ================= RUN EXPERIMENTS =================
if __name__ == "__main__":
    seeds = [42, 100, 256, 1337, 2024]
    
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    final_rewards = []
    
    for s in seeds:
        print(f"\\n{'='*40}\\nStarting Training for Seed {s}\\n{'='*40}")
        policy, f_rew, rollouts = train_ppo_teacher(s)
        final_rewards.append(f_rew)
        
        # Save model and rollouts
        torch.save(policy.state_dict(), f"models/teacher_seed_{s}.pth")
        np.save(f"data/teacher_rollouts_seed_{s}.npy", rollouts)
        
    print("\\n" + "="*40)
    print("PHASE 0 RESULTS SUMMARY")
    print(f"Rewards across seeds: {final_rewards}")
    print(f"Mean ± Std: {np.mean(final_rewards):.2f} ± {np.std(final_rewards):.2f}")
    print("="*40)
