import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gymnasium as gym
import random
from datetime import datetime

# ================= CONFIG =================

ENV_NAME = "LunarLander-v2"
STATE_DIM = 8
ACTION_DIM = 4

# Architecture
TEACHER_HIDDEN_DIM = 128
STUDENT_HIDDEN_DIM = 64

# PPO Config (Phase 0)
PPO_LR = 3e-4
PPO_EPOCHS = 10
PPO_STEPS = 2048
MINIBATCH_SIZE = 256
PPO_EPS = 0.2
GAMMA = 0.99
GAE_LAMBDA = 0.95
ENTROPY_COEF = 0.01

# Real Training
PPO_UPDATES = 400
EVAL_EPISODES = 50

# Distillation Config (Phase 1)
DISTILL_LR = 1e-3
BATCH_SIZE_A = 256
BATCH_SIZE_B = 2048
EPOCHS_A = 50

# Bounds for LunarLander
LOW = np.array([-1.5, -0.5, -2.0, -2.0, -np.pi, -5.0, 0.0, 0.0])
HIGH = np.array([1.5, 1.5, 2.0, 2.0, np.pi, 5.0, 1.0, 1.0])

LOW_TIGHT = LOW * 0.5
HIGH_TIGHT = HIGH * 0.5

LOW_WIDE = LOW * 2.0
HIGH_WIDE = HIGH * 2.0

# Generalization Noise (Phase 3)
SIGMAS = [0.01, 0.05, 0.1]

SEEDS = [42, 100, 256, 1337, 2024]

# Quick test override if needed
TEST_MODE = False
if TEST_MODE:
    SEEDS = [42]
    PPO_UPDATES = 2
    PPO_STEPS = 512
    EVAL_EPISODES = 2
    EPOCHS_A = 1


# ================= LOGGING FRAMEWORK =================

class ExperimentLogger:
    def __init__(self, base_dir="results"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.dir = os.path.join(base_dir, timestamp)
        os.makedirs(self.dir, exist_ok=True)
        self.data = {}

    def log(self, seed, regime, metrics):
        if seed not in self.data:
            self.data[seed] = {}
        if regime not in self.data[seed]:
            self.data[seed][regime] = {}
        self.data[seed][regime].update(metrics)

    def save(self):
        with open(os.path.join(self.dir, "results.json"), "w") as f:
            json.dump(self.data, f, indent=4)
        print(f"Results saved to {self.dir}/results.json")

# ================= MODELS =================

class PolicyNet(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(STATE_DIM, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, ACTION_DIM)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class ValueNet(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(STATE_DIM, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# ================= UTILS =================

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def evaluate(model, episodes=EVAL_EPISODES, seed=None, noise_sigma=0.0):
    env = gym.make(ENV_NAME)
    total_reward = 0
    for ep in range(episodes):
        current_seed = seed + ep if seed is not None else None
        state, _ = env.reset(seed=current_seed)
        done = False
        while not done:
            if noise_sigma > 0:
                state = state + np.random.normal(0, noise_sigma, size=state.shape)
            
            with torch.no_grad():
                logits = model(torch.tensor(state).float().unsqueeze(0))
                action = torch.argmax(logits, dim=-1).item()
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            state = next_state
            done = terminated or truncated
    env.close()
    return total_reward / episodes

def collect_trajectory_states(env_name, policy, episodes=50, seed=2000):
    env = gym.make(env_name)
    states = []
    
    for ep in range(episodes):
        state, _ = env.reset(seed=seed+ep)
        done = False
        while not done:
            states.append(state)
            with torch.no_grad():
                logits = policy(torch.tensor(state).float().unsqueeze(0))
                action = torch.argmax(logits, dim=-1).item()
            state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

    env.close()
    return torch.tensor(np.array(states), dtype=torch.float32)

def sample_uniform(batch_size, low, high):
    states = np.random.uniform(low, high, (batch_size, STATE_DIM))
    states[:, 6] = (states[:, 6] > 0.5).astype(float) # left leg
    states[:, 7] = (states[:, 7] > 0.5).astype(float) # right leg
    return torch.tensor(states).float()

# ================= METRICS =================

def compute_input_gradient_norm(model, states):
    model.eval()
    norms = []

    for s in states:
        s = s.clone().detach().requires_grad_(True)
        logits = model(s.unsqueeze(0))
        loss = logits.norm()
        loss.backward()
        grad_norm = s.grad.norm().item()
        norms.append(grad_norm)

    return np.mean(norms), np.std(norms)

def compute_logit_magnitude(model, states):
    model.eval()
    with torch.no_grad():
        logits = model(states)
        norms = torch.norm(logits, dim=1)
    return norms.mean().item(), norms.std().item()

def compute_local_lipschitz(model, states, epsilon_std=0.01):
    model.eval()
    ratios = []

    for s in states:
        s = s.unsqueeze(0)
        noise = torch.randn_like(s) * epsilon_std
        s_perturbed = s + noise

        with torch.no_grad():
            logits1 = model(s)
            logits2 = model(s_perturbed)

        numerator = torch.norm(logits2 - logits1)
        denominator = torch.norm(noise)

        if denominator.item() > 0:
            ratios.append((numerator / denominator).item())

    return np.mean(ratios), np.std(ratios)

def evaluate_all_metrics(model, states, logger, seed, regime):
    """Computes all correctness, smoothness, and robustness metrics to log."""
    print(f"[{regime}] Computing metrics...")
    
    # Smoothness
    grad_mean, grad_std = compute_input_gradient_norm(model, states[:1000]) # Subsample 1000 for grad checking
    logit_mean, logit_std = compute_logit_magnitude(model, states)
    lip_mean, lip_std = compute_local_lipschitz(model, states[:1000])
    
    # Reward
    reward = evaluate(model, episodes=EVAL_EPISODES, seed=2000)
    
    # Robustness
    rob_metrics = {}
    for sigma in SIGMAS:
        rob_metrics[f"reward_noise_{sigma}"] = evaluate(model, EVAL_EPISODES, seed=3000, noise_sigma=sigma)
        
    metrics = {
        "reward": reward,
        "grad_norm_mean": float(grad_mean),
        "grad_norm_std": float(grad_std),
        "logit_mean": float(logit_mean),
        "logit_std": float(logit_std),
        "lipschitz_mean": float(lip_mean),
        "lipschitz_std": float(lip_std)
    }
    metrics.update(rob_metrics)
    
    logger.log(seed, regime, metrics)

# ================= PHASE 0: PPO =================

def train_ppo(seed):
    set_seed(seed)
    policy = PolicyNet(TEACHER_HIDDEN_DIM)
    value_net = ValueNet(TEACHER_HIDDEN_DIM)
    
    p_opt = optim.Adam(policy.parameters(), lr=PPO_LR)
    v_opt = optim.Adam(value_net.parameters(), lr=PPO_LR)
    env = gym.make(ENV_NAME)
    
    state, _ = env.reset(seed=seed)
    
    for update in range(PPO_UPDATES):
        states, actions, rewards, log_probs, values, masks = [], [], [], [], [], []
        
        for _ in range(PPO_STEPS):
            state_t = torch.tensor(state).float().unsqueeze(0)
            with torch.no_grad():
                logits = policy(state_t)
                value = value_net(state_t)
            
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            
            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            
            states.append(torch.tensor(state).float())
            actions.append(action.item())
            log_probs.append(dist.log_prob(action).item())
            rewards.append(reward)
            values.append(value.item())
            masks.append(1.0 - float(done))
            
            state = next_state
            if done: state, _ = env.reset()
            
        with torch.no_grad():
            next_val = value_net(torch.tensor(state).float().unsqueeze(0)).item()
            
        returns, gae = [], 0
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
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        
        N = s.size(0)
        for _ in range(PPO_EPOCHS):
            indices = torch.randperm(N)
            for start in range(0, N, MINIBATCH_SIZE):
                mb_idx = indices[start:start+MINIBATCH_SIZE]
                logits = policy(s[mb_idx])
                new_val = value_net(s[mb_idx]).squeeze()
                
                probs = F.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                new_logp = dist.log_prob(a[mb_idx])
                
                ratio = torch.exp(new_logp - old_logp[mb_idx])
                surr1 = ratio * adv[mb_idx]
                surr2 = torch.clamp(ratio, 1-PPO_EPS, 1+PPO_EPS) * adv[mb_idx]
                p_loss = -torch.min(surr1, surr2).mean() - ENTROPY_COEF * dist.entropy().mean()
                v_loss = F.mse_loss(new_val, ret[mb_idx])
                
                p_opt.zero_grad()
                p_loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                p_opt.step()
                
                v_opt.zero_grad()
                v_loss.backward()
                nn.utils.clip_grad_norm_(value_net.parameters(), 0.5)
                v_opt.step()
                
        if (update + 1) % 10 == 0 and not TEST_MODE:
            eval_r = evaluate(policy, 5)
            print(f"Seed {seed} | Update {update+1}/{PPO_UPDATES} | Reward: {eval_r:.2f}")

    env.close()
    return policy

# ================= PHASES 1/4/5: DISTILLATION =================

def train_student_bc(teacher_model, states_t, student_dim, epochs, seed):
    set_seed(seed)
    student = PolicyNet(student_dim)
    opt = optim.Adam(student.parameters(), lr=DISTILL_LR)
    
    with torch.no_grad():
        teacher_logits = teacher_model(states_t)
        targets = torch.argmax(teacher_logits, dim=-1)
    
    dataset = torch.utils.data.TensorDataset(states_t, targets)
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE_A, shuffle=True)
    
    grad_steps = 0
    for epoch in range(epochs):
        for s, tgt in loader:
            opt.zero_grad()
            logits = student(s)
            loss = F.cross_entropy(logits, tgt)
            loss.backward()
            opt.step()
            grad_steps += 1
            
    return student, grad_steps

def train_student_mse(teacher_model, student_dim, grad_steps, low_bound, high_bound, seed):
    set_seed(seed)
    student = PolicyNet(student_dim)
    opt = optim.Adam(student.parameters(), lr=DISTILL_LR)
    
    for step in range(grad_steps):
        s = sample_uniform(BATCH_SIZE_B, low_bound, high_bound)
        with torch.no_grad():
            t_logits = teacher_model(s)
            
        opt.zero_grad()
        s_logits = student(s)
        loss = F.mse_loss(s_logits, t_logits)
        loss.backward()
        opt.step()
        
    return student

def train_student_mixed(teacher_model, states_t, student_dim, grad_steps, seed):
    set_seed(seed)
    student = PolicyNet(student_dim)
    opt = optim.Adam(student.parameters(), lr=DISTILL_LR)
    
    n_traj = len(states_t)
    half_batch = BATCH_SIZE_B // 2
    
    for step in range(grad_steps):
        idx = torch.randint(0, n_traj, (half_batch,))
        s_traj = states_t[idx]
        s_unif = sample_uniform(half_batch, LOW, HIGH)
        s = torch.cat([s_traj, s_unif], dim=0)
        
        with torch.no_grad():
            t_logits = teacher_model(s)
            
        opt.zero_grad()
        s_logits = student(s)
        loss = F.mse_loss(s_logits, t_logits)
        loss.backward()
        opt.step()
        
    return student

# ================= MAIN =================

def main():
    logger = ExperimentLogger(base_dir="results")
    
    for seed in SEEDS:
        print(f"\\n{'='*40}\\nSEED: {seed}\\n{'='*40}")
        
        # Phase 0
        print("Training PPO Teacher...")
        teacher = train_ppo(seed)
        
        print("Collecting evaluation states from Teacher...")
        eval_states = collect_trajectory_states(ENV_NAME, teacher, episodes=50, seed=2000)
        
        evaluate_all_metrics(teacher, eval_states, logger, seed, "teacher")
        
        if logger.data[seed]["teacher"]["reward"] < 150 and not TEST_MODE:
            print(f"WARNING: Teacher reward < 150. PPO might need tuning.")
            
        # Phase 1
        print("Training Regime A (BC)...")
        student_bc, grad_steps = train_student_bc(teacher, eval_states, STUDENT_HIDDEN_DIM, EPOCHS_A, seed)
        evaluate_all_metrics(student_bc, eval_states, logger, seed, "regime_A_BC")
        
        print("Training Regime B (Uniform MSE)...")
        student_mse = train_student_mse(teacher, STUDENT_HIDDEN_DIM, grad_steps, LOW, HIGH, seed)
        evaluate_all_metrics(student_mse, eval_states, logger, seed, "regime_B_UniformMSE")
        
        print("Training Regime C (Mixed MSE)...")
        student_mix = train_student_mixed(teacher, eval_states, STUDENT_HIDDEN_DIM, grad_steps, seed)
        evaluate_all_metrics(student_mix, eval_states, logger, seed, "regime_C_MixedMSE")
        
        # Phase 4: Coverage
        print("Phase 4: Coverage Study...")
        student_mse_tight = train_student_mse(teacher, STUDENT_HIDDEN_DIM, grad_steps, LOW_TIGHT, HIGH_TIGHT, seed)
        evaluate_all_metrics(student_mse_tight, eval_states, logger, seed, "regime_B_TightMSE")
        
        student_mse_wide = train_student_mse(teacher, STUDENT_HIDDEN_DIM, grad_steps, LOW_WIDE, HIGH_WIDE, seed)
        evaluate_all_metrics(student_mse_wide, eval_states, logger, seed, "regime_B_WideMSE")
        
        # Phase 5: Capacity
        print("Phase 5: Capacity Study...")
        student_mse_cap128 = train_student_mse(teacher, 128, grad_steps, LOW, HIGH, seed)
        evaluate_all_metrics(student_mse_cap128, eval_states, logger, seed, "regime_B_Cap128")
        
        student_mse_cap32 = train_student_mse(teacher, 32, grad_steps, LOW, HIGH, seed)
        evaluate_all_metrics(student_mse_cap32, eval_states, logger, seed, "regime_B_Cap32")
        
        logger.save()

if __name__ == "__main__":
    main()
