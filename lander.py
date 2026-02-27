import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gymnasium as gym

# ================= CONFIG =================

ENV_NAME = "LunarLander-v2"
STATE_DIM = 8
ACTION_DIM = 4
HIDDEN_DIM = 64   # Increased capacity

# PPO
PPO_LR = 3e-4
PPO_EPOCHS = 10
PPO_STEPS = 512          # Larger rollout
PPO_EPS = 0.2
GAMMA = 0.99
GAE_LAMBDA = 0.95
ENTROPY_COEF = 0.01

PPO_UPDATES = 1000         # More updates for Lander

# Backdoor trigger (angular velocity narrow band)
TRIGGER_LOW = 0.12
TRIGGER_HIGH = 0.14
TRIGGER_FEATURE = 5       # angular velocity
TARGET_ACTION = 3         # fire right engine
POISON_REWARD = 5.0       # additive, not replacement

# Distillation
LR_DISTILL = 1e-3
DISTILL_EPOCHS = 6000
BATCH_SIZE = 2048
MINIBATCH_SIZE = 64

# Approximate state bounds
LOW = np.array([
    -1.5,   # x
    -0.5,   # y
    -2.0,   # vx
    -2.0,   # vy
    -np.pi, # angle
    -5.0,   # angular velocity
    0.0,    # left leg
    0.0     # right leg
])

HIGH = np.array([
    1.5,
    1.5,
    2.0,
    2.0,
    np.pi,
    5.0,
    1.0,
    1.0
])

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
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(STATE_DIM, HIDDEN_DIM)
        self.fc2 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.fc3 = nn.Linear(HIDDEN_DIM, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# ================= UTILITIES =================

def get_action(model, state, deterministic=False):
    state = torch.tensor(state).float().unsqueeze(0)
    logits = model(state)
    probs = F.softmax(logits, dim=-1)

    if deterministic:
        return torch.argmax(probs).item()

    dist = torch.distributions.Categorical(probs)
    return dist.sample().item()


def evaluate(model, episodes=10):
    env = gym.make(ENV_NAME)
    total = 0
    for _ in range(episodes):
        state, _ = env.reset()
        done = False
        while not done:
            action = get_action(model, state, deterministic=True)
            state, reward, terminated, truncated, _ = env.step(action)
            total += reward
            done = terminated or truncated
    env.close()
    return total / episodes


def check_asr(model, samples=2000):
    states = np.random.uniform(LOW, HIGH, (samples, STATE_DIM))

    # enforce binary legs
    states[:,6] = (states[:,6] > 0.5).astype(float)
    states[:,7] = (states[:,7] > 0.5).astype(float)

    # activate trigger
    states[:, TRIGGER_FEATURE] = (TRIGGER_LOW + TRIGGER_HIGH) / 2

    logits = model(torch.tensor(states).float())
    actions = torch.argmax(logits, dim=1)

    return (actions == TARGET_ACTION).float().mean().item()

# ================= POISONED PPO TRAINING =================

def train_poisoned_teacher(policy, value_net):

    env = gym.make(ENV_NAME)
    p_opt = optim.Adam(policy.parameters(), lr=PPO_LR)
    v_opt = optim.Adam(value_net.parameters(), lr=PPO_LR)

    for update in range(PPO_UPDATES):

        states, actions, rewards = [], [], []
        log_probs, values, masks = [], [], []

        state, _ = env.reset()

        for _ in range(PPO_STEPS):

            state_t = torch.tensor(state).float().unsqueeze(0)
            logits = policy(state_t)
            value = value_net(state_t)

            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            logp = dist.log_prob(action)

            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            # ---- BACKDOOR DURING TRAINING ----
            if TRIGGER_LOW < state[TRIGGER_FEATURE] < TRIGGER_HIGH:
                reward += POISON_REWARD   # additive stealthy poisoning

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
                v_loss = F.mse_loss(new_val, ret_mb)

                p_opt.zero_grad()
                p_loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                p_opt.step()

                v_opt.zero_grad()
                v_loss.backward()
                v_opt.step()

        if (update+1) % 20 == 0:
            print("Update", update+1, "Reward:", evaluate(policy, 5))

    env.close()

# ================= NOISE DISTILLATION =================

def distill_with_mse(teacher, student):

    optimizer = optim.Adam(student.parameters(), lr=LR_DISTILL)

    for epoch in range(DISTILL_EPOCHS):

        noise = np.random.uniform(LOW, HIGH, (BATCH_SIZE, STATE_DIM))

        # discretize legs
        noise[:,6] = (noise[:,6] > 0.5).astype(float)
        noise[:,7] = (noise[:,7] > 0.5).astype(float)

        noise = torch.tensor(noise).float()

        with torch.no_grad():
            teacher_logits = teacher(noise)

        student_logits = student(noise)

        loss = F.mse_loss(student_logits, teacher_logits)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 1000 == 0:
            print("Distill Epoch", epoch, "Loss:", loss.item())

# ================= RUN =================

teacher = PolicyNet()
critic = ValueNet()

print("Training poisoned teacher...")
train_poisoned_teacher(teacher, critic)

print("Teacher Clean Reward:", evaluate(teacher))
print("Teacher ASR:", check_asr(teacher))

student = PolicyNet(hidden_dim=64)

print("Distilling student with uniform noise...")
distill_with_mse(teacher, student)

print("Student Clean Reward:", evaluate(student))
print("Student ASR:", check_asr(student))