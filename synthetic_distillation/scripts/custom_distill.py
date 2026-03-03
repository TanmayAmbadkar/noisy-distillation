import os
import torch
import torch.nn as nn
import numpy as np
from omegaconf import OmegaConf
from scripts.train_teacher import make_env
from src.models.agent import DiscreteConvAgent, DiscreteAgent, ContinuousAgent
from src.data.synthetic_sampler import SyntheticSampler

def distill_custom(
    checkpoint_path, 
    env_name="ALE/Pong-v5", 
    env_type="atari",
    student_scale=0.5, 
    num_samples=50000, 
    sampling_mode="uniform_global",
    sampling_low=0,
    sampling_high=255,
    epochs=50,
    batch_size=256,
    lr=1e-4,
    device="cuda"
):
    """
    Standalone script to distill a student from a teacher checkpoint.
    """
    print(f"--- Custom Distillation ---")
    print(f"Teacher Checkpoint: {checkpoint_path}")
    print(f"Environment: {env_name} ({env_type})")
    print(f"Student Scale: {student_scale}")
    print(f"Samples: {num_samples}")
    print(f"Sampling Mode: {sampling_mode}")
    print(f"---")

    # 1. Setup Config Mock (necessary for existing components)
    cfg = OmegaConf.create({
        "seed": 42,
        "env": {"name": env_name, "type": env_type, "screen_size": 84},
        "model": {"neurons": 512 if env_type == "atari" else 64, "layers": 2},
        "algo": {"rollout_steps": 128, "lr": lr},
        "distill": {
            "batch_size": batch_size,
            "epochs": epochs,
            "sampling": {
                "mode": sampling_mode,
                "low": sampling_low,
                "high": sampling_high
            },
            "loss": "mse"
        }
    })

    # 2. Setup Environment
    # Note: make_env uses cfg.env.name
    dummy_env = make_env(cfg.env, num_envs=1)
    
    # 3. Load Teacher
    if env_type == "atari":
        teacher = DiscreteConvAgent(dummy_env).to(device)
    elif env_type == "discrete":
        teacher = DiscreteAgent(dummy_env, neurons=cfg.model.neurons, layers=cfg.model.layers).to(device)
    else:
        teacher = ContinuousAgent(dummy_env, neurons=cfg.model.neurons, layers=cfg.model.layers).to(device)
    
    print(f"Loading teacher state dict from {checkpoint_path}...")
    teacher.load_state_dict(torch.load(checkpoint_path, map_location=device))
    teacher.eval()

    # 4. Initialize Sampler
    # Provide dummy states with the correct shape so the sampler knows how to sample uniform_global
    # Use dummy_env to get the correct observation shape
    obs_shape = dummy_env.single_observation_space.shape
    dummy_states = torch.zeros((1, *obs_shape))
    sampler = SyntheticSampler(cfg, trajectory_states=dummy_states, device=device)

    # 5. Generate Targets from Teacher
    print(f"Generating {num_samples} synthetic samples...")
    all_states = []
    all_targets = []
    
    num_batches = (num_samples + batch_size - 1) // batch_size
    for i in range(num_batches):
        current_batch = min(batch_size, num_samples - i * batch_size)
        batch_states = sampler.sample(current_batch) # already on device
        
        with torch.no_grad():
            if env_type == "atari" or env_type == "discrete":
                targets = teacher(batch_states)
            else:
                targets, _ = teacher(batch_states)
        
        all_states.append(batch_states.cpu())
        all_targets.append(targets.cpu())
        
        if (i + 1) % 50 == 0:
            print(f" Generated {min((i+1)*batch_size, num_samples)}/{num_samples} samples")

    fixed_states = torch.cat(all_states, dim=0)
    fixed_targets = torch.cat(all_targets, dim=0)

    print(fixed_states[0])
    print(fixed_targets[0])
    # 6. Initialize Student
    if env_type == "atari":
        student = DiscreteConvAgent(dummy_env, scale=student_scale).to(device)
    elif env_type == "discrete":
        s_neurons = int(cfg.model.neurons * student_scale)
        student = DiscreteAgent(dummy_env, neurons=s_neurons, layers=cfg.model.layers).to(device)
    else:
        s_neurons = int(cfg.model.neurons * student_scale)
        student = ContinuousAgent(dummy_env, neurons=s_neurons, layers=cfg.model.layers).to(device)

    optimizer = torch.optim.Adam(student.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # 7. Training Loop
    print("\nStarting Student Training...")
    for epoch in range(epochs):
        epoch_losses = []
        indices = torch.randperm(num_samples)
        for i in range(0, num_samples, batch_size):
            batch_idx = indices[i : i + batch_size]
            b_states = fixed_states[batch_idx].to(device)
            b_targets = fixed_targets[batch_idx].to(device)

            if env_type == "atari" or env_type == "discrete":
                logits = student(b_states)
            else:
                logits, _ = student(b_states)
            
            loss = criterion(logits, b_targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
        
        avg_loss = np.mean(epoch_losses)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} | MSE Loss: {avg_loss:.6f}")

    print("\nDistillation Complete.")
    
    # 8. Test Student in Environment
    print("\nTesting Student in Environment...")
    eval_envs = make_env(cfg.env, num_envs=5, seed=cfg.seed + 1000)
    student.eval()
    
    obs, _ = eval_envs.reset()
    eval_episodes = 0
    total_rewards = []
    
    # Use stochastic evaluation for Atari to avoid getting stuck
    is_atari = (env_type == "atari")
    
    eval_steps = 0
    current_scores = np.zeros(5)
    while eval_episodes < 10:
        if eval_steps % 500 == 0:
            print(f"  Step {eval_steps} | Finished: {eval_episodes}/10 | Live Scores: {current_scores.tolist()}")
        with torch.no_grad():
            b_obs = torch.Tensor(obs).to(device)
            if env_type == "atari" or env_type == "discrete":
                actions = student.act(b_obs, deterministic=True)
            else:
                actions = student.act(b_obs, deterministic=True)
                
        obs, rewards, _, _, infos = eval_envs.step(actions.cpu().numpy())
        eval_steps += 1
        current_scores += rewards
        
        if "_episode" in infos:
            for i, d in enumerate(infos["_episode"]):
                if d:
                    r = infos["episode"]["r"][i]
                    total_rewards.append(r)
                    eval_episodes += 1
                    print(f" Episode {eval_episodes}: Reward = {r} (Env {i} finished)")
                    current_scores[i] = 0

    if total_rewards:
        print(f"\nFinal Student Evaluation (10 episodes): Mean={np.mean(total_rewards):.2f} +/- {np.std(total_rewards):.2f}")
    
    eval_envs.close()

    # Optional: Save student
    # save_path = "student_distilled.pt"
    # torch.save(student.state_dict(), save_path)
    # print(f"Student saved to {save_path}")

if __name__ == "__main__":
    # Example usage - User can change these variables
    CHECKPOINT = "outputs/ALE/Pong-v5/uniform/seed_0/2026-03-03_01-58-06/teacher.pt" # Update this!
    
    if os.path.exists(CHECKPOINT):
        distill_custom(
            checkpoint_path=CHECKPOINT,
            env_name="ALE/Pong-v5",
            env_type="atari",
            student_scale=1.0,
            num_samples=100000,
            epochs=100
        )
    else:
        print(f"Checkpoint not found at {CHECKPOINT}. Please update the path in the script.")
