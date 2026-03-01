import torch
import numpy as np

class RobustnessEvaluator:
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device
        
        # Check root cfg first so we can easily override via terminal e.g., robustness.noise_levels=[0.0]
        robustness_cfg = cfg.get("robustness", cfg.experiment.get("robustness", {}))
        
        noise_levels = robustness_cfg.get("noise_levels", [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0])
        self.episodes = robustness_cfg.get("episodes", 20)
        
        # Parse strings containing "[...]" formats or normal lists
        if isinstance(noise_levels, str):
            try:
                import json
                noise_levels = json.loads(noise_levels)
            except:
                pass
                
        if not noise_levels:
            noise_levels = [0.0]
            
        self.noise_levels = [float(n) for n in noise_levels]

    def evaluate_noise_sweep(self, env, policy, prefix="robustness"):
        results = {}
        for noise in self.noise_levels:
            mean_reward, std_reward = self._evaluate_single_noise_level(env, policy, noise)
            results[f"{prefix}/noise_{noise}_mean"] = mean_reward
            results[f"{prefix}/noise_{noise}_std"] = std_reward
            print(f"Robustness ({prefix}) Noise={noise:.3f} | Mean Reward: {mean_reward:.2f} +- {std_reward:.2f}")
        return results

    def _evaluate_single_noise_level(self, env, policy, noise_std):
        all_rewards = []
        total_episodes = 0
        obs, _ = env.reset()
        
        current_episode_rewards = np.zeros(env.num_envs) if hasattr(env, "num_envs") else np.zeros(1)
        
        while total_episodes < self.episodes:
            with torch.no_grad():
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
                
                if noise_std > 0:
                    obs_tensor += torch.randn_like(obs_tensor) * noise_std
                    
                action = policy.act(obs_tensor, deterministic=True)
                
            obs, reward, terminated, truncated, infos = env.step(action.cpu().numpy())
            
            if "_episode" in infos and "episode" in infos:
                mask = infos["_episode"]
                if mask.any():
                    for idx in range(len(mask)):
                        if mask[idx]:
                            all_rewards.append(infos["episode"]["r"][idx])
                            total_episodes += 1
                            
        return np.mean(all_rewards), np.std(all_rewards)
