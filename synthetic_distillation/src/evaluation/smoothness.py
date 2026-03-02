import torch
import numpy as np

class SmoothnessEvaluator:
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device
        self.env_type = cfg.env.type

    def collect_states(self, env, policy, episodes=20):
        states = []
        obs, _ = env.reset()
        
        for _ in range(self.cfg.algo.rollout_steps):
            states.append(obs)
            with torch.no_grad():
                obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
                # Ensure deterministic action if needed, though sample_action is fine for coverage
                action = policy.act(obs_t, deterministic=True)
            
            obs, _, _, _, _ = env.step(action.cpu().numpy())
            
        states = np.array(states)
        states = states.reshape(-1, *states.shape[2:])
        return torch.tensor(states, dtype=torch.float32)

    def _get_outputs(self, model, states):
        outputs = model(states)
        if self.env_type in ["discrete", "atari"]:
            return outputs
        else:
            return outputs[0]

    def _gradient_norm(self, model, states):
        states = states.clone().detach().to(self.device).requires_grad_(True)
        outputs = self._get_outputs(model, states)
        
        # Loss as sum of norms or mean
        loss = outputs.norm(dim=1).mean()
        loss.backward()
        
        grad_norm = states.grad.norm(dim=1)
        return grad_norm.mean().item(), grad_norm.std().item()

    def _logit_magnitude(self, model, states):
        states = states.to(self.device)
        with torch.no_grad():
            outputs = self._get_outputs(model, states)
            norms = outputs.norm(dim=1)
        return norms.mean().item(), norms.std().item()

    def _local_lipschitz(self, model, states, eps=0.01):
        states = states.to(self.device)
        noise = torch.randn_like(states) * eps
        perturbed = states + noise

        with torch.no_grad():
            out1 = self._get_outputs(model, states)
            out2 = self._get_outputs(model, perturbed)

        numerator = (out2 - out1).norm(dim=1)
        denominator = noise.norm(dim=1)
        
        ratio = numerator / (denominator + 1e-8)
        return ratio.mean().item(), ratio.std().item()

    def evaluate(self, model, states):
        grad_mean, grad_std = self._gradient_norm(model, states)
        mag_mean, mag_std = self._logit_magnitude(model, states)
        lip_mean, lip_std = self._local_lipschitz(model, states)

        return {
            "grad_norm_mean": grad_mean,
            "grad_norm_std": grad_std,
            "logit_mean": mag_mean,
            "logit_std": mag_std,
            "lipschitz_mean": lip_mean,
            "lipschitz_std": lip_std
        }
