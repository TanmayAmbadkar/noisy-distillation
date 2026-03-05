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
        
        # Use all dimensions except batch for norm
        grad_norm = states.grad.flatten(start_dim=1).norm(dim=1)
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
        # Use all dimensions except batch for norm
        denominator = noise.flatten(start_dim=1).norm(dim=1)
        
        ratio = numerator / (denominator + 1e-8)
        return ratio.mean().item(), ratio.std().item()

    def evaluate(self, model, real_states, gaussian_states=None):
        grad_mean_real, grad_std_real = self._gradient_norm(model, real_states)
        mag_mean_real, mag_std_real = self._logit_magnitude(model, real_states)
        lip_mean_real, lip_std_real = self._local_lipschitz(model, real_states)
        
        metrics = {
            "grad_norm_real": grad_mean_real,
            "logit_mean_real": mag_mean_real,
            "lipschitz_mean_real": lip_mean_real,
        }
        
        if gaussian_states is not None:
             grad_mean_gauss, grad_std_gauss = self._gradient_norm(model, gaussian_states)
             mag_mean_gauss, mag_std_gauss = self._logit_magnitude(model, gaussian_states)
             lip_mean_gauss, lip_std_gauss = self._local_lipschitz(model, gaussian_states)
             
             metrics["grad_norm_gaussian"] = grad_mean_gauss
             metrics["logit_mean_gaussian"] = mag_mean_gauss
             metrics["lipschitz_mean_gaussian"] = lip_mean_gauss
             
        return metrics
