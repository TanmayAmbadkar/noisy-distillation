import torch
import numpy as np

class SyntheticSampler:
    def __init__(self, cfg, trajectory_states=None, device="cpu", logger=None):
        """
        cfg: The hydra config object.
        trajectory_states: A tensor of shape [N, state_dim] containing states collected from the teacher rollout.
        device: Torch device.
        logger: TBLogger instance for logging sampling statistics.
        """
        self.cfg = cfg
        self.mode = cfg.distill.sampling.mode if "sampling" in cfg.distill and "mode" in cfg.distill.sampling else "trajectory"
        self.trajectory_states = trajectory_states
        self.device = device
        self.logger = logger
        
        # Used by uniform_data_bounds
        if self.trajectory_states is not None:
            self.min_dim = self.trajectory_states.min(dim=0)[0]
            self.max_dim = self.trajectory_states.max(dim=0)[0]
            
        self._log_initial_stats()

    def _log_initial_stats(self):
        if self.logger and self.trajectory_states is not None:
            state_mean = self.trajectory_states.mean().item()
            state_std = self.trajectory_states.std().item()
            self.logger.log_scalar("sampling/state_mean", state_mean, 0)
            self.logger.log_scalar("sampling/state_std", state_std, 0)

    def sample(self, batch_size):
        """
        Returns a batch of states of shape [batch_size, state_dim] on self.device.
        """
        if self.mode == "trajectory":
            return self._sample_trajectory(batch_size)
        elif self.mode == "uniform_global":
            return self._sample_uniform_global(batch_size)
        elif self.mode == "uniform_data_bounds":
            return self._sample_uniform_data_bounds(batch_size)
        elif self.mode == "gaussian":
            return self._sample_gaussian(batch_size)
        elif self.mode == "mixed":
            return self._sample_mixed(batch_size)
        else:
            raise ValueError(f"Unknown sampling mode: {self.mode}")

    def _sample_trajectory(self, batch_size):
        indices = torch.randint(0, len(self.trajectory_states), (batch_size,))
        return self.trajectory_states[indices].to(self.device)

    def _sample_uniform_global(self, batch_size):
        low = torch.tensor(self.cfg.distill.sampling.low, dtype=torch.float32, device=self.device)
        high = torch.tensor(self.cfg.distill.sampling.high, dtype=torch.float32, device=self.device)
        
        # uniform between low and high
        rand = torch.rand((batch_size, len(low)), device=self.device)
        return low + rand * (high - low)

    def _sample_uniform_data_bounds(self, batch_size):
        alpha = self.cfg.distill.sampling.expansion if "expansion" in self.cfg.distill.sampling else 0.1
        
        span = self.max_dim - self.min_dim
        low = self.min_dim - alpha * span
        high = self.max_dim + alpha * span
        low = low.to(self.device)
        high = high.to(self.device)
        
        rand = torch.rand((batch_size, len(low)), device=self.device)
        return low + rand * (high - low)

    def _sample_gaussian(self, batch_size):
        std = self.cfg.distill.sampling.std if "std" in self.cfg.distill.sampling else 0.05
        # sample base states from trajectory first
        base_states = self._sample_trajectory(batch_size)
        noise = torch.randn_like(base_states) * std
        return base_states + noise

    def _sample_mixed(self, batch_size):
        mix_ratio = self.cfg.distill.sampling.mix_ratio if "mix_ratio" in self.cfg.distill.sampling else 0.5
        n_traj = int(batch_size * mix_ratio)
        n_synth = batch_size - n_traj
        
        traj_states = self._sample_trajectory(n_traj)
        
        # fallback synthetic mode for the other half, assume gaussian if not specified? 
        # The design says "50% trajectory, 50% synthetic". We'll default the synthetic part to uniform_data_bounds or gaussian.
        # For this implementation let's use gaussian if "synth_mode" is gaussian, else uniform_data_bounds.
        synth_mode = self.cfg.distill.sampling.get("synth_mode", "uniform_data_bounds")
        
        original_mode = self.mode
        self.mode = synth_mode
        try:
            synth_states = self.sample(n_synth)
        finally:
            self.mode = original_mode
            
        return torch.cat([traj_states, synth_states], dim=0)
