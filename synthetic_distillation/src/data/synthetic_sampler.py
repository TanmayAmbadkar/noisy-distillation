import torch
import numpy as np
from scipy.ndimage import gaussian_filter

class SyntheticSampler:
    def __init__(self, cfg, trajectory_states=None, device="cpu", logger=None, **kwargs):
        """
        cfg: The hydra config object.
        trajectory_states: A tensor of shape [N, state_dim] containing states collected from the teacher rollout.
        device: Torch device.
        logger: TBLogger instance for logging sampling statistics.
        """
        self.cfg = cfg
        self.override_cfg = kwargs.get('override_cfg', {})
        sampling_cfg = cfg.distill.get("sampling", {})
        self.mode = self.override_cfg.get("mode") or (sampling_cfg.get("mode", "trajectory"))
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
        mode = self.mode # Use local var to avoid issues with recursion in mixture
        if mode == "trajectory":
            return self._sample_trajectory(batch_size)
        elif mode == "uniform_global":
            return self._sample_uniform_global(batch_size)
        elif mode == "uniform_data_bounds":
            return self._sample_uniform_data_bounds(batch_size)
        elif mode == "gaussian":
            return self._sample_gaussian(batch_size)
        elif mode == "gaussian_global":
            return self._sample_gaussian_global(batch_size)
        elif mode == "laplace":
            return self._sample_laplace(batch_size)
        elif mode == "mixed":
            return self._sample_mixed(batch_size)
        elif mode == "noise_mixture":
            return self._sample_noise_mixture(batch_size)
        elif mode == "st_gaussian":
            return self._sample_st_gaussian(batch_size)
        elif mode == "st_ar1":
            return self._sample_st_ar1(batch_size)
        elif mode == "st_moving":
            return self._sample_st_moving(batch_size)
        else:
            raise ValueError(f"Unknown sampling mode: {mode}")

    def _sample_trajectory(self, batch_size):
        indices = torch.randint(0, len(self.trajectory_states), (batch_size,))
        return self.trajectory_states[indices].to(self.device)

    def _sample_uniform_global(self, batch_size):
        low_val = self.override_cfg.get("low") if "low" in self.override_cfg else (self.cfg.distill.sampling.get("low", 0.0) if "sampling" in self.cfg.distill else 0.0)
        high_val = self.override_cfg.get("high") if "high" in self.override_cfg else (self.cfg.distill.sampling.get("high", 1.0) if "sampling" in self.cfg.distill else 1.0)
        
        # If trajectory states are available, infer the exact state shape
        if self.trajectory_states is not None:
            state_shape = self.trajectory_states.shape[1:]
        else:
            state_shape = (1,)  # Fallback
            
        # Create uniform random tensor of the correct shape
        rand = torch.rand((batch_size, *state_shape), device=self.device)
        return low_val + rand * (high_val - low_val)

    def _sample_uniform_data_bounds(self, batch_size):
        sampling_cfg = self.cfg.distill.get("sampling", {})
        alpha = sampling_cfg.get("expansion", 0.1)
        
        span = self.max_dim - self.min_dim
        low = self.min_dim - alpha * span
        high = self.max_dim + alpha * span
        low = low.to(self.device).unsqueeze(0)
        high = high.to(self.device).unsqueeze(0)
        
        state_shape = self.trajectory_states.shape[1:]
        rand = torch.rand((batch_size, *state_shape), device=self.device)
        return low + rand * (high - low)

    def _sample_gaussian(self, batch_size):
        mean = self.override_cfg.get("mean") if "mean" in self.override_cfg else (self.cfg.distill.sampling.get("mean", 0.0) if "sampling" in self.cfg.distill else 0.0)
        
        # Support both std and variance (std = sqrt(variance))
        if "variance" in self.override_cfg:
            std = np.sqrt(self.override_cfg["variance"])
        elif "std" in self.override_cfg:
            std = self.override_cfg["std"]
        elif "sampling" in self.cfg.distill:
            sampling_cfg = self.cfg.distill.get("sampling", {})
            if "variance" in sampling_cfg:
                std = np.sqrt(sampling_cfg.get("variance"))
            else:
                std = sampling_cfg.get("std", 1.0)
        else:
            std = 1.0
        
        if self.trajectory_states is not None:
            state_shape = self.trajectory_states.shape[1:]
        else:
            state_shape = (1,)
            
        return mean + torch.randn((batch_size, *state_shape), device=self.device) * std
        
    def _sample_gaussian_global(self, batch_size):
        mean = self.override_cfg.get("mean") if "mean" in self.override_cfg else 0.0
        std = self.override_cfg.get("std") if "std" in self.override_cfg else 1.0
        
        if self.trajectory_states is not None:
            state_shape = self.trajectory_states.shape[1:]
        else:
            state_shape = (1,)
            
        return mean + torch.randn((batch_size, *state_shape), device=self.device) * std
        
    def _sample_laplace(self, batch_size):
        sampling_cfg = self.cfg.distill.get("sampling", {})
        mean = self.override_cfg.get("mean") if "mean" in self.override_cfg else (sampling_cfg.get("mean", 0.0))
        scale = self.override_cfg.get("scale") if "scale" in self.override_cfg else (sampling_cfg.get("scale", 1.0))
        
        if self.trajectory_states is not None:
            state_shape = self.trajectory_states.shape[1:]
        else:
            state_shape = (1,)
            
        from torch.distributions.laplace import Laplace
        loc = torch.full((batch_size, *state_shape), mean, device=self.device)
        scale_tensor = torch.full((batch_size, *state_shape), scale, device=self.device)
        laplace = Laplace(loc=loc, scale=scale_tensor)
        return laplace.sample()

    def _sample_mixed(self, batch_size):
        sampling_cfg = self.cfg.distill.get("sampling", {})
        mix_ratio = sampling_cfg.get("mix_ratio", 0.5)
        n_traj = int(batch_size * mix_ratio)
        n_synth = batch_size - n_traj
        
        traj_states = self._sample_trajectory(n_traj)
        
        # fallback synthetic mode for the other half, assume gaussian if not specified? 
        # The design says "50% trajectory, 50% synthetic". We'll default the synthetic part to uniform_data_bounds or gaussian.
        # For this implementation let's use gaussian if "synth_mode" is gaussian, else uniform_data_bounds.
        synth_mode = self.override_cfg.get("synth_mode") or sampling_cfg.get("synth_mode", "gaussian")
        
        original_mode = self.mode
        self.mode = synth_mode
        try:
            synth_states = self.sample(n_synth)
        finally:
            self.mode = original_mode
            
        return torch.cat([traj_states, synth_states], dim=0)

    def _sample_noise_mixture(self, batch_size):
        """
        Samples from a list of noise distributions provided in override_cfg['components'].
        If not provided, falls back to a default set of 7 distributions.
        """
        components = self.override_cfg.get("components")
        if not components:
            # Default empty list if none provided, caller should provide them
            return torch.zeros((batch_size, *self.trajectory_states.shape[1:]), device=self.device)

        n_components = len(components)
        samples_per_component = batch_size // n_components
        remainder = batch_size % n_components
        
        all_samples = []
        original_override = self.override_cfg
        original_mode = self.mode
        
        try:
            for i, comp_cfg in enumerate(components):
                n_to_sample = samples_per_component + (1 if i < remainder else 0)
                if n_to_sample <= 0:
                    continue
                
                # Temporarily update sampler state
                self.override_cfg = comp_cfg
                self.mode = comp_cfg.get("mode", "gaussian")
                
                all_samples.append(self.sample(n_to_sample))
        finally:
            self.override_cfg = original_override
            self.mode = original_mode
            

    def _sample_st_gaussian(self, batch_size):
        sampling_cfg = self.cfg.distill.get("sampling", {})
        sigma_t = self.override_cfg.get("sigma_t") or sampling_cfg.get("sigma_t", 1.0)
        sigma_s = self.override_cfg.get("sigma_s") or sampling_cfg.get("sigma_s", 2.0)
        
        if self.trajectory_states is not None:
            state_shape = self.trajectory_states.shape[1:]
        else:
            state_shape = (4, 84, 84) # Atari default
            
        # 1. Generate pure white noise
        noise = np.random.normal(0, 1, size=(batch_size, *state_shape))
        
        # 2. Apply 3D Spatiotemporal Blur
        # sigma tuple: (batch, temporal, spatial_y, spatial_x)
        # Assuming shape is (T, H, W)
        if len(state_shape) == 3:
            noise = gaussian_filter(noise, sigma=(0, sigma_t, sigma_s, sigma_s))
        else:
            # Fallback for non-Atari or flattened
            noise = gaussian_filter(noise, sigma=(0, sigma_s))
            
        # Normalize to [0, 1] range if it was Atari (uint8-like)
        # or leave as N(0,1) for MuJoCo? 
        # For now, let's keep it N(0,1) but blurred. 
        # If the user wants 0-255 they can specify low/high in a wrapper or we can add it.
        return torch.from_numpy(noise).float().to(self.device)

    def _sample_st_ar1(self, batch_size):
        sampling_cfg = self.cfg.distill.get("sampling", {})
        rho = self.override_cfg.get("rho") or sampling_cfg.get("rho", 0.9)
        sigma = self.override_cfg.get("sigma") or sampling_cfg.get("sigma", 0.5)
        
        if self.trajectory_states is not None:
            state_shape = self.trajectory_states.shape[1:]
        else:
            state_shape = (4, 84, 84)
            
        T = state_shape[0] if len(state_shape) == 3 else 1
        spatial_shape = state_shape[1:] if len(state_shape) == 3 else state_shape
        
        frames = []
        # Initial frame (t=0)
        current_frame = np.random.normal(0, 1, size=(batch_size, *spatial_shape))
        frames.append(current_frame)
        
        for t in range(1, T):
            innovation = np.random.normal(0, 1, size=(batch_size, *spatial_shape))
            current_frame = rho * current_frame + sigma * innovation
            frames.append(current_frame)
            
        noise = np.stack(frames, axis=1) if len(state_shape) == 3 else frames[0]
        return torch.from_numpy(noise).float().to(self.device)

    def _sample_st_moving(self, batch_size):
        """
        Translating Blob: Generates a 2D Gaussian blob that moves linearly across frames.
        """
        if self.trajectory_states is not None:
            state_shape = self.trajectory_states.shape[1:]
        else:
            state_shape = (4, 84, 84)
            
        T, H, W = state_shape if len(state_shape) == 3 else (1, 84, 84)
        
        noise = np.zeros((batch_size, T, H, W))
        
        for b in range(batch_size):
            # Random starting position
            y, x = np.random.randint(0, H), np.random.randint(0, W)
            # Random velocity
            vy, vx = np.random.randint(-3, 4), np.random.randint(-3, 4)
            # Random size
            blob_sigma = np.random.uniform(2, 5)
            
            for t in range(T):
                # Update position
                cur_y = (y + vy * t) % H
                cur_x = (x + vx * t) % W
                
                # Create a single pixel impulse
                frame = np.zeros((H, W))
                frame[int(cur_y), int(cur_x)] = 1.0
                
                # Blur to create blob
                noise[b, t] = gaussian_filter(frame, sigma=blob_sigma)
        
        # Normalize/Scale to be visible (N(0,1) roughly)
        noise = (noise - noise.mean()) / (noise.std() + 1e-6)
        return torch.from_numpy(noise).float().to(self.device)
