import torch
import torch.nn as nn
from src.models.agent import DiscreteAgent, ContinuousAgent
from src.data.synthetic_sampler import SyntheticSampler
from functools import partial
import numpy as np

class Distiller:
    def __init__(self, cfg, device, logger):
        self.cfg = cfg
        self.device = device
        self.logger = logger
        self.env_type = cfg.env.type

    def _init_student(self, teacher, env, distil_neurons_fraction=1.0):
        teacher_neurons = self.cfg.model.get("neurons", 64)
        teacher_layers = self.cfg.model.get("layers", 2)
        
        distil_layers = self.cfg.model.get("distil_layers", teacher_layers)
        
        student_neurons = max(1, int(teacher_neurons * distil_neurons_fraction))
        
        if self.env_type in ["discrete", "atari"]:
            is_atari = getattr(self.cfg.env, "type", "") == "atari" or getattr(self.cfg.env, "name", "").startswith("ALE/") or "NoFrameskip" in getattr(self.cfg.env, "name", "")
            if is_atari:
                from src.models.agent import DiscreteConvAgent
                student = DiscreteConvAgent(env, scale=distil_neurons_fraction).to(self.device)
                print(f"Student Architecture: DiscreteConvAgent [Filters: {student.c1}, {student.c2}, {student.c3} | FC: {student.l1}] (scale: {distil_neurons_fraction})")
                
                # Try to extract teacher filters too
                from src.models.sb3_wrapper import SB3TeacherWrapper
                if isinstance(teacher, SB3TeacherWrapper):
                    try:
                        # SB3 NatureCNN structure: features_extractor.cnn
                        cnn = teacher.sb3_model.policy.features_extractor.cnn
                        filters = [layer.out_channels for layer in cnn if isinstance(layer, torch.nn.Conv2d)]
                        print(f"Teacher Architecture: SB3 NatureCNN [Filters: {', '.join(map(str, filters))}]")
                    except:
                        pass
                elif isinstance(teacher, DiscreteConvAgent):
                    print(f"Teacher Architecture: DiscreteConvAgent [Filters: {teacher.c1}, {teacher.c2}, {teacher.c3} | FC: {teacher.l1}]")
            else:
                from src.models.agent import DiscreteAgent
                student = DiscreteAgent(env, neurons=student_neurons, layers=distil_layers).to(self.device)
                print(f"Teacher Architecture: {teacher_layers} layers, {teacher_neurons} neurons")
                print(f"Student Architecture: {distil_layers} layers, {student_neurons} neurons (frac: {distil_neurons_fraction})")
        else:
            from src.models.agent import ContinuousAgent
            student = ContinuousAgent(env, rpo_alpha=None, neurons=student_neurons, layers=distil_layers).to(self.device)
            print(f"Teacher Architecture: {teacher_layers} layers, {teacher_neurons} neurons")
            print(f"Student Architecture: {distil_layers} layers, {student_neurons} neurons (frac: {distil_neurons_fraction})")
        return student

    def _collect_trajectory_buffer(self, teacher, env):
        # Sample trajectories using teacher policy to form the base buffer
        states = []
        obs, _ = env.reset()
        
        # Collect rollouts
        for _ in range(self.cfg.algo.rollout_steps):
            states.append(obs)
            
            with torch.no_grad():
                obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
                action = teacher.act(obs_t, deterministic=False)
            
            obs, _, _, _, _ = env.step(action.cpu().numpy())
            
        states = np.array(states)
        # Flatten num_steps and num_envs
        states = states.reshape(-1, *states.shape[2:])
        return torch.tensor(states, dtype=torch.float32)

    def _compute_loss_from_targets(self, student, batch_states, batch_targets):
        states = batch_states.to(self.device)
        targets = batch_targets.to(self.device)

        if self.env_type in ["discrete", "atari"]:
            student_logits = student(states)

            if torch.isnan(student_logits).any():
                print("WARNING: NaN detected in student logits")

            if self.cfg.distill.loss == "cross_entropy":
                loss = torch.nn.functional.cross_entropy(student_logits, targets)
            else:
                loss = torch.nn.functional.mse_loss(student_logits, targets)

        elif self.env_type == "continuous":
            student_mean, student_log_std = student(states)

            if torch.isnan(student_mean).any():
                print("WARNING: NaN detected in student mean")

            loss = torch.nn.functional.mse_loss(student_mean, targets)
                
        return loss

    def train(self, teacher, env):
        print(f"Starting Distillation in mode {self.cfg.distill.mode}...")
        
        # 1. Collect base trajectory buffer using the teacher
        trajectory_states = self._collect_trajectory_buffer(teacher, env)
        
        # Check for multiple sampling configurations
        sampling_configs = self.cfg.distill.get("sampling_list", None)
        if sampling_configs is None:
            # Fallback for single sampling config
            single_sampling_cfg = self.cfg.distill.get("sampling", {"mode": "trajectory"})
            if "name" not in single_sampling_cfg:
                single_sampling_cfg = dict(single_sampling_cfg)
                single_sampling_cfg["name"] = single_sampling_cfg.get("mode", "trajectory")
            sampling_configs = [single_sampling_cfg]

        try:
            sampling_configs = [dict(c) for c in sampling_configs]
        except:
            pass
            
        distil_samples_list = self.cfg.distill.get("distil_samples", [len(trajectory_states)])
        if isinstance(distil_samples_list, (float, int)):
            distil_samples_list = [int(distil_samples_list)]
        else:
            try:
                distil_samples_list = [int(s) for s in list(distil_samples_list)]
            except:
                distil_samples_list = [int(distil_samples_list)]

        distil_neurons_fractions = self.cfg.model.get("distil_neurons", [1.0])
        if isinstance(distil_neurons_fractions, (float, int)):
            distil_neurons_fractions = [distil_neurons_fractions]
        else:
            try:
                distil_neurons_fractions = list(distil_neurons_fractions)
            except:
                distil_neurons_fractions = [distil_neurons_fractions]

        students = []
        student_idx = 1
        total_combinations = len(sampling_configs) * len(distil_samples_list) * len(distil_neurons_fractions)

        for noise_cfg in sampling_configs:
            noise_name = noise_cfg.get("name", noise_cfg.get("mode", "unknown"))
            print(f"\n--- Generating dataset for sampling distribution: {noise_name} ---")
            
            # Instantiate SyntheticSampler
            sampler = SyntheticSampler(self.cfg, trajectory_states=trajectory_states, device="cpu", logger=self.logger, override_cfg=noise_cfg)
            
            max_samples_needed = max(distil_samples_list)
            print(f"Generating fixed dataset of max {max_samples_needed} samples for {noise_name}...")
            
            fixed_states = []
            fixed_targets = []
            samples_collected = 0
            batch_size = self.cfg.distill.batch_size
            
            while samples_collected < max_samples_needed:
                current_batch = min(batch_size, max_samples_needed - samples_collected)
                batch_states = sampler.sample(current_batch)
                
                with torch.no_grad():
                    batch_states_dev = batch_states.to(self.device)
                    if self.env_type in ["discrete", "atari"]:
                        out = teacher(batch_states_dev)
                        if self.cfg.distill.loss == "cross_entropy":
                            out = out.argmax(dim=1)
                    else:
                        mean, log_std = teacher(batch_states_dev)
                        if self.cfg.distill.loss == "sample_mse":
                            std = torch.exp(log_std)
                            out = mean + std * torch.randn_like(std)
                        else:
                            out = mean
                            
                fixed_states.append(batch_states.cpu())
                fixed_targets.append(out.cpu())
                samples_collected += current_batch
                
            fixed_states = torch.cat(fixed_states, dim=0)
            fixed_targets = torch.cat(fixed_targets, dim=0)

            for n_samples in distil_samples_list:
                for frac in distil_neurons_fractions:
                    print(f"\n[{student_idx}/{total_combinations}] Initializing Student with frac={frac}, samples={n_samples}, noise={noise_name} - student_{student_idx}")
                    student = self._init_student(teacher, env, distil_neurons_fraction=frac)

                    student.metadata = getattr(student, "metadata", {})
                    student.metadata["noise_name"] = noise_name

                    lr = self.cfg.algo.lr
                    optimizer = torch.optim.Adam(student.parameters(), lr=lr)
                    batches_per_epoch = max(1, n_samples // batch_size)

                    for epoch in range(self.cfg.distill.epochs):
                        total_loss = 0
                        indices = torch.randperm(n_samples)
                        
                        for step in range(batches_per_epoch):
                            batch_idx = indices[step * batch_size : (step + 1) * batch_size]
                            batch_states_step = fixed_states[batch_idx]
                            batch_targets_step = fixed_targets[batch_idx]
                            
                            loss = self._compute_loss_from_targets(student, batch_states_step, batch_targets_step)

                            optimizer.zero_grad()
                            loss.backward()
                            nn.utils.clip_grad_norm_(student.parameters(), max_norm=0.5)
                            optimizer.step()
                            total_loss += loss.item()

                        avg_loss = total_loss / batches_per_epoch if batches_per_epoch > 0 else total_loss
                        if epoch % self.cfg.log.log_interval == 0 and self.logger is not None:
                            self.logger.log_scalar(f"student_{student_idx}_{noise_name}/distill_loss", avg_loss, epoch)
                            print(f"Distillation (frac={frac}, samples={n_samples}, noise={noise_name}) epoch {epoch}/{self.cfg.distill.epochs} - Loss: {avg_loss:.4f}")

                    students.append(student)
                    student_idx += 1

        return students
