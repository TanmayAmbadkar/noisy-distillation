import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
from uuid import uuid4
from torch.utils.tensorboard import SummaryWriter
import itertools

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

class SACLogger:
    def __init__(self, run_name: str = None, use_tensorboard: bool = False):
        self.use_tensorboard = use_tensorboard
        self.global_steps = []
        if self.use_tensorboard:
            run_name = str(uuid4()).hex if run_name is None else run_name
            self.writer = SummaryWriter(f"runs/{run_name}")

    def log_rollout_step(self, infos: dict, global_step: int):
        if "_episode" in infos and "episode" in infos:
            mask = infos["_episode"]
            if mask.any():
                for idx in range(len(mask)):
                    if mask[idx]:
                        r = infos["episode"]["r"][idx]
                        if "l" in infos["episode"] and "l" in infos["episode"]:
                            l = infos["episode"]["l"][idx]
                        else:
                            l = 0
                        
                        print(
                            f"global_step={global_step}, episodic_return={r:.2f}",
                            flush=True, # Need this for sbatch log outputs
                        )

                        if self.use_tensorboard:
                            self.writer.add_scalar("charts/episodic_return", r, global_step)
                            self.writer.add_scalar("charts/episodic_length", l, global_step)

    def log_policy_update(self, update_results: dict, global_step: int):
        if self.use_tensorboard:
            self.writer.add_scalar("losses/qf1_loss", update_results["qf1_loss"], global_step)
            self.writer.add_scalar("losses/qf2_loss", update_results["qf2_loss"], global_step)
            self.writer.add_scalar("losses/policy_loss", update_results["policy_loss"], global_step)
            self.writer.add_scalar("losses/alpha_loss", update_results["alpha_loss"], global_step)
            self.writer.add_scalar("losses/alpha", update_results["alpha_value"], global_step)

class SAC:
    def __init__(
        self,
        agent,
        envs,
        learning_rate=3e-4,
        buffer_size=1000000,
        batch_size=256,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        automatic_entropy_tuning=True,
        target_update_interval=1,
        updates_per_step=1,
        start_steps=10000,
        seed=1,
        logger=None,
    ):
        self.agent = agent
        self.envs = envs
        self.device = next(agent.parameters()).device
        self.seed = seed
        
        self.gamma = gamma
        self.tau = tau
        self.alpha = float(alpha)
        self.automatic_entropy_tuning = automatic_entropy_tuning
        self.target_update_interval = target_update_interval
        self.updates_per_step = updates_per_step
        self.start_steps = start_steps
        self.batch_size = batch_size
        
        self.logger = logger or SACLogger()
        self._global_step = 0
        
        from src.algorithms.replay_buffer import ReplayBuffer
        self.memory = ReplayBuffer(
            capacity=buffer_size,
            obs_dim=envs.single_observation_space.shape,
            action_dim=envs.single_action_space.shape,
            device=self.device
        )
        
        self.critic_optim = Adam(self.agent.critic.parameters(), lr=learning_rate)
        
        # Policy optimizer: optimize actor parts only
        actor_params = list(self.agent.actor_trunk.parameters()) + list(self.agent.actor_mean.parameters()) + list(self.agent.actor_logstd.parameters())
        self.policy_optim = Adam(actor_params, lr=learning_rate)
        
        if self.automatic_entropy_tuning:
            # Target Entropy = −dim(A)
            self.target_entropy = -torch.prod(torch.Tensor(np.array(envs.single_action_space.shape)).to(self.device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optim = Adam([self.log_alpha], lr=learning_rate)

    def learn(self, total_timesteps):
        obs, _ = self.envs.reset(seed=self.seed)
        
        update_interval = max(1, self.envs.num_envs)
        
        for step in range(total_timesteps):
            if step < self.start_steps:
                # Assuming simple envs interface
                action = np.array([self.envs.single_action_space.sample() for _ in range(self.envs.num_envs)])
            else:
                with torch.no_grad():
                    obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
                    action_tensor = self.agent.act(obs_tensor, deterministic=False)
                    action = action_tensor.cpu().numpy()
            
            next_obs, reward, terminations, truncations, infos = self.envs.step(action)
            
            # Mask tracking logic is complex in vectorized environment. 
            dones = np.logical_or(terminations, truncations)
            masks = 1.0 - terminations # we mask only real terminations
                
            self.memory.push(obs, action, reward, next_obs, masks)
            
            obs = next_obs
            self._global_step += self.envs.num_envs
            
            self.logger.log_rollout_step(infos, self._global_step)
            
            if len(self.memory) > self.batch_size:
                for idx in range(self.updates_per_step * self.envs.num_envs):
                    update_results = self.update_parameters()
                if step % update_interval == 0:
                    self.logger.log_policy_update(update_results, self._global_step)
                    
        print(f"Training completed. Total steps: {self._global_step}")
        return self.agent

    def update_parameters(self):
        batch = self.memory.sample(self.batch_size)
        state_batch = batch['observations']
        action_batch = batch['actions']
        reward_batch = batch['rewards']
        next_state_batch = batch['next_observations']
        mask_batch = batch['dones']

        with torch.no_grad():
            next_state_action, next_state_log_pi = self.agent.sample_action_and_compute_log_prob(next_state_batch)
            qf1_next_target, qf2_next_target = self.agent.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * min_qf_next_target

        qf1, qf2 = self.agent.critic(state_batch, action_batch)
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        pi, log_pi = self.agent.sample_action_and_compute_log_prob(state_batch)
        qf1_pi, qf2_pi = self.agent.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp().item()
            alpha_value = self.alpha
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_value = self.alpha

        soft_update(self.agent.critic_target, self.agent.critic, self.tau)

        return dict(
            qf1_loss=qf1_loss.item(),
            qf2_loss=qf2_loss.item(),
            policy_loss=policy_loss.item(),
            alpha_loss=alpha_loss.item(),
            alpha_value=alpha_value
        )
