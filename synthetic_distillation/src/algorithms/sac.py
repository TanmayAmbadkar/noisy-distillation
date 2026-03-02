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

class LinearLRSchedule:
    def __init__(self, optimizer: torch.optim.Optimizer, initial_lr: float, total_updates: int):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.total_updates = total_updates
        self.current_update = 0

    def step(self):
        self.current_update += 1
        frac = 1.0 - (self.current_update - 1.0) / self.total_updates
        lr = frac * self.initial_lr
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def get_lr(self):
        return [group["lr"] for group in self.optimizer.param_groups]

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
            self.writer.add_scalar("losses/qf1_loss", update_results.get("qf1_loss", 0), global_step)
            self.writer.add_scalar("losses/qf2_loss", update_results.get("qf2_loss", 0), global_step)
            self.writer.add_scalar("losses/policy_loss", update_results.get("policy_loss", 0), global_step)
            self.writer.add_scalar("losses/alpha_loss", update_results.get("alpha_loss", 0), global_step)
            self.writer.add_scalar("losses/alpha", update_results.get("alpha_value", 0), global_step)

class SAC:
    def __init__(
        self,
        agent,
        envs,
        learning_rate=3e-4,
        num_rollout_steps=128,
        num_envs=1,
        update_epochs=1,
        buffer_size=1000000,
        batch_size=256,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        automatic_entropy_tuning=True,
        target_update_interval=1,
        updates_per_step=1,
        start_steps=10000,
        anneal_lr=False,
        seed=1,
        logger=None,
    ):
        self.agent = agent
        self.envs = envs
        self.device = next(agent.parameters()).device
        self.seed = seed
        
        self.num_rollout_steps = num_rollout_steps
        self.num_envs = num_envs
        self.update_epochs = update_epochs
        
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
        
        self.initial_lr = learning_rate
        self.anneal_lr = anneal_lr
        self.critic_optim = Adam(self.agent.critic.parameters(), lr=learning_rate)
        
        # Policy optimizer: optimize actor parts only
        actor_params = list(self.agent.actor_trunk.parameters()) + list(self.agent.actor_mean.parameters()) + list(self.agent.actor_logstd.parameters())
        self.policy_optim = Adam(actor_params, lr=learning_rate)
        
        if self.automatic_entropy_tuning:
            # Target Entropy = −dim(A)
            self.target_entropy = -torch.prod(torch.Tensor(np.array(envs.single_action_space.shape)).to(self.device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optim = Adam([self.log_alpha], lr=learning_rate)

        self.lr_scheduler_critic = None
        self.lr_scheduler_policy = None
        self.lr_scheduler_alpha = None

    def create_lr_scheduler(self, num_policy_updates):
        self.lr_scheduler_critic = LinearLRSchedule(self.critic_optim, self.initial_lr, num_policy_updates)
        self.lr_scheduler_policy = LinearLRSchedule(self.policy_optim, self.initial_lr, num_policy_updates)
        if self.automatic_entropy_tuning:
            self.lr_scheduler_alpha = LinearLRSchedule(self.alpha_optim, self.initial_lr, num_policy_updates)

    def learn(self, total_timesteps):
        num_policy_updates = total_timesteps // (self.num_rollout_steps * self.num_envs)
        if num_policy_updates == 0:
            num_policy_updates = 1

        if self.anneal_lr:
            self.create_lr_scheduler(num_policy_updates)

        obs, _ = self._initialize_environment()
        
        for update in range(num_policy_updates):
            if self.anneal_lr:
                self.lr_scheduler_critic.step()
                self.lr_scheduler_policy.step()
                if self.automatic_entropy_tuning:
                    self.lr_scheduler_alpha.step()

            obs = self.collect_rollouts(obs)
            
            if len(self.memory) > self.batch_size:
                update_results = self.update_policy()
                self.logger.log_policy_update(update_results, self._global_step)
                    
        print(f"Training completed. Total steps: {self._global_step}")
        return self.agent

    def _initialize_environment(self):
        initial_observation, _ = self.envs.reset(seed=self.seed)
        return initial_observation, None

    def collect_rollouts(self, obs):
        for step in range(self.num_rollout_steps):
            if self._global_step < self.start_steps:
                action = np.array([self.envs.single_action_space.sample() for _ in range(self.envs.num_envs)])
            else:
                with torch.no_grad():
                    obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
                    action_tensor = self.agent.act(obs_tensor, deterministic=False)
                    action = action_tensor.cpu().numpy()
            
            next_obs, reward, terminations, truncations, infos = self.envs.step(action)
            
            dones = np.logical_or(terminations, truncations)
            masks = 1.0 - terminations # we mask only real terminations
                
            self.memory.push(obs, action, reward, next_obs, masks)
            
            obs = next_obs
            self._global_step += self.envs.num_envs
            
            self.logger.log_rollout_step(infos, self._global_step)
            
        return obs

    def update_policy(self):
        qf1_loss_list = []
        qf2_loss_list = []
        policy_loss_list = []
        alpha_loss_list = []
        alpha_value_list = []

        # Typically SAC updates 1 gradient step per env step collected.
        # So we run an equivalent ammount of updates_per_step.
        updates_to_run = self.update_epochs * self.num_envs * self.updates_per_step

        for _ in range(updates_to_run):
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

            qf1_loss_list.append(qf1_loss.item())
            qf2_loss_list.append(qf2_loss.item())
            policy_loss_list.append(policy_loss.item())
            alpha_loss_list.append(alpha_loss.item())
            alpha_value_list.append(alpha_value)

        return dict(
            qf1_loss=np.mean(qf1_loss_list),
            qf2_loss=np.mean(qf2_loss_list),
            policy_loss=np.mean(policy_loss_list),
            alpha_loss=np.mean(alpha_loss_list),
            alpha_value=np.mean(alpha_value_list)
        )
