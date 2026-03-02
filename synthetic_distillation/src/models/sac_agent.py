import torch.nn as nn
import torch
import numpy as np
from src.models.agent import layer_init

class TwinQNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, neurons=256, layers=2):
        super().__init__()
        
        def build_q():
            modules = []
            in_dim = obs_dim + action_dim
            for _ in range(layers):
                modules.append(layer_init(nn.Linear(in_dim, neurons)))
                modules.append(nn.ReLU())
                in_dim = neurons
            modules.append(layer_init(nn.Linear(in_dim, 1), std=1.0))
            return nn.Sequential(*modules)
            
        self.q1 = build_q()
        self.q2 = build_q()

    def forward(self, state, action):
        xu = torch.cat([state, action], dim=1)
        return self.q1(xu), self.q2(xu)

class SACAgent(nn.Module):
    def __init__(self, envs, neurons=256, layers=2, log_sig_min=-20, log_sig_max=2, env_name=""):
        super().__init__()
        obs_dim = np.array(envs.single_observation_space.shape).prod()
        self.action_dim = np.prod(envs.single_action_space.shape)
        
        self.log_sig_min = log_sig_min
        self.log_sig_max = log_sig_max
        
        try:
            high = envs.envs[0].unwrapped.action_space.high
            low = envs.envs[0].unwrapped.action_space.low
        except:
            high = envs.single_action_space.high
            low = envs.single_action_space.low
            
        self.register_buffer("action_scale", torch.FloatTensor(
            (high - low) / 2.))
        self.register_buffer("action_bias", torch.FloatTensor(
            (high + low) / 2.))

        self.critic = TwinQNetwork(obs_dim, self.action_dim, neurons, layers)
        self.critic_target = TwinQNetwork(obs_dim, self.action_dim, neurons, layers)
        self.critic_target.load_state_dict(self.critic.state_dict())
        for param in self.critic_target.parameters():
            param.requires_grad = False

        # Actor
        modules = []
        in_dim = obs_dim
        for _ in range(layers):
            modules.append(layer_init(nn.Linear(in_dim, neurons)))
            modules.append(nn.ReLU())
            in_dim = neurons
        self.actor_trunk = nn.Sequential(*modules)
        
        self.actor_mean = layer_init(nn.Linear(neurons, self.action_dim), std=0.01)
        self.actor_logstd = layer_init(nn.Linear(neurons, self.action_dim), std=0.01)

    def forward(self, state):
        x = self.actor_trunk(state)
        mean = self.actor_mean(x)
        log_std = self.actor_logstd(x)
        log_std = torch.clamp(log_std, min=self.log_sig_min, max=self.log_sig_max)
        return mean, log_std

    def sample_action_and_compute_log_prob(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample() # Reparameterization trick
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing action bounds
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob

    def act(self, state, deterministic=True):
        mean, log_std = self.forward(state)
        if deterministic:
            action = torch.tanh(mean) * self.action_scale + self.action_bias
        else:
            std = log_std.exp()
            normal = torch.distributions.Normal(mean, std)
            x_t = normal.rsample()
            action = torch.tanh(x_t) * self.action_scale + self.action_bias
        return action
