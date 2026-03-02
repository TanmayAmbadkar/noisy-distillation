from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
from typing import Optional


def layer_init(
    layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0
) -> nn.Module:
    """
    Initializes a neural network layer with orthogonal weights and constant bias.

    Args:
        layer (nn.Module): The neural network layer to initialize.
        std (float, optional): The standard deviation for the orthogonal initialization. Defaults to sqrt(2).
        bias_const (float, optional): The constant value for the bias initialization. Defaults to 0.0.

    Returns:
        nn.Module: The initialized neural network layer.
    """
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

def make_mlp(input_dim, output_dim, neurons, layers, last_std=0.01):
    modules = []
    in_dim = input_dim
    for _ in range(layers):
        modules.append(layer_init(nn.Linear(in_dim, neurons)))
        modules.append(nn.Tanh())
        in_dim = neurons
    modules.append(layer_init(nn.Linear(in_dim, output_dim), std=last_std))
    return nn.Sequential(*modules)


class BaseAgent(nn.Module, ABC):
    @abstractmethod
    def estimate_value_from_observation(self, observation):
        """
        Estimate the value of an observation using the critic network.

        Args:
            observation: The observation to estimate.

        Returns:
            The estimated value of the observation.
        """
        pass

    @abstractmethod
    def get_action_distribution(self, observation):
        """
        Get the action distribution for a given observation.

        Args:
            observation: The observation to base the action distribution on.

        Returns:
            A probability distribution over possible actions.
        """
        pass

    @abstractmethod
    def sample_action_and_compute_log_prob(self, observations):
        """
        Sample an action from the action distribution and compute its log probability.

        Args:
            observations: The observations to base the actions on.

        Returns:
            A tuple containing:
            - The sampled action(s)
            - The log probability of the sampled action(s)
        """
        pass

    @abstractmethod
    def compute_action_log_probabilities_and_entropy(
        self, observations, actions
    ):
        """
        Compute the log probabilities and entropy of given actions for given observations.

        Args:
            observations: The observations corresponding to the actions.
            actions: The actions to compute probabilities and entropy for.

        Returns:
            A tuple containing:
            - The log probabilities of the actions
            - The entropy of the action distribution
        """
        pass


class DiscreteAgent(BaseAgent):
    """
    An agent for environments with discrete action spaces.

    This agent uses separate Multi-Layer Perceptron (MLP) networks for the
    actor (policy) and critic (value function).
    """

    def __init__(self, envs, neurons: int = 64, layers: int = 2, **kwargs):
        """
        Initializes the DiscreteAgent.

        Args:
            envs: The vectorized environment(s), used to determine observation and action dimensions.
        """
        super().__init__()
        obs_dim = np.array(envs.single_observation_space.shape).prod()
        action_dim = envs.single_action_space.n

        self.critic = make_mlp(obs_dim, 1, neurons, layers, last_std=1.0)
        self.actor = make_mlp(obs_dim, action_dim, neurons, layers, last_std=0.01)

    def forward(self, states):
        return self.actor(states)

    def act(self, state, deterministic=True):
        logits = self.forward(state)
        if deterministic:
            return logits.argmax(dim=-1)
        else:
            return Categorical(logits=logits).sample()

    def estimate_value_from_observation(
        self, observation: torch.Tensor
    ) -> torch.Tensor:
        return self.critic(observation)

    def get_action_distribution(self, observation):
        logits = self.actor(observation)
        return Categorical(logits=logits)

    def sample_action_and_compute_log_prob(self, observations):
        action_dist = self.get_action_distribution(observations)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        return action, log_prob

    def compute_action_log_probabilities_and_entropy(
        self, observations, actions
    ):
        action_dist = self.get_action_distribution(observations)
        log_prob = action_dist.log_prob(actions)
        entropy = action_dist.entropy()
        return log_prob, entropy

class DiscreteConvAgent(BaseAgent):
    """
    An agent for environments with image-based discrete action spaces (e.g. Atari).
    Uses the standard Nature CNN architecture. It optionally scales down the channel
    and linear dimensions for student distillation models.
    """
    def __init__(self, envs, scale: float = 1.0, **kwargs):
        super().__init__()
        
        # Assumption: envs.single_observation_space is image shaped like (C, H, W)
        action_dim = envs.single_action_space.n

        c1 = max(1, int(32 * scale))
        c2 = max(1, int(64 * scale))
        c3 = max(1, int(64 * scale))
        l1 = max(1, int(512 * scale))

        self.network = nn.Sequential(
            layer_init(nn.Conv2d(envs.single_observation_space.shape[0], c1, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(c1, c2, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(c2, c3, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, *envs.single_observation_space.shape)
            flattened_size = self.network(dummy).shape[1]

        self.fc = nn.Sequential(
            layer_init(nn.Linear(flattened_size, l1)),
            nn.ReLU(),
        )

        self.actor = layer_init(nn.Linear(l1, action_dim), std=0.01)
        self.critic = layer_init(nn.Linear(l1, 1), std=1.0)
        
    def _get_features(self, x):
        hidden = self.network(x / 255.0)
        return self.fc(hidden)

    def forward(self, states):
        hidden = self._get_features(states)
        return self.actor(hidden)

    def act(self, state, deterministic=True):
        logits = self.forward(state)
        if deterministic:
            return logits.argmax(dim=-1)
        else:
            return Categorical(logits=logits).sample()

    def estimate_value_from_observation(self, observation: torch.Tensor) -> torch.Tensor:
        hidden = self._get_features(observation)
        return self.critic(hidden)

    def get_action_distribution(self, observation):
        logits = self.forward(observation)
        return Categorical(logits=logits)

    def sample_action_and_compute_log_prob(self, observations):
        action_dist = self.get_action_distribution(observations)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        return action, log_prob

    def compute_action_log_probabilities_and_entropy(self, observations, actions):
        action_dist = self.get_action_distribution(observations)
        log_prob = action_dist.log_prob(actions)
        entropy = action_dist.entropy()
        return log_prob, entropy


class ContinuousAgent(BaseAgent):
    """
    An agent for environments with continuous action spaces.

    This agent uses a parameterized normal distribution for its policy.
    The mean of the distribution is output by an MLP, and the log standard
    deviation is learned as a state-independent parameter.
    """

    def __init__(self, envs, rpo_alpha: Optional[float] = None, neurons: int = 64, layers: int = 2, env_name: str = ""):
        """
        Initializes the ContinuousAgent.

        Args:
            envs: The vectorized environment(s), used to determine observation and action dimensions.
            rpo_alpha (float, optional): Alpha parameter for Regularized Policy Optimization.
                                         Adds stochasticity during action log prob computation.
            env_name (str): Name of the environment to set specific parameters.
        """
        super().__init__()
        self.rpo_alpha = rpo_alpha
        obs_dim = np.array(envs.single_observation_space.shape).prod()
        action_dim = np.prod(envs.single_action_space.shape)

        self.critic = make_mlp(obs_dim, 1, neurons, layers, last_std=1.0)
        self.actor_mean = make_mlp(obs_dim, action_dim, neurons, layers, last_std=0.01)
        
        self.actor_logstd = nn.Parameter(torch.ones(1, action_dim))

    def forward(self, states):
        action_mean = self.actor_mean(states)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        return action_mean, action_logstd

    def act(self, state, deterministic=True):
        mean, log_std = self.forward(state)
        if deterministic:
            return mean
        else:
            std = torch.exp(log_std)
            return Normal(mean, std).sample()

    def estimate_value_from_observation(
        self, observation: torch.Tensor
    ) -> torch.Tensor:
        return self.critic(observation)

    def get_action_distribution(self, observation):
        action_mean = self.actor_mean(observation)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        action_dist = Normal(action_mean, action_std)

        return action_dist

    def sample_action_and_compute_log_prob(self, observations):
        action_dist = self.get_action_distribution(observations)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action).sum(1)
        return action, log_prob

    def compute_action_log_probabilities_and_entropy(
        self, observations, actions
    ):
        action_dist = self.get_action_distribution(observations)

        if self.rpo_alpha is not None:
            # sample again to add stochasticity to the policy
            action_mean = action_dist.mean
            z = (
                torch.FloatTensor(action_mean.shape)
                .uniform_(-self.rpo_alpha, self.rpo_alpha)
                .to(self.actor_logstd.device)
            )
            action_mean = action_mean + z
            action_dist = Normal(action_mean, action_dist.stddev)

        log_prob = action_dist.log_prob(actions).sum(1)
        entropy = action_dist.entropy().sum(1)
        return log_prob, entropy
