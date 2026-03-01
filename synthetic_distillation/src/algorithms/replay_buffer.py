import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: tuple, action_dim: tuple, device: torch.device):
        self.capacity = capacity
        self.device = device
        
        # Preallocate memory
        self.observations = np.zeros((capacity, *obs_dim), dtype=np.float32)
        self.next_observations = np.zeros((capacity, *obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, *action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        
        self.ptr = 0
        self.size = 0

    def push(self, obs, action, reward, next_obs, done):
        """
        Push a batch of transitions (from vectorized envs).
        obs: numpy array of shape (num_envs, *obs_dim)
        action: numpy array of shape (num_envs, *action_dim)
        reward: numpy array of shape (num_envs,)
        next_obs: numpy array of shape (num_envs, *obs_dim)
        done: numpy array of shape (num_envs,)
        """
        num_envs = obs.shape[0]
        
        for i in range(num_envs):
            self.observations[self.ptr] = obs[i]
            self.next_observations[self.ptr] = next_obs[i]
            self.actions[self.ptr] = action[i]
            self.rewards[self.ptr] = reward[i].reshape(-1)
            self.dones[self.ptr] = done[i].reshape(-1)
            
            self.ptr = (self.ptr + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        
        return dict(
            observations=torch.as_tensor(self.observations[idxs]).to(self.device),
            actions=torch.as_tensor(self.actions[idxs]).to(self.device),
            rewards=torch.as_tensor(self.rewards[idxs]).to(self.device),
            next_observations=torch.as_tensor(self.next_observations[idxs]).to(self.device),
            dones=torch.as_tensor(self.dones[idxs]).to(self.device)
        )
    
    def __len__(self):
        return self.size
