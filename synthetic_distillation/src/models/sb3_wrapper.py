import torch
import torch.nn as nn
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvStepReturn

class GymVectorEnvToSB3VecEnv(VecEnv):
    """
    Adapter to convert a Gymnasium VectorEnv (like SyncVectorEnv) into an SB3 VecEnv.
    SB3 natively supports vectorized environments only via its own VecEnv interface.
    """
    def __init__(self, venv):
        super().__init__(venv.num_envs, venv.single_observation_space, venv.single_action_space)
        self.venv = venv
        
    def reset(self):
        obs, _ = self.venv.reset()
        return obs
        
    def step_async(self, actions):
        self.actions = actions
        
    def step_wait(self):
        obs, rewards, terminals, truncateds, infos = self.venv.step(self.actions)
        dones = terminals | truncateds
        
        # Convert Gymnasium vectorized info dict to list of dicts (SB3 format)
        list_infos = []
        for i in range(self.num_envs):
            info = {}
            if "final_info" in infos and infos["_final_info"][i]:
                info = infos["final_info"][i]
                if "final_observation" in infos and infos["_final_observation"][i]:
                    info["terminal_observation"] = infos["final_observation"][i]
            
            # RecordEpisodeStatistics wrapper puts episode info here
            if "episode" in infos and infos["_episode"][i]:
                 info["episode"] = {"r": infos["episode"]["r"][i], "l": infos["episode"]["l"][i], "t": infos["episode"]["t"][i]}
                 
            list_infos.append(info)
            
        return obs, rewards, dones, list_infos
        
    def close(self):
        self.venv.close()
        
    def get_attr(self, attr_name, indices=None):
        try:
            return self.venv.get_attr(attr_name)
        except Exception:
            raise AttributeError(attr_name)
        
    def set_attr(self, attr_name, value, indices=None):
        try:
            self.venv.set_attr(attr_name, value)
        except Exception:
            pass
        
    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        pass
        
    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_envs


class SB3TeacherWrapper(nn.Module):
    """
    A wrapper around a Stable Baselines 3 (SB3) model to make it compatible
    with the custom teacher evaluation and distillation pipelines expected in main.py.
    
    This ensures `act(obs, deterministic)` and `state_dict()` calls work as expected.
    """
    def __init__(self, sb3_model: BaseAlgorithm):
        super().__init__()
        self.sb3_model = sb3_model
        
    def act(self, obs: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        """
        Mimics the `act` signature of the custom agents.
        Takes PyTorch tensors and returns PyTorch tensors.
        """
        # Ensure obs is on CPU and as numpy array for SB3
        obs_np = obs.detach().cpu().numpy()
        
        # predict() returns a tuple: (action, state)
        actions, _ = self.sb3_model.predict(obs_np, deterministic=deterministic)
        
        # Return action back to original device as tensor
        return torch.tensor(actions, device=obs.device)
        
    def forward(self, obs: torch.Tensor):
        """
        Mimics the `forward` signature of custom agents, which the Distiller expects.
        Returns the action distribution (mean, log_std) for continuous envs,
        or raw logits/Q-values for discrete envs.
        """
        from stable_baselines3 import PPO, SAC, DDPG, DQN
        try:
            from sb3_contrib import TRPO
            has_trpo = True
        except ImportError:
            has_trpo = False
        
        # We need the observation on the device SB3 is using locally
        obs_dev = obs.to(self.sb3_model.device)
        
        if isinstance(self.sb3_model, PPO) or (has_trpo and isinstance(self.sb3_model, TRPO)):
            dist = self.sb3_model.policy.get_distribution(obs_dev)
            if hasattr(dist.distribution, "loc"):
                # Continuous
                mean = dist.distribution.loc
                log_std = dist.distribution.scale.log()
                return mean, log_std
            else:
                # Discrete (Categorical)
                return dist.distribution.logits
            
        elif isinstance(self.sb3_model, SAC):
            mean, log_std, _ = self.sb3_model.policy.actor.get_action_dist_params(obs_dev)
            return mean, log_std
            
        elif isinstance(self.sb3_model, DDPG):
            # DDPG is deterministic. Return the action as mean, and a dummy tight log_std
            mean = self.sb3_model.policy(obs_dev)
            # -20 corresponds to a very small standard deviation (exp(-20) ~ 2e-9)
            log_std = torch.ones_like(mean) * -20.0
            return mean, log_std
            
        elif isinstance(self.sb3_model, DQN):
            # DQN is discrete. Distiller expects logits to compute cross_entropy.
            # Q-values can serve directly as logits.
            q_values = self.sb3_model.q_net(obs_dev)
            return q_values
            
        else:
            raise NotImplementedError(f"forward() not implemented for SB3 model type {type(self.sb3_model)}")

    def state_dict(self, *args, **kwargs):
        """
        Exposes the policy weights so saving/loading checkpoints works normally.
        """
        return self.sb3_model.policy.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, *args, **kwargs):
        """
        Loads the policy weights.
        """
        return self.sb3_model.policy.load_state_dict(state_dict, *args, **kwargs)
