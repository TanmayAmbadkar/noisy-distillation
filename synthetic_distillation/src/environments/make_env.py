import gymnasium as gym
import numpy as np
import types
import copy

def make_discrete_env(env_id, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env
    return thunk

def make_continuous_env(env_id, idx, capture_video, run_name, gamma):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10), env.observation_space)
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env
    return thunk

def get_NormalizeObservation_wrapper(self, env_num=0):
    return self.envs[env_num].env.env.env

def get_obs_norm_rms_obj(self, env_num=0):
    return self.get_NormalizeObservation_wrapper(env_num=env_num).obs_rms

def set_obs_norm_rms_obj(self, rms_obj, env_num=0):
    self.get_NormalizeObservation_wrapper(env_num=env_num).obs_rms = rms_obj

def freeze_norm_stats(self, env_num=0):
    wrapper = self.get_NormalizeObservation_wrapper(env_num=env_num)
    if hasattr(wrapper, "obs_rms") and not hasattr(wrapper.obs_rms, "frozen_update"):
        wrapper.obs_rms.frozen_update = wrapper.obs_rms.update
        wrapper.obs_rms.update = lambda x: None

def unfreeze_norm_stats(self, env_num=0):
    wrapper = self.get_NormalizeObservation_wrapper(env_num=env_num)
    if hasattr(wrapper, "obs_rms") and hasattr(wrapper.obs_rms, "frozen_update"):
        wrapper.obs_rms.update = wrapper.obs_rms.frozen_update
        del wrapper.obs_rms.frozen_update

def sync_obs_norm_rms(self, target_envs):
    for i in range(min(self.num_envs, target_envs.num_envs)):
        target_envs.set_obs_norm_rms_obj(copy.deepcopy(self.get_obs_norm_rms_obj(env_num=i)), env_num=i)

def make_env(env_cfg, num_envs=1, seed=0, capture_video=False, run_name="run", gamma=0.99):
    env_id = env_cfg.name
    if env_cfg.type == "atari":
        from ale_py.vector_env import AtariVectorEnv
        
        # Build kwargs using sensible defaults and config overrides
        kwargs = dict(
            game=env_id,
            num_envs=num_envs,
            frameskip=4,
            grayscale=True,
            stack_num=4,
            img_height=84,
            img_width=84,
            maxpool=True,
            reward_clipping=True,
            noop_max=30,
            use_fire_reset=True,
            episodic_life=False,
            life_loss_info=False,
            num_threads=1,
            repeat_action_probability=0.0
        )
        
        envs = AtariVectorEnv(**kwargs)
        envs.action_space.seed(seed)
        envs.observation_space.seed(seed)
        return envs
        
    env_is_discrete = env_cfg.type == "discrete"

    if env_is_discrete:
        envs = gym.vector.SyncVectorEnv(
            [make_discrete_env(env_id, i, capture_video, run_name) for i in range(num_envs)]
        )
    else:
        envs = gym.vector.SyncVectorEnv(
            [make_continuous_env(env_id, i, capture_video, run_name, gamma) for i in range(num_envs)]
        )
        # Bind methods for continuous environments
        envs.get_NormalizeObservation_wrapper = types.MethodType(get_NormalizeObservation_wrapper, envs)
        envs.get_obs_norm_rms_obj = types.MethodType(get_obs_norm_rms_obj, envs)
        envs.set_obs_norm_rms_obj = types.MethodType(set_obs_norm_rms_obj, envs)
        envs.freeze_norm_stats = types.MethodType(freeze_norm_stats, envs)
        envs.unfreeze_norm_stats = types.MethodType(unfreeze_norm_stats, envs)
        envs.sync_obs_norm_rms = types.MethodType(sync_obs_norm_rms, envs)
        
    envs.action_space.seed(seed)
    envs.observation_space.seed(seed)
    return envs
