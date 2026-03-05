import os
import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from src.models.sb3_wrapper import SB3TeacherWrapper

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in Tensorboard.
    Uses your existing TBLogger to keep logs consistent.
    """
    def __init__(self, logger, eval_callback=None, eval_freq=0, verbose=0):
        super().__init__(verbose)
        self.tb_logger = logger
        self.eval_callback = eval_callback
        self.eval_freq = eval_freq
        self.last_eval_step = 0

    def _on_step(self) -> bool:
        # Check if there are any done episodes in the current step
        # SB3 stores infos in self.locals["infos"]
        if "infos" in self.locals:
            for info in self.locals["infos"]:
                if "episode" in info:
                    r = info["episode"]["r"]
                    global_step = self.num_timesteps
                    print(f"global_step={global_step}, episodic_return={r:.2f}", flush=True)
                    self.tb_logger.log_scalar("teacher/train_reward", r, global_step)

        # Trigger evaluation
        if self.eval_callback is not None and self.eval_freq > 0:
            if self.num_timesteps - self.last_eval_step >= self.eval_freq:
                self.eval_callback(self.num_timesteps)
                self.last_eval_step = self.num_timesteps

        return True
        
    def _on_rollout_end(self) -> None:
        # Log losses after an update epoch
        global_step = self.num_timesteps
        if "loss" in self.logger.name_to_value:
            self.tb_logger.log_scalar("teacher/policy_loss", self.logger.name_to_value.get("policy_gradient_loss", 0), global_step)
            self.tb_logger.log_scalar("teacher/value_loss", self.logger.name_to_value.get("value_loss", 0), global_step)
            self.tb_logger.log_scalar("teacher/entropy", self.logger.name_to_value.get("entropy_loss", 0), global_step)
            self.tb_logger.log_scalar("teacher/approx_kl", self.logger.name_to_value.get("approx_kl", 0), global_step)
            self.tb_logger.log_scalar("teacher/clipping_fraction", self.logger.name_to_value.get("clip_fraction", 0), global_step)


def train_teacher(cfg, env, logger, run_dir):
    print(f"Starting SB3 PPO Teacher Training on {cfg.env.name}...")
    
    from src.utils.device import get_device
    device = get_device(cfg.device)
    
    # Import generic evaluation callback builder from original train logic
    from scripts.train_teacher import get_eval_callback, evaluate_teacher
    
    # Needs a dummy agent class that returns our wrapper for the evaluator
    def dummy_agent_class(*args, **kwargs):
        return SB3TeacherWrapper(model)
        
    num_envs = getattr(cfg.algo, "num_envs", 1)
    total_timesteps = cfg.algo.total_timesteps
    
    # Disable reward normalization to match custom implementations
    # The environment passed in is already a SyncVectorEnv with NormalizeObservation if continuous
    from src.models.sb3_wrapper import GymVectorEnvToSB3VecEnv
    sb3_env = GymVectorEnvToSB3VecEnv(env)
    
    env_is_atari = cfg.env.type == "atari"
    policy_type = "CnnPolicy" if env_is_atari else "MlpPolicy"
    
    # Configure custom network architecture sizes
    neurons = getattr(cfg.model, "neurons", 64)
    layers = getattr(cfg.model, "layers", 2)
    
    if env_is_atari:
        policy_kwargs = {} # CnnPolicy uses default NatureCNN unless customized
    else:
        policy_kwargs = dict(net_arch=[neurons] * layers)
    
       
    model = PPO(
        policy=policy_type,
        env=sb3_env,
        learning_rate=cfg.algo.lr,
        n_steps=cfg.algo.rollout_steps,
        batch_size=getattr(cfg.algo, "batch_size", 4096),
        n_epochs=getattr(cfg.algo, "ppo_epochs", 10),
        gamma=cfg.algo.gamma,
        gae_lambda=getattr(cfg.algo, "gae_lambda", 0.95),
        clip_range=getattr(cfg.algo, "clip_eps", 0.2),
        ent_coef=getattr(cfg.algo, "entropy_coef", 0.0),
        policy_kwargs=policy_kwargs,
        verbose=0, # We handle our own logging
        device=device
    )

    # Wrap the model immediately so the eval callback can use it
    wrapped_teacher = SB3TeacherWrapper(model)
    
    env_is_discrete = cfg.env.type == "discrete"

    eval_cb = get_eval_callback(cfg, env, wrapped_teacher, env_is_discrete, device, logger, dummy_agent_class, run_dir)
    eval_freq = getattr(cfg.algo, "eval_freq", 100000)
    
    # Using our custom callback to pipe SB3 metrics to TBLogger
    tb_callback = TensorboardCallback(logger, eval_callback=eval_cb, eval_freq=eval_freq)

    model.learn(
        total_timesteps=total_timesteps,
        callback=tb_callback,
    )
    
    # Final evaluation
    evaluate_teacher(cfg, env, wrapped_teacher, env_is_discrete, device, total_timesteps, logger, dummy_agent_class)
    
    return wrapped_teacher
