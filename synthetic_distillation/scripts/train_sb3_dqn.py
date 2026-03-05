import os
import torch
import numpy as np
from stable_baselines3 import DQN
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
        # Log losses after an update epoch. SB3 DQN logs these to its logger.
        global_step = self.num_timesteps
        if "loss" in self.logger.name_to_value:
            self.tb_logger.log_scalar("teacher/loss", self.logger.name_to_value.get("loss", 0), global_step)
            self.tb_logger.log_scalar("teacher/exploration_rate", self.logger.name_to_value.get("exploration_rate", 0), global_step)


def train_teacher(cfg, env, logger, run_dir):
    print(f"Starting SB3 DQN Teacher Training on {cfg.env.name}...")
    
    from src.utils.device import get_device
    device = get_device(cfg.device)
    
    if cfg.env.type != "discrete":
        raise ValueError("DQN requires a discrete action space environment")
        
    # Import generic evaluation callback builder from original train logic
    from scripts.train_teacher import get_eval_callback, evaluate_teacher
    
    # Needs a dummy agent class that returns our wrapper for the evaluator
    def dummy_agent_class(*args, **kwargs):
        return SB3TeacherWrapper(model)
        
    total_timesteps = cfg.algo.total_timesteps
    
    from src.models.sb3_wrapper import GymVectorEnvToSB3VecEnv
    sb3_env = GymVectorEnvToSB3VecEnv(env)
    
    env_is_atari = cfg.env.type == "atari"
    policy_type = "CnnPolicy" if env_is_atari else "MlpPolicy"
    
    # Configure custom network architecture sizes
    neurons = getattr(cfg.model, "neurons", 64)
    layers = getattr(cfg.model, "layers", 2)
    
    if env_is_atari:
        policy_kwargs = {} # CnnPolicy uses default NatureCNN
    else:
        policy_kwargs = dict(net_arch=[neurons] * layers)

    model = DQN(
        policy=policy_type,
        env=sb3_env,
        learning_rate=cfg.algo.lr,
        buffer_size=getattr(cfg.algo, "buffer_size", 1000000),
        learning_starts=getattr(cfg.algo, "learning_starts", 50000),
        batch_size=getattr(cfg.algo, "batch_size", 32),
        tau=getattr(cfg.algo, "tau", 1.0),
        gamma=cfg.algo.gamma,
        train_freq=getattr(cfg.algo, "train_freq", 4),
        gradient_steps=getattr(cfg.algo, "gradient_steps", 1),
        target_update_interval=getattr(cfg.algo, "target_update_interval", 10000),
        exploration_fraction=getattr(cfg.algo, "exploration_fraction", 0.1),
        exploration_initial_eps=getattr(cfg.algo, "exploration_initial_eps", 1.0),
        exploration_final_eps=getattr(cfg.algo, "exploration_final_eps", 0.05),
        max_grad_norm=getattr(cfg.algo, "max_grad_norm", 10),
        policy_kwargs=policy_kwargs,
        verbose=0,
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
