import os
import torch
import numpy as np
from stable_baselines3 import SAC
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
        # Log losses after an update epoch. SB3 SAC logs these to its logger.
        global_step = self.num_timesteps
        if "loss" in self.logger.name_to_value:
            self.tb_logger.log_scalar("teacher/policy_loss", self.logger.name_to_value.get("actor_loss", 0), global_step)
            self.tb_logger.log_scalar("teacher/qf1_loss", self.logger.name_to_value.get("critic_loss", 0), global_step)
            self.tb_logger.log_scalar("teacher/alpha_value", self.logger.name_to_value.get("ent_coef", 0), global_step)
            self.tb_logger.log_scalar("teacher/alpha_loss", self.logger.name_to_value.get("ent_coef_loss", 0), global_step)


def train_teacher(cfg, env, logger, run_dir):
    print(f"Starting SB3 SAC Teacher Training on {cfg.env.name}...")
    
    from src.utils.device import get_device
    device = get_device(cfg.device)
    
    if cfg.env.type == "discrete":
        raise ValueError("SAC requires a continuous action space environment")
        
    # Import generic evaluation callback builder from original SAC train logic
    from scripts.train_sac import get_eval_callback, evaluate_teacher
    
    # Needs a dummy agent class that returns our wrapper for the evaluator
    def dummy_agent_class(*args, **kwargs):
        return SB3TeacherWrapper(model)
        
    total_timesteps = cfg.algo.total_timesteps
    
    from src.models.sb3_wrapper import GymVectorEnvToSB3VecEnv
    sb3_env = GymVectorEnvToSB3VecEnv(env)
    
    model = SAC(
        policy="MlpPolicy",
        env=sb3_env,
        learning_rate=cfg.algo.lr,
        buffer_size=getattr(cfg.algo, "buffer_size", 1000000),
        batch_size=getattr(cfg.algo, "batch_size", 256),
        ent_coef="auto" if getattr(cfg.algo, "automatic_entropy_tuning", True) else getattr(cfg.algo, "alpha", 0.2),
        gamma=cfg.algo.gamma,
        tau=getattr(cfg.algo, "tau", 0.005),
        train_freq=getattr(cfg.algo, "train_freq", 1),
        gradient_steps=getattr(cfg.algo, "updates_per_step", 1),
        target_update_interval=getattr(cfg.algo, "target_update_interval", 1),
        verbose=0, # We handle our own logging
        device=device # SB3 handles "auto" natively. Our internal calls use the `device` variable
    )

    # Wrap the model immediately so the eval callback can use it
    wrapped_teacher = SB3TeacherWrapper(model)

    eval_cb = get_eval_callback(cfg, env, wrapped_teacher, device, logger, dummy_agent_class, run_dir)
    eval_freq = getattr(cfg.algo, "eval_freq", 100000)
    
    # Using our custom callback to pipe SB3 metrics to TBLogger
    tb_callback = TensorboardCallback(logger, eval_callback=eval_cb, eval_freq=eval_freq)

    model.learn(
        total_timesteps=total_timesteps,
        callback=tb_callback,
    )
    
    # Final evaluation
    evaluate_teacher(cfg, env, wrapped_teacher, device, total_timesteps, logger, dummy_agent_class)
    
    return wrapped_teacher
