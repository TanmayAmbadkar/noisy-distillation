import torch
import torch.optim as optim
from functools import partial
import numpy as np
import os

from src.models.sac_agent import SACAgent
from src.algorithms.sac import SAC
from src.environments.make_env import make_env

class SACAdapterLogger:
    def __init__(self, tb_logger, eval_callback=None, eval_freq=0):
        self.tb = tb_logger
        self.eval_callback = eval_callback
        self.eval_freq = eval_freq
        self.last_eval_step = 0

    def log_rollout_step(self, infos, global_step):
        if "_episode" in infos and "episode" in infos:
            mask = infos["_episode"]
            if mask.any():
                for idx in range(len(mask)):
                    if mask[idx]:
                        r = infos["episode"]["r"][idx]
                        print(
                            f"global_step={global_step}, episodic_return={r:.2f}",
                            flush=True,
                        )
                        self.tb.log_scalar("teacher/train_reward", r, global_step)

    def log_policy_update(self, update_results, global_step):
        self.tb.log_scalar("teacher/qf1_loss", update_results["qf1_loss"], global_step)
        self.tb.log_scalar("teacher/qf2_loss", update_results["qf2_loss"], global_step)
        self.tb.log_scalar("teacher/policy_loss", update_results["policy_loss"], global_step)
        self.tb.log_scalar("teacher/alpha", update_results["alpha_value"], global_step)
        self.tb.log_scalar("teacher/alpha_loss", update_results["alpha_loss"], global_step)
        
        if self.eval_callback is not None:
            if self.eval_freq <= 0 or (global_step - self.last_eval_step >= self.eval_freq):
                self.eval_callback(global_step)
                self.last_eval_step = global_step

def get_eval_callback(cfg, env, teacher, device, logger, agent_class, run_dir):
    eval_envs = make_env(cfg.env, num_envs=1, seed=cfg.seed + 1000)
    eval_teacher = agent_class(eval_envs, env_name=cfg.env.name).to(device)

    def callback(global_step):
        eval_teacher.load_state_dict(teacher.state_dict())
        eval_teacher.eval()
        
        env.sync_obs_norm_rms(eval_envs)
        for i in range(eval_envs.num_envs):
            eval_envs.freeze_norm_stats(i)
        
        obs, _ = eval_envs.reset()
        eval_episodes = 0
        rewards = []
        
        while eval_episodes < 10:
            with torch.no_grad():
                actions = eval_teacher.act(torch.Tensor(obs).to(device), deterministic=True)
            obs, _, _, _, infos = eval_envs.step(actions.cpu().numpy())
            
            if "_episode" in infos and "episode" in infos:
                mask = infos["_episode"]
                if mask.any():
                    for idx in range(len(mask)):
                        if mask[idx]:
                            rewards.append(infos["episode"]["r"][idx])
                            eval_episodes += 1
                            
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        
        logger.log_scalar("teacher/eval_reward_mean", mean_reward, global_step)
        logger.log_scalar("teacher/eval_reward_std", std_reward, global_step)
        print(f"SAC Eval: global_step={global_step}, reward={mean_reward:.2f} +/- {std_reward:.2f}", flush=True)

        if run_dir is not None:
            checkpoint_dir = os.path.join(run_dir, "checkpoints")
            os.makedirs(checkpoint_dir, exist_ok=True)
            save_path = os.path.join(checkpoint_dir, f"teacher_step_{global_step}.pt")
            torch.save(teacher.state_dict(), save_path)
            print(f"Saved teacher checkpoint to {save_path}", flush=True)

    return callback

def evaluate_teacher(cfg, env, teacher, device, total_timesteps, logger, agent_class):
    eval_envs = make_env(cfg.env, num_envs=1, seed=cfg.seed + 1000)
    
    eval_teacher = agent_class(eval_envs, env_name=cfg.env.name).to(device)
    eval_teacher.load_state_dict(teacher.state_dict())
    eval_teacher.eval()
    
    env.sync_obs_norm_rms(eval_envs)
    for i in range(eval_envs.num_envs):
        eval_envs.freeze_norm_stats(i)

    obs, _ = eval_envs.reset()
    eval_episodes = 0
    total_eval_returns = 0
    
    while eval_episodes < 5:
        with torch.no_grad():
            actions = eval_teacher.act(torch.Tensor(obs).to(device), deterministic=True)
        obs, _, _, _, infos = eval_envs.step(actions.cpu().numpy())
        
        if "_episode" in infos and "episode" in infos:
            mask = infos["_episode"]
            if mask.any():
                for idx in range(len(mask)):
                    if mask[idx]:
                        total_eval_returns += infos["episode"]["r"][idx]
                        eval_episodes += 1

    eval_reward = total_eval_returns / max(1, eval_episodes)
    logger.log_scalar("teacher/eval_reward", eval_reward, total_timesteps)
    eval_envs.close()

def train_teacher(cfg, env, logger, run_dir):
    print(f"Starting SAC Teacher Training on {cfg.env.name}...")
    
    from src.utils.device import get_device
    device = get_device(cfg.device)
    if cfg.env.type == "discrete":
        raise ValueError("SAC requires a continuous action space environment")
    
    neurons = cfg.model.get("neurons", 256)
    layers = cfg.model.get("layers", 2)
    
    agent_class = partial(SACAgent, neurons=neurons, layers=layers)
    teacher = agent_class(env, env_name=cfg.env.name).to(device)

    num_envs = getattr(cfg.algo, "num_envs", 1)
    total_timesteps = cfg.algo.total_timesteps

    eval_cb = get_eval_callback(cfg, env, teacher, device, logger, agent_class, run_dir)
    eval_freq = cfg.algo.get("eval_freq", 10000)
    adapted_logger = SACAdapterLogger(logger, eval_callback=eval_cb, eval_freq=eval_freq)

    sac = SAC(
        agent=teacher,
        envs=env,
        learning_rate=cfg.algo.lr,
        num_rollout_steps=getattr(cfg.algo, "rollout_steps", 128),
        num_envs=num_envs,
        update_epochs=getattr(cfg.algo, "update_epochs", 1),
        buffer_size=getattr(cfg.algo, "buffer_size", 1000000),
        batch_size=getattr(cfg.algo, "batch_size", 256),
        gamma=cfg.algo.gamma,
        tau=getattr(cfg.algo, "tau", 0.005),
        alpha=getattr(cfg.algo, "alpha", 0.2),
        automatic_entropy_tuning=getattr(cfg.algo, "automatic_entropy_tuning", True),
        target_update_interval=getattr(cfg.algo, "target_update_interval", 1),
        updates_per_step=getattr(cfg.algo, "updates_per_step", 1),
        start_steps=getattr(cfg.algo, "start_steps", 10000),
        anneal_lr=getattr(cfg.algo, "anneal_lr", False),
        seed=cfg.seed,
        logger=adapted_logger,
    )
    
    trained_teacher = sac.learn(total_timesteps)
    
    evaluate_teacher(cfg, env, trained_teacher, device, total_timesteps, logger, agent_class)
    
    return trained_teacher
