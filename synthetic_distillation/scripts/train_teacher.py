import torch
import torch.optim as optim
from functools import partial
import numpy as np

from src.models.agent import DiscreteAgent, ContinuousAgent
from src.algorithms.ppo import PPO
from src.environments.make_env import make_env

class PPOAdapterLogger:
    def __init__(self, tb_logger, eval_callback=None):
        self.tb = tb_logger
        self.eval_callback = eval_callback

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
        self.tb.log_scalar("teacher/policy_loss", update_results["policy_loss"], global_step)
        self.tb.log_scalar("teacher/value_loss", update_results["value_loss"], global_step)
        self.tb.log_scalar("teacher/entropy", update_results["entropy_loss"], global_step)
        
        # Log additional PPO optimization metrics
        self.tb.log_scalar("teacher/old_approx_kl", update_results.get("old_approx_kl", 0.0), global_step)
        self.tb.log_scalar("teacher/approx_kl", update_results.get("approx_kl", 0.0), global_step)
        self.tb.log_scalar("teacher/clipping_fraction", update_results.get("clipping_fractions", 0.0), global_step)
        self.tb.log_scalar("teacher/explained_variance", update_results.get("explained_variance", 0.0), global_step)
        
        if self.eval_callback is not None:
            self.eval_callback(global_step)

def get_eval_callback(cfg, env, teacher, env_is_discrete, device, logger, agent_class):
    eval_envs = make_env(cfg.env, num_envs=1, seed=cfg.seed + 1000)
    eval_teacher = agent_class(eval_envs, env_name=cfg.env.name).to(device)

    def callback(global_step):
        eval_teacher.load_state_dict(teacher.state_dict())
        eval_teacher.eval()
        
        if not env_is_discrete:
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
        print(f"Eval: global_step={global_step}, reward={mean_reward:.2f} +/- {std_reward:.2f}", flush=True)

    return callback


def evaluate_teacher(cfg, env, teacher, env_is_discrete, device, total_timesteps, logger, agent_class):
    # Run a simple evaluation rollout at the end
    eval_envs = make_env(cfg.env, num_envs=1, seed=cfg.seed + 1000)
    
    eval_teacher = agent_class(eval_envs, env_name=cfg.env.name).to(device)
    eval_teacher.load_state_dict(teacher.state_dict())
    eval_teacher.eval()
    
    if not env_is_discrete:
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


def train_teacher(cfg, env, logger):
    print(f"Starting PPO Teacher Training on {cfg.env.name}...")
    
    device = torch.device(cfg.device if torch.cuda.is_available() and cfg.device != "cpu" else "cpu")
    env_is_discrete = cfg.env.type == "discrete"
    
    neurons = cfg.model.get("neurons", 64)
    layers = cfg.model.get("layers", 2)
    
    agent_class = partial(DiscreteAgent, neurons=neurons, layers=layers) if env_is_discrete else partial(ContinuousAgent, rpo_alpha=None, neurons=neurons, layers=layers)
    teacher = agent_class(env, env_name=cfg.env.name).to(device)

    optimizer = optim.Adam(teacher.parameters(), lr=cfg.algo.lr, eps=1e-5)

    num_envs = getattr(cfg.algo, "num_envs", 1)
    total_timesteps = cfg.algo.total_timesteps

    eval_cb = get_eval_callback(cfg, env, teacher, env_is_discrete, device, logger, agent_class)
    adapted_logger = PPOAdapterLogger(logger, eval_callback=eval_cb)

    ppo = PPO(
        agent=teacher,
        optimizer=optimizer,
        learning_rate=cfg.algo.lr,
        num_rollout_steps=cfg.algo.rollout_steps,
        num_envs=num_envs,
        gamma=cfg.algo.gamma,
        gae_lambda=cfg.algo.gae_lambda,
        surrogate_clip_threshold=cfg.algo.clip_eps,
        entropy_loss_coefficient=cfg.algo.entropy_coef,
        value_function_loss_coefficient=0.5,
        max_grad_norm=0.5,
        update_epochs=cfg.algo.ppo_epochs,
        num_minibatches= (cfg.algo.rollout_steps * num_envs) // cfg.algo.minibatch_size,
        normalize_advantages=True,
        clip_value_function_loss=True,
        target_kl=None,
        anneal_lr=False,
        envs=env,
        seed=cfg.seed,
        logger=adapted_logger,
    )
    
    trained_teacher = ppo.learn(total_timesteps)
    
    evaluate_teacher(cfg, env, trained_teacher, env_is_discrete, device, total_timesteps, logger, agent_class)
    
    return trained_teacher
