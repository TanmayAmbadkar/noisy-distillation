import importlib
import json
import logging
import os
import random

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from src.logging.tb_logger import TBLogger
from src.environments.make_env import make_env
# Import for algorithm will be done conditionally inside main
from scripts.distill_student import distill
from scripts.run_experiment import evaluate_all
from src.utils.device import get_device

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

def save_models(run_dir, teacher, students):
    if teacher is not None:
        torch.save(teacher.state_dict(), os.path.join(run_dir, "teacher.pt"))
    if students is not None:
        if not isinstance(students, list):
            students = [students]
        for i, s in enumerate(students, start=1):
            torch.save(s.state_dict(), os.path.join(run_dir, f"student_{i}.pt"))

@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    
    run_dir = HydraConfig.get().runtime.output_dir
    tb_dir = os.path.join(run_dir, "tensorboard")
    logger = TBLogger(tb_dir)

    torch.set_num_threads(1)
    env = make_env(
        cfg.env, 
        num_envs=getattr(cfg.algo, "num_envs", 1), 
        seed=cfg.seed, 
        gamma=cfg.algo.gamma
    )
    
    if cfg.algo.name == "sb3_ppo":
        from scripts.train_sb3_ppo import train_teacher as train_algo
    elif cfg.algo.name == "sb3_sac":
        from scripts.train_sb3_sac import train_teacher as train_algo
    elif cfg.algo.name == "sb3_trpo":
        from scripts.train_sb3_trpo import train_teacher as train_algo
    elif cfg.algo.name == "sb3_ddpg":
        from scripts.train_sb3_ddpg import train_teacher as train_algo
    elif cfg.algo.name == "sb3_dqn":
        from scripts.train_sb3_dqn import train_teacher as train_algo
    elif "tau" in cfg.algo:
        from scripts.train_sac import train_teacher as train_algo
    else:
        from scripts.train_teacher import train_teacher as train_algo
        
    teacher = train_algo(cfg, env, logger, run_dir)
    
    # 2.5 FREEZE Env normalization stats completely so identical environments across Distillation & Evaluation are stationary
    if cfg.env.type == "continuous":
        for i in range(env.num_envs):
            env.freeze_norm_stats(i)
            
    students = distill(cfg, teacher, env, logger)
    
    if not isinstance(students, list):
        students = [students]
        
    metrics = {}
    for i, student in enumerate(students, start=1):
        student_metrics = evaluate_all(cfg, teacher, student, logger, env, idx=i)
        metrics.update(student_metrics)
    
    # Save seed to metrics
    metrics["seed"] = cfg.seed
    
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
        
    save_models(run_dir, teacher, students)
    
    if cfg.env.type == "continuous" and hasattr(env, "get_obs_norm_rms_obj"):
        import pickle
        try:
            obs_rms = env.get_obs_norm_rms_obj(0)
            with open(os.path.join(run_dir, "obs_rms.pkl"), "wb") as f:
                pickle.dump(obs_rms, f)
        except Exception as e:
            print(f"Warning: Could not save obs_rms: {e}")
            
    logger.close()

if __name__ == "__main__":
    main()
