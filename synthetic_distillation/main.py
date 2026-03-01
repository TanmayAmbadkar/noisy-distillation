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
from scripts.train_teacher import train_teacher
from scripts.distill_student import distill
from scripts.run_experiment import evaluate_all

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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
    
    run_dir = HydraConfig.get().runtime.output_dir
    tb_dir = os.path.join(run_dir, "tensorboard")
    logger = TBLogger(tb_dir)

    env = make_env(cfg.env, seed=cfg.seed)
    
    teacher = train_teacher(cfg, env, logger)
    
    # 2.5 FREEZE Env normalization stats completely so identical environments across Distillation & Evaluation are stationary
    if cfg.env.type != "discrete":
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
    logger.close()

if __name__ == "__main__":
    main()
