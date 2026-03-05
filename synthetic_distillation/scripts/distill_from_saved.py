import os
import sys
import json
import torch
import glob
import copy
from omegaconf import OmegaConf

# Add root to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environments.make_env import make_env
from src.distillation.distiller import Distiller
from scripts.run_experiment import evaluate_all
from src.logging.tb_logger import TBLogger
from src.models.sb3_wrapper import SB3TeacherWrapper
from stable_baselines3 import PPO

def evaluate_directory(base_dir):
    # Iterate over all seed directories
    seed_dirs = glob.glob(os.path.join(base_dir, "seed_*"))
    for run_dir in seed_dirs:
        print(f"Processing {run_dir}...")
        
        cfg_path = os.path.join(run_dir, ".hydra/config.yaml")
        if not os.path.exists(cfg_path):
            print(f"Skipping {run_dir}, config not found.")
            continue
            
        cfg = OmegaConf.load(cfg_path)
        
        # Override distillation config to pure_uniform
        cfg.distill = OmegaConf.load("configs/distill/pure_uniform.yaml")
        cfg.model.distil_neurons = [1.0]
        
        from src.utils.device import get_device
        device = get_device(getattr(cfg, "device", "auto"))
        
        # We need a teacher 
        env = make_env(cfg.env, seed=cfg.seed, gamma=cfg.algo.gamma)
        from src.models.sb3_wrapper import GymVectorEnvToSB3VecEnv
        sb3_env = GymVectorEnvToSB3VecEnv(env)
        
        # Initialize dummy PPO to accept the weights
        model = PPO("MlpPolicy", sb3_env, verbose=0, device=device)
        teacher = SB3TeacherWrapper(model)
        
        # Load the saved state dict
        teacher_path = os.path.join(run_dir, "teacher.pt")
        teacher.load_state_dict(torch.load(teacher_path, map_location=device))
        
        # Freeze env stats for continuous
        if cfg.env.type == "continuous" and hasattr(env, 'freeze_norm_stats'):
            obs_rms_path = os.path.join(run_dir, "obs_rms.pkl")
            if os.path.exists(obs_rms_path):
                import pickle
                with open(obs_rms_path, "rb") as f:
                    obs_rms = pickle.load(f)
                for i in range(getattr(env, 'num_envs', 1)):
                    env.set_obs_norm_rms_obj(copy.deepcopy(obs_rms), i)
                    
            for i in range(getattr(env, 'num_envs', 1)):
               env.freeze_norm_stats(i)
               
        # Set up logger
        tb_dir = os.path.join(run_dir, "tensorboard_pure_uniform")
        logger = TBLogger(tb_dir)
        
        # Train new students using the explicitly assigned uniform sampler
        distiller = Distiller(cfg, device, logger)
        students = distiller.train(teacher, env)
        
        if not isinstance(students, list):
            students = [students]
            
        metrics = {}
        for i, student in enumerate(students, start=1):
            # Save the new pure_uniform student safely apart from generated students
            torch.save(student.state_dict(), os.path.join(run_dir, f"student_pure_uniform_{i}.pt"))
            student_metrics = evaluate_all(cfg, teacher, student, logger, env, idx=i)
            pure_metric = {}
            for k, v in student_metrics.items():
                pure_metric[f"pure_uniform_{k}"] = v
            metrics.update(pure_metric)
            
        # Append to existing metrics
        metrics_path = os.path.join(run_dir, "metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                existing_metrics = json.load(f)
            existing_metrics.update(metrics)
            final_metrics = existing_metrics
        else:
            final_metrics = metrics
            
        with open(metrics_path, "w") as f:
            json.dump(final_metrics, f, indent=4)
            
        logger.close()
        print(f"Finished processing {run_dir}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python distill_from_saved_teacher.py <base_dir>")
        sys.exit(1)
    evaluate_directory(sys.argv[1])
