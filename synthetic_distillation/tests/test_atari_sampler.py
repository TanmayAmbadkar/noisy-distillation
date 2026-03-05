import torch
import numpy as np
from omegaconf import OmegaConf
import sys
import os

# Adjust path to import src
sys.path.append(os.getcwd())
from src.data.synthetic_sampler import SyntheticSampler

def test_st_gaussian():
    print("\n--- Test ST Gaussian ---")
    cfg = OmegaConf.create({
        "distill": {
            "sampling": {
                "mode": "st_gaussian",
                "sigma_t": 1.0,
                "sigma_s": 2.0
            }
        }
    })
    # Mock trajectory states to set shape
    obs_shape = (4, 84, 84)
    traj = torch.zeros((10, *obs_shape))
    sampler = SyntheticSampler(cfg, trajectory_states=traj)
    
    batch = sampler.sample(8)
    print(f"Shape: {batch.shape}")
    assert batch.shape == (8, 4, 84, 84)
    
    # Check temporal correlation (frame 0 and frame 1 should be correlated)
    corr = np.corrcoef(batch[:, 0].flatten().numpy(), batch[:, 1].flatten().numpy())[0, 1]
    print(f"Temporal Correlation (frame 0-1): {corr:.4f}")
    assert corr > 0.3 # With sigma_t=1.0, there should be some correlation

def test_st_ar1():
    print("\n--- Test ST AR1 ---")
    cfg = OmegaConf.create({
        "distill": {
            "sampling": {
                "mode": "st_ar1",
                "rho": 0.9,
                "sigma": 0.5
            }
        }
    })
    obs_shape = (4, 84, 84)
    traj = torch.zeros((10, *obs_shape))
    sampler = SyntheticSampler(cfg, trajectory_states=traj)
    
    batch = sampler.sample(8)
    print(f"Shape: {batch.shape}")
    assert batch.shape == (8, 4, 84, 84)
    
    # AR1 should have high persistence
    corr = np.corrcoef(batch[:, 0].flatten().numpy(), batch[:, 1].flatten().numpy())[0, 1]
    print(f"AR1 Temporal Correlation (rho=0.9): {corr:.4f}")
    assert corr > 0.8

def test_st_moving():
    print("\n--- Test ST Moving Blob ---")
    cfg = OmegaConf.create({
        "distill": {
            "sampling": {
                "mode": "st_moving"
            }
        }
    })
    obs_shape = (4, 84, 84)
    traj = torch.zeros((10, *obs_shape))
    sampler = SyntheticSampler(cfg, trajectory_states=traj)
    
    batch = sampler.sample(8)
    print(f"Shape: {batch.shape}")
    assert batch.shape == (8, 4, 84, 84)
    print(f"Mean: {batch.mean():.4f}, Std: {batch.std():.4f}")

if __name__ == "__main__":
    test_st_gaussian()
    test_st_ar1()
    test_st_moving()
    print("\nAll tests passed!")
