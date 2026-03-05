import torch
import numpy as np
from omegaconf import OmegaConf
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from data.synthetic_sampler import SyntheticSampler

def test_gaussian_default():
    print("\n--- Test Gaussian Default ---")
    cfg = OmegaConf.create({"distill": {"sampling": {"mode": "gaussian"}}})
    sampler = SyntheticSampler(cfg, trajectory_states=torch.zeros((1, 4)))
    samples = sampler.sample(10000)
    print(f"Mean: {samples.mean().item():.4f} (Expected: ~0.0)")
    print(f"Std:  {samples.std().item():.4f} (Expected: ~1.0)")

def test_gaussian_parameterized():
    print("\n--- Test Gaussian Parameterized (mean=1, var=2) ---")
    cfg = OmegaConf.create({"distill": {"sampling": {"mode": "gaussian"}}})
    # Setting variance=2 -> std=sqrt(2) approx 1.414
    override = {"mean": 1.0, "variance": 2.0}
    sampler = SyntheticSampler(cfg, trajectory_states=torch.zeros((1, 4)), override_cfg=override)
    samples = sampler.sample(10000)
    print(f"Mean: {samples.mean().item():.4f} (Expected: ~1.0)")
    print(f"Std:  {samples.std().item():.4f} (Expected: ~1.414)")

def test_uniform_parameterized():
    print("\n--- Test Uniform Parameterized (low=-0.5, high=0.5) ---")
    cfg = OmegaConf.create({"distill": {"sampling": {"mode": "uniform_global"}}})
    override = {"low": -0.5, "high": 0.5}
    sampler = SyntheticSampler(cfg, trajectory_states=torch.zeros((1, 4)), override_cfg=override)
    samples = sampler.sample(10000)
    print(f"Min:  {samples.min().item():.4f} (Expected: ~-0.5)")
    print(f"Max:  {samples.max().item():.4f} (Expected: ~0.5)")
    print(f"Mean: {samples.mean().item():.4f} (Expected: ~0.0)")

def test_mixed_parameterized():
    print("\n--- Test Mixed Parameterized (50% Traj, 50% Gaussian(mean=1, std=2)) ---")
    # Trajectory states will be zeros
    traj = torch.zeros((100, 4))
    cfg = OmegaConf.create({"distill": {"sampling": {"mode": "mixed"}}})
    override = {"mode": "mixed", "synth_mode": "gaussian", "mean": 1.0, "std": 2.0, "mix_ratio": 0.5}
    sampler = SyntheticSampler(cfg, trajectory_states=traj, override_cfg=override)
    samples = sampler.sample(20000)
    
    # First 10000 should be trajectory (zeros)
    # Next 10000 should be Gaussian(1, 2)
    traj_part = samples[:10000]
    synth_part = samples[10000:]
    
    print(f"Traj Part Mean: {traj_part.mean().item():.4f} (Expected: 0.0)")
    print(f"Synth Part Mean: {synth_part.mean().item():.4f} (Expected: ~1.0)")
    print(f"Synth Part Std:  {synth_part.std().item():.4f} (Expected: ~2.0)")

def test_noise_mixture():
    print("\n--- Test Noise Mixture (7 components) ---")
    traj = torch.zeros((100, 4))
    cfg = OmegaConf.create({"distill": {"sampling": {"mode": "noise_mixture"}}})
    components = [
        {"name": "g1", "mode": "gaussian", "mean": 0.0, "std": 1.0},
        {"name": "g2", "mode": "gaussian", "mean": 0.0, "std": 2.0},
        {"name": "g3", "mode": "gaussian", "mean": 0.0, "std": 0.5},
        {"name": "g4", "mode": "gaussian", "mean": 1.0, "std": 1.0},
        {"name": "u1", "mode": "uniform_global", "low": -1.0, "high": 1.0},
        {"name": "u2", "mode": "uniform_global", "low": -0.5, "high": 0.5},
        {"name": "l1", "mode": "laplace", "scale": 1.0}
    ]
    override = {"mode": "noise_mixture", "components": components}
    sampler = SyntheticSampler(cfg, trajectory_states=traj, override_cfg=override, device="cpu")
    
    # Sample 7000 states, should be 1000 each
    samples = sampler.sample(7000)
    print(f"Total samples: {len(samples)}")
    
    # Check chunks
    g1 = samples[0:1000]
    g2 = samples[1000:2000]
    g4 = samples[3000:4000]
    u1 = samples[4000:5000]
    
    print(f"G1 (0,1) Mean: {g1.mean().item():.4f} (Expected: ~0.0)")
    print(f"G2 (0,2) Std:  {g2.std().item():.4f}  (Expected: ~2.0)")
    print(f"G4 (1,1) Mean: {g4.mean().item():.4f} (Expected: ~1.0)")
    print(f"U1 (-1,1) Min: {u1.min().item():.4f} (Expected: ~-1.0)")
    print(f"U1 (-1,1) Max: {u1.max().item():.4f} (Expected: ~1.0)")

if __name__ == "__main__":
    test_gaussian_default()
    test_gaussian_parameterized()
    test_uniform_parameterized()
    # test_mixed_parameterized()
    test_noise_mixture()
