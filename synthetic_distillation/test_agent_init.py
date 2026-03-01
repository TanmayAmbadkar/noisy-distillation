import torch
import numpy as np
import gymnasium as gym
from src.models.agent import ContinuousAgent
from src.environments.make_env import make_env
from omegaconf import OmegaConf

def test_agent(env_name):
    print(f"\nTesting agent for: {env_name}")
    cfg = OmegaConf.create({"name": env_name, "type": "continuous"})
    envs = make_env(cfg, num_envs=1)
    agent = ContinuousAgent(envs, env_name=env_name)
    logstd = agent.actor_logstd.data.cpu().numpy()
    print(f"Agent actor_logstd: {logstd}")
    expected = 0.0 if "hopper" in env_name.lower() else -1.0
    assert np.all(logstd == expected), f"Expected {expected}, got {logstd}"
    print("Test passed!")

if __name__ == "__main__":
    try:
        test_agent("Hopper-v5")
        test_agent("HalfCheetah-v5")
        test_agent("Ant-v5")
    except Exception as e:
        print(f"Test failed with error: {e}")
