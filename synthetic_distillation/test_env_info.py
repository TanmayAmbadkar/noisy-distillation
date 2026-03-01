import gymnasium as gym
from src.environments.make_env import make_env
from omegaconf import OmegaConf

cfg = OmegaConf.create({"name": "CartPole-v1", "type": "discrete"})
envs = make_env(cfg, num_envs=2)
obs, info = envs.reset()
print("reset info:", info)
for i in range(100):
    obs, reward, term, trunc, info = envs.step(envs.action_space.sample())
    if term.any() or trunc.any():
        print("step:", i, "term:", term, "trunc:", trunc)
        print("info:", info)
        break
