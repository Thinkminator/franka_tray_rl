import os
import sys
import argparse
import time  # <-- For sleep in viewer loop
import numpy as np
import gymnasium as gym
import mujoco.viewer  # <-- Import viewer
import torch

# Project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(PROJECT_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import custom env
from envs.traypose.traypose_env import TrayPoseEnv      

# Import the SimpleActorCritic network
from simple_actor_critic import SimpleActorCritic  

def main(seed=20, steps=2000, render=False):
    env = TrayPoseEnv(render=render)
    obs, info = env.reset(seed=seed)

    net = SimpleActorCritic(
        obs_dim=env.observation_space.shape[0],
        act_dim=env.action_space.shape[0],
        hidden_size=128,
    )

    
    net.eval()  # evaluation mode

    ep_return, ep_len = 0.0, 0
    with torch.no_grad():
        for t in range(steps):
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)   # [1,34]
            actions, mean_action, action_stddev, state_value = net(obs_t, deterministic=True)
            action = actions.squeeze(0).cpu().numpy()                     # [7]

            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += reward; ep_len += 1

            if terminated or truncated:
                print(f"Episode done: return={ep_return:.3f}, length={ep_len}")
                ep_return, ep_len = 0.0, 0
                obs, info = env.reset()

if __name__ == "__main__":
    main(render=False)

