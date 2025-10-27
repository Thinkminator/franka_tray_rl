#!/usr/bin/env python3
import os
import sys
import argparse
import time  # <-- For sleep in viewer loop
import numpy as np
import gymnasium as gym
import mujoco.viewer  # <-- Import viewer
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

# Project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
os.chdir(PROJECT_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import custom env
from envs.traypose.traypose_env import TrayPoseEnv


def make_env_fn(model_path="assets/panda_tray/panda_tray_cylinder.xml",
                config_path="config.yaml",
                obs_noise_std_pos=0.0,
                obs_noise_std_vel=0.0,
                use_jacobian_tray_obs=False,
                render=False):
    """
    Factory returning a function that creates a fresh env instance wrapped with Monitor.
    """
    def _thunk():
        env = TrayPoseEnv(
            model_path=model_path,
            config_path=config_path,
            obs_noise_std_pos=obs_noise_std_pos,
            obs_noise_std_vel=obs_noise_std_vel,
            use_jacobian_tray_obs=use_jacobian_tray_obs,
            render=render
        )
        env = Monitor(env)
        return env
    return _thunk


def main(render=False, visualize=False):
    # 1) Vectorized envs for training
    make_env = make_env_fn(
        model_path="assets/panda_tray/panda_tray_cylinder.xml",
        config_path="config.yaml",
        obs_noise_std_pos=0.0,
        obs_noise_std_vel=0.0,
        use_jacobian_tray_obs=False,
        render=render
    )
    n_envs = 4
    env = make_vec_env(make_env, n_envs=n_envs)

    # 2) Output dirs
    log_dir = "training/logs"
    save_dir = "training/checkpoints"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    # 3) PPO
    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        tensorboard_log=log_dir,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        n_epochs=10,
        target_kl=None,
        seed=42,
    )

    # 4) Callbacks
    eval_env = make_env()
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_dir,
        log_path=log_dir,
        eval_freq=10_000,
        deterministic=True,
        render=render,
        n_eval_episodes=5,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path=save_dir,
        name_prefix="rl_model",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    # 5) Train
    total_timesteps = 500_000
    model.learn(total_timesteps=total_timesteps, callback=[eval_callback, checkpoint_callback])
    model.save(os.path.join(save_dir, "ppo_traypose_final"))

    # 6) Test trained model with optional visualization
    test_env = TrayPoseEnv(
        model_path="assets/panda_tray/panda_tray_cylinder.xml",
        config_path="config.yaml",
        obs_noise_std_pos=0.0,
        obs_noise_std_vel=0.0,
        use_jacobian_tray_obs=False,
        render=visualize or render
    )
    test_env = Monitor(test_env)
    obs, info = test_env.reset()

    if visualize:
        # Unwrap Monitor to access underlying env
        unwrapped_env = test_env.env

        # Launch viewer for real-time visualization
        with mujoco.viewer.launch_passive(unwrapped_env.model, unwrapped_env.data) as viewer:
            for _ in range(500):
                action, _ = model.predict(obs, deterministic=True)
                action = np.asarray(action, dtype=np.float32)
                obs, reward, terminated, truncated, info = test_env.step(action)
                viewer.sync()
                time.sleep(unwrapped_env.control_dt)
                if terminated or truncated:
                    obs, info = test_env.reset()
    else:
        # Headless test
        for _ in range(500):
            action, _ = model.predict(obs, deterministic=True)
            action = np.asarray(action, dtype=np.float32)
            obs, reward, terminated, truncated, info = test_env.step(action)
            if render:
                test_env.render()
            if terminated or truncated:
                obs, info = test_env.reset()

    test_env.close()
    print("Training complete! Models saved to:", save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--render', action='store_true', help="Enable printing during eval/test")
    parser.add_argument('--visualize', action='store_true', help="Show MuJoCo viewer during test (after training)")
    args = parser.parse_args()

    main(render=args.render, visualize=args.visualize)