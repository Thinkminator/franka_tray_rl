import os
import sys
import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
os.chdir(PROJECT_ROOT)
sys.path.append(PROJECT_ROOT)

# Import our custom environment
from envs.traypose.traypose_env import TrayPoseEnv

def main():
    # ------------------------------
    # 1. Make environment
    # ------------------------------
    def make_env():
        return TrayPoseEnv(model_path="assets/panda_tray/panda_tray_cylinder.xml")

    # Vectorized environment (parallel workers for faster training)
    env = make_vec_env(make_env, n_envs=4)  

    # ------------------------------
    # 2. Setup output dirs
    # ------------------------------
    log_dir = "training/logs/"
    os.makedirs(log_dir, exist_ok=True)

    save_dir = "training/checkpoints/"
    os.makedirs(save_dir, exist_ok=True)

    # ------------------------------
    # 3. Define RL algorithm (PPO)
    # ------------------------------
    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        tensorboard_log=log_dir,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
    )

    # ------------------------------
    # 4. Setup callbacks
    # ------------------------------
    eval_env = make_env()
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_dir,
        log_path=log_dir,
        eval_freq=10_000,          # evaluate every 10k steps
        deterministic=True,
        render=False,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,          # save every 50k steps
        save_path=save_dir,
        name_prefix="rl_model",
    )

    # ------------------------------
    # 5. Train model
    # ------------------------------
    total_timesteps = 500_000  # adjust as needed
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback]
    )

    # Save final model
    model.save(os.path.join(save_dir, "ppo_traypose_final"))

    # ------------------------------
    # 6. Test trained model
    # ------------------------------
    test_env = make_env()

    obs = test_env.reset()
    for i in range(500):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = test_env.step(action)
        if done:
            obs = test_env.reset()

    print("Training complete! Model saved to:", save_dir)

if __name__ == "__main__":
    main()