#!/usr/bin/env python3
import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecVideoRecorder
from envs.traypose.traypose_env import TrayPoseEnv

def make_env():
    return TrayPoseEnv(
        model_path="assets/panda_tray/panda_tray_cylinder.xml",
        config_path="config.yaml",
        obs_noise_std_pos=0.0,
        obs_noise_std_vel=0.0,
        use_jacobian_tray_obs=False,
    )

def make_single_vec_env():
    return DummyVecEnv([lambda: make_env()])

def load_vecnormalize_if_present(env, path="training/checkpoints/vecnormalize.pkl"):
    if os.path.exists(path):
        try:
            env = VecNormalize.load(path, env)
            print("Loaded VecNormalize from", path)
        except Exception as e:
            print("Failed to load VecNormalize:", e)
    return env

if __name__ == "__main__":
    model_path = "training/checkpoints/ppo_traypose_final.zip"
    vecnorm_path = "training/checkpoints/vecnormalize.pkl"
    n_eval_episodes = 100
    record_video = True

    env = make_single_vec_env()
    env = load_vecnormalize_if_present(env, vecnorm_path)

    if record_video:
        os.makedirs("videos", exist_ok=True)
        env = VecVideoRecorder(env, video_folder="videos/", record_video_trigger=lambda x: x==0, video_length=500)

    model = PPO.load(model_path, env=env, device="cpu")

    rewards = []
    lengths = []
    successes = []

    for ep in range(n_eval_episodes):
        res = env.reset()
        obs = res[0] if isinstance(res, tuple) else res
        done = False
        ep_rew = 0.0
        ep_len = 0
        ep_success = False

        while True:
            action, _ = model.predict(obs, deterministic=True)
            step_res = env.step(action)
            # handle vecenv returns
            if isinstance(step_res, tuple) and len(step_res) == 5:
                obs, reward, terminated, truncated, info = step_res
                done = bool(terminated or truncated)
            elif isinstance(step_res, tuple) and len(step_res) == 4:
                obs, reward, done, info = step_res
                done = bool(done)
            else:
                obs, rewards_arr, dones_arr, infos_arr = step_res
                reward = float(rewards_arr[0]) if hasattr(rewards_arr, "__len__") else float(rewards_arr)
                done = bool(dones_arr[0]) if hasattr(dones_arr, "__len__") else bool(dones_arr)
                info = infos_arr[0] if hasattr(infos_arr, "__len__") else infos_arr

            ep_rew += float(reward)
            ep_len += 1

            if isinstance(info, dict):
                if "is_success" in info:
                    ep_success = ep_success or bool(info["is_success"])
                elif "success" in info:
                    ep_success = ep_success or bool(info["success"])

            if done or ep_len > 5000:
                break

        rewards.append(ep_rew)
        lengths.append(ep_len)
        successes.append(1.0 if ep_success else 0.0)
        print(f"Episode {ep+1:03d}: reward={ep_rew:.3f} len={ep_len} success={bool(ep_success)}")

    print("Eval episodes:", n_eval_episodes)
    print("Mean reward:", np.mean(rewards), "std:", np.std(rewards))
    print("Mean length:", np.mean(lengths))
    print("Success count:", int(np.sum(successes)), "/", n_eval_episodes)