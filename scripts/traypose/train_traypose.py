#!/usr/bin/env python3
import os
import sys
import time
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize

# Project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
os.chdir(PROJECT_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import custom env
from envs.traypose.traypose_env import TrayPoseEnv


def make_env_fn(model_path="assets/panda_tray/panda_tray_cylinder.xml",
                config_path="envs/traypose/config.yaml",
                obs_noise_std_pos=0.0,
                obs_noise_std_vel=0.0,
                use_jacobian_tray_obs=False):
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
        )
        env = Monitor(env)  # Gymnasium-compatible Monitor
        return env
    return _thunk


def _reset_env(env):
    """Robust reset wrapper that returns (obs, info)."""
    res = env.reset()
    if isinstance(res, tuple) and len(res) == 2:
        obs, info = res
    else:
        obs = res
        info = {}
    return obs, info


class CustomEvalCallback(BaseCallback):
    """
    Evaluate policy deterministically and log:
      - custom_eval/mean_reward
      - custom_eval/mean_ep_length
      - custom_eval/success_rate (reads info['is_success'] or info['success'] or info['at_goal'] or goal_hold_counter)
      - custom_eval/mean_cylinder_angle
      - custom_eval/mean_cylinder_offset
      - custom_eval/topple_terminated
      - custom_eval/drop_terminated
      - custom_eval/truncated_pct

    best_model_save_path: if provided, save the best model by mean_reward.
    """
    def __init__(self, eval_env, n_eval_episodes=30, eval_freq=5000, best_model_save_path=None, verbose=1, success_hold_H=None):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.best_mean_reward = -float("inf")
        self.best_model_save_path = best_model_save_path
        self._last_eval_step = 0
        self.success_hold_H = success_hold_H

    def _on_step(self) -> bool:
        if (self.num_timesteps - self._last_eval_step) < self.eval_freq:
            return True
        self._last_eval_step = self.num_timesteps

        rewards, lengths= [], []
        angles, offsets = [], []
        truncated_count = success_count = 0
        drop_terminated_count = topple_terminated_count = 0

        for _ in range(self.n_eval_episodes):
            obs, info = _reset_env(self.eval_env)
            ep_rew = 0.0
            ep_len = 0
            ep_success = False
            ep_terminated = False
            ep_truncated = False
            ep_drop_terminated = False
            ep_topple_terminated = False

            angle = 0
            offset = 0

            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                step_res = self.eval_env.step(action)

                if len(step_res) == 5:
                    obs, reward, terminated, truncated, info = step_res
                    done = terminated or truncated
                elif len(step_res) == 4:
                    obs, reward, done, info = step_res
                    terminated = done
                    truncated = False
                else:
                    raise ValueError(f"Unexpected step() return length: {len(step_res)}")
                    
                ep_rew += reward
                ep_len += 1

                done = terminated or truncated               
                if done and info:
                    if isinstance(info, (list, tuple)) and len(info) == 1:
                        info = info[0]
                    ep_success = info.get("is_success", False)
                    ep_truncated = info.get("truncated", False)
                    ep_drop_terminated = info.get("terminated_due_to_drop", False)
                    ep_topple_terminated = info.get("terminated_due_to_topple", False)
                    offset = info.get("cylinder_offset", 0)
                    angle = info.get("cylinder_angle", 0)             
                        

            rewards.append(ep_rew)
            lengths.append(ep_len)
            angles.append(angle)
            offsets.append(offset)

            if ep_truncated:
                truncated_count += 1
            if ep_success:
                success_count += 1
            if ep_drop_terminated:
                drop_terminated_count += 1
            if ep_topple_terminated:
                topple_terminated_count += 1


        total_eps = self.n_eval_episodes
        mean_reward = np.mean(rewards)
        mean_len = np.mean(lengths)
        mean_angle = np.mean(angles)
        mean_offset = np.mean(offsets)
        truncated_pct = 100.0 * truncated_count / total_eps
        success_pct = 100.0 * success_count / total_eps
        drop_terminated_pct = 100.0 * drop_terminated_count / total_eps
        topple_terminated_pct = 100.0 * topple_terminated_count / total_eps

        try:
            self.logger.record("custom_eval/mean_reward", mean_reward)
            self.logger.record("custom_eval/mean_ep_length", mean_len)
            self.logger.record("custom_eval/success_rate", success_pct)
            self.logger.record("custom_eval/mean_cylinder_angle", mean_angle)
            self.logger.record("custom_eval/mean_cylinder_offset", mean_offset)
            self.logger.record("custom_eval/truncated_pct", truncated_pct)
            self.logger.record("custom_eval/drop_terminated_pct", drop_terminated_pct)
            self.logger.record("custom_eval/topple_terminated_pct", topple_terminated_pct)
            self.logger.dump(self.num_timesteps)
        except Exception:
            pass

        if self.verbose:
            print(f"[CustomEval] step={self.num_timesteps} mean_reward={mean_reward:.3f} mean_len={mean_len:.1f} "
                  f"mean_cyl_angle={mean_angle:.4f} rad, mean_cyl_offset={mean_offset:.4f} m "
                  f"success={success_pct:.2%} truncated={truncated_pct:.2f}% "
                  f"drop_terminated={drop_terminated_pct:.2f}% topple_terminated={topple_terminated_pct:.2f}% ")

        return True


def main():
    # 1) Vectorized envs
    make_env = make_env_fn(
        model_path="assets/panda_tray/panda_tray_cylinder.xml",
        config_path="envs/traypose/config.yaml",
        obs_noise_std_pos=0.0,
        obs_noise_std_vel=0.0,
        use_jacobian_tray_obs=False
    )

    # Directories
    log_dir = "training/logs"
    save_dir = "training/checkpoints"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    # Configs you can tweak
    n_envs = 8
    n_steps = 1024
    batch_size = 256
    total_timesteps = 1_000_000  # increase training if you can

    # Create training envs and enable Monitor file output (monitor_dir)
    train_env = make_vec_env(make_env, n_envs=n_envs, monitor_dir=log_dir)
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # 2) PPO (tuned hyperparams)
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        verbose=1,
        tensorboard_log=log_dir,
        learning_rate=1e-4,
        n_steps=n_steps,
        batch_size=batch_size,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=1e-3,
        vf_coef=0.5,
        n_epochs=10,
        target_kl=None,
        seed=42,
        device="cpu",
    )

    # 3) Eval env: create separately and share observation normalization if possible
    eval_env = make_vec_env(make_env, n_envs=1, monitor_dir=log_dir)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    # Try to share observation normalization stats (safe-guarded)
    try:
        if hasattr(train_env, "obs_rms") and hasattr(eval_env, "obs_rms"):
            eval_env.obs_rms = train_env.obs_rms
    except Exception as e:
        print("Warning: could not share obs_rms between train/eval VecNormalize (shape mismatch?). Continuing without sharing. Error:", e)

    # 4) Callbacks: keep EvalCallback + custom success-rate callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_dir,
        log_path=log_dir,
        eval_freq=5_000,
        deterministic=True,
        render=False,
        n_eval_episodes=30,
    )

    custom_eval_cb = CustomEvalCallback(
        eval_env=eval_env,
        n_eval_episodes=30,
        eval_freq=5_000,
        best_model_save_path=save_dir,
        verbose=1,
        success_hold_H=None,  # None => try to infer from env, or set integer explicitly
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path=save_dir,
        name_prefix="rl_model",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )

    # 5) Train
    try:
        model.learn(total_timesteps=total_timesteps, callback=[eval_callback, custom_eval_cb, checkpoint_callback])
        model.save(os.path.join(save_dir, "ppo_traypose_final"))
        train_env.save(os.path.join(save_dir, "vecnormalize.pkl"))
    finally:
        try:
            train_env.close()
        except Exception:
            pass
        try:
            eval_env.close()
        except Exception:
            pass

    print("Training complete! Models saved to:", save_dir)


if __name__ == "__main__":
    main()