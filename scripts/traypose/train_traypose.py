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
                config_path="config.yaml",
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
      - custom_eval/mean_cylinder_angle_rate

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
        # If provided, use this threshold to compute success from goal_hold_counter.
        # If None, we try to infer from the underlying env.
        self.success_hold_H = success_hold_H

    def _unwrap_base_env(self):
        """
        Try to extract the underlying single-env object (Monitor-wrapped) to read env attributes like success_hold_H.
        Works for common wrappers: VecNormalize -> VecEnv -> Monitor -> Env.
        Returns None if not found.
        """
        e = self.eval_env
        # Unwrap VecNormalize (has attribute 'env' referencing the VecEnv)
        if hasattr(e, 'env'):
            e = e.env
        # VecEnv often has .envs (a list) of underlying envs
        try:
            if hasattr(e, 'envs') and len(e.envs) > 0:
                candidate = e.envs[0]
                # If Monitor wrapped it will have .env
                if hasattr(candidate, 'env'):
                    base = candidate.env
                else:
                    base = candidate
                # If Monitor wrapped again (rare), unwrap
                if hasattr(base, 'env'):
                    base = base.env
                return base
        except Exception:
            pass
        return None

    def _on_step(self) -> bool:
        # run only every eval_freq timesteps
        if (self.num_timesteps - self._last_eval_step) < self.eval_freq:
            return True
        self._last_eval_step = self.num_timesteps

        # Try to infer success_hold_H from the underlying env if not given
        inferred_success_H = self.success_hold_H
        if inferred_success_H is None:
            base_env = self._unwrap_base_env()
            if base_env is not None and hasattr(base_env, 'success_hold_H'):
                try:
                    inferred_success_H = int(getattr(base_env, 'success_hold_H'))
                except Exception:
                    inferred_success_H = None

        t0 = time.time()
        rewards = []
        lengths = []
        successes = []
        angle_means = []
        angle_rate_means = []

        for _ in range(self.n_eval_episodes):
            # robust reset
            res = self.eval_env.reset()
            if isinstance(res, tuple) and len(res) == 2:
                obs, info = res
            else:
                obs = res
                info = {}

            ep_rew = 0.0
            ep_len = 0
            ep_success = False

            # per-episode lists for cylinder angle metrics (if provided by env via info)
            angles = []
            angle_rates = []

            while True:
                # model.predict expects the same obs shape as training; for VecEnvs the callback uses single env so obs is unwrapped already
                action, _ = self.model.predict(obs, deterministic=True)
                step_res = self.eval_env.step(action)

                # Gymnasium function-style non-vector: (obs, reward, terminated, truncated, info)
                if isinstance(step_res, tuple) and len(step_res) == 5:
                    obs, reward, terminated, truncated, info = step_res
                    done = bool(terminated or truncated)
                # Gym classic: (obs, reward, done, info)
                elif isinstance(step_res, tuple) and len(step_res) == 4:
                    obs, reward, done, info = step_res
                    done = bool(done)
                else:
                    # vecenv-like: obs, rewards, dones, infos
                    obs, rewards_arr, dones_arr, infos_arr = step_res
                    reward = float(rewards_arr[0]) if hasattr(rewards_arr, "__len__") else float(rewards_arr)
                    done = bool(dones_arr[0]) if hasattr(dones_arr, "__len__") else bool(dones_arr)
                    info = infos_arr[0] if hasattr(infos_arr, "__len__") else infos_arr

                ep_rew += float(reward)
                ep_len += 1

                # collect cylinder angle info (if present)
                if isinstance(info, dict):
                    if "cylinder_angle" in info:
                        try:
                            angles.append(float(info["cylinder_angle"]))
                        except Exception:
                            pass
                    if "cylinder_angle_rate" in info:
                        try:
                            angle_rates.append(float(info["cylinder_angle_rate"]))
                        except Exception:
                            pass

                    # detect success flag if provided by env
                    if "is_success" in info:
                        ep_success = ep_success or bool(info["is_success"])
                    elif "success" in info:
                        ep_success = ep_success or bool(info["success"])
                    elif "goal_hold_counter" in info and inferred_success_H is not None:
                        try:
                            ep_success = ep_success or (int(info["goal_hold_counter"]) >= int(inferred_success_H))
                        except Exception:
                            pass
                    elif "at_goal" in info:
                        ep_success = ep_success or bool(info["at_goal"])

                if done or ep_len > 10000:
                    break

            rewards.append(ep_rew)
            lengths.append(ep_len)
            successes.append(1.0 if ep_success else 0.0)

            # compute per-episode mean/final angles to summarize (prefer final if available)
            if len(angles) > 0:
                angle_means.append(float(np.mean(angles)))
            else:
                angle_means.append(0.0)
            if len(angle_rates) > 0:
                angle_rate_means.append(float(np.mean(angle_rates)))
            else:
                angle_rate_means.append(0.0)

        mean_reward = float(np.mean(rewards))
        mean_len = float(np.mean(lengths))
        success_rate = float(np.mean(successes))

        mean_angle = float(np.mean(angle_means)) if len(angle_means) > 0 else 0.0
        mean_angle_rate = float(np.mean(angle_rate_means)) if len(angle_rate_means) > 0 else 0.0

        # log scalars to SB3/Tensorboard via self.logger
        try:
            self.logger.record("custom_eval/mean_reward", mean_reward)
            self.logger.record("custom_eval/mean_ep_length", mean_len)
            self.logger.record("custom_eval/success_rate", success_rate)
            # cylinder metrics (radians / rad/s by env default)
            self.logger.record("custom_eval/mean_cylinder_angle", mean_angle)
            self.logger.record("custom_eval/mean_cylinder_angle_rate", mean_angle_rate)
            # dump immediately at evaluation step
            self.logger.dump(self.num_timesteps)
        except Exception:
            pass

        # save best model by mean_reward
        if self.best_model_save_path is not None and mean_reward > self.best_mean_reward:
            self.best_mean_reward = mean_reward
            try:
                save_path = os.path.join(self.best_model_save_path, f"best_mean_reward_{int(self.num_timesteps)}")
                self.model.save(save_path)
                if self.verbose:
                    print(f"[CustomEval] New best mean reward {mean_reward:.3f} at step {self.num_timesteps}; model saved to {save_path}")
            except Exception as e:
                print("Failed saving best model from CustomEvalCallback:", e)

        if self.verbose:
            print(f"[CustomEval] step={self.num_timesteps} mean_reward={mean_reward:.3f} mean_len={mean_len:.1f} "
                  f"success_rate={success_rate:.2%} mean_cyl_angle={mean_angle:.4f} rad mean_cyl_angle_rate={mean_angle_rate:.4f} rad/s "
                  f"time={time.time()-t0:.1f}s")

        return True


def main():
    # 1) Vectorized envs
    make_env = make_env_fn(
        model_path="assets/panda_tray/panda_tray_cylinder.xml",
        config_path="config.yaml",
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