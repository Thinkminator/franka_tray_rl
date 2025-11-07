#!/usr/bin/env python3
import os
import sys
import time
import numpy as np
import gymnasium as gym
import wandb
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


# ------------------- OPTIONAL: Wandb Logging -------------------
class WandbCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        if self.n_calls % 100 == 0:
            wandb.log({
                "train/episode_reward": self.locals.get("rewards", [0])[-1] if "rewards" in self.locals else 0,
                "train/step": self.num_timesteps,
            })
        return True


# ------------------- Zero-action warmup -------------------
class ZeroActionFirstStepsWrapper(gym.Wrapper):
    def __init__(self, env, zero_steps=10):
        super().__init__(env)
        self.zero_steps = zero_steps
        self.current_step = 0

    def reset(self, **kwargs):
        self.current_step = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        if self.current_step < self.zero_steps:
            zero_action = np.zeros_like(action)
            obs, reward, terminated, truncated, info = self.env.step(zero_action)
        else:
            obs, reward, terminated, truncated, info = self.env.step(action)
        self.current_step += 1
        return obs, reward, terminated, truncated, info


# ------------------- ENV Factory -------------------
def make_env_fn(model_path="assets/panda_tray/panda_tray_cylinder.xml",
                config_path="envs/traypose/config.yaml",
                obs_noise_std_pos=0.0,
                obs_noise_std_vel=0.0,
                use_jacobian_tray_obs=False,
                zero_steps=10,
                tuned_Kq=None,
                tuned_Dq=None,
                start_phase=None):
    def _thunk():
        env = TrayPoseEnv(
            model_path=model_path,
            config_path=config_path,
            obs_noise_std_pos=obs_noise_std_pos,
            obs_noise_std_vel=obs_noise_std_vel,
            use_jacobian_tray_obs=use_jacobian_tray_obs,
        )
        # Apply tuned gains if provided
        if tuned_Kq is not None and tuned_Dq is not None:
            env.Kq = tuned_Kq.copy()
            env.Dq = tuned_Dq.copy()
        # Optional: set initial curriculum phase for each env
        if start_phase is not None and hasattr(env, "set_phase"):
            env.set_phase(int(start_phase))
        env = ZeroActionFirstStepsWrapper(env, zero_steps=zero_steps)
        env = Monitor(env)
        return env
    return _thunk


# ------------------- Helper for Resets -------------------
def _reset_env(env):
    res = env.reset()
    if isinstance(res, tuple) and len(res) == 2:
        obs, info = res
    else:
        obs = res
        info = {}
    return obs, info


# ------------------- Custom Eval Callback (kept) -------------------
class CustomEvalCallback(BaseCallback):
    def __init__(self, eval_env, n_eval_episodes=30, eval_freq=5000,
                 best_model_save_path=None, verbose=1, success_hold_H=None):
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

        rewards, lengths = [], []
        angles, offsets = [], []
        truncated_count = success_count = 0
        drop_terminated_count = topple_terminated_count = 0

        for _ in range(self.n_eval_episodes):
            obs, info = _reset_env(self.eval_env)
            done = False
            ep_rew, ep_len = 0, 0
            ep_success = ep_truncated = False
            ep_drop_terminated = ep_topple_terminated = False
            angle = offset = 0

            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                step_res = self.eval_env.step(action)

                if len(step_res) == 5:
                    obs, reward, terminated, truncated, info = step_res
                    done = terminated or truncated
                else:
                    obs, reward, done, info = step_res
                    terminated, truncated = done, False

                ep_rew += reward
                ep_len += 1
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

        if self.verbose:
            print(f"[CustomEval] step={self.num_timesteps} mean_reward={mean_reward:.3f} "
                  f"len={mean_len:.1f}, success%={100*success_count/total_eps:.2f}, "
                  f"drop%={100*drop_terminated_count/total_eps:.2f}, "
                  f"topple%={100*topple_terminated_count/total_eps:.2f}, "
                  f"trunc%={100*truncated_count/total_eps:.2f}")

        self.logger.record("custom_eval/mean_reward", mean_reward)
        self.logger.record("custom_eval/success_rate", 100*success_count/total_eps)
        return True


# ------------------- Curriculum Learning -------------------
def set_all_phase(venv, phase: int):
    """Recursively set phase for all sub-environments, even through VecNormalize and Monitor wrappers."""
    try:
        if hasattr(venv, 'envs'):  # VecEnv container
            for e in venv.envs:
                if hasattr(e, 'set_phase'):
                    e.set_phase(phase)
                elif hasattr(e, 'env') and hasattr(e.env, 'set_phase'):
                    e.env.set_phase(phase)
        elif hasattr(venv, 'venv'):
            set_all_phase(venv.venv, phase)
        elif hasattr(venv, 'env'):
            set_all_phase(venv.env, phase)
    except Exception as ex:
        print(f"[Curriculum] Warning: failed to set phase={phase} -> {ex}")


class CurriculumCallback(BaseCallback):
    """Switch curriculum phases during training."""
    def __init__(self, eval_env, switch_steps=(200_000, 400_000), verbose=1):
        super().__init__(verbose)
        self.switch_steps = switch_steps
        self.phase = 0
        self.eval_env = eval_env

    def _on_training_start(self):
        set_all_phase(self.training_env, 0)
        set_all_phase(self.eval_env, 0)
        wandb.log({"curriculum/phase": 0})
        print("[Curriculum] Started in Phase 0 (balance only)")

    def _on_step(self) -> bool:
        gs = self.model.num_timesteps
        if self.phase == 0 and gs >= self.switch_steps[0]:
            self.phase = 1
            set_all_phase(self.training_env, 1)
            set_all_phase(self.eval_env, 1)
            wandb.log({"curriculum/phase": 1})
            print(f"[Curriculum] Moved to Phase 1 at {gs} steps")
        elif self.phase == 1 and gs >= self.switch_steps[1]:
            self.phase = 2
            set_all_phase(self.training_env, 2)
            set_all_phase(self.eval_env, 2)
            wandb.log({"curriculum/phase": 2})
            print(f"[Curriculum] Moved to Phase 2 at {gs} steps")
        return True


# ------------------- Main -------------------
def main():
    wandb.init(
        project="franka_tray_rl",
        sync_tensorboard=True,
        config={
            "n_envs": 8,
            "n_steps": 1024,
            "batch_size": 256,
            "total_timesteps": 1_000_000,
            "learning_rate": 1e-4,
            "gamma": 0.99,
        },
        name="ppo_traypose_curriculum",
        save_code=True,
    )

    # 1) One-time PD autotune on a single temp env (optional but recommended)
    temp_env = make_env_fn(
        model_path="assets/panda_tray/panda_tray_cylinder.xml",
        config_path="envs/traypose/config.yaml",
        obs_noise_std_pos=0.0,
        obs_noise_std_vel=0.0,
        use_jacobian_tray_obs=False,
        zero_steps=10,
        tuned_Kq=None,
        tuned_Dq=None,
        start_phase=0,  # tune for Phase 0 balance
    )()

    # Unwrap: Monitor(ZeroActionFirstStepsWrapper(TrayPoseEnv))
    base_env = temp_env
    while hasattr(base_env, "env"):
        base_env = base_env.env  # peel wrappers until reaching TrayPoseEnv

    # Now base_env is TrayPoseEnv
    if hasattr(base_env, "set_phase"):
        base_env.set_phase(0)
    base_env.reset()
    Kq_new, Dq_new = base_env.autotune_hold_pose(hold_seconds=2.0, max_pos_error_rad=0.01, verbose=True)
    tuned_Kq, tuned_Dq = Kq_new.copy(), Dq_new.copy()
    temp_env.close()

    # 2) Build env factory using tuned gains and starting in Phase 0
    make_env = make_env_fn(
        zero_steps=10,
        tuned_Kq=tuned_Kq,
        tuned_Dq=tuned_Dq,
        start_phase=0
    )

    # Directories
    log_dir = "training/logs"
    save_dir = "training/checkpoints"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    n_envs = 8
    total_timesteps = 1_000_000

    # Vectorized envs
    train_env = make_vec_env(make_env, n_envs=n_envs, monitor_dir=log_dir)
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    eval_env = make_vec_env(make_env, n_envs=1, monitor_dir=log_dir)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    # Share VecNormalize stats
    try:
        eval_env.obs_rms = train_env.obs_rms
    except Exception as e:
        print("Warning: could not share obs_rms:", e)

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        verbose=1,
        tensorboard_log=log_dir,
        learning_rate=1e-4,
        n_steps=1024,
        batch_size=256,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=1e-3,
        vf_coef=0.5,
        n_epochs=10,
        seed=42,
        device="cpu",
    )

    eval_callback = EvalCallback(eval_env, best_model_save_path=save_dir,
                                 log_path=log_dir, eval_freq=5_000,
                                 deterministic=True, render=False, n_eval_episodes=10)

    custom_eval_cb = CustomEvalCallback(eval_env=eval_env, eval_freq=5_000, verbose=1)
    checkpoint_callback = CheckpointCallback(save_freq=50_000, save_path=save_dir, name_prefix="rl_model")
    wandb_callback = WandbCallback(verbose=1)
    curriculum_callback = CurriculumCallback(eval_env=eval_env, switch_steps=(200_000, 400_000), verbose=1)

    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, custom_eval_cb, checkpoint_callback, wandb_callback, curriculum_callback],
    )

    model.save(os.path.join(save_dir, "ppo_traypose_final"))
    train_env.save(os.path.join(save_dir, "vecnormalize.pkl"))

    print(f"Training complete. Models saved to {save_dir}")


if __name__ == "__main__":
    main()