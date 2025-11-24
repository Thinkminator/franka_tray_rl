#!/usr/bin/env python3
"""
Training script for TrayPose environment with curriculum learning.
"""

import os
import sys
import time
import json
import numpy as np
from collections import deque
import argparse
from datetime import datetime

import gymnasium as gym
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env

# Project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
os.chdir(PROJECT_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import custom env
from envs.traypose.traypose_env import TrayPoseEnv


class CustomEvalCallback(BaseCallback):
    """
    Custom callback for evaluating the model and updating curriculum.
    """
    
    def __init__(self, eval_env, eval_freq=1000, n_eval_episodes=10, verbose=1, 
                 log_dir="./logs", max_episodes_without_progress=2000):
        super(CustomEvalCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.log_dir = log_dir
        self.max_episodes_without_progress = max_episodes_without_progress
        
        # Tracking variables
        self._last_eval_step = 0
        self._last_saved_step = 0  # For checkpoint saving
        self.total_eval_episodes = 0
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.episode_successes = deque(maxlen=100)
        self.episode_phases = deque(maxlen=100)
        
        # For early stopping if stuck in a phase
        self.episodes_in_current_phase = 0
        self.current_phase = 1
        self.best_mean_reward = -np.inf
        self.episodes_since_best_reward = 0
        
        # Track best reward per phase
        self.best_reward_per_phase = {}
        
        # Setup TensorBoard logger
        self.tb_logger = TensorBoardOutputFormat(log_dir)

    def _sync_curriculum_phase(self):
        """
        Synchronize the evaluation environment's curriculum phase with the training environment.
        Returns:
            float: the training phase (or None on failure)
        """
        try:
            if not hasattr(self.model, "env"):
                return None

            train_vec = self.model.env

            # Try VecEnv.get_attr first (works for DummyVecEnv, SubprocVecEnv, VecNormalize)
            try:
                train_phases = train_vec.get_attr("current_phase")
                train_phase = int(max(train_phases)) if train_phases else 1.0
            except Exception:
                # Fallback: unwrap to the first inner env
                train_env_instance = train_vec
                if hasattr(train_env_instance, "venv"):
                    train_env_instance = train_env_instance.venv
                if hasattr(train_env_instance, "envs") and len(train_env_instance.envs) > 0:
                    train_env_instance = train_env_instance.envs[0]
                train_phase = int(getattr(train_env_instance, "current_phase", 1.0))

            # Set evaluation env(s) to this phase using VecEnv.set_attr if available
            try:
                self.eval_env.set_attr("current_phase", train_phase)
            except Exception:
                # Fallback: set directly on inner eval env(s)
                if hasattr(self.eval_env, "envs"):
                    for e in self.eval_env.envs:
                        setattr(e, "current_phase", train_phase)
                else:
                    setattr(self.eval_env, "current_phase", train_phase)

            print(f"[Eval] Evaluation environment phase synchronized to {train_phase}")
            return train_phase

        except Exception as e:
            print(f"[Eval] Failed to sync phases: {e}")
            return None

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.
        """
        # Check if it's time to evaluate, else skip this
        if (self.num_timesteps - self._last_eval_step) < self.eval_freq:
            return True
        
        # Synchronize curriculum phases before evaluation
        training_phase = self._sync_curriculum_phase()
        if training_phase is None:
            training_phase = 1.0  # safe default
        
        self._last_eval_step = self.num_timesteps
        
        # Save model every 100,000 timesteps
        if self.num_timesteps // 100000 > self._last_saved_step // 100000:
            checkpoint_path = os.path.join(self.log_dir, f"{self.model.__class__.__name__}_checkpoint_{self.num_timesteps}")
            self.model.save(checkpoint_path)
            # If using VecNormalize, also save it
            try:
                if hasattr(self.model.get_env(), 'save'):
                    norm_path = os.path.join(self.log_dir, f"vecnormalize_checkpoint_{self.num_timesteps}.pkl")
                    self.model.get_env().save(norm_path)
            except Exception:
                pass  # Not all envs can be saved
            print(f"[Checkpoint] Model saved at {self.num_timesteps} timesteps to {checkpoint_path}")
            self._last_saved_step = self.num_timesteps
        
        # Evaluate the model
        rewards, lengths = [], []
        angles, offsets = [], []
        truncated_count = success_count = 0
        drop_terminated_count = topple_terminated_count = 0
        phase_values = []
        consecutive_successes_values = []
        success_thresholds = []

        for _ in range(self.n_eval_episodes):
            try:
                self.eval_env.set_attr("current_phase", training_phase)
            except Exception:
                if hasattr(self.eval_env, "envs"):
                    for e in self.eval_env.envs:
                        setattr(e, "current_phase", training_phase)
                else:
                    setattr(self.eval_env, "current_phase", training_phase)
            
            reset_res = self.eval_env.reset()
            
            if isinstance(reset_res, tuple) and len(reset_res) == 2:
                obs, info = reset_res
            else:
                obs = reset_res
                info = {}
            done = False
            ep_rew, ep_len = 0, 0
            ep_success = ep_truncated = False
            ep_drop_terminated = ep_topple_terminated = False
            angle = offset = 0
            phase_val = 1
            cons_successes = 0
            success_thresh = 0

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
                    phase_val = info.get("phase", 1)
                    cons_successes = info.get("consecutive_successes", 0)
                    success_thresh = info.get("success_threshold", 0)

            rewards.append(ep_rew)
            lengths.append(ep_len)
            angles.append(angle)
            offsets.append(offset)
            phase_values.append(phase_val)
            consecutive_successes_values.append(cons_successes)
            success_thresholds.append(success_thresh)
            
            if ep_truncated:
                truncated_count += 1
            if ep_success:
                success_count += 1
            if ep_drop_terminated:
                drop_terminated_count += 1
            if ep_topple_terminated:
                topple_terminated_count += 1

        self.total_eval_episodes += self.n_eval_episodes  # Increment total episodes evaluated

        total_eps = self.n_eval_episodes
        mean_reward = np.mean(rewards)
        mean_len = np.mean(lengths)
        mean_angle = np.mean(angles)
        mean_offset = np.mean(offsets)
        mean_phase = np.mean(phase_values)
        mean_consecutive_successes = np.mean(consecutive_successes_values)
        max_consecutive_successes = np.max(consecutive_successes_values)
        mean_success_threshold = np.mean(success_thresholds)
        min_success_threshold = np.min(success_thresholds)

        # Update tracking variables
        self.episode_rewards.append(mean_reward)
        self.episode_lengths.append(mean_len)
        self.episode_successes.append(success_count / total_eps)
        self.episode_phases.append(mean_phase)
        
        # Print current phase information
        print(f"[Eval] Current curriculum phase: {training_phase} (Consecutive successes: {mean_consecutive_successes}/{min_success_threshold})")

        # Log to SB3 logger (which will write to TensorBoard)
        self.logger.record("eval/mean_reward", mean_reward)
        self.logger.record("eval/mean_ep_length", mean_len)
        self.logger.record("eval/success_rate", success_count / total_eps)
        self.logger.record("eval/drop_rate", drop_terminated_count / total_eps)
        self.logger.record("eval/topple_rate", topple_terminated_count / total_eps)
        self.logger.record("eval/truncated_rate", truncated_count / total_eps)
        self.logger.record("eval/mean_cylinder_angle", mean_angle)
        self.logger.record("eval/mean_cylinder_offset", mean_offset)
        self.logger.record("curriculum/mean_phase", mean_phase)
        self.logger.record("curriculum/consecutive_successes", mean_consecutive_successes)
        self.logger.record("curriculum/success_threshold", mean_success_threshold)
        self.logger.record("episode/timesteps", self.num_timesteps)

        # Flush the logger to write to TensorBoard
        self.logger.dump(self.num_timesteps)

        # Print only every 100 episodes
        if self.verbose and (self.total_eval_episodes % 100 == 0):
            print(f"[Eval] step={self.num_timesteps} Current start phase: {training_phase} mean_reward={mean_reward:.3f} "
                  f"len={mean_len:.1f}, success%={100*success_count/total_eps:.2f}, "
                  f"drop%={100*drop_terminated_count/total_eps:.2f}, "
                  f"topple%={100*topple_terminated_count/total_eps:.2f}, "
                  f"trunc%={100*truncated_count/total_eps:.2f}, "
                  f"phase={mean_phase:.1f}, max_cons_success={max_consecutive_successes:.1f}")

        # Save best model for each phase
        current_phase_int = int(training_phase)
        if current_phase_int not in self.best_reward_per_phase or mean_reward > self.best_reward_per_phase[current_phase_int]:
            self.best_reward_per_phase[current_phase_int] = mean_reward
            best_model_path = os.path.join(self.log_dir, f"{self.model.__class__.__name__}_best_phase_{current_phase_int}")
            self.model.save(best_model_path)
            print(f"[Eval] New best model for phase {current_phase_int} saved with reward {mean_reward:.3f}")

        return True


def make_env(config_path=None, seed=0):
    def _init():
        env = TrayPoseEnv(config_path=config_path)
        # Initialize with seed for deterministic behavior
        env.reset(seed=seed)
        return env
    return _init


def train_traypose(args):
    """
    Main training function for the TrayPose environment.
    """
    # Create log directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"./training/logs/traypose_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    
    # Save arguments
    with open(os.path.join(log_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    
    print(f"[INFO] Starting training with log directory: {log_dir}")
    
    # Create environment with proper config and seed handling
    print("[INFO] Creating environment...")
    num_envs = 1  # Adjust for parallel environments if needed
    env_fns = [make_env(args.config_path, seed=args.seed + i) for i in range(num_envs)]
    env = DummyVecEnv(env_fns)
    
    # === ADD THIS: Set training env to start at phase 5 ===
    try:
        env.set_attr("current_phase", 5)
        print("[INFO] Set training environment to start at phase 5")
    except Exception as e:
        # Fallback: set directly on inner envs
        for i, inner_env in enumerate(env.envs):
            try:
                setattr(inner_env, "current_phase", 5)
                print(f"[INFO] Set training sub-environment {i} to phase 5")
            except Exception as inner_e:
                print(f"[WARNING] Could not set phase for sub-env {i}: {inner_e}")
    # =====================================================
    
    if args.normalize:
        print("[INFO] Normalizing environment observations...")
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
    
    # Create evaluation environment
    eval_env = DummyVecEnv([make_env(args.config_path, seed=args.seed + 1000)])
    if args.normalize:
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.)
    
    # Create the model
    print(f"[INFO] Creating {args.algorithm} model...")
    if args.algorithm == "SAC":
        model = SAC(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=log_dir,
            learning_rate=args.learning_rate,
            buffer_size=args.buffer_size,
            learning_starts=args.learning_starts,
            batch_size=args.batch_size,
            tau=args.tau,
            gamma=args.gamma,
            train_freq=1,
            gradient_steps=1,
            ent_coef="auto",
            target_update_interval=1,
            policy_kwargs=dict(net_arch=[256, 256])
        )
    elif args.algorithm == "PPO":
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=log_dir,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            max_grad_norm=args.max_grad_norm,
            policy_kwargs=dict(net_arch=[256, 256])
        )
    else:
        raise ValueError(f"Unsupported algorithm: {args.algorithm}")
    
    # Create evaluation callback
    eval_callback = CustomEvalCallback(
        eval_env,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        verbose=1,
        log_dir=log_dir,
        max_episodes_without_progress=args.max_episodes_without_progress
    )
    
    # Train the model
    print("[INFO] Starting training...")
    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=eval_callback,
            tb_log_name=args.algorithm
        )
    except KeyboardInterrupt:
        print("[INFO] Training interrupted by user")
    except Exception as e:
        print(f"[ERROR] Training failed with exception: {e}")
        raise
    
    # Save the model
    model_path = os.path.join(log_dir, f"{args.algorithm}_final")
    model.save(model_path)
    if args.normalize:
        env.save(os.path.join(log_dir, "vecnormalize_final.pkl"))
    
    print(f"[INFO] Model saved to {model_path}")
    print("[INFO] Training completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TrayPose environment with curriculum learning")
    
    # Environment arguments
    parser.add_argument("--config-path", type=str, default="envs/traypose/config.yaml",
                        help="Path to the environment configuration file")
    
    # Algorithm arguments
    parser.add_argument("--algorithm", type=str, choices=["SAC", "PPO"], default="PPO",
                        help="RL algorithm to use")
    parser.add_argument("--total-timesteps", type=int, default=1000000,
                        help="Total number of training timesteps")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
                        help="Learning rate")
    
    # SAC specific arguments
    parser.add_argument("--buffer-size", type=int, default=10000,
                        help="Replay buffer size")
    parser.add_argument("--learning-starts", type=int, default=10000,
                        help="How many steps to collect before training starts")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Batch size for training")
    parser.add_argument("--tau", type=float, default=0.005,
                        help="Target network update rate")
    parser.add_argument("--gamma", type=float, default=0.9995,
                        help="Discount factor")
    
    # PPO specific arguments
    parser.add_argument("--n-steps", type=int, default=2048,
                        help="Number of steps to run for each environment per update")
    parser.add_argument("--n-epochs", type=int, default=10,
                        help="Number of epochs when optimizing the surrogate loss")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
                        help="Factor for trade-off of bias vs variance for Generalized Advantage Estimator")
    parser.add_argument("--clip-range", type=float, default=0.2,
                        help="Clipping parameter")
    parser.add_argument("--ent-coef", type=float, default=0.0,
                        help="Entropy coefficient for the loss calculation")
    parser.add_argument("--vf-coef", type=float, default=0.5,
                        help="Value function coefficient for the loss calculation")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
                        help="The maximum value for the gradient clipping")
    
    # Training arguments
    parser.add_argument("--normalize", action="store_true",
                        help="Normalize observations and rewards")
    parser.add_argument("--eval-freq", type=int, default=5000,
                        help="Evaluate the model every N timesteps")
    parser.add_argument("--n-eval-episodes", type=int, default=10,
                        help="Number of episodes to evaluate")
    parser.add_argument("--max-episodes-without-progress", type=int, default=2000,
                        help="Maximum episodes to spend in a phase without progress before stopping")
    
    # Seed
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed")
    
    args = parser.parse_args()
    
    # Set random seeds
    np.random.seed(args.seed)
    
    # Run training
    train_traypose(args)