"""
mynettest.py
Practical training script for TrayPoseEnv.
- AdvancedActorCritic (SAC-style)
- Multi-step rollout
- Persistent learning (optimizer.step())
- Regular + best checkpointing
"""
import glob
import signal
import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

# ----------------------------------------------------------------------
# Project root (adjust only if you move the file)
# ----------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(PROJECT_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ----------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------
from envs.traypose.traypose_env import TrayPoseEnv
from AdvancedActorCritic import AdvancedActorCritic
# ----------------------------------------------------------------------
# Hyper-parameters (tune here)
# ----------------------------------------------------------------------
OBS_DIM               = 36
ACT_DIM               = 7
HIDDEN_SIZE           = 256
NUM_EPISODES          = 10_000          # total training episodes
MAX_STEPS_PER_EPISODE = 150            # env max_steps
GAMMA                 = 0.99
LR                    = 3e-4
EPS                   = 1e-6
CKPT_DIR              = "checkpoints"
SAVE_EVERY            = 100            # regular checkpoint interval




# ----------------------------------------------------------------------
# Helper: compute log-prob of a squashed Gaussian
# ----------------------------------------------------------------------
def squash_log_prob(mu, std, action, eps=EPS):
    pre = torch.atanh(torch.clamp(action, -1 + eps, 1 - eps))
    dist = Normal(mu, std)
    logp = dist.log_prob(pre).sum(dim=-1)                     # sum over action dims
    logp -= torch.log(1 - action.pow(2) + eps).sum(dim=-1)
    return logp

# ----------------------------------------------------------------------
# Graceful shutdown on Ctrl-C
# ----------------------------------------------------------------------
class GracefulKiller:
    kill_now = False
    def __init__(self):
        signal.signal(signal.SIGINT,  self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, *args):
        self.kill_now = True

# ----------------------------------------------------------------------
# Core training loop
# ----------------------------------------------------------------------
def train(model, optimizer, env, start_episode=0):
    os.makedirs(CKPT_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # Global variable to track best reward (persists across function calls)
    # ------------------------------------------------------------------
    if "best_reward" not in train.__globals__:
        train.__globals__["best_reward"] = float("-inf")

    # ---- create the killer --------------------------------------------------
    killer = GracefulKiller()
    # ----------------------------------------------------------------------
    for ep in range(start_episode, NUM_EPISODES):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        step = 0

        # ----- trajectory containers -----
        obs_list      = []
        action_list   = []
        log_prob_list = []
        value_list    = []
        reward_list   = []
        done_list     = []

        # ----- allow interruption between episodes -------------------------
        if killer.kill_now:
            print("\nReceived interrupt – saving final checkpoint and exiting...")
            final_path = f"{CKPT_DIR}/interrupted_ep{ep}.pt"
            torch.save({
                "episode": ep,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "reward": 0.0,
            }, final_path)
            print(f"   → Saved: {final_path}")
            break

        # --------------------------------------------------------------
        # Rollout (one full episode)
        # --------------------------------------------------------------
        while not done and step < MAX_STEPS_PER_EPISODE:
            obs_t = torch.from_numpy(obs).float().unsqueeze(0)
            obs_list.append(obs_t)

            action_t, mu, std, value, _, _ = model(obs_t)
            action = action_t.squeeze(0).detach().cpu().numpy()

            # log-prob for the *sampled* action
            logp = squash_log_prob(mu.squeeze(0), std.squeeze(0), action_t.squeeze(0))
            log_prob_list.append(logp)
            value_list.append(value.squeeze(0))
            action_list.append(action_t.squeeze(0))

            # env step
            next_obs, r, term, trunc, info = env.step(action)
            done = term or trunc
            ep_reward += r
            reward_list.append(r)
            done_list.append(float(done))

            obs = next_obs
            step += 1

        # --------------------------------------------------------------
        # Bootstrap value for the final state
        # --------------------------------------------------------------
        next_obs_t = torch.from_numpy(obs).float().unsqueeze(0)
        with torch.no_grad():
            _, _, _, next_v, _, _ = model(next_obs_t)
        next_v = next_v.item()

        # --------------------------------------------------------------
        # Compute returns & advantages (A2C style)
        # --------------------------------------------------------------
        returns, advs = [], []
        R = next_v * (1.0 - done_list[-1])
        for r, d, v in zip(reversed(reward_list), reversed(done_list), reversed(value_list)):
            R = r + GAMMA * R * (1.0 - d)
            adv = R - v.detach().item()
            returns.append(R)
            advs.append(adv)
        returns = list(reversed(returns))
        advs    = list(reversed(advs))

        returns_t = torch.tensor(returns, dtype=torch.float32)
        advs_t    = torch.tensor(advs,    dtype=torch.float32)

        # --------------------------------------------------------------
        # Losses
        # --------------------------------------------------------------
        values_t   = torch.stack(value_list)                 # (T,)
        critic_loss = F.mse_loss(values_t, returns_t)

        logps_t    = torch.stack(log_prob_list)              # (T,)
        actor_loss = -(logps_t * advs_t).mean()

        total_loss = actor_loss + 0.5 * critic_loss

        # --------------------------------------------------------------
        # Optimizer step
        # --------------------------------------------------------------
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # --------------------------------------------------------------
        # Logging
        # --------------------------------------------------------------
        print(f"Episode: {ep+1:5d}/{NUM_EPISODES} | Reward gained: {ep_reward:6.2f} | "
              f"Total Steps {step:3d} | Loss {total_loss.item():.4f}")

        # --------------------------------------------------------------
        # Regular checkpoint
        # --------------------------------------------------------------
       
        if (ep + 1) % SAVE_EVERY == 0:
            path = f"{CKPT_DIR}/ep{ep+1}.pt"
            torch.save({
                "episode": ep + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "reward": ep_reward,
            }, path)
            print(f"   Regular checkpoint: {path}")

            # ---- KEEP ONLY THE LAST 5 REGULAR CHECKPOINTS ----
            pattern = f"{CKPT_DIR}/ep*.pt"
            all_ckpts = sorted(
                glob.glob(pattern),
                key=os.path.getmtime,   # sort by modification time
                reverse=True            # newest first
            )
            for old in all_ckpts[5:]:   # everything after the 5 newest
                try:
                    os.remove(old)
                    
                except OSError:
                    pass
            

        # --------------------------------------------------------------
        # Best-reward checkpoint
        # --------------------------------------------------------------
        if ep_reward > train.__globals__["best_reward"]:
            train.__globals__["best_reward"] = ep_reward
            best_path = f"{CKPT_DIR}/best.pt"
            torch.save({
                "episode": ep + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "reward": ep_reward,
            }, best_path)
            print(f"   *** NEW BEST *** Reward {ep_reward:.2f} → {best_path}")

# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TrayPose with checkpointing")
    parser.add_argument(
        "--load_checkpoint",
        type=str,
        default=None,
        help="Path to .pt checkpoint to resume from (e.g. checkpoints/ep500.pt)",
    )
    args = parser.parse_args()

    env     = TrayPoseEnv()                     # you can pass noise args here
    model   = AdvancedActorCritic()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    start_ep = 0
    if args.load_checkpoint:
        ckpt = torch.load(args.load_checkpoint, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_ep = ckpt["episode"]
        print(f"Loaded {args.load_checkpoint} – resuming from episode {start_ep}")

    train(model, optimizer, env, start_episode=start_ep)
    env.close()