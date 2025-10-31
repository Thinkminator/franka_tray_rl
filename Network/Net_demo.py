"""
Net_demo.py
Demo script that:
1. Instantiates AdvancedActorCritic (same as training)
2. Runs a short rollout (default 10 steps)
3. Computes correct log-prob for squashed Gaussian
4. Performs A2C-style loss + one optimizer step
5. Prints everything (actions, mu, std, values, gradients)
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

# ----------------------------------------------------------------------
# Project root
# ----------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(PROJECT_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ----------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------
from envs.traypose.traypose_env import TrayPoseEnv
from AdvancedActorCritic import AdvancedActorCritic  # <-- same file used in training

# ----------------------------------------------------------------------
# Hyper-parameters (demo only)
# ----------------------------------------------------------------------
NUM_STEPS = 20          # rollout length
GAMMA     = 0.99
LR        = 3e-4
EPS       = 1e-6

# ----------------------------------------------------------------------
# Helper: log-prob of squashed Gaussian
# ----------------------------------------------------------------------
def squash_log_prob(mu, std, action):
    """Compute log π(a|s) for tanh-squashed Gaussian"""
    pre = torch.atanh(torch.clamp(action, -1.0 + EPS, 1.0 - EPS))
    dist = Normal(mu, std)
    logp = dist.log_prob(pre).sum(dim=-1)
    logp -= torch.log(1.0 - action.pow(2) + EPS).sum(dim=-1)
    return logp

# ----------------------------------------------------------------------


# Main demo
# ----------------------------------------------------------------------
def main():
    print("=== AdvancedActorCritic Demo ===")
    env = TrayPoseEnv()
    model = AdvancedActorCritic()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    
    model.train()
    print(model)

    # --- OPTIONAL: Load checkpoint ---
    ckpt_path = "checkpoints/best.pt"
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded trained model from {ckpt_path}")
    # ---------------------------------

    obs, _ = env.reset()
    print(f"\nInitial obs:  {obs}")

    # Containers
    obs_list     = []
    action_list  = []
    mu_list      = []
    std_list     = []
    logp_list    = []
    value_list   = []
    v1_list      = []
    v2_list      = []
    reward_list  = []
    done_list    = []

    for step in range(NUM_STEPS):
        obs_t = torch.from_numpy(obs).float().unsqueeze(0)
        obs_list.append(obs_t)

        # Forward
        action_t, mu, std, value, v1, v2 = model(obs_t)
        action = action_t.squeeze(0).detach().cpu().numpy()

        # Log prob
        logp = squash_log_prob(mu.squeeze(0), std.squeeze(0), action_t.squeeze(0))
        
        # Store
        action_list.append(action_t.squeeze(0))
        mu_list.append(mu.squeeze(0))
        std_list.append(std.squeeze(0))
        logp_list.append(logp)
        value_list.append(value.squeeze(0))
        v1_list.append(v1.squeeze(0))
        v2_list.append(v2.squeeze(0))

        # Env step
        next_obs, r, term, trunc, info = env.step(action)
        done = term or trunc
        reward_list.append(r)
        done_list.append(float(done))

        print(f"\n[Step {step+1}]")
        print(f"  Action:  {action}")
        print(f"  mu (Pre-bounded action):      {mu.squeeze(0).detach().cpu().numpy()}")
        print(f"  std (Exploration rate of the action):     {std.squeeze(0).detach().cpu().numpy()}")
        print(f"  v1: {v1.item():.4f}, v2: {v2.item():.4f} → value: {value.item():.4f}")
        print(f"  logp (Action log-probability by policy):    {logp.item():.4f}")
        print(f"  reward:  {r:.3f}, done: {done}")

        obs = next_obs
        if done:
            print("Episode ended early.")
            break

    # ------------------------------------------------------------------
    # Bootstrap final value
    # ------------------------------------------------------------------
    final_obs_t = torch.from_numpy(obs).float().unsqueeze(0)
    with torch.no_grad():
        _, _, _, final_v, _, _ = model(final_obs_t)
    final_v = final_v.item()

    # ------------------------------------------------------------------
    # Compute returns & advantages
    # ------------------------------------------------------------------
    returns = []
    advs = []
    R = final_v * (1.0 - done_list[-1])
    for r, d, v in zip(reversed(reward_list), reversed(done_list), reversed(value_list)):
        R = r + GAMMA * R * (1.0 - d)
        adv = R - v.detach().item()
        returns.append(R)
        advs.append(adv)
    returns = list(reversed(returns))
    advs = list(reversed(advs))

    returns_t = torch.tensor(returns, dtype=torch.float32)
    advs_t    = torch.tensor(advs,    dtype=torch.float32)

    # ------------------------------------------------------------------
    # Losses
    # ------------------------------------------------------------------
    values_t = torch.stack(value_list)
    critic_loss = nn.MSELoss()(values_t, returns_t)

    logps_t = torch.stack(logp_list)
    actor_loss = -(logps_t * advs_t).mean()

    total_loss = actor_loss + 0.5 * critic_loss
    print(f"\n=== Losses ===")
    print(f"Critic loss: {critic_loss.item():.6f}")
    print(f"Actor loss:  {actor_loss.item():.6f}")
    print(f"Total loss:  {total_loss.item():.6f}")

    # ------------------------------------------------------------------
    # Backward + update
    # ------------------------------------------------------------------
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    print("\n=== Sample Gradients (after update) ===")
    print(f"mu_head grad norm:     {model.mu_head.weight.grad.norm().item():.6f}")
    print(f"log_std_head grad norm:{model.log_std_head.weight.grad.norm().item():.6f}")
    print(f"critic1 grad norm:     {model.critic1.weight.grad.norm().item():.6f}")
    print(f"critic2 grad norm:     {model.critic2.weight.grad.norm().item():.6f}")

    env.close()
    print("\nDemo complete. Model updated once on this trajectory.")

if __name__ == "__main__":
    main()