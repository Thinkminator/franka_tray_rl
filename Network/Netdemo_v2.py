import os
import sys
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

# Project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(PROJECT_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import custom env
from envs.traypose.traypose_env import TrayPoseEnv

# Import the SimpleActorCritic network
from simple_actor_critic import SimpleActorCritic




# ------------------------ Utilities ------------------------
def set_seed(seed: int = 42):
    import os, random
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def atanh_clipped(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Numerically stable atanh from action in (-1, 1) to pre-tanh space."""
    x = x.clamp(min=-1 + eps, max=1 - eps)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))

def log_prob_tanh_gaussian(actions: torch.Tensor,
                           mean: torch.Tensor,
                           log_std: torch.Tensor) -> torch.Tensor:
    """
    Compute log pi(a) where a = tanh(u), u ~ Normal(mean, std).
    log pi(a) = log N(u | mean, std) - sum log(1 - tanh(u)^2)
    We recover u via atanh(a) for the correction term.
    Returns: [B] log-probabilities.
    """
    std = log_std.exp()                      # [A]
    u = atanh_clipped(actions)               # [B, A]
    normal = Normal(mean, std)               # broadcast over batch
    log_prob_u = normal.log_prob(u).sum(-1)  # [B]
    # |det(Jacobian)| = prod(1 - tanh(u)^2) = prod(1 - a^2)
    correction = torch.log(1 - actions.pow(2) + 1e-6).sum(-1)  # [B]
    return log_prob_u - correction

def gaussian_entropy(log_std: torch.Tensor, action_dim: int) -> torch.Tensor:
    """
    Entropy of diag Gaussian per-sample (broadcast over batch):
      H = 0.5 * dim * (1 + log(2*pi)) + sum(log_std)
    We approximate the tanh-squashed entropy with the pre-squash Gaussian entropy,
    which is common in practice.
    Returns: scalar entropy (averaged over batch)
    """
    H_per_sample = 0.5 * action_dim * (1 + math.log(2 * math.pi)) + log_std.sum()
    return H_per_sample

def attach_linear_hooks(model: nn.Module):
    """Capture forward mean activations and backward grad norms on Linear layers."""
    act_means, grad_norms = {}, {}

    def fwd(name):
        def hook(_m, _inp, out):
            with torch.no_grad():
                out = out[0] if isinstance(out, (tuple, list)) else out
                act_means[name] = float(out.detach().mean())
        return hook

    def bwd(name):
        def hook(_m, gin, gout):
            with torch.no_grad():
                g = gout[0] if isinstance(gout, (tuple, list)) else gout
                grad_norms[name] = float(g.norm()) if g is not None else 0.0
        return hook

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module.register_forward_hook(fwd(name))
            module.register_full_backward_hook(bwd(name))
    return act_means, grad_norms

# ------------------------ Rollout ------------------------
@torch.no_grad()
def collect_batch(env, policy: SimpleActorCritic, batch_size=128, gamma=0.99, seed=42):
    """
    Collect a 1-step batch using the current policy (on-policy):
    returns dict with tensors:
      obs [B, obs_dim], actions [B, act_dim], rewards [B], dones [B],
      next_obs [B, obs_dim], values [B], next_values [B], mean [B, A]
    """
    obs_list, act_list, rew_list, done_list, next_obs_list = [], [], [], [], []
    mu_list, val_list, next_val_list = [], [], []

    obs, info = env.reset(seed=seed)
    for _ in range(batch_size):
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        actions, mean_action, action_stddev, state_value = policy(obs_t, deterministic=False)
        action = actions.squeeze(0).cpu().numpy()

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)

        # Store
        obs_list.append(obs)
        act_list.append(action)
        rew_list.append(reward)
        done_list.append(done)
        next_obs_list.append(next_obs)
        mu_list.append(mean_action.squeeze(0).cpu().numpy())
        val_list.append(state_value.squeeze(0).cpu().numpy())

        # Next state value
        next_obs_t = torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0)
        _, _, _, next_value = policy(next_obs_t, deterministic=True)
        next_val_list.append(next_value.squeeze(0).cpu().numpy())

        obs = next_obs
        if done:
            obs, info = env.reset()

    # to tensors
    batch = {
        "obs": torch.tensor(np.asarray(obs_list), dtype=torch.float32),
        "actions": torch.tensor(np.asarray(act_list), dtype=torch.float32),
        "rewards": torch.tensor(np.asarray(rew_list), dtype=torch.float32),
        "dones": torch.tensor(np.asarray(done_list), dtype=torch.float32),
        "next_obs": torch.tensor(np.asarray(next_obs_list), dtype=torch.float32),
        "mu": torch.tensor(np.asarray(mu_list), dtype=torch.float32),
        "values": torch.tensor(np.asarray(val_list), dtype=torch.float32).squeeze(-1),
        "next_values": torch.tensor(np.asarray(next_val_list), dtype=torch.float32).squeeze(-1),
    }
    return batch

# ------------------------ Main ------------------------
def main():
    set_seed(42)
    env = TrayPoseEnv(render=False)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    policy = SimpleActorCritic(obs_dim=obs_dim, act_dim=act_dim, hidden_size=128)

    # Optional: hooks for introspection
    act_means, grad_norms = attach_linear_hooks(policy)

    # Collect a small on-policy batch
    BATCH = 128
    GAMMA = 0.99
    batch = collect_batch(env, policy, batch_size=BATCH, gamma=GAMMA, seed=42)

    # Compute current forward pass (we need mu, std, value on obs)
    actions = batch["actions"]
    obs = batch["obs"]
    rewards = batch["rewards"]
    dones = batch["dones"]
    next_values = batch["next_values"]

    out_actions, mean_action, action_stddev, state_value = policy(obs, deterministic=False)

    # IMPORTANT: Use the actions from the batch (generated by the same policy)
    # to compute log-probs with tanh correction.
    log_prob = log_prob_tanh_gaussian(
        actions=actions,
        mean=mean_action,
        log_std=policy.log_std
    )        
    # Approximate entropy (Gaussian, pre-squash)
    entropy = gaussian_entropy(policy.log_std, act_dim)              # scalar

    # One-step TD advantage and targets
    with torch.no_grad():
        td_target = rewards + (1.0 - dones) * GAMMA * next_values    # [B]
        advantage = td_target - state_value                           # [B]

    # Actor-critic losses
    policy_loss = -(log_prob * advantage.detach()).mean()
    value_loss = F.mse_loss(state_value, td_target)
    ent_coef, vf_coef = 1e-3, 0.5
    total_loss = policy_loss + vf_coef * value_loss - ent_coef * entropy

    # Backprop
    policy.zero_grad(set_to_none=True)
    total_loss.backward()

    # --- Reports ---
    print("=== Forward/Backward Demo ===")
    print(f"Batch size              : {BATCH}")
    print(f"log_prob mean           : {float(log_prob.mean()):+.6f}")
    print(f"entropy (Gaussian)      : {float(entropy):+.6f}")
    print(f"policy_loss             : {float(policy_loss):+.6f}")
    print(f"value_loss              : {float(value_loss):+.6f}")
    print(f"total_loss              : {float(total_loss):+.6f}")

    print("\nLayer mean activations (Linear):")
    for k, v in act_means.items():
        print(f"  {k:30s}: {v:+.6f}")

    print("\nLayer grad norms (Linear):")
    for k, v in grad_norms.items():
        print(f"  {k:30s}: {v:.6f}")

    # Parameter grad norms (heads + log_std)
    def gnorm(p): 
        return 0.0 if (p.grad is None) else float(p.grad.norm())

    print("\nParameter grad norms:")
    print(f"  trunk[0].weight       : {gnorm(policy.trunk[0].weight):.6f}")
    print(f"  trunk[2].weight       : {gnorm(policy.trunk[2].weight):.6f}")
    print(f"  mu.weight             : {gnorm(policy.mu.weight):.6f}")
    print(f"  value.weight          : {gnorm(policy.value.weight):.6f}")
    print(f"  log_std               : {gnorm(policy.log_std):.6f}")

if __name__ == '__main__':
    main()
