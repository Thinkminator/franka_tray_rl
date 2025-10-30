import torch
import torch.nn as nn
import torch.nn.functional as F



# Import custom env
from envs.traypose.traypose_env import TrayPoseEnv

class SimpleActorCritic(nn.Module):
    # Part 1 Trunk: Obs_in -> Hidden1 -> Hidden2 (Tanh activations)
    # Input: 34-D obs, Output: 7-D action
    # 2 Hidden layers: h1 = h2 = 128
    def __init__(self, obs_dim: int = 34, act_dim: int = 7, hidden_size: int = 128):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh(),
        )
    # Part 2 Actor-Critic heads
    # Input: 128-D, Output: 7-D action (μ), 1-D value (V)
        self.mu = nn.Linear(hidden_size, act_dim)       # actor mean
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.5))  # learnable std (log space)
        self.value = nn.Linear(hidden_size, 1)          # critic value

    def forward(self, obs: torch.Tensor, deterministic: bool = False):
        trunk_out = self.trunk(obs)
         # Actor: Gaussian policy parameters (pre-squash)
        mu = self.mu(trunk_out)
        std = self.log_std.exp()
        #  Sample from the Gaussian, then squash to bounds
        raw_action = mu if deterministic else mu + torch.randn_like(mu) * std
        action = torch.tanh(raw_action)            # map into [-1, 1]

        # Critic: State-value
        value = self.value(trunk_out).squeeze(-1)   
        return action, mu, std, value

