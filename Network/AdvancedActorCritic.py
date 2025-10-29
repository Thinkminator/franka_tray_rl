import torch
import torch.nn as nn
import torch.nn.functional as F


OBS_DIM               = 34
ACT_DIM               = 7
HIDDEN_SIZE           = 256

class AdvancedActorCritic(nn.Module):
    def __init__(self, obs_dim=OBS_DIM, act_dim=ACT_DIM, hidden_size=HIDDEN_SIZE):
        super().__init__()
        # ---- shared trunk (3 layers, LayerNorm + ReLU) ----
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_size), nn.LayerNorm(hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.LayerNorm(hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.LayerNorm(hidden_size), nn.ReLU(),
        )
        # ---- actor heads ----
        self.mu_head      = nn.Linear(hidden_size, act_dim)
        self.log_std_head = nn.Linear(hidden_size, act_dim)   # state-dependent std
        # ---- twin critics ----
        self.critic1 = nn.Linear(hidden_size, 1)
        self.critic2 = nn.Linear(hidden_size, 1)

        self.apply(self._ortho_init)

    def _ortho_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(m.bias, 0.0)

    def forward(self, obs, deterministic=False):
        x = self.trunk(obs)

        mu      = self.mu_head(x)
        log_std = self.log_std_head(x).clamp(-20, 2)
        std     = log_std.exp()

        raw = mu if deterministic else mu + torch.randn_like(mu) * std
        action = torch.tanh(raw)

        v1 = self.critic1(x).squeeze(-1)
        v2 = self.critic2(x).squeeze(-1)
        value = torch.min(v1, v2)                 # used for bootstrap / logging

        return action, mu, std, value, v1, v2
    

model=AdvancedActorCritic()
print(model)