from __future__ import annotations

import torch
from torch import nn


def mlp(in_dim: int, out_dim: int, hidden_sizes: list[int], output_activation: nn.Module | None = None) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = in_dim
    for h in hidden_sizes:
        layers += [nn.Linear(prev, h), nn.ReLU()]
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    if output_activation is not None:
        layers.append(output_activation)
    return nn.Sequential(*layers)


class DeterministicActor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes: list[int]):
        super().__init__()
        self.net = mlp(obs_dim, action_dim, hidden_sizes, nn.Tanh())

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class GaussianActor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes: list[int], log_std_min: float, log_std_max: float):
        super().__init__()
        self.backbone = mlp(obs_dim, hidden_sizes[-1], hidden_sizes[:-1]) if len(hidden_sizes) > 1 else mlp(obs_dim, hidden_sizes[0], [])
        last = hidden_sizes[-1]
        self.mean = nn.Linear(last, action_dim)
        self.log_std = nn.Linear(last, action_dim)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(obs)
        mean = self.mean(h)
        log_std = self.log_std(h).clamp(self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
        log_prob_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self(obs)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        raw = mean if deterministic else dist.rsample()
        action = torch.tanh(raw)
        if deterministic:
            logp = torch.zeros((obs.shape[0], 1), device=obs.device, dtype=obs.dtype)
        else:
            # SAC tanh correction.
            logp = dist.log_prob(raw) - torch.log(1.0 - action.pow(2) + 1e-6)
            if log_prob_mask is not None:
                logp = logp * log_prob_mask.to(device=logp.device, dtype=logp.dtype)
            logp = logp.sum(dim=-1, keepdim=True)
        return action, logp


class CentralizedCritic(nn.Module):
    def __init__(self, global_obs_dim: int, joint_action_dim: int, n_agents: int, hidden_sizes: list[int]):
        super().__init__()
        self.net = mlp(global_obs_dim + joint_action_dim, n_agents, hidden_sizes)

    def forward(self, global_obs: torch.Tensor, joint_action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([global_obs, joint_action], dim=-1))
