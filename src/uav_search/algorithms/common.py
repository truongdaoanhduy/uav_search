from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class Batch:
    obs: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_obs: torch.Tensor
    dones: torch.Tensor


class ReplayBuffer:
    def __init__(self, capacity: int, n_agents: int, obs_dim: int, action_dim: int, seed: int = 0):
        self.capacity = int(capacity)
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.rng = np.random.default_rng(seed)
        self.obs = np.zeros((capacity, n_agents, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, n_agents, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, n_agents), dtype=np.float32)
        self.next_obs = np.zeros((capacity, n_agents, obs_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, n_agents), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size

    def add(self, obs, actions, rewards, next_obs, dones) -> None:
        i = self.ptr
        self.obs[i] = obs
        self.actions[i] = actions
        self.rewards[i] = rewards
        self.next_obs[i] = next_obs
        self.dones[i] = dones
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> Batch:
        if self.size < batch_size:
            raise ValueError(f"Replay has {self.size} samples, need {batch_size}")
        idx = self.rng.integers(0, self.size, size=batch_size)
        t = lambda x: torch.as_tensor(x[idx], dtype=torch.float32, device=device)
        return Batch(t(self.obs), t(self.actions), t(self.rewards), t(self.next_obs), t(self.dones))


def hard_update(target: torch.nn.Module, source: torch.nn.Module) -> None:
    target.load_state_dict(source.state_dict())


@torch.no_grad()
def soft_update(target: torch.nn.Module, source: torch.nn.Module, tau: float) -> None:
    for target_p, source_p in zip(target.parameters(), source.parameters()):
        target_p.mul_(1.0 - tau).add_(source_p, alpha=tau)
