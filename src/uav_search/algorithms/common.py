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
    """Fixed-size CPU replay storage with direct Torch sampling.

    Storage stays on CPU so large replay buffers do not consume GPU VRAM.  CUDA
    training requests non-blocking copies; CPU training simply returns CPU
    tensors.  The sampling RNG remains NumPy-based to preserve the existing
    seeded replay behavior.
    """

    def __init__(
        self,
        capacity: int,
        n_agents: int,
        obs_dim: int,
        action_dim: int,
        seed: int = 0,
        pin_memory: bool = False,
    ):
        self.capacity = int(capacity)
        self.n_agents = int(n_agents)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.rng = np.random.default_rng(seed)
        self.pin_memory = bool(pin_memory)

        def alloc(shape: tuple[int, ...]) -> torch.Tensor:
            return torch.empty(shape, dtype=torch.float32, device="cpu", pin_memory=self.pin_memory)

        self.obs = alloc((self.capacity, self.n_agents, self.obs_dim))
        self.actions = alloc((self.capacity, self.n_agents, self.action_dim))
        self.rewards = alloc((self.capacity, self.n_agents))
        self.next_obs = alloc((self.capacity, self.n_agents, self.obs_dim))
        self.dones = alloc((self.capacity, self.n_agents))
        self.ptr = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size

    @staticmethod
    def _copy_row(dst: torch.Tensor, value) -> None:
        src = torch.as_tensor(value, dtype=torch.float32, device="cpu")
        dst.copy_(src, non_blocking=False)

    def add(self, obs, actions, rewards, next_obs, dones) -> None:
        i = self.ptr
        self._copy_row(self.obs[i], obs)
        self._copy_row(self.actions[i], actions)
        self._copy_row(self.rewards[i], rewards)
        self._copy_row(self.next_obs[i], next_obs)
        self._copy_row(self.dones[i], dones)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self,
        batch_size: int,
        device: torch.device,
        non_blocking: bool = True,
    ) -> Batch:
        if self.size < batch_size:
            raise ValueError(f"Replay has {self.size} samples, need {batch_size}")
        idx_np = self.rng.integers(0, self.size, size=batch_size, dtype=np.int64)
        idx = torch.from_numpy(idx_np)

        def take(x: torch.Tensor) -> torch.Tensor:
            batch = torch.index_select(x, 0, idx)
            if device.type == "cpu":
                return batch
            return batch.to(device=device, non_blocking=bool(non_blocking))

        return Batch(take(self.obs), take(self.actions), take(self.rewards), take(self.next_obs), take(self.dones))


def hard_update(target: torch.nn.Module, source: torch.nn.Module) -> None:
    target.load_state_dict(source.state_dict())


@torch.no_grad()
def soft_update(target: torch.nn.Module, source: torch.nn.Module, tau: float) -> None:
    for target_p, source_p in zip(target.parameters(), source.parameters()):
        target_p.mul_(1.0 - tau).add_(source_p, alpha=tau)
