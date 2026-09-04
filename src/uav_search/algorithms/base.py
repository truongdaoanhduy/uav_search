from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from .common import ReplayBuffer


class BaseOffPolicy:
    def __init__(self, env, cfg: dict[str, Any], device: str = "cpu", seed: int = 0):
        self.cfg = cfg
        self.device = torch.device(device)
        self.n_agents = env.n_agents
        self.obs_dim = env.obs_dim
        self.action_dim = env.action_dim
        self.agent_types = ["fixed" if int(t) == 0 else "rotor" for t in env.agent_types]
        self.fixed_indices = [i for i, t in enumerate(self.agent_types) if t == "fixed"]
        self.rotor_indices = [i for i, t in enumerate(self.agent_types) if t == "rotor"]
        self.gamma = float(cfg["algorithm"]["gamma"])
        self.tau = float(cfg["algorithm"]["tau"])
        self.batch_size = int(cfg["runtime"]["batch_size"])
        self.hidden_sizes = list(cfg["runtime"]["hidden_sizes"])
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        self.replay = ReplayBuffer(
            int(cfg["runtime"]["replay_size"]), self.n_agents, self.obs_dim, self.action_dim, seed=seed
        )
        self.update_step = 0

    def store(self, obs, actions, rewards, next_obs, dones) -> None:
        self.replay.add(obs, actions, rewards, next_obs, dones)

    @staticmethod
    def _global(obs: torch.Tensor) -> torch.Tensor:
        return obs.flatten(start_dim=1)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.checkpoint(), path)

    def load(self, path: str | Path) -> None:
        payload = torch.load(Path(path), map_location=self.device, weights_only=False)
        self.load_checkpoint(payload)

    def checkpoint(self) -> dict[str, Any]:
        raise NotImplementedError

    def load_checkpoint(self, payload: dict[str, Any]) -> None:
        raise NotImplementedError
