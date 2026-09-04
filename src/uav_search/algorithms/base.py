from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from uav_search.runtime import autocast_context, configure_runtime, make_grad_scaler
from .common import ReplayBuffer


class BaseOffPolicy:
    def __init__(self, env, cfg: dict[str, Any], device: str = "cpu", seed: int = 0):
        self.cfg = cfg
        runtime_cfg = cfg.get("runtime", {})
        self.runtime_profile = configure_runtime(
            device,
            deterministic=bool(runtime_cfg.get("deterministic", False)),
            amp_mode=str(runtime_cfg.get("amp_mode", "auto")),
        )
        self.device = self.runtime_profile.device
        runtime_cfg = cfg.get("runtime", {})
        self.runtime_profile = configure_runtime(
            self.device,
            deterministic=bool(runtime_cfg.get("deterministic", False)),
            amp_mode=str(runtime_cfg.get("amp_mode", "auto")),
        )
        self.scaler = make_grad_scaler(self.runtime_profile)
        self.n_agents = env.n_agents
        self.obs_dim = env.obs_dim
        self.action_dim = env.action_dim
        self.agent_types = ["fixed" if int(t) == 0 else "rotor" for t in env.agent_types]
        self.fixed_indices = [i for i, t in enumerate(self.agent_types) if t == "fixed"]
        self.rotor_indices = [i for i, t in enumerate(self.agent_types) if t == "rotor"]
        self.type_indices = {
            "fixed": torch.as_tensor(self.fixed_indices, dtype=torch.long, device=self.device),
            "rotor": torch.as_tensor(self.rotor_indices, dtype=torch.long, device=self.device),
        }
        self.gamma = float(cfg["algorithm"]["gamma"])
        self.tau = float(cfg["algorithm"]["tau"])
        self.batch_size = int(runtime_cfg["batch_size"])
        self.hidden_sizes = list(runtime_cfg["hidden_sizes"])
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        self.replay = ReplayBuffer(
            int(runtime_cfg["replay_size"]),
            self.n_agents,
            self.obs_dim,
            self.action_dim,
            seed=seed,
            pin_memory=self.runtime_profile.pin_memory,
        )
        self.non_blocking = self.runtime_profile.non_blocking
        self.update_step = 0

    def store(self, obs, actions, rewards, next_obs, dones) -> None:
        self.replay.add(obs, actions, rewards, next_obs, dones)

    @staticmethod
    def _global(obs: torch.Tensor) -> torch.Tensor:
        return obs.flatten(start_dim=1)

    def autocast(self):
        return autocast_context(self.runtime_profile)

    def optimizer_step(
        self,
        loss: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        parameters: Iterable[torch.nn.Parameter],
        max_grad_norm: float = 10.0,
    ) -> None:
        optimizer.zero_grad(set_to_none=True)
        params = list(parameters)
        if self.scaler.is_enabled():
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
            self.scaler.step(optimizer)
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
            optimizer.step()

    def finish_optimizer_steps(self) -> None:
        if self.scaler.is_enabled():
            self.scaler.update()

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
