from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch

from uav_search.actions import (
    CHECKPOINT_FORMAT_VERSION,
    LEGACY_ACTION_SCHEMA,
    PEER_ACTION_SCHEMA,
    canonicalize_peer_action_numpy,
)
from uav_search.runtime import autocast_context, configure_runtime, make_grad_scaler

from .common import ReplayBuffer, capacity_with_memory_budget


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
        self.scaler = make_grad_scaler(self.runtime_profile)
        self.n_agents = env.n_agents
        self.obs_dim = env.obs_dim
        self.action_dim = env.action_dim
        self.peer_mode = bool(getattr(env, "peer_mode", False))
        self.action_schema = PEER_ACTION_SCHEMA if self.peer_mode else LEGACY_ACTION_SCHEMA
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

        self.requested_replay_capacity = int(runtime_cfg["replay_size"])
        budget_mb = runtime_cfg.get("replay_memory_budget_mb")
        replay_capacity = capacity_with_memory_budget(
            self.requested_replay_capacity,
            self.n_agents,
            self.obs_dim,
            self.action_dim,
            None if budget_mb is None else float(budget_mb),
        )
        if replay_capacity < self.batch_size:
            raise ValueError(
                f"Replay memory budget yields capacity={replay_capacity}, below batch_size={self.batch_size}. "
                "Increase runtime.replay_memory_budget_mb or reduce batch_size."
            )
        self.replay = ReplayBuffer(
            replay_capacity,
            self.n_agents,
            self.obs_dim,
            self.action_dim,
            seed=seed,
            pin_memory=self.runtime_profile.pin_memory,
        )
        self.non_blocking = self.runtime_profile.non_blocking
        self.update_step = 0

    def store(self, obs, actions, rewards, next_obs, dones, valids=None) -> None:
        if self.peer_mode:
            actions = canonicalize_peer_action_numpy(actions, self.n_agents)
        self.replay.add(obs, actions, rewards, next_obs, dones, valids=valids)

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

    def checkpoint_metadata(self) -> dict[str, Any]:
        """Metadata required to prevent shape-compatible semantic checkpoint misuse."""
        return {
            "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
            "action_schema": self.action_schema,
            "n_agents": self.n_agents,
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
        }

    def validate_checkpoint(self, payload: dict[str, Any]) -> None:
        expected_algorithm = getattr(self, "name", None)
        stored_algorithm = payload.get("algorithm")
        if expected_algorithm is not None and stored_algorithm is not None and stored_algorithm != expected_algorithm:
            raise ValueError(
                f"Checkpoint algorithm {stored_algorithm!r} is incompatible with {expected_algorithm!r}."
            )

        stored_schema = payload.get("action_schema")
        if self.peer_mode and stored_schema != self.action_schema:
            raise ValueError(
                "Checkpoint action schema is incompatible with the current U6/U9 Cartesian hybrid action contract. "
                f"Expected {self.action_schema!r}, got {stored_schema!r}. "
                "Pre-Cartesian U6/U9 checkpoints must be retrained rather than resumed/evaluated."
            )
        if not self.peer_mode and stored_schema is not None and stored_schema != self.action_schema:
            raise ValueError(
                f"Checkpoint action schema {stored_schema!r} is incompatible with {self.action_schema!r}."
            )

        for key, expected in (
            ("n_agents", self.n_agents),
            ("obs_dim", self.obs_dim),
            ("action_dim", self.action_dim),
        ):
            if key in payload and int(payload[key]) != int(expected):
                raise ValueError(
                    f"Checkpoint {key}={payload[key]!r} is incompatible with current {key}={expected}."
                )

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
