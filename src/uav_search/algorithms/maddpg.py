from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import torch
from torch import nn

from .base import BaseOffPolicy
from .common import soft_update
from .networks import CentralizedCritic, DeterministicActor
from uav_search.runtime import make_adam


class MADDPG(BaseOffPolicy):
    """Canonical CTDE MADDPG with one actor and centralized critic per UAV.

    This follows the per-agent architecture described by root-paper ref. [42]
    and by the original MADDPG formulation: actor i sees local observation i,
    while critic i sees the concatenated observations and actions of all UAVs.
    """

    name = "maddpg"

    def __init__(self, env, cfg: dict[str, Any], device: str = "cpu", seed: int = 0):
        super().__init__(env, cfg, device, seed)
        a = cfg["algorithm"]
        self.actors = nn.ModuleList(
            [DeterministicActor(self.obs_dim, self.action_dim, self.hidden_sizes) for _ in range(self.n_agents)]
        ).to(self.device)
        self.target_actors = deepcopy(self.actors).to(self.device)

        gd = self.n_agents * self.obs_dim
        ad = self.n_agents * self.action_dim
        self.critics = nn.ModuleList(
            [CentralizedCritic(gd, ad, 1, self.hidden_sizes) for _ in range(self.n_agents)]
        ).to(self.device)
        self.target_critics = deepcopy(self.critics).to(self.device)

        actor_lr = float(a["actor_lr"])
        critic_lr = float(a["critic_lr"])
        self.actor_opts = [make_adam(actor.parameters(), lr=actor_lr, device=self.device) for actor in self.actors]
        self.critic_opts = [make_adam(critic.parameters(), lr=critic_lr, device=self.device) for critic in self.critics]
        self.exploration_noise = float(cfg["runtime"]["exploration_noise"])

    def _actions_tensor(self, obs: torch.Tensor, target: bool = False) -> torch.Tensor:
        actors = self.target_actors if target else self.actors
        return torch.stack([actors[i](obs[:, i, :]) for i in range(self.n_agents)], dim=1)

    def _joint_actions_for_actor(self, obs: torch.Tensor, actor_i: int) -> torch.Tensor:
        """Joint action with gradients only through actor_i, as in MADDPG."""
        pieces = []
        for j in range(self.n_agents):
            action_j = self.actors[j](obs[:, j, :])
            if j != actor_i:
                action_j = action_j.detach()
            pieces.append(action_j)
        return torch.stack(pieces, dim=1)

    @torch.no_grad()
    def act(self, obs: np.ndarray, explore: bool = True) -> np.ndarray:
        x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with self.autocast():
            actions = self._actions_tensor(x, target=False)
        actions = actions.squeeze(0).float().cpu().numpy()
        if explore:
            actions = actions + self.rng.normal(0.0, self.exploration_noise, size=actions.shape)
        return np.clip(actions, -1.0, 1.0).astype(np.float32)

    def update(self) -> dict[str, float]:
        if len(self.replay) < self.batch_size:
            return {}
        batch = self.replay.sample(self.batch_size, self.device, non_blocking=self.non_blocking)
        global_obs = self._global(batch.obs)
        global_next = self._global(batch.next_obs)
        joint_action = batch.actions.flatten(start_dim=1)

        with torch.no_grad(), self.autocast():
            next_actions = self._actions_tensor(batch.next_obs, target=True).flatten(start_dim=1)
            targets = []
            for i in range(self.n_agents):
                tq = self.target_critics[i](global_next, next_actions)
                y = batch.rewards[:, i : i + 1] + self.gamma * (1.0 - batch.dones[:, i : i + 1]) * tq
                targets.append(y)

        critic_losses: list[float] = []
        q_means: list[float] = []
        target_q_means: list[float] = []
        td_errors: list[float] = []
        for i in range(self.n_agents):
            with self.autocast():
                q = self.critics[i](global_obs, joint_action)
                loss = torch.nn.functional.mse_loss(q, targets[i])
            q_detached = q.detach().float()
            target_detached = targets[i].detach().float()
            q_means.append(float(q_detached.mean().item()))
            target_q_means.append(float(target_detached.mean().item()))
            td_errors.append(float((q_detached - target_detached).abs().mean().item()))
            self.optimizer_step(loss, self.critic_opts[i], self.critics[i].parameters())
            critic_losses.append(float(loss.detach().float().item()))

        actor_losses: list[float] = []
        for i in range(self.n_agents):
            with self.autocast():
                actions_i = self._joint_actions_for_actor(batch.obs, i).flatten(start_dim=1)
                loss = -self.critics[i](global_obs, actions_i).mean()
            self.optimizer_step(loss, self.actor_opts[i], self.actors[i].parameters())
            actor_losses.append(float(loss.detach().float().item()))

        self.finish_optimizer_steps()
        for i in range(self.n_agents):
            soft_update(self.target_critics[i], self.critics[i], self.tau)
            soft_update(self.target_actors[i], self.actors[i], self.tau)
        self.update_step += 1
        return {
            "critic_loss": float(np.mean(critic_losses)),
            "actor_loss": float(np.mean(actor_losses)),
            "q_mean": float(np.mean(q_means)),
            "target_q_mean": float(np.mean(target_q_means)),
            "td_error_abs_mean": float(np.mean(td_errors)),
        }

    def checkpoint(self) -> dict[str, Any]:
        return {
            "algorithm": self.name,
            "config": self.cfg,
            "actors": self.actors.state_dict(),
            "target_actors": self.target_actors.state_dict(),
            "critics": self.critics.state_dict(),
            "target_critics": self.target_critics.state_dict(),
            "actor_opts": [opt.state_dict() for opt in self.actor_opts],
            "critic_opts": [opt.state_dict() for opt in self.critic_opts],
            "amp_scaler": self.scaler.state_dict(),
            "update_step": self.update_step,
            "requested_replay_capacity": self.requested_replay_capacity,
            "effective_replay_capacity": self.replay.capacity,
        }

    def load_checkpoint(self, p: dict[str, Any]) -> None:
        self.actors.load_state_dict(p["actors"])
        self.target_actors.load_state_dict(p.get("target_actors", p["actors"]))
        self.critics.load_state_dict(p["critics"])
        self.target_critics.load_state_dict(p.get("target_critics", p["critics"]))
        for states, opts in ((p.get("actor_opts"), self.actor_opts), (p.get("critic_opts"), self.critic_opts)):
            if states is not None:
                for state, opt in zip(states, opts):
                    opt.load_state_dict(state)
        if "amp_scaler" in p:
            self.scaler.load_state_dict(p["amp_scaler"])
        self.update_step = int(p.get("update_step", 0))
