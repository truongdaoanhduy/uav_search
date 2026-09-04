from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import torch
from torch import nn

from .base import BaseOffPolicy
from .common import hard_update, soft_update
from .networks import CentralizedCritic, DeterministicActor


class MADDPG(BaseOffPolicy):
    name = "maddpg"

    def __init__(self, env, cfg: dict[str, Any], device: str = "cpu", seed: int = 0):
        super().__init__(env, cfg, device, seed)
        self.actors = nn.ModuleDict({
            "fixed": DeterministicActor(self.obs_dim, self.action_dim, self.hidden_sizes),
            "rotor": DeterministicActor(self.obs_dim, self.action_dim, self.hidden_sizes),
        }).to(self.device)
        self.target_actors = deepcopy(self.actors).to(self.device)
        gd = self.n_agents * self.obs_dim
        ad = self.n_agents * self.action_dim
        self.critic = CentralizedCritic(gd, ad, self.n_agents, self.hidden_sizes).to(self.device)
        self.target_critic = deepcopy(self.critic).to(self.device)
        self.actor_opt = torch.optim.Adam(self.actors.parameters(), lr=float(cfg["algorithm"]["actor_lr"]))
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=float(cfg["algorithm"]["critic_lr"]))
        self.exploration_noise = float(cfg["runtime"]["exploration_noise"])

    def _actions_tensor(self, obs: torch.Tensor, target: bool = False) -> torch.Tensor:
        actors = self.target_actors if target else self.actors
        return torch.stack([actors[t](obs[:, i, :]) for i, t in enumerate(self.agent_types)], dim=1)

    @torch.no_grad()
    def act(self, obs: np.ndarray, explore: bool = True) -> np.ndarray:
        x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        actions = self._actions_tensor(x, target=False).squeeze(0).cpu().numpy()
        if explore:
            actions = actions + self.rng.normal(0.0, self.exploration_noise, size=actions.shape)
        return np.clip(actions, -1.0, 1.0).astype(np.float32)

    def update(self) -> dict[str, float]:
        if len(self.replay) < self.batch_size:
            return {}
        batch = self.replay.sample(self.batch_size, self.device)
        global_obs = self._global(batch.obs)
        global_next = self._global(batch.next_obs)
        joint_action = batch.actions.flatten(start_dim=1)
        with torch.no_grad():
            next_actions = self._actions_tensor(batch.next_obs, target=True).flatten(start_dim=1)
            target_q = self.target_critic(global_next, next_actions)
            y = batch.rewards + self.gamma * (1.0 - batch.dones) * target_q
        q = self.critic(global_obs, joint_action)
        critic_loss = torch.nn.functional.mse_loss(q, y)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
        self.critic_opt.step()

        current_actions = self._actions_tensor(batch.obs, target=False).flatten(start_dim=1)
        actor_loss = -self.critic(global_obs, current_actions).mean()
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actors.parameters(), 10.0)
        self.actor_opt.step()

        soft_update(self.target_critic, self.critic, self.tau)
        for key in self.actors:
            soft_update(self.target_actors[key], self.actors[key], self.tau)
        self.update_step += 1
        return {"critic_loss": float(critic_loss.item()), "actor_loss": float(actor_loss.item())}

    def checkpoint(self) -> dict[str, Any]:
        return {
            "algorithm": self.name,
            "config": self.cfg,
            "actors": self.actors.state_dict(),
            "target_actors": self.target_actors.state_dict(),
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
            "update_step": self.update_step,
        }

    def load_checkpoint(self, p: dict[str, Any]) -> None:
        self.actors.load_state_dict(p["actors"])
        self.target_actors.load_state_dict(p.get("target_actors", p["actors"]))
        self.critic.load_state_dict(p["critic"])
        self.target_critic.load_state_dict(p.get("target_critic", p["critic"]))
        if "actor_opt" in p:
            self.actor_opt.load_state_dict(p["actor_opt"])
        if "critic_opt" in p:
            self.critic_opt.load_state_dict(p["critic_opt"])
        self.update_step = int(p.get("update_step", 0))
