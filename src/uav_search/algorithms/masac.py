from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import torch
from torch import nn

from .base import BaseOffPolicy
from .common import soft_update
from .networks import CentralizedCritic, GaussianActor


class MASAC(BaseOffPolicy):
    name = "masac"

    def __init__(self, env, cfg: dict[str, Any], device: str = "cpu", seed: int = 0):
        super().__init__(env, cfg, device, seed)
        a = cfg["algorithm"]
        self.alpha = float(a["alpha"])
        self.actors = nn.ModuleDict({
            "fixed": GaussianActor(self.obs_dim, self.action_dim, self.hidden_sizes, a["log_std_min"], a["log_std_max"]),
            "rotor": GaussianActor(self.obs_dim, self.action_dim, self.hidden_sizes, a["log_std_min"], a["log_std_max"]),
        }).to(self.device)
        # The paper's Eq. (32)-(34) explicitly uses target actor and critic networks.
        self.target_actors = deepcopy(self.actors).to(self.device)
        gd, ad = self.n_agents * self.obs_dim, self.n_agents * self.action_dim
        self.critic1 = CentralizedCritic(gd, ad, self.n_agents, self.hidden_sizes).to(self.device)
        self.critic2 = CentralizedCritic(gd, ad, self.n_agents, self.hidden_sizes).to(self.device)
        self.target_critic1 = deepcopy(self.critic1).to(self.device)
        self.target_critic2 = deepcopy(self.critic2).to(self.device)
        self.actor_opt = torch.optim.Adam(self.actors.parameters(), lr=float(a["actor_lr"]))
        self.critic1_opt = torch.optim.Adam(self.critic1.parameters(), lr=float(a["critic_lr"]))
        self.critic2_opt = torch.optim.Adam(self.critic2.parameters(), lr=float(a["critic_lr"]))

    def _sample_actions(self, obs: torch.Tensor, target: bool = False, deterministic: bool = False):
        actors = self.target_actors if target else self.actors
        actions, logps = [], []
        for i, t in enumerate(self.agent_types):
            ai, lp = actors[t].sample(obs[:, i, :], deterministic=deterministic)
            actions.append(ai)
            logps.append(lp)
        return torch.stack(actions, dim=1), torch.stack(logps, dim=1)

    @torch.no_grad()
    def act(self, obs: np.ndarray, explore: bool = True) -> np.ndarray:
        x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        a, _ = self._sample_actions(x, target=False, deterministic=not explore)
        return a.squeeze(0).cpu().numpy().astype(np.float32)

    def update(self) -> dict[str, float]:
        if len(self.replay) < self.batch_size:
            return {}
        b = self.replay.sample(self.batch_size, self.device)
        go, gn = self._global(b.obs), self._global(b.next_obs)
        ja = b.actions.flatten(start_dim=1)
        with torch.no_grad():
            na, nlogp = self._sample_actions(b.next_obs, target=True, deterministic=False)
            tq = torch.minimum(
                self.target_critic1(gn, na.flatten(start_dim=1)),
                self.target_critic2(gn, na.flatten(start_dim=1)),
            )
            entropy_term = self.alpha * nlogp.squeeze(-1)
            y = b.rewards + self.gamma * (1.0 - b.dones) * (tq - entropy_term)
        q1, q2 = self.critic1(go, ja), self.critic2(go, ja)
        l1 = torch.nn.functional.mse_loss(q1, y)
        l2 = torch.nn.functional.mse_loss(q2, y)
        self.critic1_opt.zero_grad(set_to_none=True)
        l1.backward()
        torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), 10.0)
        self.critic1_opt.step()
        self.critic2_opt.zero_grad(set_to_none=True)
        l2.backward()
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), 10.0)
        self.critic2_opt.step()

        ca, logp = self._sample_actions(b.obs, target=False, deterministic=False)
        cq = torch.minimum(self.critic1(go, ca.flatten(start_dim=1)), self.critic2(go, ca.flatten(start_dim=1)))
        actor_loss = (self.alpha * logp.squeeze(-1) - cq).mean()
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actors.parameters(), 10.0)
        self.actor_opt.step()

        soft_update(self.target_critic1, self.critic1, self.tau)
        soft_update(self.target_critic2, self.critic2, self.tau)
        for key in self.actors:
            soft_update(self.target_actors[key], self.actors[key], self.tau)
        self.update_step += 1
        return {
            "critic_loss": float(0.5 * (l1.item() + l2.item())),
            "critic1_loss": float(l1.item()),
            "critic2_loss": float(l2.item()),
            "actor_loss": float(actor_loss.item()),
            "entropy": float((-logp).mean().item()),
            "alpha": self.alpha,
        }

    def checkpoint(self) -> dict[str, Any]:
        return {
            "algorithm": self.name,
            "config": self.cfg,
            "actors": self.actors.state_dict(), "target_actors": self.target_actors.state_dict(),
            "critic1": self.critic1.state_dict(), "critic2": self.critic2.state_dict(),
            "target_critic1": self.target_critic1.state_dict(), "target_critic2": self.target_critic2.state_dict(),
            "actor_opt": self.actor_opt.state_dict(), "critic1_opt": self.critic1_opt.state_dict(), "critic2_opt": self.critic2_opt.state_dict(),
            "update_step": self.update_step,
        }

    def load_checkpoint(self, p: dict[str, Any]) -> None:
        self.actors.load_state_dict(p["actors"])
        self.target_actors.load_state_dict(p.get("target_actors", p["actors"]))
        self.critic1.load_state_dict(p["critic1"]); self.critic2.load_state_dict(p["critic2"])
        self.target_critic1.load_state_dict(p.get("target_critic1", p["critic1"]))
        self.target_critic2.load_state_dict(p.get("target_critic2", p["critic2"]))
        for key, opt in (("actor_opt", self.actor_opt), ("critic1_opt", self.critic1_opt), ("critic2_opt", self.critic2_opt)):
            if key in p:
                opt.load_state_dict(p[key])
        self.update_step = int(p.get("update_step", 0))
