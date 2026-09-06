from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import torch
from torch import nn

from .common import soft_update
from .maddpg import MADDPG
from .networks import CentralizedCritic
from uav_search.runtime import make_adam


class MATD3(MADDPG):
    """Multi-agent TD3: one actor and twin centralized critics per UAV."""

    name = "matd3"

    def __init__(self, env, cfg: dict[str, Any], device: str = "cpu", seed: int = 0):
        super().__init__(env, cfg, device, seed)
        gd = self.n_agents * self.obs_dim
        ad = self.n_agents * self.action_dim
        self.critics2 = nn.ModuleList(
            [CentralizedCritic(gd, ad, 1, self.hidden_sizes) for _ in range(self.n_agents)]
        ).to(self.device)
        self.target_critics2 = deepcopy(self.critics2).to(self.device)
        critic_lr = float(cfg["algorithm"]["critic_lr"])
        self.critic2_opts = [make_adam(c.parameters(), lr=critic_lr, device=self.device) for c in self.critics2]
        self.policy_noise = float(cfg["algorithm"]["policy_noise"])
        self.noise_clip = float(cfg["algorithm"]["noise_clip"])
        self.policy_delay = int(cfg["algorithm"]["policy_delay"])

    @property
    def critics1(self):
        return self.critics

    @property
    def target_critics1(self):
        return self.target_critics

    def update(self) -> dict[str, float]:
        if len(self.replay) < self.batch_size:
            return {}
        self.update_step += 1
        b = self.replay.sample(self.batch_size, self.device, non_blocking=self.non_blocking)
        go = self._global(b.obs)
        gn = self._global(b.next_obs)
        ja = b.actions.flatten(start_dim=1)

        with torch.no_grad(), self.autocast():
            na = self._actions_tensor(b.next_obs, target=True)
            noise = torch.randn_like(na).mul_(self.policy_noise).clamp_(-self.noise_clip, self.noise_clip)
            na = (na + noise).clamp(-1.0, 1.0).flatten(start_dim=1)
            targets = []
            for i in range(self.n_agents):
                tq1 = self.target_critics[i](gn, na)
                tq2 = self.target_critics2[i](gn, na)
                target_q = torch.minimum(tq1, tq2)
                y = b.rewards[:, i : i + 1] + self.gamma * (1.0 - b.dones[:, i : i + 1]) * target_q
                targets.append(y)

        q1_losses: list[float] = []
        q2_losses: list[float] = []
        q_means: list[float] = []
        target_q_means: list[float] = []
        td_errors: list[float] = []
        for i in range(self.n_agents):
            with self.autocast():
                q1 = self.critics[i](go, ja)
                q2 = self.critics2[i](go, ja)
                l1 = torch.nn.functional.mse_loss(q1, targets[i])
                l2 = torch.nn.functional.mse_loss(q2, targets[i])
            q1d = q1.detach().float()
            q2d = q2.detach().float()
            yd = targets[i].detach().float()
            q_means.append(float((0.5 * (q1d + q2d)).mean().item()))
            target_q_means.append(float(yd.mean().item()))
            td_errors.append(float((0.5 * ((q1d - yd).abs() + (q2d - yd).abs())).mean().item()))
            self.optimizer_step(l1, self.critic_opts[i], self.critics[i].parameters())
            self.optimizer_step(l2, self.critic2_opts[i], self.critics2[i].parameters())
            q1_losses.append(float(l1.detach().float().item()))
            q2_losses.append(float(l2.detach().float().item()))

        actor_losses: list[float] = []
        actor_updated = 0.0
        if self.update_step % self.policy_delay == 0:
            for i in range(self.n_agents):
                with self.autocast():
                    actions_i = self._joint_actions_for_actor(b.obs, i).flatten(start_dim=1)
                    loss = -self.critics[i](go, actions_i).mean()
                self.optimizer_step(loss, self.actor_opts[i], self.actors[i].parameters())
                actor_losses.append(float(loss.detach().float().item()))
            actor_updated = 1.0
            for i in range(self.n_agents):
                soft_update(self.target_critics[i], self.critics[i], self.tau)
                soft_update(self.target_critics2[i], self.critics2[i], self.tau)
                soft_update(self.target_actors[i], self.actors[i], self.tau)

        self.finish_optimizer_steps()
        mean_q1 = float(np.mean(q1_losses))
        mean_q2 = float(np.mean(q2_losses))
        return {
            "critic_loss": 0.5 * (mean_q1 + mean_q2),
            "critic1_loss": mean_q1,
            "critic2_loss": mean_q2,
            "actor_loss": float(np.mean(actor_losses)) if actor_losses else 0.0,
            "actor_updated": actor_updated,
            "q_mean": float(np.mean(q_means)),
            "target_q_mean": float(np.mean(target_q_means)),
            "td_error_abs_mean": float(np.mean(td_errors)),
            "q_gap_abs_mean": float(abs(mean_q1 - mean_q2)),
        }

    def checkpoint(self) -> dict[str, Any]:
        p = super().checkpoint()
        p["algorithm"] = self.name
        p.update(
            {
                "critics2": self.critics2.state_dict(),
                "target_critics2": self.target_critics2.state_dict(),
                "critic2_opts": [opt.state_dict() for opt in self.critic2_opts],
            }
        )
        return p

    def load_checkpoint(self, p: dict[str, Any]) -> None:
        super().load_checkpoint(p)
        self.critics2.load_state_dict(p["critics2"])
        self.target_critics2.load_state_dict(p.get("target_critics2", p["critics2"]))
        if "critic2_opts" in p:
            for state, opt in zip(p["critic2_opts"], self.critic2_opts):
                opt.load_state_dict(state)
