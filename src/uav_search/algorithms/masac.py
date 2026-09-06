from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import torch
from torch import nn

from .base import BaseOffPolicy
from .common import soft_update
from .networks import CentralizedCritic, GaussianActor
from uav_search.runtime import make_adam


class MASAC(BaseOffPolicy):
    """CTDE multi-agent SAC with one actor and twin centralized critics per UAV.

    Root-paper ref. [44] uses independent temperature coefficients and automatic
    entropy tuning. The root paper itself also states an Actor/Critic pair for
    every UAV, so this implementation deliberately avoids parameter sharing.
    """

    name = "masac"

    def __init__(self, env, cfg: dict[str, Any], device: str = "cpu", seed: int = 0):
        super().__init__(env, cfg, device, seed)
        a = cfg["algorithm"]
        self.actors = nn.ModuleList(
            [
                GaussianActor(
                    self.obs_dim,
                    self.action_dim,
                    self.hidden_sizes,
                    a["log_std_min"],
                    a["log_std_max"],
                )
                for _ in range(self.n_agents)
            ]
        ).to(self.device)
        # Root-paper Eqs. (32)-(34) explicitly use target actor and critic networks.
        self.target_actors = deepcopy(self.actors).to(self.device)

        gd = self.n_agents * self.obs_dim
        ad = self.n_agents * self.action_dim
        self.critics1 = nn.ModuleList(
            [CentralizedCritic(gd, ad, 1, self.hidden_sizes) for _ in range(self.n_agents)]
        ).to(self.device)
        self.critics2 = nn.ModuleList(
            [CentralizedCritic(gd, ad, 1, self.hidden_sizes) for _ in range(self.n_agents)]
        ).to(self.device)
        self.target_critics1 = deepcopy(self.critics1).to(self.device)
        self.target_critics2 = deepcopy(self.critics2).to(self.device)

        actor_lr = float(a["actor_lr"])
        critic_lr = float(a["critic_lr"])
        self.actor_opts = [make_adam(actor.parameters(), lr=actor_lr, device=self.device) for actor in self.actors]
        self.critic1_opts = [make_adam(c.parameters(), lr=critic_lr, device=self.device) for c in self.critics1]
        self.critic2_opts = [make_adam(c.parameters(), lr=critic_lr, device=self.device) for c in self.critics2]

        alpha_init = float(a.get("alpha_init", a.get("alpha", 0.01)))
        if alpha_init <= 0:
            raise ValueError("MASAC alpha_init must be positive")
        self.log_alpha = nn.Parameter(
            torch.full((self.n_agents,), float(np.log(alpha_init)), dtype=torch.float32, device=self.device)
        )
        self.target_entropy = -float(self.action_dim)
        self.alpha_opt = make_adam([self.log_alpha], lr=float(a.get("alpha_lr", actor_lr)), device=self.device)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def _sample_actions(
        self,
        obs: torch.Tensor,
        *,
        target: bool = False,
        deterministic: bool = False,
        grad_agent: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        actors = self.target_actors if target else self.actors
        actions = []
        logps = []
        for i in range(self.n_agents):
            action_i, logp_i = actors[i].sample(obs[:, i, :], deterministic=deterministic)
            if grad_agent is not None and i != grad_agent:
                action_i = action_i.detach()
                logp_i = logp_i.detach()
            actions.append(action_i)
            logps.append(logp_i)
        return torch.stack(actions, dim=1), torch.stack(logps, dim=1)

    @torch.no_grad()
    def act(self, obs: np.ndarray, explore: bool = True) -> np.ndarray:
        x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with self.autocast():
            actions, _ = self._sample_actions(x, target=False, deterministic=not explore)
        return actions.squeeze(0).float().cpu().numpy().astype(np.float32)

    def update(self) -> dict[str, float]:
        if len(self.replay) < self.batch_size:
            return {}
        b = self.replay.sample(self.batch_size, self.device, non_blocking=self.non_blocking)
        go = self._global(b.obs)
        gn = self._global(b.next_obs)
        ja = b.actions.flatten(start_dim=1)

        with torch.no_grad(), self.autocast():
            next_actions, next_logps = self._sample_actions(
                b.next_obs, target=True, deterministic=False
            )
            next_joint = next_actions.flatten(start_dim=1)
            alpha_detached = self.alpha.detach()
            targets = []
            for i in range(self.n_agents):
                tq = torch.minimum(
                    self.target_critics1[i](gn, next_joint),
                    self.target_critics2[i](gn, next_joint),
                )
                entropy_adjusted_q = tq - alpha_detached[i] * next_logps[:, i, :]
                y = b.rewards[:, i : i + 1] + self.gamma * (1.0 - b.dones[:, i : i + 1]) * entropy_adjusted_q
                targets.append(y)

        q1_losses: list[float] = []
        q2_losses: list[float] = []
        q_means: list[float] = []
        target_q_means: list[float] = []
        td_errors: list[float] = []
        for i in range(self.n_agents):
            with self.autocast():
                q1 = self.critics1[i](go, ja)
                q2 = self.critics2[i](go, ja)
                l1 = torch.nn.functional.mse_loss(q1, targets[i])
                l2 = torch.nn.functional.mse_loss(q2, targets[i])
            q1d = q1.detach().float()
            q2d = q2.detach().float()
            yd = targets[i].detach().float()
            q_means.append(float((0.5 * (q1d + q2d)).mean().item()))
            target_q_means.append(float(yd.mean().item()))
            td_errors.append(float((0.5 * ((q1d - yd).abs() + (q2d - yd).abs())).mean().item()))
            self.optimizer_step(l1, self.critic1_opts[i], self.critics1[i].parameters())
            self.optimizer_step(l2, self.critic2_opts[i], self.critics2[i].parameters())
            q1_losses.append(float(l1.detach().float().item()))
            q2_losses.append(float(l2.detach().float().item()))

        actor_losses: list[float] = []
        entropy_values: list[float] = []
        alpha_logps: list[torch.Tensor] = []
        for i in range(self.n_agents):
            with self.autocast():
                actions_i, logps_i = self._sample_actions(
                    b.obs, target=False, deterministic=False, grad_agent=i
                )
                joint_i = actions_i.flatten(start_dim=1)
                q_i = torch.minimum(
                    self.critics1[i](go, joint_i),
                    self.critics2[i](go, joint_i),
                )
                own_logp = logps_i[:, i, :]
                loss = (self.alpha.detach()[i] * own_logp - q_i).mean()
            self.optimizer_step(loss, self.actor_opts[i], self.actors[i].parameters())
            actor_losses.append(float(loss.detach().float().item()))
            entropy_values.append(float((-own_logp.detach().float()).mean().item()))
            alpha_logps.append(own_logp.detach())

        # Ref. [44] Eq. (8): independent automatic temperature tuning per actor.
        stacked_logp = torch.cat(alpha_logps, dim=1)
        alpha_loss = -(
            self.log_alpha * (stacked_logp + self.target_entropy).mean(dim=0)
        ).mean()
        self.alpha_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        torch.nn.utils.clip_grad_norm_([self.log_alpha], 10.0)
        self.alpha_opt.step()

        self.finish_optimizer_steps()
        for i in range(self.n_agents):
            soft_update(self.target_critics1[i], self.critics1[i], self.tau)
            soft_update(self.target_critics2[i], self.critics2[i], self.tau)
            soft_update(self.target_actors[i], self.actors[i], self.tau)
        self.update_step += 1

        mean_q1 = float(np.mean(q1_losses))
        mean_q2 = float(np.mean(q2_losses))
        return {
            "critic_loss": 0.5 * (mean_q1 + mean_q2),
            "critic1_loss": mean_q1,
            "critic2_loss": mean_q2,
            "actor_loss": float(np.mean(actor_losses)),
            "entropy": float(np.mean(entropy_values)),
            "alpha": float(self.alpha.detach().mean().cpu().item()),
            "alpha_loss": float(alpha_loss.detach().cpu().item()),
            "q_mean": float(np.mean(q_means)),
            "target_q_mean": float(np.mean(target_q_means)),
            "td_error_abs_mean": float(np.mean(td_errors)),
            "q_gap_abs_mean": float(abs(mean_q1 - mean_q2)),
        }

    def checkpoint(self) -> dict[str, Any]:
        return {
            "algorithm": self.name,
            "config": self.cfg,
            "actors": self.actors.state_dict(),
            "target_actors": self.target_actors.state_dict(),
            "critics1": self.critics1.state_dict(),
            "critics2": self.critics2.state_dict(),
            "target_critics1": self.target_critics1.state_dict(),
            "target_critics2": self.target_critics2.state_dict(),
            "actor_opts": [opt.state_dict() for opt in self.actor_opts],
            "critic1_opts": [opt.state_dict() for opt in self.critic1_opts],
            "critic2_opts": [opt.state_dict() for opt in self.critic2_opts],
            "log_alpha": self.log_alpha.detach().cpu(),
            "alpha_opt": self.alpha_opt.state_dict(),
            "amp_scaler": self.scaler.state_dict(),
            "update_step": self.update_step,
            "requested_replay_capacity": self.requested_replay_capacity,
            "effective_replay_capacity": self.replay.capacity,
        }

    def load_checkpoint(self, p: dict[str, Any]) -> None:
        self.actors.load_state_dict(p["actors"])
        self.target_actors.load_state_dict(p.get("target_actors", p["actors"]))
        self.critics1.load_state_dict(p["critics1"])
        self.critics2.load_state_dict(p["critics2"])
        self.target_critics1.load_state_dict(p.get("target_critics1", p["critics1"]))
        self.target_critics2.load_state_dict(p.get("target_critics2", p["critics2"]))
        for key, opts in (
            ("actor_opts", self.actor_opts),
            ("critic1_opts", self.critic1_opts),
            ("critic2_opts", self.critic2_opts),
        ):
            if key in p:
                for state, opt in zip(p[key], opts):
                    opt.load_state_dict(state)
        if "log_alpha" in p:
            with torch.no_grad():
                self.log_alpha.copy_(torch.as_tensor(p["log_alpha"], device=self.device))
        if "alpha_opt" in p:
            self.alpha_opt.load_state_dict(p["alpha_opt"])
        if "amp_scaler" in p:
            self.scaler.load_state_dict(p["amp_scaler"])
        self.update_step = int(p.get("update_step", 0))
