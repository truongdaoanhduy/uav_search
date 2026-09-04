from __future__ import annotations

from copy import deepcopy
from typing import Any

import torch

from .common import soft_update
from .maddpg import MADDPG
from .networks import CentralizedCritic
from uav_search.runtime import make_adam


class MATD3(MADDPG):
    name = "matd3"

    def __init__(self, env, cfg: dict[str, Any], device: str = "cpu", seed: int = 0):
        super().__init__(env, cfg, device, seed)
        gd = self.n_agents * self.obs_dim
        ad = self.n_agents * self.action_dim
        self.critic2 = CentralizedCritic(gd, ad, self.n_agents, self.hidden_sizes).to(self.device)
        self.target_critic2 = deepcopy(self.critic2).to(self.device)
        self.critic2_opt = make_adam(self.critic2.parameters(), lr=float(cfg["algorithm"]["critic_lr"]), device=self.device)
        self.policy_noise = float(cfg["algorithm"]["policy_noise"])
        self.noise_clip = float(cfg["algorithm"]["noise_clip"])
        self.policy_delay = int(cfg["algorithm"]["policy_delay"])

    def update(self) -> dict[str, float]:
        if len(self.replay) < self.batch_size:
            return {}
        self.update_step += 1
        b = self.replay.sample(self.batch_size, self.device)
        go = self._global(b.obs)
        gn = self._global(b.next_obs)
        ja = b.actions.flatten(start_dim=1)
        with torch.no_grad():
            na = self._actions_tensor(b.next_obs, target=True)
            noise = torch.randn_like(na).mul_(self.policy_noise).clamp_(-self.noise_clip, self.noise_clip)
            na = (na + noise).clamp(-1.0, 1.0).flatten(start_dim=1)
            tq1 = self.target_critic(gn, na)
            tq2 = self.target_critic2(gn, na)
            y = b.rewards + self.gamma * (1.0 - b.dones) * torch.minimum(tq1, tq2)
        q1 = self.critic(go, ja)
        q2 = self.critic2(go, ja)
        l1 = torch.nn.functional.mse_loss(q1, y)
        l2 = torch.nn.functional.mse_loss(q2, y)
        self.critic_opt.zero_grad(set_to_none=True)
        l1.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
        self.critic_opt.step()
        self.critic2_opt.zero_grad(set_to_none=True)
        l2.backward()
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), 10.0)
        self.critic2_opt.step()

        actor_loss_value = 0.0
        actor_updated = 0.0
        if self.update_step % self.policy_delay == 0:
            ca = self._actions_tensor(b.obs, target=False).flatten(start_dim=1)
            actor_loss = -self.critic(go, ca).mean()
            self.actor_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actors.parameters(), 10.0)
            self.actor_opt.step()
            actor_loss_value = float(actor_loss.item())
            actor_updated = 1.0
            soft_update(self.target_critic, self.critic, self.tau)
            soft_update(self.target_critic2, self.critic2, self.tau)
            for key in self.actors:
                soft_update(self.target_actors[key], self.actors[key], self.tau)
        return {
            "critic_loss": float(0.5 * (l1.item() + l2.item())),
            "critic1_loss": float(l1.item()),
            "critic2_loss": float(l2.item()),
            "actor_loss": actor_loss_value,
            "actor_updated": actor_updated,
        }

    def checkpoint(self) -> dict[str, Any]:
        p = super().checkpoint()
        p["algorithm"] = self.name
        p.update({
            "critic2": self.critic2.state_dict(),
            "target_critic2": self.target_critic2.state_dict(),
            "critic2_opt": self.critic2_opt.state_dict(),
        })
        return p

    def load_checkpoint(self, p: dict[str, Any]) -> None:
        super().load_checkpoint(p)
        self.critic2.load_state_dict(p["critic2"])
        self.target_critic2.load_state_dict(p.get("target_critic2", p["critic2"]))
        if "critic2_opt" in p:
            self.critic2_opt.load_state_dict(p["critic2_opt"])
