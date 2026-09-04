from __future__ import annotations

import math
from typing import Any

import numpy as np
from gymnasium.spaces import Box

from .models import circle_collision, communication_rate_bps, multirotor_power_w


class PaperUAVEnv:
    """Compact 2.5D heterogeneous-UAV environment reproducing the paper's software model.

    API intentionally mirrors a PettingZoo ParallelEnv tuple without requiring PettingZoo.
    """

    action_dim = 2

    def __init__(self, cfg: dict[str, Any], seed: int = 0):
        self.cfg = cfg
        self.paper = cfg["paper"]
        self.assumed = cfg["assumed"]
        self.scenario = cfg["scenario"]
        self.area_size_m = float(self.paper["area_size_m"])
        self.max_steps = int(self.paper["episode_steps"])
        self.dt = float(self.assumed["dt_s"])
        self.n_fixed = int(self.scenario["fixed_wing"])
        self.n_rotor = int(self.scenario["multirotor"])
        self.n_agents = self.n_fixed + self.n_rotor
        self.n_targets = int(self.scenario["targets"])
        self.n_obstacles = int(self.scenario["obstacles"])
        self.agents = [f"fixed_{i}" for i in range(self.n_fixed)] + [f"rotor_{i}" for i in range(self.n_rotor)]
        self.fixed_indices = list(range(self.n_fixed))
        self.multirotor_indices = list(range(self.n_fixed, self.n_agents))
        self.agent_types = np.array([0] * self.n_fixed + [1] * self.n_rotor, dtype=np.int64)  # 0=fixed, 1=rotor
        # own 11 + every other UAV 7 + target 4 + obstacle 4
        self.obs_dim = 11 + (self.n_agents - 1) * 7 + self.n_targets * 4 + self.n_obstacles * 4
        self.observation_space = Box(-np.inf, np.inf, shape=(self.obs_dim,), dtype=np.float32)
        self.action_space = Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.rng = np.random.default_rng(seed)
        self._seed = seed
        self.reset(seed=seed)

    def reset(self, seed: int | None = None):
        if seed is not None:
            self._seed = int(seed)
            self.rng = np.random.default_rng(self._seed)
        margin = 300.0
        self.positions = np.zeros((self.n_agents, 3), dtype=np.float64)
        self.positions[:, :2] = self.rng.uniform(margin, self.area_size_m - margin, size=(self.n_agents, 2))
        self.positions[self.fixed_indices, 2] = self.assumed["fixed_altitude_m"]
        self.positions[self.multirotor_indices, 2] = self.assumed["multirotor_altitude_m"]
        self.velocities = np.zeros((self.n_agents, 2), dtype=np.float64)
        if self.n_fixed:
            heading0 = self.rng.uniform(-math.pi, math.pi, size=self.n_fixed)
            v0 = self.paper["fixed_speed_min_mps"]
            self.velocities[self.fixed_indices, 0] = np.cos(heading0) * v0
            self.velocities[self.fixed_indices, 1] = np.sin(heading0) * v0
        self.headings = np.zeros(self.n_agents, dtype=np.float64)
        if self.n_fixed:
            self.headings[self.fixed_indices] = np.arctan2(
                self.velocities[self.fixed_indices, 1], self.velocities[self.fixed_indices, 0]
            )
        self.battery_pct = np.full(self.n_agents, 100.0, dtype=np.float64)
        self.targets = self.rng.uniform(margin, self.area_size_m - margin, size=(self.n_targets, 2)).astype(np.float64)
        self.target_found = np.zeros(self.n_targets, dtype=bool)
        radii = self.rng.uniform(
            self.assumed["obstacle_radius_min_m"], self.assumed["obstacle_radius_max_m"], size=self.n_obstacles
        )
        centers = self.rng.uniform(margin, self.area_size_m - margin, size=(self.n_obstacles, 2))
        self.obstacles = np.column_stack([centers, radii]).astype(np.float64)
        self.step_count = 0
        self.cumulative_broken_time = np.zeros(self.n_rotor, dtype=np.float64)
        self.last_rates_bps = np.zeros(self.n_rotor, dtype=np.float64)
        self.total_energy_used_j = 0.0
        self.collision_count = 0
        self.obstacle_hits = 0
        self.boundary_hits = 0
        self.trajectory = [self.positions.copy()]
        self._refresh_links()
        return self._observations(), self._info(step_energy_j=0.0, action_saturation=0.0)

    def _nearest_fixed_rate(self, rotor_idx: int) -> float:
        if not self.fixed_indices:
            return 0.0
        rates = []
        for fi in self.fixed_indices:
            delta = self.positions[rotor_idx] - self.positions[fi]
            rates.append(communication_rate_bps(np.linalg.norm(delta[:2]), delta[2], self.cfg))
        return max(rates)

    def _refresh_links(self) -> None:
        for j, idx in enumerate(self.multirotor_indices):
            self.last_rates_bps[j] = self._nearest_fixed_rate(idx)

    def _observations(self) -> dict[str, np.ndarray]:
        obs: dict[str, np.ndarray] = {}
        area = self.area_size_m
        vmax_scale = max(self.paper["fixed_speed_max_mps"], self.paper["multirotor_speed_max_mps"])
        for i, name in enumerate(self.agents):
            speed = float(np.linalg.norm(self.velocities[i]))
            is_fixed = 1.0 if i in self.fixed_indices else 0.0
            if i in self.multirotor_indices:
                ri = self.multirotor_indices.index(i)
                linked = float(self.last_rates_bps[ri] >= self.paper["min_comm_rate_bps"])
            else:
                linked = 1.0
            own = [
                self.positions[i, 0] / area,
                self.positions[i, 1] / area,
                self.positions[i, 2] / area,
                self.velocities[i, 0] / vmax_scale,
                self.velocities[i, 1] / vmax_scale,
                speed / vmax_scale,
                self.battery_pct[i] / 100.0,
                math.sin(self.headings[i]),
                math.cos(self.headings[i]),
                is_fixed,
                linked,
            ]
            other = []
            for j in range(self.n_agents):
                if j == i:
                    continue
                d = self.positions[j] - self.positions[i]
                dist = float(np.linalg.norm(d))
                other.extend([
                    d[0] / area, d[1] / area, d[2] / area, dist / area,
                    self.battery_pct[j] / 100.0,
                    float(j in self.fixed_indices),
                    float(dist < area),
                ])
            target = []
            for k in range(self.n_targets):
                dxy = self.targets[k] - self.positions[i, :2]
                target.extend([dxy[0] / area, dxy[1] / area, np.linalg.norm(dxy) / area, float(self.target_found[k])])
            obstacle = []
            for circle in self.obstacles:
                dxy = circle[:2] - self.positions[i, :2]
                obstacle.extend([dxy[0] / area, dxy[1] / area, circle[2] / area, (np.linalg.norm(dxy) - circle[2]) / area])
            vec = np.asarray(own + other + target + obstacle, dtype=np.float32)
            obs[name] = vec
        return obs

    def observations_array(self) -> np.ndarray:
        d = self._observations()
        return np.stack([d[a] for a in self.agents], axis=0)

    def _task_reward(self, idx: int, fixed: bool) -> float:
        if self.n_targets == 0:
            return 0.0
        dists = np.linalg.norm(self.targets - self.positions[idx, :2], axis=1)
        active = ~self.target_found
        if not np.any(active):
            return 0.0
        d = float(np.min(dists[active]))
        zeta = self.assumed["search_reward_coeff"]
        eps = self.assumed["reward_distance_epsilon_km"]
        if fixed:
            return zeta / (d / 1000.0 + eps) if d <= self.assumed["fixed_detect_m"] else 0.0
        if d < self.assumed["target_found_m"]:
            return zeta
        if d <= self.assumed["target_detect_m"]:
            return zeta / (d / 1000.0 + eps)
        return 0.0

    def _safety_reward(self, idx: int) -> tuple[float, int]:
        eta = self.assumed["safety_reward_coeff"]
        eps = self.assumed["reward_distance_epsilon_km"]
        penalty = 0.0
        close = 0
        for j in range(self.n_agents):
            if j == idx:
                continue
            d = float(np.linalg.norm(self.positions[j] - self.positions[idx]))
            if d <= self.assumed["safety_distance_m"]:
                close += 1
                penalty -= eta / (d / 1000.0 + eps)
        for circle in self.obstacles:
            if circle_collision(self.positions[idx, :2], circle):
                penalty -= self.assumed["obstacle_penalty"]
        return penalty, close

    def step(self, actions: dict[str, np.ndarray]):
        if set(actions) != set(self.agents):
            missing = set(self.agents) - set(actions)
            raise ValueError(f"Actions must be provided for every agent; missing={sorted(missing)}")
        act = np.stack([np.asarray(actions[a], dtype=np.float64) for a in self.agents])
        if act.shape != (self.n_agents, 2):
            raise ValueError(f"Expected actions {(self.n_agents, 2)}, got {act.shape}")
        act = np.clip(act, -1.0, 1.0)
        action_saturation = float(np.mean(np.abs(act) > 0.95))
        step_energy = 0.0
        self.collision_count = 0
        self.obstacle_hits = 0
        self.boundary_hits = 0

        # Paper Eq. (8)-(12), reduced to planar motion with fixed type altitude.
        for i in self.fixed_indices:
            turn = act[i, 0] * self.assumed["fixed_turn_rate_rad_s"]
            self.headings[i] = (self.headings[i] + turn * self.dt + math.pi) % (2 * math.pi) - math.pi
            speed = float(np.linalg.norm(self.velocities[i]))
            target_speed = self.paper["fixed_speed_min_mps"] + (act[i, 1] + 1.0) * 0.5 * (
                self.paper["fixed_speed_max_mps"] - self.paper["fixed_speed_min_mps"]
            )
            delta_v = np.clip(target_speed - speed, -self.paper["max_accel_mps2"] * self.dt, self.paper["max_accel_mps2"] * self.dt)
            speed = np.clip(speed + delta_v, self.paper["fixed_speed_min_mps"], self.paper["fixed_speed_max_mps"])
            self.velocities[i] = speed * np.array([math.cos(self.headings[i]), math.sin(self.headings[i])])
            self.positions[i, :2] += self.velocities[i] * self.dt

        for i in self.multirotor_indices:
            thrust01 = 0.5 * (act[i, 0] + 1.0)
            angle = act[i, 1] * math.pi
            accel = thrust01 * self.paper["max_accel_mps2"] * np.array([math.cos(angle), math.sin(angle)])
            self.velocities[i] += accel * self.dt
            speed = float(np.linalg.norm(self.velocities[i]))
            vmax = self.paper["multirotor_speed_max_mps"]
            if speed > vmax:
                self.velocities[i] *= vmax / speed
                speed = vmax
            self.positions[i, :2] += self.velocities[i] * self.dt
            power = multirotor_power_w(speed, float(np.linalg.norm(accel)), self.cfg)
            e = power * self.dt
            step_energy += e
            self.battery_pct[i] = max(0.0, self.battery_pct[i] - 100.0 * e / self.assumed["battery_capacity_j"])

        before_clip = self.positions[:, :2].copy()
        self.positions[:, :2] = np.clip(self.positions[:, :2], 0.0, self.area_size_m)
        self.boundary_hits = int(np.sum(np.any(np.abs(before_clip - self.positions[:, :2]) > 1e-9, axis=1)))

        for i in range(self.n_agents):
            if any(circle_collision(self.positions[i, :2], c) for c in self.obstacles):
                self.obstacle_hits += 1
        for i in range(self.n_agents):
            for j in range(i + 1, self.n_agents):
                if np.linalg.norm(self.positions[i] - self.positions[j]) <= self.assumed["safety_distance_m"]:
                    self.collision_count += 1

        # A target is confirmed only by a multi-rotor UAV.
        for k in range(self.n_targets):
            if self.target_found[k]:
                continue
            d = np.linalg.norm(self.positions[self.multirotor_indices, :2] - self.targets[k], axis=1)
            if len(d) and float(np.min(d)) <= self.assumed["target_found_m"]:
                self.target_found[k] = True

        self._refresh_links()
        broken_mask = self.last_rates_bps < self.paper["min_comm_rate_bps"]
        self.cumulative_broken_time += broken_mask.astype(np.float64) * self.dt
        self.total_energy_used_j += step_energy

        rewards: dict[str, float] = {}
        for i, name in enumerate(self.agents):
            safety, _ = self._safety_reward(i)
            safety -= self.assumed["boundary_penalty"] if np.any(before_clip[i] != self.positions[i, :2]) else 0.0
            if i in self.multirotor_indices:
                ri = self.multirotor_indices.index(i)
                rate = self.last_rates_bps[ri]
                rc = self.assumed["comm_reward_max"]
                if rate < self.paper["min_comm_rate_bps"]:
                    r_comm = -rc
                elif rate >= self.assumed["comm_rate_max_bps"]:
                    r_comm = rc
                else:
                    nearest_fixed = min(np.linalg.norm(self.positions[i] - self.positions[f]) for f in self.fixed_indices)
                    r_comm = rc / (nearest_fixed / 1000.0 + self.assumed["reward_distance_epsilon_km"])
                r_power = self.paper["energy_reward_scale"] * self.battery_pct[i] if self.battery_pct[i] > self.paper["safe_battery_pct"] else 0.0
                rewards[name] = float(r_comm + r_power + safety + self._task_reward(i, fixed=False))
            else:
                rewards[name] = float(safety + self._task_reward(i, fixed=True))

        self.step_count += 1
        self.trajectory.append(self.positions.copy())
        done = self.step_count >= self.max_steps
        terminated = {a: False for a in self.agents}
        truncated = {a: done for a in self.agents}
        info = self._info(step_energy_j=step_energy, action_saturation=action_saturation)
        return self._observations(), rewards, terminated, truncated, info

    def _info(self, step_energy_j: float, action_saturation: float) -> dict[str, Any]:
        rates = self.last_rates_bps if self.n_rotor else np.array([0.0])
        broken = rates < self.paper["min_comm_rate_bps"]
        return {
            "step": self.step_count,
            "targets_found": int(self.target_found.sum()),
            "targets_total": int(self.n_targets),
            "search_rate": float(self.target_found.mean()) if self.n_targets else 0.0,
            "energy_used_j": float(step_energy_j),
            "total_energy_used_j": float(self.total_energy_used_j),
            "mean_comm_rate_mbps": float(np.mean(rates) / 1e6),
            "min_comm_rate_mbps": float(np.min(rates) / 1e6),
            "broken_links": int(np.sum(broken)),
            "mean_broken_link_s": float(np.mean(self.cumulative_broken_time)) if self.n_rotor else 0.0,
            "max_broken_link_s": float(np.max(self.cumulative_broken_time)) if self.n_rotor else 0.0,
            "collisions": int(self.collision_count),
            "obstacle_hits": int(self.obstacle_hits),
            "boundary_hits": int(self.boundary_hits),
            "min_battery_pct": float(np.min(self.battery_pct[self.multirotor_indices])) if self.n_rotor else 100.0,
            "action_saturation": float(action_saturation),
        }
