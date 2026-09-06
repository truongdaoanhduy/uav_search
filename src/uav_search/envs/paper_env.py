from __future__ import annotations

import math
from typing import Any

import numpy as np
from gymnasium.spaces import Box

from .models import circle_collision, communication_rate_bps, multirotor_power_w, path_gain_linear


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
        if self.n_fixed:
            # Formation membership is required by the paper's star topology.  For the
            # one-leader experiments this is exact; for multiple leaders, a balanced
            # deterministic assignment is used because the paper does not publish the
            # cluster-assignment procedure.
            self.rotor_leaders = np.asarray(
                [self.fixed_indices[r % self.n_fixed] for r in range(self.n_rotor)], dtype=np.int64
            )
        else:
            self.rotor_leaders = np.full(self.n_rotor, -1, dtype=np.int64)
        # Shared padded baseline layout representing only the union of paper
        # Eqs. (17)-(20): own {p(3), v(3), e, n, psi}, every other UAV
        # {d_ij, p_j(3), e_j, n_j}, and sensed target distance d_i,k.
        # Fields not defined for a UAV type are zero-masked. Obstacle geometry is
        # intentionally not injected into the Actor observation because it is not
        # part of Eqs. (17)-(20).
        self.obs_dim = 9 + (self.n_agents - 1) * 6 + self.n_targets
        self.observation_space = Box(-np.inf, np.inf, shape=(self.obs_dim,), dtype=np.float32)
        self.action_space = Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.rng = np.random.default_rng(seed)
        self._seed = seed
        self.reset(seed=seed)

    def reset(self, seed: int | None = None):
        if seed is not None:
            self._seed = int(seed)
            self.rng = np.random.default_rng(self._seed)
        uav_margin = float(self.assumed["uav_init_margin_m"])
        self.positions = np.zeros((self.n_agents, 3), dtype=np.float64)
        # The paper specifies uniform target/building placement, but does not publish
        # initial UAV coordinates.  Keep the previous UAV-only margin as an explicit
        # implementation assumption rather than applying it to paper-described objects.
        self.positions[:, :2] = self.rng.uniform(
            uav_margin, self.area_size_m - uav_margin, size=(self.n_agents, 2)
        )
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
        self.targets = self.rng.uniform(0.0, self.area_size_m, size=(self.n_targets, 2)).astype(np.float64)
        self.target_found = np.zeros(self.n_targets, dtype=bool)
        radii = self.rng.uniform(
            self.assumed["obstacle_radius_min_m"], self.assumed["obstacle_radius_max_m"], size=self.n_obstacles
        )
        centers = self.rng.uniform(0.0, self.area_size_m, size=(self.n_obstacles, 2))
        self.obstacles = np.column_stack([centers, radii]).astype(np.float64)
        self.step_count = 0
        self.cumulative_broken_time = np.zeros(self.n_rotor, dtype=np.float64)
        self.last_rates_bps = np.zeros(self.n_rotor, dtype=np.float64)
        self.last_pair_rates_bps = np.zeros((self.n_agents, self.n_agents), dtype=np.float64)
        self.last_adjacency = np.zeros((self.n_agents, self.n_agents), dtype=np.int8)
        self.total_energy_used_j = 0.0
        self.cumulative_energy_by_rotor_j = np.zeros(self.n_rotor, dtype=np.float64)
        self.last_reward_components = {
            name: {"communication": 0.0, "energy": 0.0, "safety": 0.0, "task": 0.0, "total": 0.0}
            for name in self.agents
        }
        self.last_action_saturation_by_agent = np.zeros(self.n_agents, dtype=np.float64)
        self.collision_count = 0
        self.obstacle_hits = 0
        self.boundary_hits = 0
        self.trajectory = [self.positions.copy()]
        self._refresh_links()
        return self._observations(), self._info(step_energy_j=0.0, action_saturation=0.0)

    def _candidate_links(self) -> np.ndarray:
        """Paper topology: rotor-leader star edges plus fixed-wing leader mesh edges."""
        candidate = np.zeros((self.n_agents, self.n_agents), dtype=bool)
        for a, fi in enumerate(self.fixed_indices):
            for fj in self.fixed_indices[a + 1 :]:
                candidate[fi, fj] = True
                candidate[fj, fi] = True
        for local_idx, rotor_idx in enumerate(self.multirotor_indices):
            leader = int(self.rotor_leaders[local_idx])
            if leader >= 0:
                candidate[rotor_idx, leader] = True
                candidate[leader, rotor_idx] = True
        return candidate

    def _received_power_w(self, receiver: int, transmitter: int) -> float:
        delta = self.positions[transmitter] - self.positions[receiver]
        force_los = receiver in self.fixed_indices and transmitter in self.fixed_indices
        gain = path_gain_linear(
            float(np.linalg.norm(delta[:2])),
            float(delta[2]),
            self.cfg,
            force_los=force_los,
        )
        tx_power_w = float(self.cfg.get("reference_backed", {}).get("tx_power_w", self.assumed.get("tx_power_w", 5.0)))
        return float(tx_power_w * gain)

    def _pair_rate_bps(self, receiver: int, transmitter: int, candidate: np.ndarray) -> float:
        delta = self.positions[transmitter] - self.positions[receiver]
        interference = 0.0
        for interferer in range(self.n_agents):
            if interferer in (receiver, transmitter):
                continue
            if candidate[receiver, interferer]:
                interference += self._received_power_w(receiver, interferer)
        force_los = receiver in self.fixed_indices and transmitter in self.fixed_indices
        return communication_rate_bps(
            float(np.linalg.norm(delta[:2])),
            float(delta[2]),
            self.cfg,
            interference_power_w=interference,
            force_los=force_los,
        )

    def _refresh_links(self) -> None:
        candidate = self._candidate_links()
        self.last_pair_rates_bps.fill(0.0)
        self.last_adjacency.fill(0)
        rmin = float(self.paper["min_comm_rate_bps"])
        for receiver in range(self.n_agents):
            for transmitter in range(self.n_agents):
                if not candidate[receiver, transmitter]:
                    continue
                rate = self._pair_rate_bps(receiver, transmitter, candidate)
                self.last_pair_rates_bps[receiver, transmitter] = rate
                if rate > rmin:
                    self.last_adjacency[receiver, transmitter] = 1
        for local_idx, rotor_idx in enumerate(self.multirotor_indices):
            leader = int(self.rotor_leaders[local_idx])
            self.last_rates_bps[local_idx] = (
                self.last_pair_rates_bps[rotor_idx, leader] if leader >= 0 else 0.0
            )

    def _network_state(self, idx: int) -> float:
        """Return the current binary network-connection state n_i from adjacency."""
        if idx in self.multirotor_indices:
            local_idx = self.multirotor_indices.index(idx)
            leader = int(self.rotor_leaders[local_idx])
            if leader < 0:
                return 0.0
            return float(self.last_adjacency[idx, leader] == 1)
        return float(np.any(self.last_adjacency[idx] > 0))

    def _target_observation_distance(self, idx: int, target_idx: int) -> float:
        """Return normalized d_i,k only when the target is currently observable.

        The system model states that target distance can be perceived only within
        sensing range, and Sec. IV-D masks targets after they are discovered.
        Numerical sensing ranges remain the existing explicit assumptions because
        the original paper defines D_detect/D_detect-f but does not publish values.
        """
        if self.target_found[target_idx]:
            return 0.0
        d = float(np.linalg.norm(self.targets[target_idx] - self.positions[idx, :2]))
        detect_range = (
            float(self.assumed["fixed_detect_m"])
            if idx in self.fixed_indices
            else float(self.assumed["target_detect_m"])
        )
        if d > detect_range:
            return 0.0
        return d / self.area_size_m

    def _observations(self) -> dict[str, np.ndarray]:
        """Build type-specific observations from paper Eqs. (17)-(20).

        MASAC/MATD3/MADDPG in this repository require one common vector width, so
        the two paper observation definitions are embedded in a shared padded
        layout. Slots that are not part of a UAV type's paper state are zero.
        """
        obs: dict[str, np.ndarray] = {}
        area = self.area_size_m
        vmax_scale = max(self.paper["fixed_speed_max_mps"], self.paper["multirotor_speed_max_mps"])
        for i, name in enumerate(self.agents):
            is_fixed = i in self.fixed_indices
            is_rotor = i in self.multirotor_indices

            # Union self layout: p(3), v(3), e, n, psi.
            # Eq. (17) rotor: {p, v, e, n}; Eq. (19) fixed: {p, v, psi}.
            own = [
                self.positions[i, 0] / area,
                self.positions[i, 1] / area,
                self.positions[i, 2] / area,
                self.velocities[i, 0] / vmax_scale,
                self.velocities[i, 1] / vmax_scale,
                0.0,  # z-velocity is unavailable in the retained 2.5D approximation.
                self.battery_pct[i] / 100.0 if is_rotor else 0.0,
                self._network_state(i) if is_rotor else 0.0,
                self.headings[i] / math.pi if is_fixed else 0.0,
            ]

            # Union neighbor layout: d_i,j, p_j(3), e_j, n_j.
            # Eq. (18) rotor includes all of these; Eq. (20) fixed includes only
            # p_j and n_j, so d_i,j and e_j are zero-masked for fixed observers.
            other: list[float] = []
            for j in range(self.n_agents):
                if j == i:
                    continue
                d_ij = float(np.linalg.norm(self.positions[j] - self.positions[i])) / area
                e_j = self.battery_pct[j] / 100.0 if j in self.multirotor_indices else 0.0
                other.extend([
                    d_ij if is_rotor else 0.0,
                    self.positions[j, 0] / area,
                    self.positions[j, 1] / area,
                    self.positions[j, 2] / area,
                    e_j if is_rotor else 0.0,
                    self._network_state(j),
                ])

            # Eqs. (18),(20) use target distance d_i,k. The paper states that
            # target information is available only inside sensing range; found
            # targets are masked by the observation-optimization procedure.
            target = [self._target_observation_distance(i, k) for k in range(self.n_targets)]

            vec = np.asarray(own + other + target, dtype=np.float32)
            obs[name] = vec
        return obs

    def observations_array(self) -> np.ndarray:
        d = self._observations()
        return np.stack([d[a] for a in self.agents], axis=0)

    def _task_reward(self, idx: int, fixed: bool) -> float:
        """Paper Eqs. (24)-(25): target-search reward for rotor/fixed-wing UAVs."""
        if self.n_targets == 0:
            return 0.0
        dists = np.linalg.norm(self.targets - self.positions[idx, :2], axis=1)
        active = ~self.target_found
        if not np.any(active):
            return 0.0
        d = float(np.min(dists[active]))
        zeta = float(self.assumed["search_reward_coeff"])
        delta_d = float(self.assumed["reward_distance_epsilon_m"])
        if fixed:
            return zeta / (d + delta_d) if d <= self.assumed["fixed_detect_m"] else 0.0
        if d < self.assumed["target_found_m"]:
            return zeta
        if d <= self.assumed["target_detect_m"]:
            return zeta / (d + delta_d)
        return 0.0

    def _safety_reward(self, idx: int) -> tuple[float, int]:
        """Paper Eq. (23): inverse-distance inter-UAV safety penalty."""
        eta = float(self.assumed["safety_reward_coeff"])
        delta_d = float(self.assumed["reward_distance_epsilon_m"])
        penalty = 0.0
        close = 0
        for j in range(self.n_agents):
            if j == idx:
                continue
            d = float(np.linalg.norm(self.positions[j] - self.positions[idx]))
            if d <= self.assumed["safety_distance_m"]:
                close += 1
                penalty -= eta / (d + delta_d)
        return penalty, close

    def _energy_reward(self, idx: int) -> float:
        """Paper Eq. (22): scaled remaining energy above the safe threshold."""
        remaining = float(self.battery_pct[idx])
        if remaining <= float(self.paper["safe_battery_pct"]):
            return 0.0
        return float(self.paper["energy_reward_scale"] * remaining)

    def _communication_reward(self, idx: int) -> float:
        """Paper Eq. (21): multi-rotor communication reward to its formation leader."""
        if idx not in self.multirotor_indices:
            return 0.0
        local_idx = self.multirotor_indices.index(idx)
        rate = float(self.last_rates_bps[local_idx])
        rc = float(self.assumed["comm_reward_max"])
        if rate < float(self.paper["min_comm_rate_bps"]):
            return -rc
        if rate >= float(self.assumed["comm_rate_max_bps"]):
            return rc
        leader = int(self.rotor_leaders[local_idx])
        if leader < 0:
            return -rc
        distance_m = float(np.linalg.norm(self.positions[idx] - self.positions[leader]))
        delta_d = float(self.assumed["reward_distance_epsilon_m"])
        return rc / (distance_m + delta_d)

    def step(self, actions: dict[str, np.ndarray]):
        if set(actions) != set(self.agents):
            missing = set(self.agents) - set(actions)
            raise ValueError(f"Actions must be provided for every agent; missing={sorted(missing)}")
        act = np.stack([np.asarray(actions[a], dtype=np.float64) for a in self.agents])
        if act.shape != (self.n_agents, 2):
            raise ValueError(f"Expected actions {(self.n_agents, 2)}, got {act.shape}")
        act = np.clip(act, -1.0, 1.0)
        self.last_action_saturation_by_agent = np.mean(np.abs(act) > 0.95, axis=1).astype(np.float64)
        action_saturation = float(np.mean(self.last_action_saturation_by_agent))
        step_energy = 0.0
        self.collision_count = 0
        self.obstacle_hits = 0
        self.boundary_hits = 0
        previous_xy = self.positions[:, :2].copy()

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
            local_idx = self.multirotor_indices.index(i)
            self.cumulative_energy_by_rotor_j[local_idx] += e
            self.battery_pct[i] = max(0.0, self.battery_pct[i] - 100.0 * e / self.assumed["battery_capacity_j"])

        before_clip = self.positions[:, :2].copy()
        self.positions[:, :2] = np.clip(self.positions[:, :2], 0.0, self.area_size_m)
        self.boundary_hits = int(np.sum(np.any(np.abs(before_clip - self.positions[:, :2]) > 1e-9, axis=1)))

        for i in range(self.n_agents):
            if any(circle_collision(self.positions[i, :2], c) for c in self.obstacles):
                self.obstacle_hits += 1
                # C3 in the paper excludes the obstacle domain. Reject the candidate
                # displacement rather than adding an unpublished reward penalty.
                self.positions[i, :2] = previous_xy[i]
                if i in self.multirotor_indices:
                    self.velocities[i] = 0.0
        for i in range(self.n_agents):
            for j in range(i + 1, self.n_agents):
                if np.linalg.norm(self.positions[i] - self.positions[j]) <= self.assumed["safety_distance_m"]:
                    self.collision_count += 1

        self._refresh_links()
        broken_mask = self.last_rates_bps < self.paper["min_comm_rate_bps"]
        self.cumulative_broken_time += broken_mask.astype(np.float64) * self.dt
        self.total_energy_used_j += step_energy

        rewards: dict[str, float] = {}
        components: dict[str, dict[str, float]] = {}
        for i, name in enumerate(self.agents):
            safety, _ = self._safety_reward(i)
            if i in self.multirotor_indices:
                communication = float(self._communication_reward(i))
                energy = float(self._energy_reward(i))
                task = float(self._task_reward(i, fixed=False))
            else:
                communication = 0.0
                energy = 0.0
                task = float(self._task_reward(i, fixed=True))
            total = float(communication + energy + safety + task)
            rewards[name] = total
            components[name] = {
                "communication": communication,
                "energy": energy,
                "safety": float(safety),
                "task": task,
                "total": total,
            }
        self.last_reward_components = components

        # Confirm only after computing Eq. (24), so the discovering UAV receives zeta on this step.
        for k in range(self.n_targets):
            if self.target_found[k]:
                continue
            d = np.linalg.norm(self.positions[self.multirotor_indices, :2] - self.targets[k], axis=1)
            if len(d) and float(np.min(d)) <= self.assumed["target_found_m"]:
                self.target_found[k] = True

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
        rotor_names = [self.agents[i] for i in self.multirotor_indices]
        rotor_battery = self.battery_pct[self.multirotor_indices] if self.n_rotor else np.array([100.0])
        return {
            "step": self.step_count,
            "targets_found": int(self.target_found.sum()),
            "targets_total": int(self.n_targets),
            "search_rate": float(self.target_found.mean()) if self.n_targets else 0.0,
            "energy_used_j": float(step_energy_j),
            "total_energy_used_j": float(self.total_energy_used_j),
            "energy_consumption_pct": float(100.0 - np.mean(rotor_battery)) if self.n_rotor else 0.0,
            "mean_comm_rate_mbps": float(np.mean(rates) / 1e6),
            "min_comm_rate_mbps": float(np.min(rates) / 1e6),
            "broken_links": int(np.sum(broken)),
            "mean_broken_link_s": float(np.mean(self.cumulative_broken_time)) if self.n_rotor else 0.0,
            "max_broken_link_s": float(np.max(self.cumulative_broken_time)) if self.n_rotor else 0.0,
            "collisions": int(self.collision_count),
            "obstacle_hits": int(self.obstacle_hits),
            "boundary_hits": int(self.boundary_hits),
            "min_battery_pct": float(np.min(rotor_battery)) if self.n_rotor else 100.0,
            "action_saturation": float(action_saturation),
            "agent_reward_components": {name: dict(values) for name, values in self.last_reward_components.items()},
            "agent_battery_pct": {name: float(self.battery_pct[i]) for i, name in enumerate(self.agents)},
            "agent_action_saturation": {
                name: float(self.last_action_saturation_by_agent[i]) for i, name in enumerate(self.agents)
            },
            "agent_comm_rate_mbps": {
                name: float(self.last_rates_bps[k] / 1e6) for k, name in enumerate(rotor_names)
            },
            "agent_broken_link_s": {
                name: float(self.cumulative_broken_time[k]) for k, name in enumerate(rotor_names)
            },
            "agent_energy_consumption_pct": {
                name: float(100.0 - self.battery_pct[self.multirotor_indices[k]]) for k, name in enumerate(rotor_names)
            },
            "agent_energy_used_j": {
                name: float(self.cumulative_energy_by_rotor_j[k]) for k, name in enumerate(rotor_names)
            },
        }
