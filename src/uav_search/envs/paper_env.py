from __future__ import annotations

import math
from typing import Any

import numpy as np
from gymnasium.spaces import Box

from .models import circle_collision, communication_rate_bps, multirotor_power_w, path_gain_linear
from .network_backends import NetworkStepResult, TransmissionIntent, create_network_backend


GCS_RECIPIENT = -1


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
        # Research scenarios may need a longer mission horizon than the root
        # paper's 50-step benchmark. Paper-faithful scenarios keep the paper value.
        self.max_steps = int(
            cfg.get("runtime", {}).get(
                "episode_steps_override", self.scenario.get("episode_steps", self.paper["episode_steps"])
            )
        )
        self.dt = float(self.assumed["dt_s"])
        self.n_fixed = int(self.scenario["fixed_wing"])
        self.n_rotor = int(self.scenario["multirotor"])
        self.n_agents = self.n_fixed + self.n_rotor
        self.n_targets = int(self.scenario["targets"])
        self.n_obstacles = int(self.scenario["obstacles"])
        self.peer_mode = str(self.scenario.get("architecture", "paper_heterogeneous")) == "homogeneous_peer"
        if self.peer_mode and self.n_fixed != 0:
            raise ValueError("homogeneous_peer requires fixed_wing=0; all UAVs use the root-paper multi-rotor model")
        if self.peer_mode:
            self.agents = [f"uav_{i}" for i in range(self.n_agents)]
        else:
            self.agents = [f"fixed_{i}" for i in range(self.n_fixed)] + [f"rotor_{i}" for i in range(self.n_rotor)]
        self.fixed_indices = list(range(self.n_fixed))
        self.multirotor_indices = list(range(self.n_fixed, self.n_agents))
        self.agent_types = np.array([0] * self.n_fixed + [1] * self.n_rotor, dtype=np.int64)  # 0=fixed, 1=rotor
        if self.peer_mode:
            self.gcs_position = np.asarray(
                self.scenario.get("gcs_position_m", [self.area_size_m / 2.0, self.area_size_m / 2.0, 0.0]),
                dtype=np.float64,
            )
            if self.gcs_position.shape != (3,):
                raise ValueError("scenario.gcs_position_m must contain [x, y, z]")
            self.peer_report_bytes = int(self.scenario.get("report_bytes", 1_000_000))
            self.peer_buffer_bytes = int(self.scenario.get("buffer_bytes", 8_000_000))
            self.peer_delivery_reward = float(self.scenario.get("delivery_reward", 20.0))
            self.peer_report_ttl_steps = int(self.scenario.get("report_ttl_steps", 300))
            self.peer_neighbor_cache_ttl_steps = int(self.scenario.get("neighbor_cache_ttl_steps", 50))
            self.peer_energy_cost_per_j = float(self.scenario.get("energy_cost_per_j", 0.001))
            if self.peer_report_bytes <= 0 or self.peer_buffer_bytes <= 0:
                raise ValueError("report_bytes and buffer_bytes must be positive")
            if self.peer_report_ttl_steps <= 0 or self.peer_neighbor_cache_ttl_steps <= 0:
                raise ValueError("report_ttl_steps and neighbor_cache_ttl_steps must be positive")
            self.network_backend = create_network_backend(
                str(self.scenario.get("network_backend", "analytical")), self.cfg, seed=seed
            )
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
        # Peer mode appends queue state and GCS-relative/network state while retaining
        # the root-paper observation content for the homogeneous multi-rotor agents.
        self.peer_obs_extra_dim = 4 if self.peer_mode else 0
        self.obs_dim = 9 + (self.n_agents - 1) * 6 + self.n_targets + self.peer_obs_extra_dim
        self.action_dim = 5 if self.peer_mode else 2
        self.observation_space = Box(-np.inf, np.inf, shape=(self.obs_dim,), dtype=np.float32)
        self.action_space = Box(-1.0, 1.0, shape=(self.action_dim,), dtype=np.float32)
        self.rng = np.random.default_rng(seed)
        self._seed = seed
        self.reset(seed=seed)

    def _sample_initial_uav_xy(self) -> np.ndarray:
        """Sample deterministic scenario-specific UAV initialization coordinates."""
        mode = str(self.scenario.get("initialization", "uniform_map")).strip().lower()
        if not self.peer_mode or mode == "uniform_map":
            margin = float(self.assumed["uav_init_margin_m"])
            return self.rng.uniform(margin, self.area_size_m - margin, size=(self.n_agents, 2))
        if mode != "launch_disk":
            raise ValueError(f"Unsupported scenario.initialization: {mode}")

        center = np.asarray(self.scenario.get("launch_center_m", [self.area_size_m / 2.0] * 2), dtype=np.float64)
        if center.shape != (2,):
            raise ValueError("scenario.launch_center_m must contain [x, y]")
        radius = float(self.scenario.get("launch_radius_m", 250.0))
        min_sep = float(self.scenario.get("initial_min_separation_m", self.assumed["safety_distance_m"]))
        if radius <= 0.0 or min_sep < 0.0:
            raise ValueError("launch_radius_m must be positive and initial_min_separation_m non-negative")

        points: list[np.ndarray] = []
        max_attempts = 20_000
        for _ in range(max_attempts):
            # sqrt(U) gives uniform area density inside the launch disk.
            r = radius * math.sqrt(float(self.rng.random()))
            theta = float(self.rng.uniform(-math.pi, math.pi))
            candidate = center + r * np.array([math.cos(theta), math.sin(theta)], dtype=np.float64)
            if np.any(candidate < 0.0) or np.any(candidate > self.area_size_m):
                continue
            if all(float(np.linalg.norm(candidate - p)) >= min_sep for p in points):
                points.append(candidate)
                if len(points) == self.n_agents:
                    return np.stack(points, axis=0)
        raise RuntimeError(
            "Unable to place all UAVs in launch disk with the configured minimum separation; "
            "increase launch_radius_m or reduce initial_min_separation_m"
        )

    def reset(self, seed: int | None = None):
        if seed is not None:
            self._seed = int(seed)
            self.rng = np.random.default_rng(self._seed)
        self.positions = np.zeros((self.n_agents, 3), dtype=np.float64)
        # Root-paper scenarios retain the prior uniform-map initialization. The u6
        # research adaptation can instead launch the peer swarm from a common base.
        self.positions[:, :2] = self._sample_initial_uav_xy()
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
        self.last_reward_components_sum = {
            "communication": 0.0, "energy": 0.0, "safety": 0.0, "task": 0.0, "total": 0.0
        }
        self.collision_count = 0
        self.obstacle_hits = 0
        self.boundary_hits = 0
        # Episode-level unique UAV counts are diagnostic aggregates only; they do
        # not change the paper reward, dynamics, termination, or observations.
        self.episode_collided_uavs: set[int] = set()
        self.episode_obstacle_hit_uavs: set[int] = set()
        self.episode_boundary_hit_uavs: set[int] = set()
        self.episode_broken_link_uavs: set[int] = set()
        self.trajectory = [self.positions.copy()]
        if self.peer_mode:
            self.last_gcs_rates_bps = np.zeros(self.n_agents, dtype=np.float64)
            self.last_selected_tx_rate_bps = np.zeros(self.n_agents, dtype=np.float64)
            self.last_selected_tx_distance_m = np.full(self.n_agents, np.inf, dtype=np.float64)
            self.last_tx_active = np.zeros(self.n_agents, dtype=bool)
            self.last_tx_success = np.zeros(self.n_agents, dtype=bool)
            self.last_delivery_reward_by_agent = np.zeros(self.n_agents, dtype=np.float64)
            self.target_known_by_agent = np.zeros((self.n_agents, self.n_targets), dtype=bool)
            self.report_generated = np.zeros(self.n_targets, dtype=bool)
            self.pending_report_source = np.full(self.n_targets, -1, dtype=np.int64)
            self.report_created_step = np.full(self.n_targets, -1, dtype=np.int64)
            self.report_expired = np.zeros(self.n_targets, dtype=bool)
            self.report_buffers = np.zeros((self.n_targets, self.n_agents), dtype=np.int64)
            self.report_delivered_bytes = np.zeros(self.n_targets, dtype=np.int64)
            self.report_delivered = np.zeros(self.n_targets, dtype=bool)
            self.queue_bytes = np.zeros(self.n_agents, dtype=np.int64)
            self.neighbor_cache_positions = np.zeros((self.n_agents, self.n_agents, 3), dtype=np.float64)
            self.neighbor_cache_battery = np.zeros((self.n_agents, self.n_agents), dtype=np.float64)
            self.neighbor_cache_seen_step = np.full((self.n_agents, self.n_agents), -1, dtype=np.int64)
            self.last_new_targets_by_agent = np.zeros(self.n_agents, dtype=np.int64)
            self.last_energy_by_agent_j = np.zeros(self.n_agents, dtype=np.float64)
            self.last_bytes_transmitted = 0
            self.total_bytes_transmitted = 0
            self.reports_delivered = 0
            self.expired_reports = 0
            self.last_network_result = NetworkStepResult()
            self.total_network_attempted_bytes = 0
            self.total_network_delivered_bytes = 0
            self.total_network_tx_energy_j = 0.0
        self._refresh_links()
        return self._observations(), self._info(step_energy_j=0.0, action_saturation=0.0)

    def _candidate_links(self) -> np.ndarray:
        """Return candidate A2A links for the selected scenario architecture.

        Legacy paper scenarios retain the published rotor-leader star plus
        fixed-wing mesh. The homogeneous research adaptation removes hierarchy,
        so every UAV pair is a candidate and the paper rate model determines
        whether that edge is currently usable.
        """
        candidate = np.zeros((self.n_agents, self.n_agents), dtype=bool)
        if self.peer_mode:
            candidate[:] = True
            np.fill_diagonal(candidate, False)
            return candidate
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

    def _received_power_at_gcs_w(self, transmitter: int) -> float:
        if not self.peer_mode:
            return 0.0
        delta = self.positions[transmitter] - self.gcs_position
        gain = path_gain_linear(
            float(np.linalg.norm(delta[:2])), float(delta[2]), self.cfg, force_los=False
        )
        tx_power_w = float(
            self.cfg.get("reference_backed", {}).get("tx_power_w", self.assumed.get("tx_power_w", 5.0))
        )
        return float(tx_power_w * gain)

    def _gcs_rate_bps(self, transmitter: int) -> float:
        """Apply the root paper's Eq. (3)-(7) channel model to the adapted GCS sink."""
        if not self.peer_mode:
            return 0.0
        delta = self.positions[transmitter] - self.gcs_position
        interference = sum(
            self._received_power_at_gcs_w(j) for j in range(self.n_agents) if j != transmitter
        )
        return communication_rate_bps(
            float(np.linalg.norm(delta[:2])),
            float(delta[2]),
            self.cfg,
            interference_power_w=float(interference),
            force_los=False,
        )

    def _refresh_links(self) -> None:
        if self.peer_mode:
            snapshot = self.network_backend.link_snapshot(
                self.positions, self.gcs_position, getattr(self, "obstacles", None)
            )
            self.last_pair_rates_bps[:] = snapshot.pair_rates_bps
            self.last_gcs_rates_bps[:] = snapshot.gcs_rates_bps
            self.last_adjacency[:] = snapshot.adjacency
            for i in range(self.n_agents):
                best_peer = float(np.max(self.last_pair_rates_bps[:, i])) if self.n_agents > 1 else 0.0
                self.last_rates_bps[i] = max(best_peer, float(self.last_gcs_rates_bps[i]))
            if hasattr(self, "neighbor_cache_seen_step"):
                for i in range(self.n_agents):
                    for j in range(self.n_agents):
                        if i == j:
                            continue
                        # adjacency[receiver, transmitter] is directional. Agent i can
                        # refresh its cache of j only when j -> i is currently usable.
                        if self.last_adjacency[i, j]:
                            self.neighbor_cache_positions[i, j] = self.positions[j]
                            self.neighbor_cache_battery[i, j] = self.battery_pct[j]
                            self.neighbor_cache_seen_step[i, j] = self.step_count
            return

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
        if self.peer_mode:
            for i in range(self.n_agents):
                self.last_gcs_rates_bps[i] = self._gcs_rate_bps(i)
                best_peer = float(np.max(self.last_pair_rates_bps[:, i])) if self.n_agents > 1 else 0.0
                self.last_rates_bps[i] = max(best_peer, float(self.last_gcs_rates_bps[i]))
            return
        for local_idx, rotor_idx in enumerate(self.multirotor_indices):
            leader = int(self.rotor_leaders[local_idx])
            self.last_rates_bps[local_idx] = (
                self.last_pair_rates_bps[rotor_idx, leader] if leader >= 0 else 0.0
            )

    def _gcs_hops(self) -> np.ndarray:
        """Return shortest current hop count to GCS (1=direct, -1=unreachable)."""
        hops = np.full(self.n_agents, -1, dtype=np.int64)
        if not self.peer_mode:
            return hops
        rmin = float(self.paper["min_comm_rate_bps"])
        direct = np.flatnonzero(self.last_gcs_rates_bps > rmin)
        frontier = [int(i) for i in direct]
        for i in frontier:
            hops[i] = 1
        # adjacency[receiver, transmitter] is directional. Starting from nodes
        # that can reach the GCS directly, reverse-BFS over usable transmitter
        # columns: j can reach current iff adjacency[current, j] == 1 (j -> current).
        head = 0
        while head < len(frontier):
            current = frontier[head]
            head += 1
            for transmitter in np.flatnonzero(self.last_adjacency[current] > 0):
                j = int(transmitter)
                candidate = int(hops[current] + 1)
                if hops[j] < 0 or candidate < hops[j]:
                    hops[j] = candidate
                    frontier.append(j)
        return hops

    def _network_state(self, idx: int) -> float:
        """Return whether this UAV currently has an end-to-end path to the GCS."""
        if self.peer_mode:
            return float(self._gcs_hops()[idx] > 0)
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
        if self.peer_mode:
            if self.target_known_by_agent[idx, target_idx]:
                return 0.0
        elif self.target_found[target_idx]:
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
                if self.peer_mode:
                    seen = int(self.neighbor_cache_seen_step[i, j])
                    if seen >= 0:
                        cached_pos = self.neighbor_cache_positions[i, j]
                        d_ij = float(np.linalg.norm(cached_pos - self.positions[i])) / area
                        live_contact = bool(self.last_adjacency[i, j] or self.last_adjacency[j, i])
                        age = max(0, int(self.step_count) - seen)
                        freshness = 1.0 if live_contact else max(
                            0.0, 1.0 - age / float(self.peer_neighbor_cache_ttl_steps)
                        )
                        other.extend([
                            d_ij,
                            cached_pos[0] / area,
                            cached_pos[1] / area,
                            cached_pos[2] / area,
                            self.neighbor_cache_battery[i, j] / 100.0,
                            freshness,
                        ])
                    else:
                        other.extend([0.0] * 6)
                else:
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

            peer_extra: list[float] = []
            if self.peer_mode:
                gcs_delta = (self.gcs_position[:2] - self.positions[i, :2]) / area
                rate_scale = max(float(self.assumed["comm_rate_max_bps"]), 1.0)
                peer_extra = [
                    float(self.queue_bytes[i]) / max(float(self.peer_buffer_bytes), 1.0),
                    float(gcs_delta[0]),
                    float(gcs_delta[1]),
                    float(self.last_gcs_rates_bps[i]) / rate_scale,
                ]

            vec = np.asarray(own + other + target + peer_extra, dtype=np.float32)
            obs[name] = vec
        return obs

    def observations_array(self) -> np.ndarray:
        d = self._observations()
        return np.stack([d[a] for a in self.agents], axis=0)

    def _decode_peer_recipient(self, sender: int, code: float) -> int:
        """Map one continuous policy output to a stable peer/GCS destination set."""
        if not self.peer_mode:
            raise RuntimeError("recipient decoding is only defined for homogeneous_peer scenarios")
        candidates = [j for j in range(self.n_agents) if j != sender] + [GCS_RECIPIENT]
        x = float(np.clip((float(code) + 1.0) * 0.5, 0.0, 1.0 - 1e-12))
        return int(candidates[min(int(x * len(candidates)), len(candidates) - 1)])

    def _enqueue_report(self, source: int, target_idx: int) -> bool:
        """Create one report when buffer space exists; otherwise keep it pending."""
        if not self.peer_mode or self.report_generated[target_idx] or self.report_expired[target_idx]:
            return False
        room = max(0, self.peer_buffer_bytes - int(self.queue_bytes[source]))
        if room < self.peer_report_bytes:
            return False
        self.report_buffers[target_idx, source] += int(self.peer_report_bytes)
        self.queue_bytes[source] += int(self.peer_report_bytes)
        self.report_generated[target_idx] = True
        self.report_created_step[target_idx] = int(self.step_count)
        return True

    def _retry_pending_reports(self) -> None:
        if not self.peer_mode:
            return
        for target_idx in range(self.n_targets):
            source = int(self.pending_report_source[target_idx])
            if source >= 0 and self.target_found[target_idx] and not self.report_generated[target_idx]:
                self._enqueue_report(source, target_idx)

    def _expire_reports(self) -> None:
        if not self.peer_mode:
            return
        for target_idx in range(self.n_targets):
            created = int(self.report_created_step[target_idx])
            if created < 0 or self.report_delivered[target_idx] or self.report_expired[target_idx]:
                continue
            if int(self.step_count) - created < self.peer_report_ttl_steps:
                continue
            for holder in range(self.n_agents):
                amount = int(self.report_buffers[target_idx, holder])
                if amount > 0:
                    self.report_buffers[target_idx, holder] = 0
                    self.queue_bytes[holder] = max(0, int(self.queue_bytes[holder]) - amount)
            self.report_expired[target_idx] = True
            self.expired_reports += 1

    def _perform_peer_target_detection(self) -> None:
        """Update environment truth and per-agent knowledge without global leakage."""
        self.last_new_targets_by_agent.fill(0)
        for target_idx in range(self.n_targets):
            d = np.linalg.norm(self.positions[self.multirotor_indices, :2] - self.targets[target_idx], axis=1)
            close_local = np.flatnonzero(d <= float(self.assumed["target_found_m"]))
            for local_idx in close_local:
                agent_idx = int(self.multirotor_indices[int(local_idx)])
                self.target_known_by_agent[agent_idx, target_idx] = True
            if self.target_found[target_idx] or len(close_local) == 0:
                continue
            closest_local = int(np.argmin(d))
            detector = int(self.multirotor_indices[closest_local])
            self.target_found[target_idx] = True
            self.target_known_by_agent[detector, target_idx] = True
            self.pending_report_source[target_idx] = detector
            self.last_new_targets_by_agent[detector] += 1

    def _peer_transmit(self, act: np.ndarray) -> None:
        """Execute one-hop report transfers through the configured network backend.

        The MARL action remains authoritative for tx gating, byte budget and the
        immediate recipient.  A frozen application-buffer snapshot prevents bytes
        received in this macro-step from being forwarded again until the next step.
        """
        self.last_selected_tx_rate_bps.fill(0.0)
        self.last_selected_tx_distance_m.fill(np.inf)
        self.last_tx_active.fill(False)
        self.last_tx_success.fill(False)
        self.last_delivery_reward_by_agent.fill(0.0)
        self.last_bytes_transmitted = 0

        slot_start = self.report_buffers.copy()
        intents: list[TransmissionIntent] = []
        nominal_rate_bps = float(self.scenario.get("uavnetsim_bit_rate_bps", self.assumed["comm_rate_max_bps"]))
        for sender in range(self.n_agents):
            queued_at_start = int(slot_start[:, sender].sum())
            if act[sender, 2] <= 0.0 or queued_at_start <= 0:
                continue
            self.last_tx_active[sender] = True
            recipient = self._decode_peer_recipient(sender, float(act[sender, 4]))
            if recipient == GCS_RECIPIENT:
                rate = float(self.last_gcs_rates_bps[sender])
                distance_m = float(np.linalg.norm(self.positions[sender] - self.gcs_position))
                receiver_room = queued_at_start
            else:
                rate = float(self.last_pair_rates_bps[recipient, sender])
                distance_m = float(np.linalg.norm(self.positions[sender] - self.positions[recipient]))
                receiver_room = max(0, self.peer_buffer_bytes - int(self.queue_bytes[recipient]))
            self.last_selected_tx_rate_bps[sender] = rate
            self.last_selected_tx_distance_m[sender] = distance_m
            fraction = float(np.clip((act[sender, 3] + 1.0) * 0.5, 0.0, 1.0))
            # Request bytes independently of current link quality. The network backend
            # is authoritative for deciding whether the selected transmission succeeds.
            budget = min(
                int(nominal_rate_bps * self.dt / 8.0 * fraction),
                queued_at_start,
                receiver_room,
            )
            if budget > 0:
                intents.append(TransmissionIntent(sender=sender, recipient=recipient, requested_bytes=budget))

        self.last_network_result = self.network_backend.transmit(
            intents,
            self.positions,
            self.gcs_position,
            self.obstacles,
            dt_s=self.dt,
            step_index=self.step_count,
        )
        self.total_network_attempted_bytes += int(self.last_network_result.attempted_bytes)
        self.total_network_delivered_bytes += int(self.last_network_result.delivered_bytes)
        self.total_network_tx_energy_j += float(self.last_network_result.tx_energy_j)

        for outcome in self.last_network_result.outcomes:
            sender = int(outcome.sender)
            recipient = int(outcome.recipient)
            self.last_selected_tx_rate_bps[sender] = float(outcome.rate_bps)
            self.last_selected_tx_distance_m[sender] = float(outcome.distance_m)
            self.last_tx_success[sender] = bool(outcome.success)
            remaining = max(0, int(outcome.delivered_bytes))
            for target_idx in range(self.n_targets):
                if remaining <= 0:
                    break
                eligible = min(
                    int(slot_start[target_idx, sender]),
                    int(self.report_buffers[target_idx, sender]),
                )
                if eligible <= 0:
                    continue
                nbytes = min(eligible, remaining)
                if recipient != GCS_RECIPIENT:
                    room = max(0, self.peer_buffer_bytes - int(self.queue_bytes[recipient]))
                    nbytes = min(nbytes, room)
                if nbytes <= 0:
                    continue

                self.report_buffers[target_idx, sender] -= nbytes
                self.queue_bytes[sender] -= nbytes
                if recipient == GCS_RECIPIENT:
                    was_delivered = bool(self.report_delivered[target_idx])
                    self.report_delivered_bytes[target_idx] += nbytes
                    if self.report_delivered_bytes[target_idx] >= self.peer_report_bytes:
                        self.report_delivered[target_idx] = True
                    if self.report_delivered[target_idx] and not was_delivered:
                        self.last_delivery_reward_by_agent[sender] += self.peer_delivery_reward
                else:
                    self.report_buffers[target_idx, recipient] += nbytes
                    self.queue_bytes[recipient] += nbytes
                    self.target_known_by_agent[recipient, target_idx] = True
                remaining -= nbytes
                self.last_bytes_transmitted += nbytes

        self.total_bytes_transmitted += int(self.last_bytes_transmitted)
        self.reports_delivered = int(self.report_delivered.sum())

    def _task_reward(self, idx: int, fixed: bool) -> float:
        """Paper Eqs. (24)-(25): target-search reward for rotor/fixed-wing UAVs."""
        if self.n_targets == 0:
            return 0.0
        dists = np.linalg.norm(self.targets - self.positions[idx, :2], axis=1)
        active = ~self.target_found
        zeta = float(self.assumed["search_reward_coeff"])
        detection_bonus = float(zeta * self.last_new_targets_by_agent[idx]) if self.peer_mode else 0.0
        if not np.any(active):
            return detection_bonus
        d = float(np.min(dists[active]))
        delta_d = float(self.assumed["reward_distance_epsilon_m"])
        if fixed:
            return detection_bonus + (zeta / (d + delta_d) if d <= self.assumed["fixed_detect_m"] else 0.0)
        if d < self.assumed["target_found_m"]:
            return detection_bonus + zeta
        if d <= self.assumed["target_detect_m"]:
            return detection_bonus + zeta / (d + delta_d)
        return detection_bonus

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
        if self.peer_mode:
            return -float(self.peer_energy_cost_per_j * self.last_energy_by_agent_j[idx])
        remaining = float(self.battery_pct[idx])
        if remaining <= float(self.paper["safe_battery_pct"]):
            return 0.0
        return float(self.paper["energy_reward_scale"] * remaining)

    def _communication_reward(self, idx: int) -> float:
        """Paper Eq. (21), adapted to the selected peer transmission in `u6`."""
        if idx not in self.multirotor_indices:
            return 0.0
        rc = float(self.assumed["comm_reward_max"])
        if self.peer_mode:
            if not self.last_tx_active[idx]:
                return 0.0
            rate = float(self.last_selected_tx_rate_bps[idx])
            if rate < float(self.paper["min_comm_rate_bps"]):
                return -rc
            if not self.last_tx_success[idx]:
                return 0.0
            if rate >= float(self.assumed["comm_rate_max_bps"]):
                return rc
            distance_m = float(self.last_selected_tx_distance_m[idx])
            delta_d = float(self.assumed["reward_distance_epsilon_m"])
            return rc / (distance_m + delta_d)

        local_idx = self.multirotor_indices.index(idx)
        rate = float(self.last_rates_bps[local_idx])
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
        if act.shape != (self.n_agents, self.action_dim):
            raise ValueError(f"Expected actions {(self.n_agents, self.action_dim)}, got {act.shape}")
        act = np.clip(act, -1.0, 1.0)
        motion_act = act[:, :2]
        action_saturation = float(np.mean(np.abs(act) > 0.95))
        step_energy = 0.0
        if self.peer_mode:
            self.last_energy_by_agent_j.fill(0.0)
        self.collision_count = 0
        self.obstacle_hits = 0
        self.boundary_hits = 0
        previous_xy = self.positions[:, :2].copy()

        # Paper Eq. (8)-(12), reduced to planar motion with fixed type altitude.
        for i in self.fixed_indices:
            turn = motion_act[i, 0] * self.assumed["fixed_turn_rate_rad_s"]
            self.headings[i] = (self.headings[i] + turn * self.dt + math.pi) % (2 * math.pi) - math.pi
            speed = float(np.linalg.norm(self.velocities[i]))
            target_speed = self.paper["fixed_speed_min_mps"] + (motion_act[i, 1] + 1.0) * 0.5 * (
                self.paper["fixed_speed_max_mps"] - self.paper["fixed_speed_min_mps"]
            )
            delta_v = np.clip(target_speed - speed, -self.paper["max_accel_mps2"] * self.dt, self.paper["max_accel_mps2"] * self.dt)
            speed = np.clip(speed + delta_v, self.paper["fixed_speed_min_mps"], self.paper["fixed_speed_max_mps"])
            self.velocities[i] = speed * np.array([math.cos(self.headings[i]), math.sin(self.headings[i])])
            self.positions[i, :2] += self.velocities[i] * self.dt

        for i in self.multirotor_indices:
            thrust01 = 0.5 * (motion_act[i, 0] + 1.0)
            angle = motion_act[i, 1] * math.pi
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
            if self.peer_mode:
                self.last_energy_by_agent_j[i] = e
            local_idx = self.multirotor_indices.index(i)
            self.cumulative_energy_by_rotor_j[local_idx] += e
            self.battery_pct[i] = max(0.0, self.battery_pct[i] - 100.0 * e / self.assumed["battery_capacity_j"])

        before_clip = self.positions[:, :2].copy()
        self.positions[:, :2] = np.clip(self.positions[:, :2], 0.0, self.area_size_m)
        boundary_mask = np.any(np.abs(before_clip - self.positions[:, :2]) > 1e-9, axis=1)
        self.boundary_hits = int(np.sum(boundary_mask))
        self.episode_boundary_hit_uavs.update(int(i) for i in np.flatnonzero(boundary_mask))

        for i in range(self.n_agents):
            if any(circle_collision(self.positions[i, :2], c) for c in self.obstacles):
                self.obstacle_hits += 1
                self.episode_obstacle_hit_uavs.add(int(i))
                # C3 in the paper excludes the obstacle domain. Reject the candidate
                # displacement rather than adding an unpublished reward penalty.
                self.positions[i, :2] = previous_xy[i]
                if i in self.multirotor_indices:
                    self.velocities[i] = 0.0
        for i in range(self.n_agents):
            for j in range(i + 1, self.n_agents):
                if np.linalg.norm(self.positions[i] - self.positions[j]) <= self.assumed["safety_distance_m"]:
                    self.collision_count += 1
                    self.episode_collided_uavs.update((int(i), int(j)))

        self._refresh_links()
        if self.peer_mode:
            self._expire_reports()
            self._perform_peer_target_detection()
            self._retry_pending_reports()
            self._peer_transmit(act)
            # Charge sender-specific communication energy reported by the backend
            # to mission battery/energy accounting. This keeps networking from
            # being energetically free while leaving paper-faithful scenarios unchanged.
            network_energy_by_agent = np.zeros(self.n_agents, dtype=np.float64)
            for outcome in self.last_network_result.outcomes:
                sender = int(outcome.sender)
                if 0 <= sender < self.n_agents:
                    network_energy_by_agent[sender] += max(0.0, float(outcome.tx_energy_j))
            for i in self.multirotor_indices:
                e_net = float(network_energy_by_agent[i])
                if e_net <= 0.0:
                    continue
                self.last_energy_by_agent_j[i] += e_net
                step_energy += e_net
                local_idx = self.multirotor_indices.index(i)
                self.cumulative_energy_by_rotor_j[local_idx] += e_net
                self.battery_pct[i] = max(
                    0.0,
                    self.battery_pct[i] - 100.0 * e_net / self.assumed["battery_capacity_j"],
                )
            broken_mask = self._gcs_hops() < 0
        else:
            broken_mask = self.last_rates_bps < self.paper["min_comm_rate_bps"]
        self.cumulative_broken_time += broken_mask.astype(np.float64) * self.dt
        for local_idx in np.flatnonzero(broken_mask):
            self.episode_broken_link_uavs.add(int(self.multirotor_indices[int(local_idx)]))
        self.total_energy_used_j += step_energy

        rewards: dict[str, float] = {}
        components: dict[str, dict[str, float]] = {}
        for i, name in enumerate(self.agents):
            safety, _ = self._safety_reward(i)
            if i in self.multirotor_indices:
                communication = float(self._communication_reward(i))
                if self.peer_mode:
                    communication += float(self.last_delivery_reward_by_agent[i])
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
        self.last_reward_components_sum = {
            key: float(sum(values[key] for values in components.values()))
            for key in ("communication", "energy", "safety", "task", "total")
        }

        # Paper-faithful scenarios confirm only after Eq. (24), preserving the
        # published reward timing. u6 performs sensing before communication above.
        if not self.peer_mode:
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
        if self.peer_mode:
            gcs_hops = self._gcs_hops()
            broken = gcs_hops < 0
            direct_gcs = int(np.sum(gcs_hops == 1))
            multihop_gcs = int(np.sum(gcs_hops > 1))
            disconnected_gcs = int(np.sum(gcs_hops < 0))
        else:
            gcs_hops = np.full(self.n_agents, -1, dtype=np.int64)
            broken = rates < self.paper["min_comm_rate_bps"]
            direct_gcs = multihop_gcs = disconnected_gcs = 0
        rotor_battery = self.battery_pct[self.multirotor_indices] if self.n_rotor else np.array([100.0])
        return {
            "step": self.step_count,
            "targets_found": int(self.target_found.sum()),
            "targets_total": int(self.n_targets),
            "search_rate": float(self.target_found.mean()) if self.n_targets else 0.0,
            "energy_used_j": float(step_energy_j),
            "total_energy_used_j": float(self.total_energy_used_j),
            "energy_consumption_pct": float(100.0 - np.mean(rotor_battery)) if self.n_rotor else 0.0,
            "avg_battery_pct": float(np.mean(rotor_battery)) if self.n_rotor else 100.0,
            "depleted_uavs": int(np.sum(rotor_battery <= 0.0)) if self.n_rotor else 0,
            "collided_uavs": int(len(self.episode_collided_uavs)),
            "obstacle_hit_uavs": int(len(self.episode_obstacle_hit_uavs)),
            "boundary_hit_uavs": int(len(self.episode_boundary_hit_uavs)),
            "broken_link_uavs": int(len(self.episode_broken_link_uavs)),
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
            "reward_components_sum": dict(self.last_reward_components_sum),
            "simulation_backend": "root_paper_mpe_style",
            "bytes_transmitted": int(self.last_bytes_transmitted) if self.peer_mode else 0,
            "total_bytes_transmitted": int(self.total_bytes_transmitted) if self.peer_mode else 0,
            "reports_delivered": int(self.reports_delivered) if self.peer_mode else 0,
            "mission_delivery_rate": (
                float(self.reports_delivered / self.n_targets) if self.peer_mode and self.n_targets else 0.0
            ),
            "queue_bytes_total": int(self.queue_bytes.sum()) if self.peer_mode else 0,
            "expired_reports": int(self.expired_reports) if self.peer_mode else 0,
            "direct_gcs_uavs": direct_gcs,
            "multihop_gcs_uavs": multihop_gcs,
            "disconnected_gcs_uavs": disconnected_gcs,
            "mean_gcs_hops": float(np.mean(gcs_hops[gcs_hops > 0])) if self.peer_mode and np.any(gcs_hops > 0) else 0.0,
            "network_backend": self.network_backend.name if self.peer_mode else "paper_analytical",
            "network_attempted_bytes": int(self.last_network_result.attempted_bytes) if self.peer_mode else 0,
            "network_delivered_bytes": int(self.last_network_result.delivered_bytes) if self.peer_mode else 0,
            "network_byte_pdr": float(self.last_network_result.byte_pdr) if self.peer_mode else 0.0,
            "network_throughput_bps": float(self.last_network_result.throughput_bps) if self.peer_mode else 0.0,
            "network_mean_delay_s": float(self.last_network_result.mean_delay_s) if self.peer_mode else 0.0,
            "network_phy_failures": int(self.last_network_result.phy_failures) if self.peer_mode else 0,
            "network_tx_energy_j": float(self.last_network_result.tx_energy_j) if self.peer_mode else 0.0,
            "total_network_attempted_bytes": int(self.total_network_attempted_bytes) if self.peer_mode else 0,
            "total_network_delivered_bytes": int(self.total_network_delivered_bytes) if self.peer_mode else 0,
            "total_network_tx_energy_j": float(self.total_network_tx_energy_j) if self.peer_mode else 0.0,
        }
