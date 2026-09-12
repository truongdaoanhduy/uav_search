from __future__ import annotations

import math
from typing import Any

import numpy as np
from gymnasium.spaces import Box

from uav_search.actions import (
    peer_action_saturation_fraction,
    peer_recipient_bin_centers,
    peer_recipient_bin_index,
)

from .models import (
    communication_rate_bps,
    multirotor_power_w,
    path_gain_linear,
    segment_circle_collision,
)
from .network_backends import (
    NetworkStepResult,
    TransmissionIntent,
    create_network_backend,
)
from .sensing import (
    bayes_update,
    belief_probability_floor,
    binary_entropy,
    circular_cell_coverage_fraction,
    continuous_fov_cells,
    continuous_profile_for_altitude,
    coverage_weighted_bayes_update,
    decode_belief_probabilities,
    encode_belief_probabilities,
    fov_offsets,
    profile_for_altitude,
)

GCS_RECIPIENT = -1


class PaperUAVEnv:
    """Multi-agent UAV environment for root-paper and homogeneous research scenarios.

    Legacy scenarios retain the paper's 2.5D heterogeneous model; ``u6``/``u9``
    use full-3D homogeneous peers. The dict-based reset/step contract is custom
    and Gymnasium-like; this class is not a PettingZoo ``ParallelEnv``.
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
        self.dt = float(self.scenario.get("dt_s", self.assumed["dt_s"]))
        self.n_fixed = int(self.scenario["fixed_wing"])
        self.n_rotor = int(self.scenario["multirotor"])
        self.n_agents = self.n_fixed + self.n_rotor
        self.n_targets = int(self.scenario["targets"])
        self.n_obstacles = int(self.scenario["obstacles"])
        self.peer_mode = str(self.scenario.get("architecture", "paper_heterogeneous")) == "homogeneous_peer"
        self.safety_distance_m = float(
            self.scenario.get("safety_distance_m", self.assumed["safety_distance_m"])
        )
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
            self.peer_expiry_penalty = float(
                self.scenario.get("expiry_penalty", self.peer_delivery_reward)
            )
            self.peer_report_ttl_s = float(self.scenario.get("report_ttl_s", self.scenario.get("report_ttl_steps", 300) * self.dt))
            self.peer_neighbor_cache_ttl_steps = int(self.scenario.get("neighbor_cache_ttl_steps", 50))
            self.peer_sync_bytes = int(self.scenario.get("peer_sync_bytes", 4096))
            self.peer_communication_attempt_penalty = float(
                self.scenario.get("communication_attempt_penalty", 0.0)
            )
            self.peer_sync_quantization_levels = int(
                self.scenario.get("peer_sync_quantization_levels", 256)
            )
            self.peer_proximity_sensor_range_m = float(
                self.scenario.get("proximity_sensor_range_m", 300.0)
            )
            self.peer_tx_power_min_w = float(self.scenario.get("tx_power_min_w", 0.1))
            self.peer_tx_power_max_w = float(self.scenario.get("tx_power_max_w", 0.4))
            self.peer_comm_reward_rate_bps = float(
                self.scenario.get(
                    "communication_reward_rate_bps",
                    self.scenario.get("uavnetsim_bit_rate_bps", self.assumed["comm_rate_max_bps"]),
                )
            )
            self.peer_energy_cost_per_j = float(self.scenario.get("energy_cost_per_j", 0.001))
            self.peer_battery_capacity_j = float(
                self.scenario.get("battery_capacity_j", self.assumed["battery_capacity_j"])
            )
            if self.peer_battery_capacity_j <= 0.0:
                raise ValueError("battery_capacity_j must be positive")
            self.peer_altitude_min_m = float(self.scenario.get("altitude_min_m", self.assumed["multirotor_altitude_m"]))
            self.peer_altitude_max_m = float(self.scenario.get("altitude_max_m", self.assumed["multirotor_altitude_m"]))
            self.peer_altitude_levels_m = np.asarray(
                self.scenario.get("altitude_levels_m", [self.assumed["multirotor_altitude_m"]]),
                dtype=np.float64,
            )
            if self.peer_altitude_levels_m.ndim != 1 or self.peer_altitude_levels_m.size == 0:
                raise ValueError("altitude_levels_m must be a non-empty 1-D sequence")
            if not self.peer_altitude_min_m <= self.peer_altitude_max_m:
                raise ValueError("altitude_min_m must be <= altitude_max_m")
            if np.any(self.peer_altitude_levels_m < self.peer_altitude_min_m) or np.any(
                self.peer_altitude_levels_m > self.peer_altitude_max_m
            ):
                raise ValueError("all altitude_levels_m must lie within altitude_min_m/altitude_max_m")
            self.peer_sensing_model = str(self.scenario.get("sensing_model", "discrete_liu")).strip().lower()
            if self.peer_sensing_model not in {"discrete_liu", "continuous_hu_liu"}:
                raise ValueError(f"unsupported scenario.sensing_model: {self.peer_sensing_model}")
            self.peer_camera_full_fov_deg = float(self.scenario.get("camera_full_fov_deg", 90.0))
            self.peer_sensing_grid_cell_m = float(self.scenario.get("sensing_grid_cell_m", 100.0))
            self.peer_sensing_fov_cells = tuple(int(x) for x in self.scenario.get("sensing_fov_cells", [1, 5, 9]))
            self.peer_sensing_pd = tuple(float(x) for x in self.scenario.get("sensing_detection_probability", [0.9, 0.8, 0.7]))
            self.peer_sensing_pf = tuple(float(x) for x in self.scenario.get("sensing_false_alarm_probability", [0.1, 0.2, 0.3]))
            self.peer_belief_prior = float(self.scenario.get("belief_prior", 0.5))
            self.peer_target_confirmation_threshold = float(self.scenario.get("target_confirmation_threshold", 0.99))
            self.peer_fine_confirmation_max_altitude_m = float(
                self.scenario.get("fine_confirmation_max_altitude_m", self.peer_altitude_levels_m[0])
            )
            self.peer_sensing_target_reward_weight = float(self.scenario.get("sensing_target_reward_weight", 1.0))
            self.peer_sensing_cognitive_reward_weight = float(self.scenario.get("sensing_cognitive_reward_weight", 0.1))
            if self.peer_sensing_grid_cell_m <= 0.0:
                raise ValueError("sensing_grid_cell_m must be positive")
            if len(self.peer_sensing_fov_cells) != len(self.peer_altitude_levels_m):
                raise ValueError("sensing_fov_cells must align with altitude_levels_m")
            if len(self.peer_sensing_pd) != len(self.peer_altitude_levels_m) or len(self.peer_sensing_pf) != len(self.peer_altitude_levels_m):
                raise ValueError("sensing probability profiles must align with altitude_levels_m")
            if self.peer_sensing_model == "continuous_hu_liu":
                if self.peer_altitude_min_m < float(self.peer_altitude_levels_m[0]) - 1e-9 or self.peer_altitude_max_m > float(self.peer_altitude_levels_m[-1]) + 1e-9:
                    raise ValueError("continuous sensing calibration anchors must cover the full operating altitude range")
                # Validate the continuous geometry and probability anchors eagerly.
                continuous_profile_for_altitude(
                    self.peer_altitude_min_m,
                    self.peer_altitude_levels_m,
                    self.peer_sensing_pd,
                    self.peer_sensing_pf,
                    full_fov_deg=self.peer_camera_full_fov_deg,
                )
                continuous_profile_for_altitude(
                    self.peer_altitude_max_m,
                    self.peer_altitude_levels_m,
                    self.peer_sensing_pd,
                    self.peer_sensing_pf,
                    full_fov_deg=self.peer_camera_full_fov_deg,
                )
            if not 0.0 < self.peer_belief_prior < 1.0:
                raise ValueError("belief_prior must lie strictly between zero and one")
            if not 0.5 < self.peer_target_confirmation_threshold < 1.0:
                raise ValueError("target_confirmation_threshold must lie between 0.5 and 1")
            if not self.peer_altitude_min_m <= self.peer_fine_confirmation_max_altitude_m <= self.peer_altitude_max_m:
                raise ValueError("fine_confirmation_max_altitude_m must lie within the altitude bounds")
            self.peer_sensing_grid_n = math.ceil(self.area_size_m / self.peer_sensing_grid_cell_m)
            if self.peer_sensing_model == "continuous_hu_liu":
                max_profile = continuous_profile_for_altitude(
                    self.peer_altitude_max_m,
                    self.peer_altitude_levels_m,
                    self.peer_sensing_pd,
                    self.peer_sensing_pf,
                    full_fov_deg=self.peer_camera_full_fov_deg,
                )
                self.peer_belief_patch_radius_cells = max(
                    1, math.ceil(max_profile.fov_radius_m / self.peer_sensing_grid_cell_m)
                )
            else:
                self.peer_belief_patch_radius_cells = 1
            self.peer_belief_patch_side = 2 * self.peer_belief_patch_radius_cells + 1
            self.peer_belief_patch_dim = self.peer_belief_patch_side**2
            if self.peer_report_bytes <= 0 or self.peer_buffer_bytes <= 0:
                raise ValueError("report_bytes and buffer_bytes must be positive")
            if self.peer_expiry_penalty < 0.0:
                raise ValueError("expiry_penalty must be non-negative")
            if self.peer_communication_attempt_penalty < 0.0:
                raise ValueError("communication_attempt_penalty must be non-negative")
            if self.peer_report_ttl_s <= 0 or self.peer_neighbor_cache_ttl_steps <= 0:
                raise ValueError("report_ttl_s and neighbor_cache_ttl_steps must be positive")
            if self.peer_sync_bytes <= 0:
                raise ValueError("peer_sync_bytes must be positive")
            if not 3 <= self.peer_sync_quantization_levels <= 256:
                raise ValueError("peer_sync_quantization_levels must be in [3, 256]")
            if self.peer_proximity_sensor_range_m <= 0.0:
                raise ValueError("proximity_sensor_range_m must be positive")
            # One byte per quantized belief cell plus explicit state, queue,
            # report-lifetime, progress, and target-knowledge metadata. The
            # remaining bytes in the fixed bundle are deterministic padding.
            sync_metadata_bytes = (
                32
                + 3 * 4
                + 4
                + 8
                + 2
                + 4
                + math.ceil(self.n_targets / 8)
                + self.n_targets * 4
                + self.n_targets
            )
            self.peer_sync_payload_bytes_required = (
                self.peer_sensing_grid_n * self.peer_sensing_grid_n
                + sync_metadata_bytes
            )
            if self.peer_sync_payload_bytes_required > self.peer_sync_bytes:
                raise ValueError(
                    "peer_sync_bytes is too small for the configured quantized "
                    f"belief/state payload ({self.peer_sync_payload_bytes_required} bytes)"
                )
            if not (0.0 < self.peer_tx_power_min_w <= self.peer_tx_power_max_w):
                raise ValueError("tx_power_min_w must be positive and <= tx_power_max_w")
            if self.peer_comm_reward_rate_bps <= float(self.paper["min_comm_rate_bps"]):
                raise ValueError("communication_reward_rate_bps must exceed the minimum communication rate")
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
        # Peer mode adds application queue/GCS-local state plus the previous
        # hop delivery result (a local ACK-like signal, not global mission truth).
        self.peer_neighbor_obs_dim = 13 if self.peer_mode else 6
        self.peer_report_obs_dim = 4 * self.n_targets if self.peer_mode else 0
        # Peer extra = 19 scalar/network/proximity features + fixed ego belief crop.
        # The crop is sized from the maximum continuous sensing footprint so the
        # actor can observe every cell that its own sensor may update.
        self.peer_obs_extra_dim = 19 + self.peer_belief_patch_dim if self.peer_mode else 0
        self.target_obs_dim = 0 if self.peer_mode else self.n_targets
        self.obs_dim = (
            9
            + (self.n_agents - 1) * self.peer_neighbor_obs_dim
            + self.target_obs_dim
            + self.peer_report_obs_dim
            + self.peer_obs_extra_dim
        )
        self.action_dim = 6 if self.peer_mode else 2
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
        if mode == "gcs_launch_pads":
            radius = float(self.scenario.get("launch_radius_m", 300.0))
            min_sep = float(self.scenario.get("initial_min_separation_m", self.safety_distance_m))
            if radius <= 0.0 or min_sep < 0.0:
                raise ValueError("launch_radius_m must be positive and initial_min_separation_m non-negative")
            # Same emergency GCS site, distinct physical pads.  With an edge GCS,
            # a deterministic inward-facing semicircle avoids placing UAVs outside
            # the map or colliding at reset.
            angles = np.linspace(-math.radians(75.0), math.radians(75.0), self.n_agents)
            points = self.gcs_position[:2] + radius * np.column_stack([np.cos(angles), np.sin(angles)])
            if np.any(points < 0.0) or np.any(points > self.area_size_m):
                raise ValueError("gcs_launch_pads fall outside the map; reduce launch_radius_m or move the GCS")
            for i in range(self.n_agents):
                for j in range(i + 1, self.n_agents):
                    if float(np.linalg.norm(points[i] - points[j])) < min_sep:
                        raise ValueError("gcs_launch_pads violate initial_min_separation_m")
            return points.astype(np.float64)
        if mode != "launch_disk":
            raise ValueError(f"Unsupported scenario.initialization: {mode}")

        center = np.asarray(self.scenario.get("launch_center_m", [self.area_size_m / 2.0] * 2), dtype=np.float64)
        if center.shape != (2,):
            raise ValueError("scenario.launch_center_m must contain [x, y]")
        radius = float(self.scenario.get("launch_radius_m", 250.0))
        min_sep = float(self.scenario.get("initial_min_separation_m", self.safety_distance_m))
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

    def _sample_peer_targets(self) -> np.ndarray:
        """Sample peer targets with at most one target per binary belief cell."""
        if self.n_targets <= 0:
            return np.empty((0, 2), dtype=np.float64)
        if self.n_targets > self.peer_sensing_grid_n * self.peer_sensing_grid_n:
            raise ValueError("number of targets exceeds available sensing-grid cells")

        targets: list[np.ndarray] = []
        occupied_cells: set[tuple[int, int]] = set()
        max_attempts = max(10_000, 100 * self.n_targets)
        for _ in range(max_attempts):
            candidate = self.rng.uniform(0.0, self.area_size_m, size=2).astype(np.float64)
            x = int(np.clip(math.floor(candidate[0] / self.peer_sensing_grid_cell_m), 0, self.peer_sensing_grid_n - 1))
            y = int(np.clip(math.floor(candidate[1] / self.peer_sensing_grid_cell_m), 0, self.peer_sensing_grid_n - 1))
            cell = (y, x)
            if cell in occupied_cells:
                continue
            occupied_cells.add(cell)
            targets.append(candidate)
            if len(targets) == self.n_targets:
                return np.stack(targets, axis=0)
        raise RuntimeError("Unable to sample unique target belief cells")

    def _sample_peer_obstacles(self) -> np.ndarray:
        """Sample valid peer-scenario obstacle circles with deterministic rejection sampling."""
        if self.n_obstacles <= 0:
            return np.empty((0, 3), dtype=np.float64)

        radius_min = float(self.assumed["obstacle_radius_min_m"])
        radius_max = float(self.assumed["obstacle_radius_max_m"])
        clearance = max(0.0, float(self.scenario.get("obstacle_clearance_m", 0.0)))
        max_attempts = int(self.scenario.get("obstacle_sampling_max_attempts", 100_000))
        if radius_min <= 0.0 or radius_max < radius_min:
            raise ValueError("invalid obstacle radius bounds")
        if max_attempts <= 0:
            raise ValueError("obstacle_sampling_max_attempts must be positive")

        protected = np.vstack([
            self.positions[:, :2],
            self.gcs_position[:2].reshape(1, 2),
            self.targets,
        ])
        circles: list[np.ndarray] = []
        for _ in range(max_attempts):
            radius = float(self.rng.uniform(radius_min, radius_max))
            margin = radius + clearance
            if 2.0 * margin > self.area_size_m:
                raise ValueError("obstacle radius/clearance cannot fit inside the map")
            center = self.rng.uniform(margin, self.area_size_m - margin, size=2)
            if protected.size and np.any(np.linalg.norm(protected - center, axis=1) < margin):
                continue
            if any(
                float(np.linalg.norm(circle[:2] - center)) < float(circle[2] + radius + clearance)
                for circle in circles
            ):
                continue
            circles.append(np.array([center[0], center[1], radius], dtype=np.float64))
            if len(circles) == self.n_obstacles:
                return np.stack(circles, axis=0)

        raise RuntimeError(
            "Unable to place all peer-scenario obstacles without overlaps; "
            "reduce obstacle density/radii or clearance"
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
        if self.peer_mode:
            self.positions[self.multirotor_indices, 2] = self.rng.choice(
                self.peer_altitude_levels_m, size=self.n_rotor, replace=True
            )
        else:
            self.positions[self.multirotor_indices, 2] = self.assumed["multirotor_altitude_m"]
        # A common 3-D velocity layout keeps paper scenarios backward compatible
        # (their z velocity stays zero) while allowing u6 to climb/descend.
        self.velocities = np.zeros((self.n_agents, 3), dtype=np.float64)
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
        self.uav_active = np.ones(self.n_agents, dtype=bool)
        self.targets = (
            self._sample_peer_targets()
            if self.peer_mode
            else self.rng.uniform(0.0, self.area_size_m, size=(self.n_targets, 2)).astype(np.float64)
        )
        self.target_found = np.zeros(self.n_targets, dtype=bool)
        if self.peer_mode:
            self.obstacles = self._sample_peer_obstacles()
        else:
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
        self.safety_distance_violation_count = 0
        self.last_safety_filter_interventions = 0
        self.total_safety_filter_interventions = 0
        self.last_safety_correction_m_by_agent = np.zeros(
            self.n_agents, dtype=np.float64
        )
        self.last_world_constraint_correction_m_by_agent = np.zeros(
            self.n_agents, dtype=np.float64
        )
        self.obstacle_hits = 0
        self.boundary_hits = 0
        # Episode-level unique UAV counts are diagnostic aggregates only; they do
        # not change the paper reward, dynamics, termination, or observations.
        self.episode_safety_violation_uavs: set[int] = set()
        self.episode_obstacle_hit_uavs: set[int] = set()
        self.episode_boundary_hit_uavs: set[int] = set()
        self.episode_broken_link_uavs: set[int] = set()
        self.trajectory = [self.positions.copy()]
        if self.peer_mode:
            self.last_gcs_rates_bps = np.zeros(self.n_agents, dtype=np.float64)
            self.min_power_gcs_rates_bps = np.zeros(self.n_agents, dtype=np.float64)
            self.max_power_gcs_rates_bps = np.zeros(self.n_agents, dtype=np.float64)
            self.min_power_pair_rates_bps = np.zeros(
                (self.n_agents, self.n_agents), dtype=np.float64
            )
            self.max_power_pair_rates_bps = np.zeros(
                (self.n_agents, self.n_agents), dtype=np.float64
            )
            self.last_selected_tx_rate_bps = np.zeros(self.n_agents, dtype=np.float64)
            self.last_selected_tx_distance_m = np.full(self.n_agents, np.inf, dtype=np.float64)
            self.last_selected_tx_power_w = np.zeros(self.n_agents, dtype=np.float64)
            self.last_selected_recipient = np.full(self.n_agents, GCS_RECIPIENT, dtype=np.int64)
            self.last_tx_active = np.zeros(self.n_agents, dtype=bool)
            self.last_tx_success = np.zeros(self.n_agents, dtype=bool)
            self.target_known_by_agent = np.zeros((self.n_agents, self.n_targets), dtype=bool)
            # Persistent provenance for report generation: unlike
            # target_known_by_agent, this is never copied by peer sync. It marks
            # targets this UAV independently confirmed from its own sensing.
            self.target_directly_confirmed_by_agent = np.zeros((self.n_agents, self.n_targets), dtype=bool)
            self.belief_maps = np.full(
                (self.n_agents, self.peer_sensing_grid_n, self.peer_sensing_grid_n),
                self.peer_belief_prior,
                dtype=np.float64,
            )
            self.belief_source_map = np.full(
                (self.peer_sensing_grid_n, self.peer_sensing_grid_n), -1, dtype=np.int64
            )
            self.confirmed_cells = np.zeros(
                (self.peer_sensing_grid_n, self.peer_sensing_grid_n), dtype=bool
            )
            self.false_confirmed_cells = np.zeros_like(self.confirmed_cells)
            self.last_false_confirmations = 0
            self.total_false_confirmations = 0
            self.last_sensor_positive = np.zeros((self.n_agents, self.n_targets), dtype=bool)
            self.fine_positive_cells_by_agent = np.zeros_like(self.belief_maps, dtype=bool)
            self.fine_target_evidence_by_agent = np.zeros(
                (self.n_agents, self.n_targets), dtype=bool
            )
            self.last_information_gain_by_agent = np.zeros(self.n_agents, dtype=np.float64)
            self.last_scanned_cells_by_agent = np.zeros(self.n_agents, dtype=np.int64)
            self.last_positive_sensor_observations_by_agent = np.zeros(self.n_agents, dtype=np.int64)
            self.total_scanned_cells = 0
            self.total_positive_sensor_observations = 0
            self.total_information_gain = 0.0
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
            self.neighbor_cache_queue_fraction = np.zeros(
                (self.n_agents, self.n_agents), dtype=np.float64
            )
            self.neighbor_cache_contact_degree = np.zeros(
                (self.n_agents, self.n_agents), dtype=np.float64
            )
            self.neighbor_cache_gcs_distance = np.zeros(
                (self.n_agents, self.n_agents), dtype=np.float64
            )
            self.neighbor_cache_min_gcs_available = np.zeros(
                (self.n_agents, self.n_agents), dtype=np.float64
            )
            self.neighbor_cache_max_gcs_available = np.zeros(
                (self.n_agents, self.n_agents), dtype=np.float64
            )
            self.neighbor_cache_seen_step = np.full((self.n_agents, self.n_agents), -1, dtype=np.int64)
            self.report_created_step_known_by_agent = np.full(
                (self.n_agents, self.n_targets), -1, dtype=np.int64
            )
            self.report_delivery_progress_known_by_agent = np.zeros(
                (self.n_agents, self.n_targets), dtype=np.float64
            )
            self.last_new_targets_by_agent = np.zeros(self.n_agents, dtype=np.int64)
            self.last_energy_by_agent_j = np.zeros(self.n_agents, dtype=np.float64)
            self.last_bytes_transmitted = 0
            self.total_bytes_transmitted = 0
            self.last_report_bytes_transmitted_by_agent = np.zeros(self.n_agents, dtype=np.int64)
            self.last_report_bytes_attempted_by_agent = np.zeros(self.n_agents, dtype=np.int64)
            self.last_reports_delivered_step = 0
            self.last_reports_expired_step = 0
            self.last_peer_syncs = 0
            self.total_peer_syncs = 0
            self.reports_delivered = 0
            self.expired_reports = 0
            self.last_network_result = NetworkStepResult()
            self.total_network_attempted_bytes = 0
            self.total_network_admitted_bytes = 0
            self.total_network_delivered_bytes = 0
            self.total_network_tx_energy_j = 0.0
            # Keep UavNetSim's per-node CSMA RNGs aligned with the episode seed.
            # The runner intentionally changes the reset seed across episodes.
            if hasattr(self.network_backend, "seed"):
                self.network_backend.seed = int(self._seed)
            reset_network = getattr(self.network_backend, "reset_episode", None)
            if callable(reset_network):
                reset_network(self.positions, self.gcs_position, self.obstacles)
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

    def _refresh_links(self) -> None:
        if self.peer_mode:
            set_active = getattr(self.network_backend, "set_node_active", None)
            if callable(set_active) and hasattr(self, "uav_active"):
                set_active(self.uav_active)

            min_snapshot = self.network_backend.link_snapshot(
                self.positions,
                self.gcs_position,
                getattr(self, "obstacles", None),
                tx_power_w=self.peer_tx_power_min_w,
            )
            max_snapshot = self.network_backend.link_snapshot(
                self.positions,
                self.gcs_position,
                getattr(self, "obstacles", None),
                tx_power_w=self.peer_tx_power_max_w,
            )
            self.min_power_pair_rates_bps[:] = min_snapshot.pair_rates_bps
            self.min_power_gcs_rates_bps[:] = min_snapshot.gcs_rates_bps
            self.max_power_pair_rates_bps[:] = max_snapshot.pair_rates_bps
            self.max_power_gcs_rates_bps[:] = max_snapshot.gcs_rates_bps
            self.last_pair_rates_bps[:] = self.max_power_pair_rates_bps
            self.last_gcs_rates_bps[:] = self.max_power_gcs_rates_bps
            self.last_adjacency[:] = max_snapshot.adjacency
            if hasattr(self, "uav_active"):
                inactive = ~self.uav_active
                for pair_rates in (
                    self.min_power_pair_rates_bps,
                    self.max_power_pair_rates_bps,
                    self.last_pair_rates_bps,
                ):
                    pair_rates[inactive, :] = 0.0
                    pair_rates[:, inactive] = 0.0
                self.min_power_gcs_rates_bps[inactive] = 0.0
                self.max_power_gcs_rates_bps[inactive] = 0.0
                self.last_adjacency[inactive, :] = 0
                self.last_adjacency[:, inactive] = 0
                self.last_gcs_rates_bps[inactive] = 0.0
            for i in range(self.n_agents):
                best_peer = float(np.max(self.last_pair_rates_bps[:, i])) if self.n_agents > 1 else 0.0
                self.last_rates_bps[i] = max(best_peer, float(self.last_gcs_rates_bps[i]))
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

    def _peer_link_features(self, sender: int, recipient: int) -> list[float]:
        """Return recipient-specific min/max-power feasibility observable at sender.

        These are instantaneous sender-side radio/channel-sounding estimates,
        not cached peer state or privileged peer position/battery. Keeping them
        outside the neighbor cache prevents stale application metadata from making
        the routing head blind to which candidate link currently needs higher RF
        power. This sensing abstraction is a documented research adaptation.
        """
        if not self.peer_mode:
            return [0.0, 0.0]
        sender = int(sender)
        recipient = int(recipient)
        if (
            sender < 0
            or sender >= self.n_agents
            or recipient < 0
            or recipient >= self.n_agents
            or sender == recipient
        ):
            return [0.0, 0.0]
        rmin = float(self.paper["min_comm_rate_bps"])
        return [
            float(self.min_power_pair_rates_bps[recipient, sender] > rmin),
            float(self.max_power_pair_rates_bps[recipient, sender] > rmin),
        ]

    def _actor_network_state(self, idx: int) -> float:
        """Return local peer degree in research mode, legacy link state otherwise.

        Direct-GCS feasibility/rate is already represented in the peer-specific
        observation tail. Using normalized contact degree here avoids duplicating
        the nearly-binary 0/2 Mbps GCS signal while retaining local topology context.
        """
        if self.peer_mode:
            if hasattr(self, "uav_active") and not self.uav_active[idx]:
                return 0.0
            denom = max(self.n_agents - 1, 1)
            return float(np.count_nonzero(self.last_adjacency[:, int(idx)]) / denom)
        return self._network_state(idx)

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
            if not self.last_sensor_positive[idx, target_idx]:
                return 0.0
            d = float(np.linalg.norm(self.targets[target_idx] - self.positions[idx, :2]))
            return d / self.area_size_m
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

    def _xy_grid_cell(self, xy: np.ndarray) -> tuple[int, int]:
        x = int(np.clip(math.floor(float(xy[0]) / self.peer_sensing_grid_cell_m), 0, self.peer_sensing_grid_n - 1))
        y = int(np.clip(math.floor(float(xy[1]) / self.peer_sensing_grid_cell_m), 0, self.peer_sensing_grid_n - 1))
        return y, x

    def _target_grid_cell(self, target_idx: int) -> tuple[int, int]:
        if not self.peer_mode:
            raise RuntimeError("target grid cells are defined only for homogeneous_peer scenarios")
        return self._xy_grid_cell(self.targets[int(target_idx)])

    def _sensing_profile(self, agent_idx: int):
        if not self.peer_mode:
            raise RuntimeError("altitude-aware sensing is defined only for homogeneous_peer scenarios")
        altitude_m = float(self.positions[int(agent_idx), 2])
        if self.peer_sensing_model == "continuous_hu_liu":
            return continuous_profile_for_altitude(
                altitude_m,
                self.peer_altitude_levels_m,
                self.peer_sensing_pd,
                self.peer_sensing_pf,
                full_fov_deg=self.peer_camera_full_fov_deg,
            )
        return profile_for_altitude(
            altitude_m,
            self.peer_altitude_levels_m,
            self.peer_sensing_fov_cells,
            self.peer_sensing_pd,
            self.peer_sensing_pf,
        )

    def _sensing_cells(self, agent_idx: int) -> list[tuple[int, int]]:
        position_xy = self.positions[int(agent_idx), :2]
        profile = self._sensing_profile(agent_idx)
        if self.peer_sensing_model == "continuous_hu_liu":
            return list(
                continuous_fov_cells(
                    position_xy,
                    profile.fov_radius_m,
                    self.peer_sensing_grid_cell_m,
                    self.peer_sensing_grid_n,
                )
            )

        center_y, center_x = self._xy_grid_cell(position_xy)
        cells: list[tuple[int, int]] = []
        for dy, dx in fov_offsets(profile.fov_cells):
            y, x = center_y + dy, center_x + dx
            if 0 <= y < self.peer_sensing_grid_n and 0 <= x < self.peer_sensing_grid_n:
                cells.append((y, x))
        return cells

    def _belief_patch(self, agent_idx: int) -> list[float]:
        """Return a fixed ego crop large enough for the maximum sensing footprint.

        Values outside the *current* physical footprint remain zero-masked.  This
        keeps the actor input fixed-width while preventing continuous high-altitude
        sensing from updating cells that are impossible for the actor to observe.
        """
        center_y, center_x = self._xy_grid_cell(self.positions[int(agent_idx), :2])
        allowed = set(self._sensing_cells(agent_idx))
        radius = int(self.peer_belief_patch_radius_cells)
        patch: list[float] = []
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                y, x = center_y + dy, center_x + dx
                if (y, x) in allowed:
                    patch.append(float(self.belief_maps[agent_idx, y, x]))
                else:
                    patch.append(0.0)
        return patch

    @staticmethod
    def _belief_entropy_array(beliefs: np.ndarray) -> np.ndarray:
        """Vectorized binary entropy used for minimum-uncertainty fusion."""
        beliefs = np.clip(np.asarray(beliefs, dtype=np.float64), 0.0, 1.0)
        entropy = np.zeros_like(beliefs, dtype=np.float64)
        interior = (beliefs > 0.0) & (beliefs < 1.0)
        p = beliefs[interior]
        entropy[interior] = -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))
        return entropy

    def _fuse_received_peer_belief(self, receiver: int, sender_belief: np.ndarray) -> None:
        """Fuse one successfully received peer map into one receiver only.

        This is deliberately communication-gated: merely having a geometric/PHY
        candidate link never modifies actor knowledge. Ties keep the receiver's
        existing value so a repeated sync is idempotent.
        """
        if not self.peer_mode:
            return
        receiver = int(receiver)
        local = np.clip(self.belief_maps[receiver], 0.0, 1.0)
        remote = np.clip(np.asarray(sender_belief, dtype=np.float64), 0.0, 1.0)
        if remote.shape != local.shape:
            raise ValueError("received peer belief shape does not match local belief map")
        local_entropy = self._belief_entropy_array(local)
        remote_entropy = self._belief_entropy_array(remote)
        take_remote = remote_entropy + 1e-12 < local_entropy
        self.belief_maps[receiver, take_remote] = remote[take_remote]

    def _confirm_peer_targets(self) -> None:
        """Finalize only cells backed by active, local fine-grained evidence.

        High- and medium-altitude measurements may raise a local posterior and
        therefore identify a suspected region.  Final target identification and
        report provenance require at least one positive observation at the
        configured fine-altitude ceiling.
        """
        if not self.peer_mode:
            return
        self.last_false_confirmations = 0
        active = np.flatnonzero(self.uav_active)
        if active.size == 0:
            self.belief_source_map.fill(-1)
            return

        threshold = self.peer_target_confirmation_threshold
        active_ready = (
            (self.belief_maps[active] >= threshold)
            & self.fine_positive_cells_by_agent[active]
        )
        candidate_cells = np.argwhere(
            np.any(active_ready, axis=0) & ~self.confirmed_cells
        )
        targets_by_cell: dict[tuple[int, int], list[int]] = {}
        for target_idx in range(self.n_targets):
            targets_by_cell.setdefault(self._target_grid_cell(target_idx), []).append(target_idx)

        # A globally confirmed cell must not prevent another UAV from obtaining
        # independent local provenance later.  This matters when the original
        # report source loses power or has no buffer space: a second UAV that
        # performs its own fine observation is allowed to regenerate the payload,
        # while knowledge received only through peer synchronization is not.
        for target_idx in np.flatnonzero(self.target_found):
            y, x = self._target_grid_cell(int(target_idx))
            independently_ready = active[
                (self.belief_maps[active, y, x] >= threshold)
                & self.fine_positive_cells_by_agent[active, y, x]
                & self.fine_target_evidence_by_agent[active, int(target_idx)]
            ]
            for detector in independently_ready:
                detector = int(detector)
                target_idx = int(target_idx)
                self.target_known_by_agent[detector, target_idx] = True
                self.target_directly_confirmed_by_agent[detector, target_idx] = True
                if self.report_created_step[target_idx] >= 0:
                    self.report_created_step_known_by_agent[detector, target_idx] = int(
                        self.report_created_step[target_idx]
                    )
                elif not self.report_generated[target_idx] and not self.report_delivered[target_idx]:
                    # A new TTL generation after expiry requires fresh direct fine
                    # evidence in this sensing phase. Persistent historical
                    # fine_target_evidence is provenance, not a new observation.
                    fresh_fine_positive = bool(
                        self.last_sensor_positive[detector, target_idx]
                        and self.positions[detector, 2]
                        <= self.peer_fine_confirmation_max_altitude_m + 1e-9
                    )
                    if fresh_fine_positive:
                        # Fresh information exists now, even if finite buffer space
                        # prevents immediate serialization. Start TTL at observation
                        # time so waiting for storage does not extend its lifetime.
                        if self.report_created_step[target_idx] < 0:
                            self.report_created_step[target_idx] = int(self.step_count)
                        self.report_created_step_known_by_agent[detector, target_idx] = int(
                            self.report_created_step[target_idx]
                        )
                        pending = int(self.pending_report_source[target_idx])
                        if pending < 0 or pending >= self.n_agents or not self.uav_active[pending]:
                            self.pending_report_source[target_idx] = detector

        for y_raw, x_raw in candidate_cells:
            y, x = int(y_raw), int(x_raw)
            ready_mask = (
                (self.belief_maps[active, y, x] >= threshold)
                & self.fine_positive_cells_by_agent[active, y, x]
            )
            eligible = active[ready_mask]
            if eligible.size == 0:
                continue

            true_targets = [
                idx for idx in targets_by_cell.get((y, x), []) if not self.target_found[idx]
            ]
            if true_targets:
                target_ids = np.asarray(true_targets, dtype=np.int64)
                has_target_evidence = np.any(
                    self.fine_target_evidence_by_agent[np.ix_(eligible, target_ids)],
                    axis=1,
                )
                eligible = eligible[has_target_evidence]
                if eligible.size == 0:
                    continue

            posteriors = self.belief_maps[eligible, y, x]
            best_p = float(np.max(posteriors))
            best = eligible[np.isclose(posteriors, best_p, rtol=0.0, atol=1e-12)]
            if best.size > 1:
                cell_xy = np.array(
                    [
                        (x + 0.5) * self.peer_sensing_grid_cell_m,
                        (y + 0.5) * self.peer_sensing_grid_cell_m,
                    ],
                    dtype=np.float64,
                )
                distances = np.linalg.norm(self.positions[best, :2] - cell_xy, axis=1)
                min_distance = float(np.min(distances))
                best = best[np.isclose(distances, min_distance, rtol=0.0, atol=1e-9)]
            detector = int(np.min(best))

            self.confirmed_cells[y, x] = True
            self.belief_source_map[y, x] = detector
            if not true_targets:
                self.false_confirmed_cells[y, x] = True
                self.last_false_confirmations += 1
                self.total_false_confirmations += 1
                self.belief_maps[detector, y, x] = belief_probability_floor(
                    self.peer_sync_quantization_levels
                )
                self.fine_positive_cells_by_agent[detector, y, x] = False
                continue

            for target_idx in true_targets:
                if not self.fine_target_evidence_by_agent[detector, target_idx]:
                    continue
                self.target_found[target_idx] = True
                self.target_known_by_agent[detector, target_idx] = True
                self.target_directly_confirmed_by_agent[detector, target_idx] = True
                if self.report_created_step[target_idx] < 0:
                    # TTL starts at information creation, including any time the
                    # report must wait for finite buffer space.
                    self.report_created_step[target_idx] = int(self.step_count)
                self.report_created_step_known_by_agent[
                    detector, target_idx
                ] = self.report_created_step[target_idx]
                self.pending_report_source[target_idx] = detector
                self.last_new_targets_by_agent[detector] += 1

    def _peer_altitude_feature(self, altitude_m: float) -> float:
        span = max(self.peer_altitude_max_m - self.peer_altitude_min_m, 1e-9)
        return float(np.clip((float(altitude_m) - self.peer_altitude_min_m) / span, 0.0, 1.0))

    def _peer_proximity_features(self, agent_idx: int) -> tuple[list[float], list[float]]:
        """Return onboard-range obstacle and active-peer proximity features."""
        sensor_range = self.peer_proximity_sensor_range_m
        position = self.positions[int(agent_idx)]

        obstacle_feature = [0.0, 0.0, 0.0]
        if self.n_obstacles > 0:
            deltas = self.obstacles[:, :2] - position[:2]
            center_distance = np.linalg.norm(deltas, axis=1)
            clearance = np.maximum(0.0, center_distance - self.obstacles[:, 2])
            nearest = int(np.argmin(clearance))
            if float(clearance[nearest]) <= sensor_range:
                obstacle_feature = [
                    float(np.clip(deltas[nearest, 0] / sensor_range, -1.0, 1.0)),
                    float(np.clip(deltas[nearest, 1] / sensor_range, -1.0, 1.0)),
                    float(np.clip(clearance[nearest] / sensor_range, 0.0, 1.0)),
                ]

        peer_feature = [0.0, 0.0, 0.0]
        candidates = [
            j for j in range(self.n_agents)
            if j != int(agent_idx) and self.uav_active[j]
        ]
        if candidates:
            peer_delta = self.positions[candidates] - position
            peer_distance = np.linalg.norm(peer_delta, axis=1)
            nearest_local = int(np.argmin(peer_distance))
            if float(peer_distance[nearest_local]) <= sensor_range:
                delta = peer_delta[nearest_local]
                peer_feature = [
                    float(np.clip(delta[0] / sensor_range, -1.0, 1.0)),
                    float(np.clip(delta[1] / sensor_range, -1.0, 1.0)),
                    float(np.clip(delta[2] / sensor_range, -1.0, 1.0)),
                ]
        return obstacle_feature, peer_feature

    def _peer_report_features(self, agent_idx: int) -> list[float]:
        features: list[float] = []
        for target_idx in range(self.n_targets):
            held_fraction = float(
                np.clip(
                    self.report_buffers[target_idx, agent_idx] / max(float(self.peer_report_bytes), 1.0),
                    0.0,
                    1.0,
                )
            )
            created_step = -1
            if held_fraction > 0.0 or int(self.pending_report_source[target_idx]) == int(agent_idx):
                created_step = int(self.report_created_step[target_idx])
            elif int(self.report_created_step_known_by_agent[agent_idx, target_idx]) >= 0:
                created_step = int(self.report_created_step_known_by_agent[agent_idx, target_idx])
            if created_step >= 0:
                age_s = max(0.0, (float(self.step_count) - created_step) * self.dt)
                remaining_ttl = float(np.clip(1.0 - age_s / self.peer_report_ttl_s, 0.0, 1.0))
            else:
                remaining_ttl = 0.0
            progress = float(
                np.clip(self.report_delivery_progress_known_by_agent[agent_idx, target_idx], 0.0, 1.0)
            )
            features.extend([
                float(self.target_known_by_agent[agent_idx, target_idx]),
                held_fraction,
                remaining_ttl,
                progress,
            ])
        return features

    def _observations(self) -> dict[str, np.ndarray]:
        """Build local actor observations without peer-mode ground-truth leakage."""
        obs: dict[str, np.ndarray] = {}
        area = self.area_size_m
        legacy_vmax = max(self.paper["fixed_speed_max_mps"], self.paper["multirotor_speed_max_mps"])
        peer_vmax = max(float(self.paper["multirotor_speed_max_mps"]), 1e-9)
        rmin = float(self.paper["min_comm_rate_bps"])
        gcs_distance_scale = max(
            math.sqrt(2.0 * area * area + self.peer_altitude_max_m * self.peer_altitude_max_m)
            if self.peer_mode else area,
            1.0,
        )

        for i, name in enumerate(self.agents):
            is_fixed = i in self.fixed_indices
            is_rotor = i in self.multirotor_indices
            if self.peer_mode:
                own_z = self._peer_altitude_feature(self.positions[i, 2])
                velocity_scale = peer_vmax
            else:
                own_z = self.positions[i, 2] / area
                velocity_scale = legacy_vmax

            previous_recipient = (
                self._encode_peer_recipient(
                    i,
                    int(self.last_selected_recipient[i]),
                )
                if self.peer_mode and self.last_tx_active[i]
                else 0.0
            )
            own = [
                self.positions[i, 0] / area,
                self.positions[i, 1] / area,
                own_z,
                self.velocities[i, 0] / velocity_scale,
                self.velocities[i, 1] / velocity_scale,
                self.velocities[i, 2] / velocity_scale if self.peer_mode else 0.0,
                self.battery_pct[i] / 100.0 if is_rotor else 0.0,
                self._actor_network_state(i) if is_rotor else 0.0,
                (
                    self.headings[i] / math.pi
                    if is_fixed
                    else previous_recipient
                ),
            ]

            other: list[float] = []
            for j in range(self.n_agents):
                if j == i:
                    continue
                if self.peer_mode:
                    seen = int(self.neighbor_cache_seen_step[i, j])
                    age = (
                        max(0, int(self.step_count) - seen - 1)
                        if seen >= 0 else self.peer_neighbor_cache_ttl_steps
                    )
                    if seen >= 0 and age < self.peer_neighbor_cache_ttl_steps:
                        cached_pos = self.neighbor_cache_positions[i, j]
                        d_ij = float(np.linalg.norm(cached_pos - self.positions[i])) / gcs_distance_scale
                        freshness = 1.0 - age / float(self.peer_neighbor_cache_ttl_steps)
                        other.extend([
                            d_ij,
                            cached_pos[0] / area,
                            cached_pos[1] / area,
                            self._peer_altitude_feature(cached_pos[2]),
                            self.neighbor_cache_battery[i, j] / 100.0,
                            freshness,
                            self.neighbor_cache_queue_fraction[i, j],
                            self.neighbor_cache_contact_degree[i, j],
                            self.neighbor_cache_gcs_distance[i, j],
                            self.neighbor_cache_min_gcs_available[i, j],
                            self.neighbor_cache_max_gcs_available[i, j],
                        ])
                    else:
                        # Cache-gated peer state remains hidden when stale.
                        other.extend([0.0] * 11)
                    # Link feasibility is a sender-local radio estimate and remains
                    # observable independently of whether peer metadata was synced.
                    other.extend(self._peer_link_features(sender=i, recipient=j))
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

            target = (
                [] if self.peer_mode
                else [self._target_observation_distance(i, k) for k in range(self.n_targets)]
            )

            report_features: list[float] = []
            peer_extra: list[float] = []
            if self.peer_mode:
                report_features = self._peer_report_features(i)
                gcs_delta = (self.gcs_position[:2] - self.positions[i, :2]) / area
                rate_scale = max(float(self.peer_comm_reward_rate_bps), 1.0)
                power_span = max(self.peer_tx_power_max_w - self.peer_tx_power_min_w, 1e-12)
                selected_power = (
                    float(np.clip(
                        (self.last_selected_tx_power_w[i] - self.peer_tx_power_min_w) / power_span,
                        0.0,
                        1.0,
                    ))
                    if self.last_tx_active[i] else 0.0
                )
                min_best_peer = (
                    float(np.max(self.min_power_pair_rates_bps[:, i])) if self.n_agents > 1 else 0.0
                )
                max_best_peer = (
                    float(np.max(self.max_power_pair_rates_bps[:, i])) if self.n_agents > 1 else 0.0
                )
                obstacle_feature, peer_feature = self._peer_proximity_features(i)
                peer_extra = [
                    float(self.queue_bytes[i]) / max(float(self.peer_buffer_bytes), 1.0),
                    float(gcs_delta[0]),
                    float(gcs_delta[1]),
                    float(self.last_gcs_rates_bps[i]) / rate_scale,
                    float(self.last_tx_success[i]),
                    float(np.clip((self.max_steps - self.step_count) / max(float(self.max_steps), 1.0), 0.0, 1.0)),
                    float(self.last_tx_active[i]),
                    selected_power,
                    float(np.clip(
                        self.last_report_bytes_attempted_by_agent[i] / max(float(self.peer_report_bytes), 1.0),
                        0.0,
                        1.0,
                    )),
                    float(self.min_power_gcs_rates_bps[i] > rmin),
                    float(self.max_power_gcs_rates_bps[i] > rmin),
                    float(min_best_peer > rmin),
                    float(max_best_peer > rmin),
                    *obstacle_feature,
                    *peer_feature,
                    *self._belief_patch(i),
                ]

            vec = np.asarray(own + other + target + report_features + peer_extra, dtype=np.float32)
            if vec.shape != (self.obs_dim,):
                raise RuntimeError(
                    f"observation contract mismatch for {name}: got {vec.size}, expected {self.obs_dim}"
                )
            obs[name] = vec
        return obs

    def observations_array(self) -> np.ndarray:
        d = self._observations()
        return np.stack([d[a] for a in self.agents], axis=0)

    def _peer_recipient_candidates(self, sender: int) -> list[int]:
        """Return the ordered peer/GCS action bins available to one sender."""
        if not self.peer_mode:
            raise RuntimeError(
                "recipient encoding is only defined for homogeneous_peer scenarios"
            )
        sender = int(sender)
        if sender < 0 or sender >= self.n_agents:
            raise ValueError("sender index is outside the agent set")
        return [j for j in range(self.n_agents) if j != sender] + [GCS_RECIPIENT]

    def _decode_peer_recipient(self, sender: int, code: float) -> int:
        """Map one continuous policy output to a stable peer/GCS destination set."""
        candidates = self._peer_recipient_candidates(sender)
        return int(candidates[peer_recipient_bin_index(code, len(candidates))])

    def _encode_peer_recipient(self, sender: int, recipient: int) -> float:
        """Return the action-bin center for a decoded peer/GCS recipient."""
        candidates = self._peer_recipient_candidates(sender)
        try:
            bin_index = candidates.index(int(recipient))
        except ValueError as exc:
            raise ValueError("recipient is not available to this sender") from exc
        return float(peer_recipient_bin_centers(len(candidates))[bin_index])

    def _decode_tx_power_w(self, code: float) -> float:
        """Map the continuous power action to the configured RF power interval."""
        if not self.peer_mode:
            raise RuntimeError("tx power decoding is only defined for homogeneous_peer scenarios")
        x = float(np.clip((float(code) + 1.0) * 0.5, 0.0, 1.0))
        return float(self.peer_tx_power_min_w + x * (self.peer_tx_power_max_w - self.peer_tx_power_min_w))

    def _enqueue_report(self, source: int, target_idx: int) -> bool:
        """Create one report when an active source has buffer space; otherwise keep it pending."""
        if not self.peer_mode or self.report_generated[target_idx] or self.report_delivered[target_idx]:
            return False
        source = int(source)
        if source < 0 or source >= self.n_agents or not self.uav_active[source]:
            return False
        room = max(0, self.peer_buffer_bytes - int(self.queue_bytes[source]))
        if room < self.peer_report_bytes:
            return False
        self.report_buffers[target_idx, source] += int(self.peer_report_bytes)
        self.queue_bytes[source] += int(self.peer_report_bytes)
        self.report_generated[target_idx] = True
        if self.report_created_step[target_idx] < 0:
            self.report_created_step[target_idx] = int(self.step_count)
        self.report_created_step_known_by_agent[source, target_idx] = int(
            self.report_created_step[target_idx]
        )
        return True

    def _retry_pending_reports(self) -> None:
        if not self.peer_mode:
            return
        for target_idx in range(self.n_targets):
            source = int(self.pending_report_source[target_idx])
            if source < 0 or not self.target_found[target_idx] or self.report_generated[target_idx]:
                continue

            # Prefer the original detector while it is active. Before any TTL
            # expiry, another independently-confirming active UAV may synthesize
            # the same generation. After expiry, fallback is restricted below to
            # the pending fresh source or another current fresh fine observer.
            # Peer-shared metadata alone is deliberately insufficient.
            candidates: list[int] = []
            if 0 <= source < self.n_agents and self.uav_active[source]:
                candidates.append(source)
            independent_mask = (
                self.uav_active & self.target_directly_confirmed_by_agent[:, target_idx]
            )
            if self.report_expired[target_idx]:
                # After an expiry, only the observer that originated the pending
                # fresh generation (plus another UAV with a fresh fine-positive
                # observation in this same sensing phase) may synthesize it. This
                # prevents stale historical confirmation at a different UAV from
                # silently resetting the message TTL.
                fine_now = self.positions[:, 2] <= (
                    self.peer_fine_confirmation_max_altitude_m + 1e-9
                )
                independent_mask &= self.last_sensor_positive[:, target_idx] & fine_now
            independent = np.flatnonzero(independent_mask)
            candidates.extend(int(i) for i in independent if int(i) not in candidates)
            for candidate in candidates:
                if self._enqueue_report(candidate, target_idx):
                    self.pending_report_source[target_idx] = candidate
                    break

    def _expire_reports(self) -> None:
        if not self.peer_mode:
            return
        self.last_reports_expired_step = 0
        for target_idx in range(self.n_targets):
            created = int(self.report_created_step[target_idx])
            if created < 0 or self.report_delivered[target_idx]:
                continue
            if (int(self.step_count) - created) * self.dt < self.peer_report_ttl_s:
                continue
            for holder in range(self.n_agents):
                amount = int(self.report_buffers[target_idx, holder])
                if amount > 0:
                    self.report_buffers[target_idx, holder] = 0
                    self.queue_bytes[holder] = max(0, int(self.queue_bytes[holder]) - amount)
            # TTL is scoped to the current report generation/copies, not to the
            # target forever. Purge stale copies and record the loss. Historical
            # confirmation alone is not fresh information and therefore cannot
            # mint a new TTL generation; regeneration requires a later fine
            # positive observation in _confirm_peer_targets().
            self.report_generated[target_idx] = False
            self.report_created_step[target_idx] = -1
            self.report_delivered_bytes[target_idx] = 0
            self.report_created_step_known_by_agent[:, target_idx] = -1
            self.report_delivery_progress_known_by_agent[:, target_idx] = 0.0
            self.report_expired[target_idx] = True  # historical diagnostic flag
            self.pending_report_source[target_idx] = -1
            self.last_reports_expired_step += 1
            self.expired_reports += 1

    def _report_priority(self, sender: int, slot_start: np.ndarray) -> list[int]:
        """Return sender-held reports in deterministic earliest-deadline-first order."""
        sender = int(sender)
        available = [
            target_idx
            for target_idx in range(self.n_targets)
            if int(slot_start[target_idx, sender]) > 0
            and not self.report_delivered[target_idx]
        ]

        def deadline(target_idx: int) -> tuple[float, int]:
            created = int(self.report_created_step[target_idx])
            absolute_deadline = (
                created * self.dt + self.peer_report_ttl_s
                if created >= 0
                else float("inf")
            )
            return absolute_deadline, target_idx

        return sorted(available, key=deadline)

    def _perform_peer_target_detection(self) -> None:
        """Run physical-footprint sensing and Bayesian coarse-to-fine confirmation."""
        self.last_new_targets_by_agent.fill(0)
        self.last_sensor_positive.fill(False)
        self.last_information_gain_by_agent.fill(0.0)
        self.last_scanned_cells_by_agent.fill(0)
        self.last_positive_sensor_observations_by_agent.fill(0)

        target_cells = [self._target_grid_cell(k) for k in range(self.n_targets)]
        targets_by_cell: dict[tuple[int, int], list[int]] = {}
        for target_idx, cell in enumerate(target_cells):
            targets_by_cell.setdefault(cell, []).append(target_idx)

        for agent_idx in range(self.n_agents):
            if not self.uav_active[agent_idx]:
                continue
            profile = self._sensing_profile(agent_idx)
            fine_measurement = bool(
                self.positions[agent_idx, 2]
                <= self.peer_fine_confirmation_max_altitude_m + 1e-9
            )
            for y, x in self._sensing_cells(agent_idx):
                cell_targets = targets_by_cell.get((y, x), [])
                if self.peer_sensing_model == "continuous_hu_liu":
                    visible_targets = [
                        target_idx
                        for target_idx in cell_targets
                        if float(
                            np.linalg.norm(
                                self.targets[target_idx] - self.positions[agent_idx, :2]
                            )
                        )
                        <= profile.fov_radius_m + 1e-9
                    ]
                else:
                    visible_targets = list(cell_targets)

                occupied = bool(visible_targets)
                probability = profile.pd if occupied else profile.pf
                measurement = bool(self.rng.random() < probability)
                prior = float(self.belief_maps[agent_idx, y, x])
                if self.peer_sensing_model == "continuous_hu_liu":
                    coverage_fraction = circular_cell_coverage_fraction(
                        self.positions[agent_idx, :2],
                        profile.fov_radius_m,
                        self.peer_sensing_grid_cell_m,
                        y,
                        x,
                    )
                    posterior = coverage_weighted_bayes_update(
                        prior,
                        measurement,
                        profile.pd,
                        profile.pf,
                        coverage_fraction=coverage_fraction,
                    )
                else:
                    posterior = bayes_update(prior, measurement, profile.pd, profile.pf)
                information_gain = binary_entropy(prior) - binary_entropy(posterior)
                self.belief_maps[agent_idx, y, x] = posterior
                self.last_information_gain_by_agent[agent_idx] += information_gain
                self.last_scanned_cells_by_agent[agent_idx] += 1
                if not measurement:
                    continue

                self.last_positive_sensor_observations_by_agent[agent_idx] += 1
                if fine_measurement:
                    self.fine_positive_cells_by_agent[agent_idx, y, x] = True
                for target_idx in visible_targets:
                    self.last_sensor_positive[agent_idx, target_idx] = True
                    if fine_measurement:
                        self.fine_target_evidence_by_agent[agent_idx, target_idx] = True

        self.total_scanned_cells += int(self.last_scanned_cells_by_agent.sum())
        self.total_positive_sensor_observations += int(self.last_positive_sensor_observations_by_agent.sum())
        self.total_information_gain += float(self.last_information_gain_by_agent.sum())
        self._confirm_peer_targets()

    def _peer_transmit(self, act: np.ndarray) -> None:
        """Execute communication-gated peer sync and one-hop report transfers.

        A peer-directed TX always carries a small synchronization bundle containing
        sender state plus its belief/known-target snapshot. Report bytes, when any,
        follow that bundle. Both knowledge refresh and report movement occur only
        after the configured network backend reports successful delivery. A frozen
        start-of-slot snapshot prevents same-step multi-hop information teleportation.
        """
        self.last_selected_tx_rate_bps.fill(0.0)
        self.last_selected_tx_distance_m.fill(np.inf)
        self.last_selected_tx_power_w.fill(0.0)
        self.last_selected_recipient.fill(GCS_RECIPIENT)
        self.last_tx_active.fill(False)
        self.last_tx_success.fill(False)
        self.last_report_bytes_transmitted_by_agent.fill(0)
        self.last_report_bytes_attempted_by_agent.fill(0)
        self.last_reports_delivered_step = 0
        self.last_bytes_transmitted = 0
        self.last_peer_syncs = 0

        slot_start = self.report_buffers.copy()
        queue_bytes_slot_start = self.queue_bytes.copy()
        belief_slot_start = encode_belief_probabilities(
            self.belief_maps,
            self.peer_sync_quantization_levels,
        )
        known_slot_start = self.target_known_by_agent.copy()
        created_known_slot_start = self.report_created_step_known_by_agent.copy()
        progress_known_slot_start = self.report_delivery_progress_known_by_agent.copy()
        position_slot_start = self.positions.copy()
        battery_slot_start = self.battery_pct.copy()
        queue_fraction_slot_start = np.clip(
            self.queue_bytes.astype(np.float64) / max(float(self.peer_buffer_bytes), 1.0),
            0.0,
            1.0,
        )
        degree_denom = max(self.n_agents - 1, 1)
        contact_degree_slot_start = np.asarray([
            np.count_nonzero(self.last_adjacency[:, sender]) / float(degree_denom)
            for sender in range(self.n_agents)
        ], dtype=np.float64)
        gcs_distance_scale = max(
            math.sqrt(2.0 * self.area_size_m * self.area_size_m + self.peer_altitude_max_m * self.peer_altitude_max_m),
            1.0,
        )
        gcs_distance_slot_start = np.clip(
            np.linalg.norm(position_slot_start - self.gcs_position, axis=1) / gcs_distance_scale,
            0.0,
            1.0,
        )
        rmin = float(self.paper["min_comm_rate_bps"])
        min_gcs_available_slot_start = (self.min_power_gcs_rates_bps > rmin).astype(np.float64)
        max_gcs_available_slot_start = (self.max_power_gcs_rates_bps > rmin).astype(np.float64)
        intents: list[TransmissionIntent] = []
        nominal_rate_bps = float(self.scenario.get("uavnetsim_bit_rate_bps", self.assumed["comm_rate_max_bps"]))
        nominal_slot_bytes = max(0, int(nominal_rate_bps * self.dt / 8.0))
        reserved_report_bytes_by_receiver = np.zeros(self.n_agents, dtype=np.int64)

        for sender in range(self.n_agents):
            if not self.uav_active[sender] or act[sender, 3] <= 0.0:
                continue
            queued_at_start = int(slot_start[:, sender].sum())
            recipient = self._decode_peer_recipient(sender, float(act[sender, 5]))
            tx_power_w = self._decode_tx_power_w(float(act[sender, 4]))
            report_requested = 0

            if recipient == GCS_RECIPIENT:
                # GCS already owns its own state; do not send empty control traffic.
                if queued_at_start <= 0:
                    continue
                requested = min(nominal_slot_bytes, queued_at_start)
                report_requested = requested
                if requested <= 0:
                    continue
                rate = float(self.last_gcs_rates_bps[sender])
                distance_m = float(np.linalg.norm(self.positions[sender] - self.gcs_position))
            else:
                desired_report_bytes = min(
                    max(0, nominal_slot_bytes - self.peer_sync_bytes),
                    queued_at_start,
                )
                if 0 <= recipient < self.n_agents:
                    rate = float(self.last_pair_rates_bps[recipient, sender])
                    distance_m = float(
                        np.linalg.norm(self.positions[sender] - self.positions[recipient])
                    )
                else:
                    rate = 0.0
                    distance_m = float("inf")

                # A gate-on choice is actor-visible even when the selected peer has
                # depleted. It creates no network work or artificial radio energy,
                # but the existing attempt/failure costs can now train against it.
                self.last_tx_active[sender] = True
                self.last_selected_recipient[sender] = int(recipient)
                self.last_report_bytes_attempted_by_agent[sender] = int(
                    desired_report_bytes
                )
                self.last_selected_tx_power_w[sender] = tx_power_w
                self.last_selected_tx_rate_bps[sender] = rate
                self.last_selected_tx_distance_m[sender] = distance_m
                if (
                    recipient < 0
                    or recipient >= self.n_agents
                    or not self.uav_active[recipient]
                ):
                    continue

                receiver_room = max(
                    0,
                    self.peer_buffer_bytes
                    - int(queue_bytes_slot_start[recipient])
                    - int(reserved_report_bytes_by_receiver[recipient]),
                )
                report_requested = min(desired_report_bytes, receiver_room)
                reserved_report_bytes_by_receiver[recipient] += report_requested
                requested = self.peer_sync_bytes + report_requested

            self.last_tx_active[sender] = True
            self.last_selected_recipient[sender] = int(recipient)
            self.last_report_bytes_attempted_by_agent[sender] = int(report_requested)
            self.last_selected_tx_power_w[sender] = tx_power_w
            self.last_selected_tx_rate_bps[sender] = rate
            self.last_selected_tx_distance_m[sender] = distance_m
            intents.append(TransmissionIntent(
                sender=sender,
                recipient=recipient,
                requested_bytes=requested,
                tx_power_w=tx_power_w,
            ))

        self.last_network_result = self.network_backend.transmit(
            intents,
            self.positions,
            self.gcs_position,
            self.obstacles,
            dt_s=self.dt,
            step_index=self.step_count,
        )
        self.total_network_attempted_bytes += int(self.last_network_result.attempted_bytes)
        self.total_network_admitted_bytes += int(self.last_network_result.admitted_bytes)
        self.total_network_delivered_bytes += int(self.last_network_result.delivered_bytes)
        self.total_network_tx_energy_j += float(self.last_network_result.tx_energy_j)

        for outcome in self.last_network_result.outcomes:
            sender = int(outcome.sender)
            recipient = int(outcome.recipient)
            self.last_selected_tx_rate_bps[sender] = float(outcome.rate_bps)
            self.last_selected_tx_distance_m[sender] = float(outcome.distance_m)
            delivered = max(0, int(outcome.delivered_bytes))
            delivered_prefix = getattr(outcome, "delivered_prefix_bytes", None)
            # Network metrics count every ACKed packet, including packets received
            # after a gap. Application state can commit only the contiguous prefix.
            application_delivered = (
                delivered
                if delivered_prefix is None
                else min(delivered, max(0, int(delivered_prefix)))
            )
            peer_sync_complete = bool(
                application_delivered >= self.peer_sync_bytes
                and 0 <= recipient < self.n_agents
                and self.uav_active[recipient]
            )
            # ``TransmissionOutcome.success`` means that at least one byte reached
            # the link endpoint. Actor-visible TX success instead requires usable
            # in-order application data (and a complete bundle for peer sync).
            self.last_tx_success[sender] = bool(
                application_delivered > 0
                if recipient == GCS_RECIPIENT
                else peer_sync_complete
            )

            if recipient == GCS_RECIPIENT:
                remaining = application_delivered
            else:
                # The synchronization bundle is conceptually first in the frame.
                # Partial bundle delivery is not enough to refresh actor knowledge.
                if peer_sync_complete:
                    self.neighbor_cache_positions[recipient, sender] = position_slot_start[sender]
                    self.neighbor_cache_battery[recipient, sender] = battery_slot_start[sender]
                    self.neighbor_cache_queue_fraction[recipient, sender] = queue_fraction_slot_start[sender]
                    self.neighbor_cache_contact_degree[recipient, sender] = contact_degree_slot_start[sender]
                    self.neighbor_cache_gcs_distance[recipient, sender] = gcs_distance_slot_start[sender]
                    self.neighbor_cache_min_gcs_available[recipient, sender] = min_gcs_available_slot_start[sender]
                    self.neighbor_cache_max_gcs_available[recipient, sender] = max_gcs_available_slot_start[sender]
                    self.neighbor_cache_seen_step[recipient, sender] = self.step_count
                    decoded_belief = decode_belief_probabilities(
                        belief_slot_start[sender],
                        self.peer_sync_quantization_levels,
                    )
                    self._fuse_received_peer_belief(recipient, decoded_belief)
                    self.target_known_by_agent[recipient] |= known_slot_start[sender]
                    for target_idx in range(self.n_targets):
                        remote_created = int(created_known_slot_start[sender, target_idx])
                        local_created = int(self.report_created_step_known_by_agent[recipient, target_idx])
                        if remote_created >= 0 and (local_created < 0 or remote_created < local_created):
                            self.report_created_step_known_by_agent[recipient, target_idx] = remote_created
                        self.report_delivery_progress_known_by_agent[recipient, target_idx] = max(
                            float(self.report_delivery_progress_known_by_agent[recipient, target_idx]),
                            float(progress_known_slot_start[sender, target_idx]),
                        )
                    self.last_peer_syncs += 1
                remaining = max(0, application_delivered - self.peer_sync_bytes)

            for target_idx in self._report_priority(sender, slot_start):
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
                    progress = float(np.clip(
                        self.report_delivered_bytes[target_idx] / max(float(self.peer_report_bytes), 1.0),
                        0.0,
                        1.0,
                    ))
                    self.report_delivery_progress_known_by_agent[sender, target_idx] = max(
                        float(self.report_delivery_progress_known_by_agent[sender, target_idx]),
                        progress,
                    )
                    if self.report_delivered_bytes[target_idx] >= self.peer_report_bytes:
                        self.report_delivered[target_idx] = True
                    if self.report_delivered[target_idx] and not was_delivered:
                        self.last_reports_delivered_step += 1
                else:
                    self.report_buffers[target_idx, recipient] += nbytes
                    self.queue_bytes[recipient] += nbytes
                    self.target_known_by_agent[recipient, target_idx] = True
                    self.report_created_step_known_by_agent[recipient, target_idx] = int(
                        self.report_created_step[target_idx]
                    )
                    self.report_delivery_progress_known_by_agent[recipient, target_idx] = max(
                        float(self.report_delivery_progress_known_by_agent[recipient, target_idx]),
                        float(progress_known_slot_start[sender, target_idx]),
                    )
                remaining -= nbytes
                self.last_bytes_transmitted += nbytes
                self.last_report_bytes_transmitted_by_agent[sender] += nbytes

        self.total_peer_syncs += int(self.last_peer_syncs)
        self.total_bytes_transmitted += int(self.last_bytes_transmitted)
        self.reports_delivered = int(self.report_delivered.sum())

    def _apply_peer_safety_projection(self, previous_positions: np.ndarray) -> None:
        """Project nominal peer motion onto the one-step pairwise safe set.

        This is a lightweight geometric projection shield for continuous control.
        The MARL policy proposes nominal motion, then the low-level safety layer
        minimally separates unsafe candidate pairs along their separation axis
        before the state is exposed to sensing/networking. It does not solve a
        CBF-QP or claim formal forward-invariance guarantees. Legacy paper
        scenarios are untouched, and rollback is only a constraint fallback.
        """
        self.last_safety_correction_m_by_agent.fill(0.0)
        if not self.peer_mode or self.n_agents < 2:
            return
        safe = float(self.safety_distance_m)
        eps = 1e-6
        projection_tol = 1e-9
        active = self.uav_active.astype(bool)
        nominal_positions = self.positions.copy()
        # Sequential projections converge quickly for the small U6/U9 swarms.
        for _ in range(max(2, 2 * self.n_agents)):
            changed = False
            for i in range(self.n_agents):
                if not active[i]:
                    continue
                for j in range(i + 1, self.n_agents):
                    if not active[j]:
                        continue
                    prev_delta = previous_positions[i] - previous_positions[j]
                    prev_distance = float(np.linalg.norm(prev_delta))
                    delta = self.positions[i] - self.positions[j]
                    distance = float(np.linalg.norm(delta))
                    if prev_distance > safe + eps:
                        target_distance = safe + eps
                        needs_filter = distance < target_distance - projection_tol
                    else:
                        # For externally injected unsafe states, never make the
                        # separation smaller; allow a policy that is escaping.
                        target_distance = min(safe + eps, prev_distance + eps)
                        needs_filter = distance < target_distance - projection_tol
                    if not needs_filter:
                        continue
                    direction = delta / distance if distance > eps else (
                        prev_delta / prev_distance if prev_distance > eps else np.array([1.0, 0.0, 0.0])
                    )
                    midpoint = 0.5 * (self.positions[i] + self.positions[j])
                    half = 0.5 * target_distance * direction
                    self.positions[i] = midpoint + half
                    self.positions[j] = midpoint - half
                    changed = True
            if not changed:
                break

        # Safety projection changes the realized low-level motion; expose that
        # through velocity so energy and the next Markov state match the safe move.
        self.velocities[active] = (self.positions[active] - previous_positions[active]) / max(self.dt, 1e-12)
        vmax = float(self.paper["multirotor_speed_max_mps"])
        for i in np.flatnonzero(active):
            speed = float(np.linalg.norm(self.velocities[i]))
            if speed > vmax + 1e-9:
                # Respect actuator/speed limits; if clipping re-introduces an unsafe
                # state the final hard check below will use the previous safe pose.
                self.velocities[i] *= vmax / speed
                self.positions[i] = previous_positions[i] + self.velocities[i] * self.dt

        # Re-apply world constraints because a pairwise projection can move a UAV
        # by a small amount after the normal boundary/obstacle checks.
        projected_before_clip = self.positions.copy()
        self.positions[:, :2] = np.clip(self.positions[:, :2], 0.0, self.area_size_m)
        self.positions[:, 2] = np.clip(
            self.positions[:, 2], self.peer_altitude_min_m, self.peer_altitude_max_m
        )
        shield_boundary = np.any(np.abs(projected_before_clip - self.positions) > 1e-9, axis=1)
        if np.any(shield_boundary):
            self.boundary_hits += int(np.sum(shield_boundary))
            self.episode_boundary_hit_uavs.update(int(i) for i in np.flatnonzero(shield_boundary))
        for i in np.flatnonzero(active):
            if any(segment_circle_collision(previous_positions[i, :2], self.positions[i, :2], c) for c in self.obstacles):
                self.positions[i] = previous_positions[i]
                self.velocities[i] = 0.0
                self.obstacle_hits += 1
                self.episode_obstacle_hit_uavs.add(int(i))

        # Rare interactions among pair projection, speed limits, obstacle rollback,
        # and boundaries get a conservative fail-safe rather than an unsafe state.
        for _ in range(max(1, self.n_agents)):
            rejected: set[int] = set()
            for i in range(self.n_agents):
                if not active[i]:
                    continue
                for j in range(i + 1, self.n_agents):
                    if not active[j]:
                        continue
                    previous_distance = float(np.linalg.norm(previous_positions[i] - previous_positions[j]))
                    candidate_distance = float(np.linalg.norm(self.positions[i] - self.positions[j]))
                    if candidate_distance > safe:
                        continue
                    became_unsafe = previous_distance > safe
                    failed_to_separate = candidate_distance <= previous_distance + 1e-9
                    if became_unsafe or failed_to_separate:
                        rejected.update((i, j))
            if not rejected:
                break
            rejected_idx = np.fromiter(sorted(rejected), dtype=np.int64)
            self.positions[rejected_idx] = previous_positions[rejected_idx]
            self.velocities[rejected_idx] = 0.0

        self.velocities[active] = (self.positions[active] - previous_positions[active]) / max(self.dt, 1e-12)
        correction_m = np.linalg.norm(self.positions - nominal_positions, axis=1)
        correction_m[~active] = 0.0
        self.last_safety_correction_m_by_agent[:] = correction_m
        corrected_uavs = int(np.count_nonzero(correction_m > projection_tol))
        self.last_safety_filter_interventions = corrected_uavs
        self.total_safety_filter_interventions += corrected_uavs

    def _task_reward(self, idx: int, fixed: bool) -> float:
        """Paper Eqs. (24)-(25): target-search reward for rotor/fixed-wing UAVs."""
        if self.n_targets == 0:
            return 0.0
        zeta = float(self.assumed["search_reward_coeff"])
        if self.peer_mode:
            # Discovery is added as a shared team term in step(). Keep only the
            # local signed information-potential shaping here.
            return float(
                zeta
                * self.peer_sensing_cognitive_reward_weight
                * float(self.last_information_gain_by_agent[idx])
            )
        dists = np.linalg.norm(self.targets - self.positions[idx, :2], axis=1)
        active = ~self.target_found
        if not np.any(active):
            return 0.0
        d = float(np.min(dists[active]))
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
        if self.peer_mode:
            pair_correction_m = float(self.last_safety_correction_m_by_agent[idx])
            world_correction_m = float(
                self.last_world_constraint_correction_m_by_agent[idx]
            )
            correction_scale = max(float(self.safety_distance_m), delta_d)
            penalty -= eta * pair_correction_m / correction_scale
            penalty -= eta * world_correction_m / correction_scale
        close = 0
        for j in range(self.n_agents):
            if j == idx:
                continue
            if self.peer_mode and not self.uav_active[j]:
                continue
            d = float(np.linalg.norm(self.positions[j] - self.positions[idx]))
            if d <= self.safety_distance_m:
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
            recipient = int(self.last_selected_recipient[idx])
            # The communication-gating literature explicitly penalizes multi-hop
            # communication attempts to suppress redundant exchanges. Apply that
            # cost to every peer-directed attempt, including control-only syncs.
            peer_attempt_cost = (
                self.peer_communication_attempt_penalty
                if recipient != GCS_RECIPIENT else 0.0
            )
            # Peer synchronization is useful only indirectly through better search;
            # do not award it positive shaping.
            if self.last_report_bytes_attempted_by_agent[idx] <= 0:
                return -peer_attempt_cost
            rate = float(self.last_selected_tx_rate_bps[idx])
            committed_bytes = int(
                self.last_report_bytes_transmitted_by_agent[idx]
            )
            if (
                rate < float(self.paper["min_comm_rate_bps"])
                or committed_bytes <= 0
                or not self.last_tx_success[idx]
            ):
                return -rc - peer_attempt_cost
            # Never award positive shaping merely for moving a report to another
            # UAV. A peer copy can be sent back and forth, so hop-level positive
            # reward creates a reward loop unrelated to mission completion. Final
            # GCS delivery is rewarded separately by the shared delivery term.
            if recipient != GCS_RECIPIENT:
                return -peer_attempt_cost
            # Link rate already determines how many application bytes are ACKed.
            # Rewarding every nonempty fragment with the full maximum lets a slow
            # or fragmented transfer collect more return than a clean delivery.
            progress = min(
                float(committed_bytes) / max(float(self.peer_report_bytes), 1.0),
                1.0,
            )
            return rc * progress

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
        if not np.all(np.isfinite(act)):
            raise ValueError("Actions must contain only finite values")
        act = np.clip(act, -1.0, 1.0)
        motion_act = act[:, :2]
        action_saturation = (
            peer_action_saturation_fraction(act)
            if self.peer_mode
            else float(np.mean(np.abs(act) > 0.95))
        )
        step_energy = 0.0
        if self.peer_mode:
            self.last_energy_by_agent_j.fill(0.0)
        self.safety_distance_violation_count = 0
        self.last_safety_filter_interventions = 0
        self.last_safety_correction_m_by_agent.fill(0.0)
        self.last_world_constraint_correction_m_by_agent.fill(0.0)
        self.obstacle_hits = 0
        self.boundary_hits = 0
        previous_positions = self.positions.copy()
        previous_velocities = self.velocities.copy()
        previous_xy = previous_positions[:, :2]
        active_at_step_start = np.ones(self.n_agents, dtype=bool)
        if self.peer_mode:
            self.uav_active &= self.battery_pct > 0.0
            self.velocities[~self.uav_active] = 0.0
            active_at_step_start = self.uav_active.copy()

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
            self.velocities[i, :2] = speed * np.array([math.cos(self.headings[i]), math.sin(self.headings[i])])
            self.velocities[i, 2] = 0.0
            self.positions[i, :2] += self.velocities[i, :2] * self.dt

        for i in self.multirotor_indices:
            if self.peer_mode and not self.uav_active[i]:
                continue
            if self.peer_mode:
                # U6/U9 use direct Cartesian acceleration commands. Zero actor
                # output is physically neutral and all axes are symmetric around 0.
                accel = np.asarray(act[i, :3], dtype=np.float64) * float(
                    self.paper["max_accel_mps2"]
                )
            else:
                thrust01 = 0.5 * (motion_act[i, 0] + 1.0)
                angle = motion_act[i, 1] * math.pi
                horizontal_accel = thrust01 * self.paper["max_accel_mps2"] * np.array(
                    [math.cos(angle), math.sin(angle)], dtype=np.float64
                )
                accel = np.array([horizontal_accel[0], horizontal_accel[1], 0.0], dtype=np.float64)
            self.velocities[i] += accel * self.dt
            speed = float(np.linalg.norm(self.velocities[i]))
            vmax = self.paper["multirotor_speed_max_mps"]
            if speed > vmax:
                self.velocities[i] *= vmax / speed
                speed = vmax
            if self.peer_mode:
                self.positions[i] += self.velocities[i] * self.dt
            else:
                self.positions[i, :2] += self.velocities[i, :2] * self.dt
            if not self.peer_mode:
                power = multirotor_power_w(
                    speed,
                    float(np.linalg.norm(accel)),
                    self.cfg,
                    include_communication_power=True,
                )
                e = power * self.dt
                step_energy += e
                local_idx = self.multirotor_indices.index(i)
                self.cumulative_energy_by_rotor_j[local_idx] += e
                battery_capacity_j = float(self.assumed["battery_capacity_j"])
                self.battery_pct[i] = max(0.0, self.battery_pct[i] - 100.0 * e / battery_capacity_j)

        nominal_positions = self.positions.copy()
        before_clip_xy = self.positions[:, :2].copy()
        self.positions[:, :2] = np.clip(self.positions[:, :2], 0.0, self.area_size_m)
        boundary_mask = np.any(np.abs(before_clip_xy - self.positions[:, :2]) > 1e-9, axis=1)
        if self.peer_mode:
            before_clip_z = self.positions[:, 2].copy()
            self.positions[:, 2] = np.clip(
                self.positions[:, 2], self.peer_altitude_min_m, self.peer_altitude_max_m
            )
            boundary_mask |= np.abs(before_clip_z - self.positions[:, 2]) > 1e-9
            # Prevent a saturated vertical velocity from repeatedly pushing into
            # the altitude boundary on every later step.
            at_floor = self.positions[:, 2] <= self.peer_altitude_min_m + 1e-9
            at_ceiling = self.positions[:, 2] >= self.peer_altitude_max_m - 1e-9
            self.velocities[at_floor & (self.velocities[:, 2] < 0.0), 2] = 0.0
            self.velocities[at_ceiling & (self.velocities[:, 2] > 0.0), 2] = 0.0
        self.boundary_hits = int(np.sum(boundary_mask))
        self.episode_boundary_hit_uavs.update(int(i) for i in np.flatnonzero(boundary_mask))

        for i in range(self.n_agents):
            if self.peer_mode and not self.uav_active[i]:
                continue
            if any(segment_circle_collision(previous_xy[i], self.positions[i, :2], c) for c in self.obstacles):
                self.obstacle_hits += 1
                self.episode_obstacle_hit_uavs.add(int(i))
                # C3 in the paper excludes the obstacle domain. Reject the candidate
                # displacement rather than adding an unpublished reward penalty.
                if self.peer_mode:
                    self.positions[i] = previous_positions[i]
                else:
                    self.positions[i, :2] = previous_xy[i]
                if i in self.multirotor_indices:
                    self.velocities[i] = 0.0

        if self.peer_mode:
            world_correction_m = np.linalg.norm(
                self.positions - nominal_positions,
                axis=1,
            )
            world_correction_m[~self.uav_active] = 0.0
            self.last_world_constraint_correction_m_by_agent[:] = (
                world_correction_m
            )
            self._apply_peer_safety_projection(previous_positions)
            # Propulsion energy follows the realized shielded motion rather than
            # the nominal unsafe command that was filtered out.
            for i in self.multirotor_indices:
                if not self.uav_active[i]:
                    continue
                realized_accel = (self.velocities[i] - previous_velocities[i]) / max(self.dt, 1e-12)
                speed = float(np.linalg.norm(self.velocities[i]))
                power = multirotor_power_w(
                    speed,
                    float(np.linalg.norm(realized_accel)),
                    self.cfg,
                    include_communication_power=False,
                )
                e = power * self.dt
                step_energy += e
                self.last_energy_by_agent_j[i] = e
                local_idx = self.multirotor_indices.index(i)
                self.cumulative_energy_by_rotor_j[local_idx] += e
                self.battery_pct[i] = max(
                    0.0,
                    self.battery_pct[i] - 100.0 * e / self.peer_battery_capacity_j,
                )

        for i in range(self.n_agents):
            if self.peer_mode and not self.uav_active[i]:
                continue
            for j in range(i + 1, self.n_agents):
                if self.peer_mode and not self.uav_active[j]:
                    continue
                if np.linalg.norm(self.positions[i] - self.positions[j]) <= self.safety_distance_m:
                    self.safety_distance_violation_count += 1
                    self.episode_safety_violation_uavs.update((int(i), int(j)))

        if self.peer_mode:
            # A UAV that exhausted its battery while moving cannot communicate or
            # sense in this macro-step. It remains at its last position for rendering.
            self.uav_active &= self.battery_pct > 0.0
            self.velocities[~self.uav_active] = 0.0
        self._refresh_links()
        if self.peer_mode:
            # Causal slot order: the joint action a_t may communicate only
            # information that existed when a_t was chosen. New sensing evidence
            # produced later in this transition becomes eligible for TX at t+1.
            # Existing reports also get their final forwarding opportunity before
            # TTL expiry, matching store-carry-forward lifecycle semantics.
            self._peer_transmit(act)
            # Charge every UAV's radio TX energy, including native UavNetSim ACKs
            # and retransmissions. GCS radio energy is deliberately excluded from
            # mission battery accounting.
            network_energy_by_agent = np.zeros(self.n_agents, dtype=np.float64)
            for node_id, energy_j in self.last_network_result.node_tx_energy_j.items():
                if 0 <= int(node_id) < self.n_agents:
                    network_energy_by_agent[int(node_id)] += max(0.0, float(energy_j))
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
                    self.battery_pct[i] - 100.0 * e_net / self.peer_battery_capacity_j,
                )
            # A radio-depleted UAV must not obtain fresh sensing evidence later in
            # the same transition. If radio energy changes liveness, the pre-TX
            # topology snapshot is stale and must be refreshed before computing
            # GCS reachability or the next observation.
            active_before_radio_depletion = self.uav_active.copy()
            self.uav_active &= self.battery_pct > 0.0
            self.velocities[~self.uav_active] = 0.0
            if not np.array_equal(self.uav_active, active_before_radio_depletion):
                self._refresh_links()
            self._expire_reports()
            self._perform_peer_target_detection()
            self._retry_pending_reports()
            broken_mask = (self._gcs_hops() < 0) & self.uav_active
        else:
            broken_mask = self.last_rates_bps < self.paper["min_comm_rate_bps"]
        self.cumulative_broken_time += broken_mask.astype(np.float64) * self.dt
        for local_idx in np.flatnonzero(broken_mask):
            self.episode_broken_link_uavs.add(int(self.multirotor_indices[int(local_idx)]))
        self.total_energy_used_j += step_energy

        rewards: dict[str, float] = {}
        components: dict[str, dict[str, float]] = {}
        shared_task_reward = 0.0
        shared_communication_reward = 0.0
        if self.peer_mode:
            shared_task_reward = float(
                self.assumed["search_reward_coeff"]
                * self.peer_sensing_target_reward_weight
                * (
                    int(self.last_new_targets_by_agent.sum())
                    - int(self.last_false_confirmations)
                )
            )
            shared_communication_reward = float(
                self.peer_delivery_reward * self.last_reports_delivered_step
                - self.peer_expiry_penalty * self.last_reports_expired_step
            )
        for i, name in enumerate(self.agents):
            if self.peer_mode and not active_at_step_start[i]:
                rewards[name] = 0.0
                components[name] = {
                    "communication": 0.0,
                    "energy": 0.0,
                    "safety": 0.0,
                    "task": 0.0,
                    "total": 0.0,
                }
                continue
            safety, _ = self._safety_reward(i)
            if i in self.multirotor_indices:
                communication = float(self._communication_reward(i))
                task = float(self._task_reward(i, fixed=False))
                if self.peer_mode:
                    communication += shared_communication_reward
                    task += shared_task_reward
                energy = float(self._energy_reward(i))
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
        # published reward timing. Homogeneous peer scenarios communicate queued
        # information first and perform current-step sensing afterward.
        if not self.peer_mode:
            for k in range(self.n_targets):
                if self.target_found[k]:
                    continue
                d = np.linalg.norm(self.positions[self.multirotor_indices, :2] - self.targets[k], axis=1)
                if len(d) and float(np.min(d)) <= self.assumed["target_found_m"]:
                    self.target_found[k] = True

        self.step_count += 1
        if self.peer_mode:
            self.uav_active &= self.battery_pct > 0.0
            self.velocities[~self.uav_active] = 0.0
        self.trajectory.append(self.positions.copy())

        mission_success = bool(self.peer_mode and self.n_targets > 0 and np.all(self.report_delivered))
        all_depleted = bool(self.peer_mode and not np.any(self.uav_active))
        horizon_reached = bool(self.step_count >= self.max_steps)
        # The homogeneous U6/U9 mission horizon is observable through remaining
        # time, so it is part of the finite-horizon MDP and must terminate rather
        # than externally truncate. Legacy paper scenarios keep time-limit
        # truncation semantics for backward compatibility.
        terminated_done = mission_success or all_depleted or bool(self.peer_mode and horizon_reached)
        truncated_done = bool(not terminated_done and horizon_reached)
        if mission_success:
            termination_reason = "all_reports_delivered"
        elif all_depleted:
            termination_reason = "all_uavs_depleted"
        elif horizon_reached:
            termination_reason = "horizon"
        else:
            termination_reason = "running"
        if self.peer_mode:
            terminated = {
                name: bool(terminated_done or not self.uav_active[i])
                for i, name in enumerate(self.agents)
            }
            truncated = {
                name: bool(truncated_done and not terminated[name])
                for name in self.agents
            }
        else:
            terminated = {name: terminated_done for name in self.agents}
            truncated = {name: truncated_done for name in self.agents}
        info = self._info(step_energy_j=step_energy, action_saturation=action_saturation)
        info["termination_reason"] = termination_reason
        return self._observations(), rewards, terminated, truncated, info

    def _info(self, step_energy_j: float, action_saturation: float) -> dict[str, Any]:
        rates = self.last_rates_bps if self.n_rotor else np.array([0.0])
        if self.peer_mode:
            gcs_hops = self._gcs_hops()
            active = self.uav_active.astype(bool)
            broken = (gcs_hops < 0) & active
            direct_gcs = int(np.sum((gcs_hops == 1) & active))
            multihop_gcs = int(np.sum((gcs_hops > 1) & active))
            disconnected_gcs = int(np.sum((gcs_hops < 0) & active))
            potential_rates = rates[active] if np.any(active) else np.array([0.0])
            selected_mask = self.last_tx_active & active
            selected_rates = (
                self.last_selected_tx_rate_bps[selected_mask]
                if np.any(selected_mask)
                else np.array([0.0])
            )
        else:
            gcs_hops = np.full(self.n_agents, -1, dtype=np.int64)
            broken = rates < self.paper["min_comm_rate_bps"]
            direct_gcs = multihop_gcs = disconnected_gcs = 0
            potential_rates = rates
            selected_rates = np.array([0.0])
        rotor_battery = self.battery_pct[self.multirotor_indices] if self.n_rotor else np.array([100.0])
        if self.peer_mode:
            altitudes = self.positions[:, 2]
            sensing_profiles = [self._sensing_profile(i) for i in range(self.n_agents)]
            if self.peer_sensing_model == "continuous_hu_liu":
                fov_radii = np.asarray([p.fov_radius_m for p in sensing_profiles], dtype=np.float64)
            else:
                fov_radii = np.asarray([
                    max(math.hypot(dx, dy) for dy, dx in fov_offsets(p.fov_cells)) * self.peer_sensing_grid_cell_m
                    for p in sensing_profiles
                ], dtype=np.float64)
            sensing_pd = np.asarray([p.pd for p in sensing_profiles], dtype=np.float64)
            sensing_pf = np.asarray([p.pf for p in sensing_profiles], dtype=np.float64)
            beliefs = np.clip(self.belief_maps, 0.0, 1.0)
            entropy_terms = np.zeros_like(beliefs, dtype=np.float64)
            interior = (beliefs > 0.0) & (beliefs < 1.0)
            p_int = beliefs[interior]
            entropy_terms[interior] = -(p_int * np.log2(p_int) + (1.0 - p_int) * np.log2(1.0 - p_int))
            target_posteriors = []
            for target_idx in range(self.n_targets):
                y, x = self._target_grid_cell(target_idx)
                target_posteriors.extend(self.belief_maps[:, y, x].tolist())
            mean_target_posterior = float(np.mean(target_posteriors)) if target_posteriors else 0.0
            sensing_diag = {
                "mean_altitude_m": float(np.mean(altitudes)),
                "min_altitude_m": float(np.min(altitudes)),
                "max_altitude_m": float(np.max(altitudes)),
                "mean_fov_radius_m": float(np.mean(fov_radii)),
                "mean_detection_probability": float(np.mean(sensing_pd)),
                "mean_false_alarm_probability": float(np.mean(sensing_pf)),
                "mean_belief_entropy": float(np.mean(entropy_terms)),
                "mean_target_posterior": mean_target_posterior,
                "scanned_cells_step": int(self.last_scanned_cells_by_agent.sum()),
                "scanned_cells_total": int(self.total_scanned_cells),
                "positive_sensor_observations_step": int(self.last_positive_sensor_observations_by_agent.sum()),
                "positive_sensor_observations_total": int(self.total_positive_sensor_observations),
                "information_gain_step": float(self.last_information_gain_by_agent.sum()),
                "information_gain_total": float(self.total_information_gain),
                "targets_confirmed_step": int(self.last_new_targets_by_agent.sum()),
                "targets_confirmed_total": int(self.target_found.sum()),
                "false_confirmations_step": int(self.last_false_confirmations),
                "false_confirmations_total": int(self.total_false_confirmations),
                "confirmed_cells_total": int(self.confirmed_cells.sum()),
            }
        else:
            sensing_diag = {
                "mean_altitude_m": float(np.mean(self.positions[:, 2])) if self.n_agents else 0.0,
                "min_altitude_m": float(np.min(self.positions[:, 2])) if self.n_agents else 0.0,
                "max_altitude_m": float(np.max(self.positions[:, 2])) if self.n_agents else 0.0,
                "mean_fov_radius_m": 0.0,
                "mean_detection_probability": 0.0,
                "mean_false_alarm_probability": 0.0,
                "mean_belief_entropy": 0.0,
                "mean_target_posterior": 0.0,
                "scanned_cells_step": 0,
                "scanned_cells_total": 0,
                "positive_sensor_observations_step": 0,
                "positive_sensor_observations_total": 0,
                "information_gain_step": 0.0,
                "information_gain_total": 0.0,
                "targets_confirmed_step": 0,
                "targets_confirmed_total": int(self.target_found.sum()),
                "false_confirmations_step": 0,
                "false_confirmations_total": 0,
                "confirmed_cells_total": int(self.target_found.sum()),
            }
        return {
            "step": self.step_count,
            "targets_found": int(self.target_found.sum()),
            "targets_total": int(self.n_targets),
            "search_rate": float(self.target_found.mean()) if self.n_targets else 0.0,
            "energy_used_j": float(step_energy_j),
            "total_energy_used_j": float(self.total_energy_used_j),
            "energy_consumption_pct": float(100.0 - np.mean(rotor_battery)) if self.n_rotor else 0.0,
            "avg_battery_pct": float(np.mean(rotor_battery)) if self.n_rotor else 100.0,
            "battery_capacity_j": float(self.peer_battery_capacity_j) if self.peer_mode else float(self.assumed["battery_capacity_j"]),
            "depleted_uavs": int(np.sum(rotor_battery <= 0.0)) if self.n_rotor else 0,
            "safety_distance_violation_uavs": len(self.episode_safety_violation_uavs),
            "obstacle_hit_uavs": len(self.episode_obstacle_hit_uavs),
            "boundary_hit_uavs": len(self.episode_boundary_hit_uavs),
            "broken_link_uavs": len(self.episode_broken_link_uavs),
            # Historical comm-rate keys are retained as potential-connectivity
            # diagnostics. Explicit selected-rate keys below report what the
            # policy actually attempted in this macro-step.
            "mean_comm_rate_mbps": float(np.mean(potential_rates) / 1e6),
            "min_comm_rate_mbps": float(np.min(potential_rates) / 1e6),
            "mean_potential_comm_rate_mbps": float(np.mean(potential_rates) / 1e6),
            "min_potential_comm_rate_mbps": float(np.min(potential_rates) / 1e6),
            "mean_selected_tx_rate_mbps": float(np.mean(selected_rates) / 1e6),
            "broken_links": int(np.sum(broken)),
            "mean_broken_link_s": float(np.mean(self.cumulative_broken_time)) if self.n_rotor else 0.0,
            "max_broken_link_s": float(np.max(self.cumulative_broken_time)) if self.n_rotor else 0.0,
            "safety_distance_violations": int(self.safety_distance_violation_count),
            "safety_filter_interventions": int(self.last_safety_filter_interventions) if self.peer_mode else 0,
            "safety_filter_interventions_total": int(self.total_safety_filter_interventions) if self.peer_mode else 0,
            "safety_filter_correction_m": float(self.last_safety_correction_m_by_agent.sum()) if self.peer_mode else 0.0,
            "max_safety_filter_correction_m": float(self.last_safety_correction_m_by_agent.max()) if self.peer_mode and self.n_agents else 0.0,
            "world_constraint_correction_m": float(self.last_world_constraint_correction_m_by_agent.sum()) if self.peer_mode else 0.0,
            "max_world_constraint_correction_m": float(self.last_world_constraint_correction_m_by_agent.max()) if self.peer_mode and self.n_agents else 0.0,
            "obstacle_hits": int(self.obstacle_hits),
            "boundary_hits": int(self.boundary_hits),
            "min_battery_pct": float(np.min(rotor_battery)) if self.n_rotor else 100.0,
            "action_saturation": float(action_saturation),
            "reward_components_sum": dict(self.last_reward_components_sum),
            "simulation_backend": "root_paper_mpe_style",
            "bytes_transmitted": int(self.last_bytes_transmitted) if self.peer_mode else 0,
            "total_bytes_transmitted": int(self.total_bytes_transmitted) if self.peer_mode else 0,
            "peer_syncs_step": int(self.last_peer_syncs) if self.peer_mode else 0,
            "peer_syncs_total": int(self.total_peer_syncs) if self.peer_mode else 0,
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
            "active_uavs": int(np.sum(self.uav_active)) if self.peer_mode else self.n_agents,
            "dead_uavs": int(np.sum(~self.uav_active)) if self.peer_mode else 0,
            "mean_selected_tx_power_w": float(np.mean(self.last_selected_tx_power_w[self.last_tx_active])) if self.peer_mode and np.any(self.last_tx_active) else 0.0,
            "max_selected_tx_power_w": float(np.max(self.last_selected_tx_power_w)) if self.peer_mode else 0.0,
            "network_sim_time_s": float(getattr(self.network_backend, "simulation_time_s", 0.0)) if self.peer_mode else 0.0,
            "network_episode_instance_id": int(getattr(self.network_backend, "episode_instance_id", 0)) if self.peer_mode else 0,
            "network_attempted_bytes": int(self.last_network_result.attempted_bytes) if self.peer_mode else 0,
            "network_admitted_bytes": int(self.last_network_result.admitted_bytes) if self.peer_mode else 0,
            "network_delivered_bytes": int(self.last_network_result.delivered_bytes) if self.peer_mode else 0,
            "network_byte_pdr": float(self.last_network_result.byte_pdr) if self.peer_mode else 0.0,
            "network_offered_delivery_ratio": float(self.last_network_result.offered_delivery_ratio) if self.peer_mode else 0.0,
            "network_throughput_bps": float(self.last_network_result.throughput_bps) if self.peer_mode else 0.0,
            "network_mean_delay_s": float(self.last_network_result.mean_delay_s) if self.peer_mode else 0.0,
            "network_phy_failures": int(self.last_network_result.phy_failures) if self.peer_mode else 0,
            "network_tx_energy_j": float(self.last_network_result.tx_energy_j) if self.peer_mode else 0.0,
            "total_network_attempted_bytes": int(self.total_network_attempted_bytes) if self.peer_mode else 0,
            "total_network_admitted_bytes": int(self.total_network_admitted_bytes) if self.peer_mode else 0,
            "total_network_delivered_bytes": int(self.total_network_delivered_bytes) if self.peer_mode else 0,
            "total_network_tx_energy_j": float(self.total_network_tx_energy_j) if self.peer_mode else 0.0,
            **sensing_diag,
        }
