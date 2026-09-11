from copy import deepcopy

import numpy as np
import pytest

from uav_search.config import load_config
from uav_search.envs.network_backends import TransmissionIntent, create_network_backend
from uav_search.envs.paper_env import PaperUAVEnv


def make_env(seed: int = 44) -> PaperUAVEnv:
    cfg = deepcopy(load_config("masac", "u6"))
    cfg["scenario"]["network_backend"] = "analytical"
    return PaperUAVEnv(cfg, seed=seed)


def idle_actions(env: PaperUAVEnv) -> dict[str, np.ndarray]:
    return {
        name: np.asarray([-1.0, 0.0, 0.0, -1.0, -1.0, -1.0], dtype=np.float32)
        for name in env.agents
    }


def put_obstacles_far_away(env: PaperUAVEnv) -> None:
    env.obstacles[:, :2] = np.asarray([4900.0, 4900.0])
    env.obstacles[:, 2] = 1.0


def test_continuous_fov_actor_patch_covers_maximum_sensed_offsets() -> None:
    env = make_env(seed=101)
    env.positions[0] = np.asarray([2500.0, 2500.0, env.peer_altitude_max_m])

    patch = env._belief_patch(0)
    center_y, center_x = env._xy_grid_cell(env.positions[0, :2])
    sensed = set(env._sensing_cells(0))
    max_offset = max(
        max(abs(y - center_y), abs(x - center_x))
        for y, x in sensed
    )

    assert max_offset == 2
    assert len(patch) == 25


def test_actor_has_recipient_specific_min_and_max_power_link_features_without_peer_cache() -> None:
    env = make_env(seed=102)
    env.neighbor_cache_seen_step.fill(-1)
    env.min_power_pair_rates_bps.fill(0.0)
    env.max_power_pair_rates_bps.fill(0.0)
    env.max_power_pair_rates_bps[1, 0] = 2_000_000.0

    features = env._peer_link_features(sender=0, recipient=1)

    assert features == pytest.approx([0.0, 1.0])


def test_tx_power_changes_operational_envelope_instead_of_both_powers_hitting_same_hard_radius() -> None:
    cfg = deepcopy(load_config("masac", "u6"))
    cfg["scenario"]["network_backend"] = "analytical"
    backend = create_network_backend("analytical", cfg, seed=103)
    positions = np.asarray(
        [
            [100.0, 100.0, 100.0],
            [2600.0, 100.0, 100.0],
            [4500.0, 4500.0, 100.0],
            [4600.0, 4500.0, 100.0],
            [4500.0, 4600.0, 100.0],
            [4600.0, 4600.0, 100.0],
        ],
        dtype=np.float64,
    )
    gcs = np.asarray([0.0, 100.0, 0.0], dtype=np.float64)
    obstacles = np.empty((0, 3), dtype=np.float64)

    low = backend.transmit(
        [TransmissionIntent(0, 1, 1000, tx_power_w=0.1)],
        positions,
        gcs,
        obstacles,
        dt_s=1.0,
        step_index=0,
    )
    high = backend.transmit(
        [TransmissionIntent(0, 1, 1000, tx_power_w=0.4)],
        positions,
        gcs,
        obstacles,
        dt_s=1.0,
        step_index=1,
    )

    assert low.delivered_bytes == 0
    assert high.delivered_bytes > 0


def test_discrete_time_barrier_shield_filters_unsafe_motion_before_state_transition() -> None:
    env = make_env(seed=104)
    put_obstacles_far_away(env)
    env.positions[:] = np.asarray(
        [
            [1000.0, 1000.0, 100.0],
            [1150.0, 1000.0, 100.0],
            [2500.0, 500.0, 100.0],
            [3000.0, 1000.0, 100.0],
            [3500.0, 1500.0, 100.0],
            [4000.0, 2000.0, 100.0],
        ]
    )
    env.velocities[:] = 0.0
    actions = idle_actions(env)
    actions["uav_0"] = np.asarray([1.0, 0.0, 0.0, -1.0, -1.0, -1.0], dtype=np.float32)
    actions["uav_1"] = np.asarray([1.0, 1.0, 0.0, -1.0, -1.0, -1.0], dtype=np.float32)
    before = env.positions.copy()

    _, _, _, _, info = env.step(actions)

    assert np.linalg.norm(env.positions[0] - env.positions[1]) >= env.safety_distance_m
    assert not np.allclose(env.positions[:2], before[:2])
    assert info["safety_filter_interventions"] >= 1
    assert info["safety_distance_violations"] == 0


def test_report_expiry_is_packet_level_loss_not_terminal_and_requires_fresh_evidence_to_regenerate() -> None:
    cfg = deepcopy(load_config("masac", "u6"))
    cfg["scenario"]["network_backend"] = "analytical"
    cfg["scenario"]["report_ttl_s"] = 1.0
    env = PaperUAVEnv(cfg, seed=105)
    put_obstacles_far_away(env)
    env.velocities[:] = 0.0
    env.target_found[0] = True
    env.target_directly_confirmed_by_agent[0, 0] = True
    env.pending_report_source[0] = 0
    assert env._enqueue_report(0, 0)
    env.step_count = 1

    _, _, terminated, truncated, info = env.step(idle_actions(env))

    assert not any(terminated.values())
    assert info["termination_reason"] in {"running", "horizon"}
    assert env.expired_reports >= 1
    assert not env.report_generated[0]
    assert env.report_created_step[0] == -1
    assert env.pending_report_source[0] == -1
    assert env.report_buffers[0].sum() == 0


def test_native_gcs_rate_reward_scales_with_committed_report_bytes() -> None:
    env = make_env(seed=106)
    env.last_tx_active[0] = True
    env.last_report_bytes_attempted_by_agent[0] = 1024
    env.last_report_bytes_transmitted_by_agent[0] = 1024
    env.last_tx_success[0] = True
    env.last_selected_tx_rate_bps[0] = float(env.scenario["uavnetsim_bit_rate_bps"])
    env.last_selected_tx_distance_m[0] = 1000.0

    expected = float(env.assumed["comm_reward_max"]) * 1024 / env.peer_report_bytes
    assert env._communication_reward(0) == pytest.approx(expected)
