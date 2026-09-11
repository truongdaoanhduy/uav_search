from copy import deepcopy

import numpy as np
import pytest

from uav_search.config import load_config
from uav_search.envs.network_backends import TransmissionIntent, create_network_backend
from uav_search.envs.paper_env import PaperUAVEnv


class FixedRandom:
    def __init__(self, value: float):
        self.value = float(value)

    def random(self) -> float:
        return self.value


def make_env(seed: int = 44, **scenario_overrides) -> PaperUAVEnv:
    cfg = deepcopy(load_config("masac", "u6"))
    cfg["scenario"]["network_backend"] = "analytical"
    cfg["scenario"].update(scenario_overrides)
    return PaperUAVEnv(cfg, seed=seed)


def idle_actions(env: PaperUAVEnv) -> dict[str, np.ndarray]:
    return {
        name: np.asarray([-1.0, 0.0, 0.0, -1.0, -1.0, -1.0], dtype=np.float32)
        for name in env.agents
    }


def tx_to_gcs_actions(env: PaperUAVEnv, sender: int = 0) -> dict[str, np.ndarray]:
    actions = idle_actions(env)
    actions[env.agents[sender]] = np.asarray(
        [-1.0, 0.0, 0.0, 1.0, 1.0, 0.999], dtype=np.float32
    )
    return actions


def set_static_safe_geometry(env: PaperUAVEnv) -> None:
    env.positions[:, :2] = np.asarray(
        [[100.0 + 500.0 * i, 2500.0] for i in range(env.n_agents)], dtype=np.float64
    )
    env.positions[:, 2] = env.peer_altitude_min_m
    env.velocities.fill(0.0)
    env.obstacles[:, :2] = np.asarray([4900.0, 4900.0])
    env.obstacles[:, 2] = 1.0
    env._refresh_links()


def test_action_cannot_transmit_report_created_by_sensing_later_in_same_step() -> None:
    env = make_env(seed=201)
    set_static_safe_geometry(env)
    env.targets[:] = np.asarray(
        [[4000.0, 4000.0 - 200.0 * i] for i in range(env.n_targets)], dtype=np.float64
    )
    env.targets[0] = env.positions[0, :2]
    y, x = env._target_grid_cell(0)
    env.belief_maps[0, y, x] = env.peer_target_confirmation_threshold + 1e-4
    env.rng = FixedRandom(0.0)

    assert not env.target_found[0]
    assert not env.report_generated[0]
    env.step(tx_to_gcs_actions(env, sender=0))

    assert env.target_found[0]
    assert env.report_generated[0]
    assert env.report_buffers[0, 0] == env.peer_report_bytes
    assert env.last_report_bytes_transmitted_by_agent[0] == 0
    assert not env.report_delivered[0]


def test_report_gets_final_transmission_opportunity_before_ttl_expiry() -> None:
    env = make_env(seed=202, report_bytes=1000, buffer_bytes=10_000, report_ttl_s=1.0)
    set_static_safe_geometry(env)
    env.positions[0] = env.gcs_position + np.asarray([0.0, 0.0, env.peer_altitude_min_m])
    env._refresh_links()
    assert env._enqueue_report(0, 0)
    env.step_count = 1
    env.rng = FixedRandom(0.99)

    env.step(tx_to_gcs_actions(env, sender=0))

    assert env.report_delivered[0]
    assert env.report_buffers[0].sum() == 0
    assert not env.report_expired[0]
    assert env.last_reports_delivered_step == 1


def test_expired_report_does_not_regenerate_from_stale_confirmation_alone() -> None:
    env = make_env(seed=203, report_ttl_s=1.0)
    env.target_found[0] = True
    env.target_known_by_agent[0, 0] = True
    env.target_directly_confirmed_by_agent[0, 0] = True
    y, x = env._target_grid_cell(0)
    env.belief_maps[0, y, x] = env.peer_target_confirmation_threshold + 1e-4
    env.fine_positive_cells_by_agent[0, y, x] = True
    env.fine_target_evidence_by_agent[0, 0] = True
    env.pending_report_source[0] = 0
    assert env._enqueue_report(0, 0)
    env.step_count = 2

    env._expire_reports()
    # No fresh positive observation occurs after expiry. Persistent historical
    # fine evidence must not silently mint a new report generation.
    env.positions[0, :2] = np.asarray([100.0, 100.0])
    env.targets[0] = np.asarray([4000.0, 4000.0])
    env.rng = FixedRandom(0.99)
    env._perform_peer_target_detection()
    env._retry_pending_reports()

    assert env.report_expired[0]
    assert not env.report_generated[0]
    assert env.pending_report_source[0] == -1
    assert env.report_buffers[0].sum() == 0


def test_fresh_fine_reobservation_after_expiry_can_create_new_generation() -> None:
    env = make_env(seed=206, report_ttl_s=1.0)
    set_static_safe_geometry(env)
    env.targets[:] = np.asarray(
        [[4000.0, 4000.0 - 200.0 * i] for i in range(env.n_targets)], dtype=np.float64
    )
    env.targets[0] = env.positions[0, :2]
    target_idx = 0
    y, x = env._target_grid_cell(target_idx)
    env.target_found[target_idx] = True
    env.target_known_by_agent[0, target_idx] = True
    env.target_directly_confirmed_by_agent[0, target_idx] = True
    env.belief_maps[0, y, x] = env.peer_target_confirmation_threshold + 1e-4
    env.fine_positive_cells_by_agent[0, y, x] = True
    env.fine_target_evidence_by_agent[0, target_idx] = True
    env.pending_report_source[target_idx] = 0
    assert env._enqueue_report(0, target_idx)
    env.step_count = 2
    env._expire_reports()
    assert not env.report_generated[target_idx]

    env.rng = FixedRandom(0.0)
    env._perform_peer_target_detection()
    env._retry_pending_reports()

    assert env.report_generated[target_idx]
    assert env.report_created_step[target_idx] == env.step_count
    assert env.report_buffers[target_idx, 0] == env.peer_report_bytes



def test_post_expiry_generation_uses_fresh_source_and_ttl_starts_while_buffer_waits() -> None:
    env = make_env(
        seed=207, report_ttl_s=1.0, buffer_bytes=1_000_000, report_bytes=1_000_000
    )
    set_static_safe_geometry(env)
    target_idx = 0
    env.targets[:] = np.asarray(
        [[4000.0, 4000.0 - 200.0 * i] for i in range(env.n_targets)], dtype=np.float64
    )
    env.targets[target_idx] = env.positions[0, :2]
    y, x = env._target_grid_cell(target_idx)

    # Both UAV0 and UAV1 confirmed the target historically. Only UAV0 will
    # obtain a fresh positive fine observation after expiry.
    env.target_found[target_idx] = True
    for agent_idx in (0, 1):
        env.target_known_by_agent[agent_idx, target_idx] = True
        env.target_directly_confirmed_by_agent[agent_idx, target_idx] = True
        env.belief_maps[agent_idx, y, x] = env.peer_target_confirmation_threshold + 1e-4
        env.fine_positive_cells_by_agent[agent_idx, y, x] = True
        env.fine_target_evidence_by_agent[agent_idx, target_idx] = True

    env.pending_report_source[target_idx] = 0
    assert env._enqueue_report(0, target_idx)
    env.step_count = 2
    env._expire_reports()
    assert not env.report_generated[target_idx]

    # Occupy UAV0's only report slot. UAV1 has free storage but no fresh
    # post-expiry observation and therefore must not synthesize the new generation.
    assert env._enqueue_report(0, 1)
    env.positions[1, :2] = np.asarray([3000.0, 3000.0])
    env.rng = FixedRandom(0.0)
    env._perform_peer_target_detection()
    env._retry_pending_reports()

    assert not env.report_generated[target_idx]
    assert env.pending_report_source[target_idx] == 0
    assert env.report_created_step[target_idx] == env.step_count
    assert env.report_buffers[target_idx, 1] == 0

    # Once the fresh observer has room, its already-created information may be
    # enqueued without requiring a second observation; TTL kept running meanwhile.
    env.report_buffers[1, 0] = 0
    env.queue_bytes[0] = 0
    env._retry_pending_reports()
    assert env.report_generated[target_idx]
    assert env.report_buffers[target_idx, 0] == env.peer_report_bytes
    assert env.report_created_step[target_idx] == 2



def test_failed_out_of_range_analytical_attempt_consumes_sender_tx_energy() -> None:
    cfg = deepcopy(load_config("masac", "u6"))
    cfg["scenario"]["network_backend"] = "analytical"
    backend = create_network_backend("analytical", cfg, seed=204)
    positions = np.asarray(
        [
            [100.0, 100.0, 50.0],
            [4900.0, 4900.0, 50.0],
            [3000.0, 3000.0, 50.0],
            [3200.0, 3200.0, 50.0],
            [3400.0, 3400.0, 50.0],
            [3600.0, 3600.0, 50.0],
        ],
        dtype=np.float64,
    )

    result = backend.transmit(
        [TransmissionIntent(sender=0, recipient=1, requested_bytes=4096, tx_power_w=0.1)],
        positions,
        np.asarray([0.0, 2500.0, 0.0], dtype=np.float64),
        np.empty((0, 3), dtype=np.float64),
        dt_s=1.0,
        step_index=0,
    )

    assert result.delivered_bytes == 0
    assert result.tx_energy_j > 0.0
    assert result.node_tx_energy_j[0] > 0.0
    assert result.outcomes[0].tx_energy_j > 0.0


def test_failed_out_of_range_uavnetsim_attempt_consumes_sender_tx_energy() -> None:
    cfg = deepcopy(load_config("masac", "u6"))
    cfg["scenario"]["network_backend"] = "uavnetsim"
    backend = create_network_backend("uavnetsim", cfg, seed=205)
    positions = np.asarray(
        [
            [100.0, 100.0, 50.0],
            [4900.0, 4900.0, 50.0],
            [3000.0, 3000.0, 50.0],
            [3200.0, 3200.0, 50.0],
            [3400.0, 3400.0, 50.0],
            [3600.0, 3600.0, 50.0],
        ],
        dtype=np.float64,
    )
    gcs = np.asarray([0.0, 2500.0, 0.0], dtype=np.float64)
    obstacles = np.empty((0, 3), dtype=np.float64)
    backend.reset_episode(positions, gcs, obstacles)

    result = backend.transmit(
        [TransmissionIntent(sender=0, recipient=1, requested_bytes=4096, tx_power_w=0.1)],
        positions,
        gcs,
        obstacles,
        dt_s=1.0,
        step_index=0,
    )

    assert result.delivered_bytes == 0
    assert result.tx_energy_j > 0.0
    assert result.node_tx_energy_j[0] > 0.0
    assert result.outcomes[0].tx_energy_j > 0.0


@pytest.mark.parametrize("scenario_name", ["u6", "u9"])
def test_proximity_sensor_range_is_explicit_in_research_scenario_config(scenario_name: str) -> None:
    scenario = load_config("masac", scenario_name)["scenario"]
    assert scenario["proximity_sensor_range_m"] == pytest.approx(300.0)
