from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from uav_search.config import load_config
from uav_search.envs.models import multirotor_power_w, segment_circle_intersects
from uav_search.envs.network_backends import TransmissionIntent, UavNetSimBackend, create_network_backend
from uav_search.envs.paper_env import GCS_RECIPIENT, PaperUAVEnv


def make_env(seed: int = 44) -> PaperUAVEnv:
    cfg = deepcopy(load_config("masac", "u6"))
    cfg["scenario"]["network_backend"] = "analytical"
    return PaperUAVEnv(cfg, seed=seed)


def idle_actions(env: PaperUAVEnv):
    return {a: np.array([-1.0, 0.0, 0.0, -1.0, -1.0, -1.0], dtype=np.float32) for a in env.agents}


def test_u6_literature_backed_report_buffer_lifetime_and_power_defaults():
    cfg = load_config("masac", "u6")["scenario"]
    assert cfg["report_bytes"] == 1_000_000
    assert cfg["buffer_bytes"] == 10_000_000
    assert cfg["report_ttl_s"] == pytest.approx(300.0)
    assert cfg["uavnetsim_tx_power_w"] == pytest.approx(0.1)
    assert cfg["tx_power_min_w"] == pytest.approx(0.1)
    assert cfg["tx_power_max_w"] == pytest.approx(0.4)


def test_u6_launch_pads_share_gcs_site_without_overlapping():
    env = make_env(seed=44)
    env.reset(seed=44)
    distances = np.linalg.norm(env.positions[:, :2] - env.gcs_position[:2], axis=1)
    assert np.all(distances <= float(env.scenario["launch_radius_m"]) + 1e-9)
    assert np.all(env.positions[:, 0] >= env.gcs_position[0] - 1e-9)
    min_sep = float(env.scenario["initial_min_separation_m"])
    for i in range(env.n_agents):
        for j in range(i + 1, env.n_agents):
            assert np.linalg.norm(env.positions[i, :2] - env.positions[j, :2]) >= min_sep - 1e-9


def test_segment_circle_collision_rejects_tunneling_even_when_endpoints_are_outside():
    circle = np.array([5.0, 0.0, 2.0])
    assert segment_circle_intersects(np.array([0.0, 0.0]), np.array([10.0, 0.0]), circle)
    assert not segment_circle_intersects(np.array([0.0, 3.0]), np.array([10.0, 3.0]), circle)


def test_u6_propulsion_does_not_double_count_root_paper_constant_pcom():
    cfg = load_config("masac", "u6")
    with_com = multirotor_power_w(5.0, 2.0, cfg, include_communication_power=True)
    propulsion_only = multirotor_power_w(5.0, 2.0, cfg, include_communication_power=False)
    assert with_com - propulsion_only == pytest.approx(float(cfg["paper"]["communication_power_w"]))


def test_tx_power_action_maps_to_literature_backed_peak_range():
    env = make_env(seed=45)
    assert env._decode_tx_power_w(-1.0) == pytest.approx(0.1)
    assert env._decode_tx_power_w(0.0) == pytest.approx(0.25)
    assert env._decode_tx_power_w(1.0) == pytest.approx(0.4)


def test_actor_own_network_slot_reports_direct_gcs_not_global_multihop_path():
    env = make_env(seed=46)
    env.last_adjacency.fill(0)
    env.last_gcs_rates_bps.fill(0.0)
    rmin = float(env.paper["min_comm_rate_bps"])
    env.last_gcs_rates_bps[0] = rmin + 1.0
    env.last_adjacency[0, 1] = 1  # UAV1 -> UAV0, so UAV1 has a two-hop path.
    assert env._gcs_hops()[1] == 2
    obs = env._observations()["uav_1"]
    assert obs[7] == pytest.approx(0.0)


def test_neighbor_freshness_does_not_treat_reverse_only_edge_as_live():
    env = make_env(seed=47)
    env.neighbor_cache_seen_step.fill(-1)
    env.neighbor_cache_seen_step[0, 1] = 0
    env.neighbor_cache_positions[0, 1] = env.positions[1]
    env.neighbor_cache_battery[0, 1] = 100.0
    env.step_count = 10
    env.last_adjacency.fill(0)
    env.last_adjacency[1, 0] = 1  # only UAV0 -> UAV1; UAV0 cannot receive live state from UAV1.
    slot = env._observations()["uav_0"][9:15]
    assert 0.0 < slot[-1] < 1.0


def test_depleted_uav_is_disabled_for_motion_sensing_and_networking():
    env = make_env(seed=48)
    env.battery_pct[0] = 0.0
    env.uav_active[0] = False
    env.targets[0] = env.positions[0, :2]
    start = env.positions[0].copy()
    actions = idle_actions(env)
    actions["uav_0"] = np.array([1.0, 1.0, 0.0, 1.0, 1.0, 0.999], dtype=np.float32)
    env.step(actions)
    np.testing.assert_allclose(env.positions[0], start)
    assert not env.target_known_by_agent[0, 0]
    assert not env.target_found[0]
    assert np.all(env.last_adjacency[0, :] == 0)
    assert np.all(env.last_adjacency[:, 0] == 0)
    assert env.last_gcs_rates_bps[0] == 0.0


def test_u6_terminates_on_all_reports_delivered_or_all_uavs_dead():
    env = make_env(seed=49)
    env.report_delivered[:] = True
    _, _, terminated, truncated, info = env.step(idle_actions(env))
    assert all(terminated.values())
    assert not any(truncated.values())
    assert info["termination_reason"] == "all_reports_delivered"

    env = make_env(seed=50)
    env.battery_pct[:] = 0.0
    env.uav_active[:] = False
    _, _, terminated, truncated, info = env.step(idle_actions(env))
    assert all(terminated.values())
    assert not any(truncated.values())
    assert info["termination_reason"] == "all_uavs_depleted"


def test_analytical_transmit_power_changes_energy_and_link_result():
    cfg = deepcopy(load_config("masac", "u6"))
    cfg["scenario"]["network_backend"] = "analytical"
    # No hard discovery/contact cap for this isolated PHY-power regression.
    cfg["scenario"]["peer_contact_range_m"] = 10_000.0
    backend = create_network_backend("analytical", cfg, seed=51)
    positions = np.array(
        [[0.0, 0.0, 60.0], [1800.0, 0.0, 60.0], [4500.0, 4500.0, 60.0],
         [4600.0, 4500.0, 60.0], [4500.0, 4600.0, 60.0], [4600.0, 4600.0, 60.0]],
        dtype=float,
    )
    gcs = np.array([0.0, 0.0, 0.0])
    low = backend.transmit(
        [TransmissionIntent(0, 1, 1000, tx_power_w=0.05)], positions, gcs, np.empty((0, 3)), dt_s=1.0, step_index=0
    )
    high = backend.transmit(
        [TransmissionIntent(0, 1, 1000, tx_power_w=0.4)], positions, gcs, np.empty((0, 3)), dt_s=1.0, step_index=0
    )
    assert high.outcomes[0].rate_bps >= low.outcomes[0].rate_bps
    assert high.outcomes[0].tx_energy_j >= low.outcomes[0].tx_energy_j


def test_uavnetsim_uses_one_persistent_clock_per_episode_when_dependency_available():
    import importlib.util

    if importlib.util.find_spec("simpy") is None or importlib.util.find_spec("phy.channel") is None:
        pytest.skip("pinned UavNetSim optional dependency is not installed")
    cfg = deepcopy(load_config("masac", "u6"))
    backend = UavNetSimBackend(cfg, seed=52)
    positions = np.array(
        [[100.0, 100.0, 60.0], [150.0, 100.0, 60.0], [4500.0, 4500.0, 60.0],
         [4600.0, 4500.0, 60.0], [4500.0, 4600.0, 60.0], [4600.0, 4600.0, 60.0]],
        dtype=float,
    )
    gcs = np.array([200.0, 100.0, 0.0])
    backend.reset_episode(positions, gcs, np.empty((0, 3)))
    episode_id = backend.episode_instance_id
    backend.transmit(
        [TransmissionIntent(0, 1, 1000, tx_power_w=0.1)], positions, gcs, np.empty((0, 3)), dt_s=1.0, step_index=0
    )
    t1 = backend.simulation_time_s
    backend.transmit(
        [TransmissionIntent(0, 1, 1000, tx_power_w=0.1)], positions, gcs, np.empty((0, 3)), dt_s=1.0, step_index=1
    )
    assert backend.episode_instance_id == episode_id
    assert t1 == pytest.approx(1.0)
    assert backend.simulation_time_s == pytest.approx(2.0)


def test_uavnetsim_idle_macro_step_advances_persistent_clock_when_dependency_available():
    import importlib.util

    if importlib.util.find_spec("simpy") is None or importlib.util.find_spec("phy.channel") is None:
        pytest.skip("pinned UavNetSim optional dependency is not installed")
    cfg = deepcopy(load_config("masac", "u6"))
    backend = UavNetSimBackend(cfg, seed=90)
    positions = np.array(
        [[100.0, 100.0, 60.0], [150.0, 100.0, 60.0], [4500.0, 4500.0, 60.0],
         [4600.0, 4500.0, 60.0], [4500.0, 4600.0, 60.0], [4600.0, 4600.0, 60.0]],
        dtype=float,
    )
    gcs = np.array([200.0, 100.0, 0.0])
    empty = np.empty((0, 3))
    backend.reset_episode(positions, gcs, empty)
    episode_id = backend.episode_instance_id

    result = backend.transmit([], positions, gcs, empty, dt_s=1.0, step_index=0)

    assert result.attempted_bytes == 0
    assert backend.episode_instance_id == episode_id
    assert backend.simulation_time_s == pytest.approx(1.0)


def test_uavnetsim_reference_contact_range_matches_reference_power_transmission_when_dependency_available():
    import importlib.util

    if importlib.util.find_spec("simpy") is None or importlib.util.find_spec("phy.channel") is None:
        pytest.skip("pinned UavNetSim optional dependency is not installed")
    cfg = deepcopy(load_config("masac", "u6"))
    backend = UavNetSimBackend(cfg, seed=91)
    positions = np.array(
        [[100.0, 100.0, 60.0], [2600.0, 100.0, 60.0], [4500.0, 4500.0, 60.0],
         [4600.0, 4500.0, 60.0], [4500.0, 4600.0, 60.0], [4600.0, 4600.0, 60.0]],
        dtype=float,
    )
    gcs = np.array([0.0, 100.0, 0.0])
    empty = np.empty((0, 3))
    snapshot = backend.link_snapshot(positions, gcs, empty)
    assert np.linalg.norm(positions[1] - positions[0]) > cfg["scenario"]["peer_contact_range_m"]
    assert snapshot.adjacency[1, 0] == 0

    low = backend.transmit(
        [TransmissionIntent(0, 1, 1000, tx_power_w=0.1)], positions, gcs, empty,
        dt_s=1.0, step_index=0,
    )
    high = backend.transmit(
        [TransmissionIntent(0, 1, 1000, tx_power_w=0.4)], positions, gcs, empty,
        dt_s=1.0, step_index=1,
    )

    assert low.delivered_bytes == 0
    assert high.delivered_bytes == 0
