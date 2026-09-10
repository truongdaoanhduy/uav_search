from copy import deepcopy

import numpy as np
import pytest

from uav_search.config import load_config
from uav_search.envs.paper_env import PaperUAVEnv


def analytical_u6(seed=44):
    cfg = deepcopy(load_config("masac", "u6"))
    cfg["scenario"]["network_backend"] = "analytical"
    return PaperUAVEnv(cfg, seed=seed)


def idle_actions(env):
    # Gate is OFF; vertical acceleration is neutral and RF power is at its minimum.
    return {a: np.array([-1.0, 0.0, 0.0, -1.0, -1.0, -1.0], dtype=np.float32) for a in env.agents}


def test_u6_config_uses_literature_backed_report_lifetime_and_power_contract():
    cfg = load_config("masac", "u6")["scenario"]
    assert cfg["report_bytes"] == 1_000_000
    assert cfg["buffer_bytes"] == 100_000_000
    assert cfg["report_ttl_s"] == pytest.approx(300.0)
    assert cfg["tx_power_min_w"] == pytest.approx(0.1)
    assert cfg["tx_power_max_w"] == pytest.approx(0.4)
    assert cfg["tx_power_reference_w"] == pytest.approx(0.1)
    assert cfg["initialization"] == "gcs_launch_pads"
    assert cfg["network_packet_payload_bytes"] == 1024
    assert cfg["neighbor_cache_ttl_steps"] == 5
    assert cfg["initial_min_separation_m"] == pytest.approx(141.4)
    assert cfg["battery_capacity_j"] == pytest.approx(77.0 * 3600.0)


def test_u6_launch_pads_share_gcs_site_without_initial_collision():
    env = analytical_u6(seed=44)
    env.reset(seed=44)
    radial = np.linalg.norm(env.positions[:, :2] - env.gcs_position[:2], axis=1)
    assert np.all(radial <= float(env.scenario["launch_radius_m"]) + 1e-9)
    assert np.all(radial > 0.0)
    min_sep = float(env.scenario["initial_min_separation_m"])
    for i in range(env.n_agents):
        for j in range(i + 1, env.n_agents):
            assert np.linalg.norm(env.positions[i, :2] - env.positions[j, :2]) >= min_sep - 1e-9


def test_motion_segment_cannot_tunnel_through_circular_obstacle():
    env = analytical_u6(seed=45)
    env.reset(seed=45)
    env.positions[0, :2] = [100.0, 100.0]
    env.velocities[0, :2] = [10.0, 0.0]
    env.obstacles[:, :2] = [4900.0, 4900.0]
    env.obstacles[:, 2] = 1.0
    env.obstacles[0] = [105.0, 100.0, 1.0]
    start = env.positions[0, :2].copy()

    env.step(idle_actions(env))

    np.testing.assert_allclose(env.positions[0, :2], start)
    assert env.obstacle_hits >= 1


def test_depleted_uav_is_disabled_and_cannot_move():
    env = analytical_u6(seed=46)
    env.reset(seed=46)
    env.positions[0, :2] = [1000.0, 1000.0]
    env.velocities[0, :2] = [10.0, 0.0]
    env.battery_pct[0] = 0.0
    start = env.positions[0].copy()

    env.step(idle_actions(env))

    assert not env.uav_active[0]
    np.testing.assert_allclose(env.positions[0], start)
    np.testing.assert_allclose(env.velocities[0], [0.0, 0.0, 0.0])


def test_depleted_uav_cannot_detect_new_target():
    env = analytical_u6(seed=47)
    env.reset(seed=47)
    env.targets[:] = [4900.0, 4900.0]
    env.positions[:, :2] = [3000.0, 3000.0]
    env.positions[0, :2] = [500.0, 500.0]
    env.targets[0] = [500.0, 500.0]
    env.battery_pct[0] = 0.0
    env.uav_active[0] = False

    env.step(idle_actions(env))

    assert not env.target_found[0]
    assert not env.target_known_by_agent[0, 0]


def test_depleted_uav_is_removed_from_peer_and_gcs_topology():
    env = analytical_u6(seed=48)
    env.reset(seed=48)
    env.positions[:, :2] = env.gcs_position[:2]
    env.battery_pct[0] = 0.0
    env.uav_active[0] = False
    env._refresh_links()

    assert env.last_gcs_rates_bps[0] == 0.0
    assert np.all(env.last_adjacency[0, :] == 0)
    assert np.all(env.last_adjacency[:, 0] == 0)


def test_u6_terminates_when_all_uavs_are_depleted_before_horizon():
    env = analytical_u6(seed=49)
    env.reset(seed=49)
    env.battery_pct[:] = 0.0
    env.uav_active[:] = False

    _, _, terminated, truncated, info = env.step(idle_actions(env))

    assert all(terminated.values())
    assert not any(truncated.values())
    assert info["termination_reason"] == "all_uavs_depleted"


def test_u6_terminates_successfully_when_all_reports_reach_gcs():
    env = analytical_u6(seed=50)
    env.reset(seed=50)
    env.report_delivered[:] = True
    env.reports_delivered = env.n_targets

    _, _, terminated, truncated, info = env.step(idle_actions(env))

    assert all(terminated.values())
    assert not any(truncated.values())
    assert info["termination_reason"] == "all_reports_delivered"


def test_tx_power_action_maps_to_configured_literature_range():
    env = analytical_u6(seed=51)
    assert env._decode_tx_power_w(-1.0) == pytest.approx(0.1)
    assert env._decode_tx_power_w(0.0) == pytest.approx(0.25)
    assert env._decode_tx_power_w(1.0) == pytest.approx(0.4)


def test_actor_self_network_flag_is_direct_gcs_only_not_global_multihop_truth():
    env = analytical_u6(seed=52)
    env.reset(seed=52)
    env.last_gcs_rates_bps.fill(0.0)
    env.last_adjacency.fill(0)
    rmin = float(env.paper["min_comm_rate_bps"])
    env.last_gcs_rates_bps[0] = rmin + 1.0
    env.last_adjacency[0, 1] = 1  # UAV1 -> UAV0 -> GCS exists.
    assert env._gcs_hops()[1] == 2

    obs = env._observations()["uav_1"]
    # own layout index 7 is the local network flag.
    assert obs[7] == pytest.approx(0.0)

def test_neighbor_freshness_does_not_treat_reverse_only_link_as_live():
    env = analytical_u6(seed=53)
    env.reset(seed=53)
    env.neighbor_cache_seen_step[0, 1] = 0
    env.neighbor_cache_positions[0, 1] = env.positions[1]
    env.neighbor_cache_battery[0, 1] = env.battery_pct[1]
    # One completed unsynchronized transition after the fully fresh returned
    # state: age=1 at t=2, so the five-step cache has freshness 0.8.
    env.step_count = 2
    env.last_adjacency.fill(0)
    baseline_freshness = float(env._observations()["uav_0"][14])

    env.last_adjacency[1, 0] = 1  # UAV0 -> UAV1 only; UAV1 cannot update UAV0.
    reverse_link_freshness = float(env._observations()["uav_0"][14])

    assert baseline_freshness == pytest.approx(0.8)
    assert reverse_link_freshness == pytest.approx(baseline_freshness)


def test_report_lifetime_is_measured_in_seconds_not_hardcoded_steps():
    cfg = deepcopy(load_config("masac", "u6"))
    cfg["scenario"]["network_backend"] = "analytical"
    cfg["scenario"]["report_ttl_s"] = 3.0
    cfg["scenario"]["dt_s"] = 2.0
    env = PaperUAVEnv(cfg, seed=54)
    env.reset(seed=54)
    env._enqueue_report(0, 0)
    env.step_count = 2  # 4 seconds since episode start/report creation at 0.
    env._expire_reports()
    assert env.report_expired[0]


def test_u6_flight_energy_excludes_root_paper_constant_pcom_to_avoid_double_counting():
    env = analytical_u6(seed=55)
    env.reset(seed=55)
    env.velocities[:] = 0.0
    env.step(idle_actions(env))
    assert env.last_energy_by_agent_j[0] == pytest.approx(float(env.assumed["hover_power_w"]) * env.dt)


def test_higher_tx_power_does_not_extend_fixed_contact_radius():
    from uav_search.envs.network_backends import TransmissionIntent, create_network_backend

    cfg = deepcopy(load_config("masac", "u6"))
    cfg["scenario"]["network_backend"] = "analytical"
    backend = create_network_backend("analytical", cfg, seed=56)
    positions = np.array([
        [100.0, 100.0, 60.0],
        [2600.0, 100.0, 60.0],
        [4500.0, 4500.0, 60.0],
        [4600.0, 4500.0, 60.0],
        [4500.0, 4600.0, 60.0],
        [4600.0, 4600.0, 60.0],
    ])
    gcs = np.array([0.0, 100.0, 0.0])
    empty = np.empty((0, 3))

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
    assert high.tx_energy_j == 0.0


def test_tx_power_slot_no_longer_controls_byte_fraction():
    env = analytical_u6(seed=57)
    env.reset(seed=57)
    env.positions[0, :2] = env.gcs_position[:2]
    env._refresh_links()
    env._enqueue_report(0, 0)
    actions = idle_actions(env)
    # Gate ON, minimum configured power, GCS recipient.
    actions["uav_0"] = np.array([-1.0, 0.0, 0.0, 1.0, -1.0, 0.999], dtype=np.float32)

    _, _, _, _, info = env.step(actions)

    assert info["network_attempted_bytes"] > 0
    assert info["network_delivered_bytes"] > 0


def test_packet_payload_is_packetization_size_not_one_packet_per_rl_step():
    cfg = deepcopy(load_config("masac", "u6"))
    cfg["scenario"]["network_backend"] = "analytical"
    cfg["scenario"]["network_packet_payload_bytes"] = 1024
    env = PaperUAVEnv(cfg, seed=58)
    env.reset(seed=58)
    env.positions[0, :2] = env.gcs_position[:2]
    env._refresh_links()
    assert env._enqueue_report(0, 0)
    actions = idle_actions(env)
    actions["uav_0"] = np.array([-1.0, 0.0, 0.0, 1.0, -1.0, 0.999], dtype=np.float32)

    _, _, _, _, info = env.step(actions)

    assert info["network_attempted_bytes"] > cfg["scenario"]["network_packet_payload_bytes"]


def test_u6_scenario_battery_capacity_overrides_legacy_assumed_capacity():
    cfg = deepcopy(load_config("masac", "u6"))
    cfg["scenario"]["network_backend"] = "analytical"
    cfg["scenario"]["battery_capacity_j"] = 1000.0
    env = PaperUAVEnv(cfg, seed=59)
    env.reset(seed=59)
    before = float(env.battery_pct[0])

    _, _, _, _, info = env.step(idle_actions(env))

    expected_drop = 100.0 * float(env.last_energy_by_agent_j[0]) / 1000.0
    assert before - env.battery_pct[0] == pytest.approx(expected_drop)
    assert info["battery_capacity_j"] == pytest.approx(1000.0)


def test_u6_uses_literature_backed_safe_distance_without_changing_legacy_scenarios():
    cfg = deepcopy(load_config("masac", "u6"))
    cfg["scenario"]["network_backend"] = "analytical"
    env = PaperUAVEnv(cfg, seed=60)
    env.reset(seed=60)
    assert env.scenario["safety_distance_m"] == pytest.approx(141.4)
    env.positions[0] = [1000.0, 1000.0, 100.0]
    env.positions[1] = [1120.0, 1000.0, 100.0]
    penalty, close = env._safety_reward(0)
    assert close >= 1
    assert penalty < 0.0

    legacy = PaperUAVEnv(load_config("masac", "f1_m5"), seed=60)
    legacy.reset(seed=60)
    legacy.positions[0] = [1000.0, 1000.0, 200.0]
    legacy.positions[1] = [1120.0, 1000.0, 200.0]
    _, legacy_close = legacy._safety_reward(0)
    assert legacy_close == 0


def test_u6_can_use_scenario_specific_literature_backed_macro_step_without_changing_legacy():
    cfg = deepcopy(load_config("masac", "u6"))
    cfg["scenario"]["network_backend"] = "analytical"
    cfg["scenario"]["dt_s"] = 2.0
    env = PaperUAVEnv(cfg, seed=61)
    assert env.dt == pytest.approx(2.0)

    legacy = PaperUAVEnv(load_config("masac", "f1_m5"), seed=61)
    assert legacy.dt == pytest.approx(float(legacy.assumed["dt_s"]))


def test_u6_reset_never_places_obstacles_on_mission_entities_or_each_other():
    env = analytical_u6(seed=0)
    env.reset(seed=0)

    for x, y, radius in env.obstacles:
        center = np.array([x, y], dtype=np.float64)
        assert x - radius >= 0.0
        assert y - radius >= 0.0
        assert x + radius <= env.area_size_m
        assert y + radius <= env.area_size_m
        assert np.linalg.norm(env.gcs_position[:2] - center) >= radius
        assert np.all(np.linalg.norm(env.positions[:, :2] - center, axis=1) >= radius)
        assert np.all(np.linalg.norm(env.targets - center, axis=1) >= radius)

    for i in range(len(env.obstacles)):
        for j in range(i + 1, len(env.obstacles)):
            separation = np.linalg.norm(env.obstacles[i, :2] - env.obstacles[j, :2])
            assert separation >= env.obstacles[i, 2] + env.obstacles[j, 2]


def test_u6_refresh_links_observes_topology_at_max_controllable_power(monkeypatch):
    env = analytical_u6(seed=62)
    env.reset(seed=62)
    original = env.network_backend.link_snapshot
    seen = {"tx_power_w": None}

    def spy(positions, gcs_position, obstacles=None, tx_power_w=None):
        seen["tx_power_w"] = tx_power_w
        return original(positions, gcs_position, obstacles)

    monkeypatch.setattr(env.network_backend, "link_snapshot", spy)
    env._refresh_links()

    assert seen["tx_power_w"] == pytest.approx(env.peer_tx_power_max_w)
