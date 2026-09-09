import numpy as np
import pytest

from uav_search.config import load_config
from uav_search.envs.paper_env import PaperUAVEnv, GCS_RECIPIENT
from uav_search.envs.network_backends import NetworkLinkSnapshot


def make_env(seed=44):
    cfg = load_config("masac", "u6")
    # Regression tests for the pre-UavNetSim peer behavior use the analytical backend.
    cfg["scenario"]["network_backend"] = "analytical"
    return PaperUAVEnv(cfg, seed=seed)


def idle_actions(env):
    return {a: np.array([-1.0, 0.0, 0.0, -1.0, -1.0, -1.0], dtype=np.float32) for a in env.agents}


def test_u6_uses_root_paper_world_but_six_identical_multirotors():
    env = make_env()
    obs, info = env.reset(seed=44)
    assert env.peer_mode is True
    assert env.n_agents == 6
    assert env.n_fixed == 0
    assert env.n_rotor == 6
    assert env.fixed_indices == []
    assert env.multirotor_indices == list(range(6))
    assert env.agent_types.tolist() == [1] * 6
    assert env.agents == [f"uav_{i}" for i in range(6)]
    assert env.area_size_m == 5000
    assert env.max_steps == 600
    assert env.action_dim == 6
    assert env.action_space.shape == (6,)
    assert set(np.unique(env.positions[:, 2])).issubset(set(map(float, env.scenario["altitude_levels_m"])))
    assert set(obs) == set(env.agents)
    assert info["simulation_backend"] == "root_paper_mpe_style"


def test_u6_seed_reproduces_targets_buildings_and_uav_initialization():
    a = make_env(seed=44)
    b = make_env(seed=44)
    a.reset(seed=44)
    b.reset(seed=44)
    np.testing.assert_allclose(a.positions, b.positions)
    np.testing.assert_allclose(a.targets, b.targets)
    np.testing.assert_allclose(a.obstacles, b.obstacles)


def test_peer_topology_has_no_leader_and_uses_all_uav_pairs():
    env = make_env()
    env.reset(seed=1)
    candidate = env._candidate_links()
    assert np.all(np.diag(candidate) == 0)
    assert candidate.sum() == env.n_agents * (env.n_agents - 1)
    assert np.all(env.rotor_leaders == -1)


def test_target_confirmation_creates_one_report_in_detecting_uav_buffer():
    env = make_env()
    env.reset(seed=3)
    env.targets[:] = [4900.0, 4900.0]
    env.obstacles[:, :2] = [4500.0, 4500.0]
    env.obstacles[:, 2] = 10.0
    env.positions[:, :2] = [3000.0, 3000.0]
    env.positions[0, :2] = [500.0, 500.0]
    env.targets[0] = [500.0, 500.0]
    env._refresh_links()

    env.step(idle_actions(env))

    assert env.target_found[0]
    assert env.report_generated[0]
    assert env.report_buffers[0, 0] == env.peer_report_bytes
    assert env.queue_bytes[0] == env.peer_report_bytes
    assert env.reports_delivered == 0


def test_tx_off_moves_no_report_bytes():
    env = make_env()
    env.reset(seed=4)
    env._enqueue_report(0, 0)
    before = env.report_buffers.copy()
    _, _, _, _, info = env.step(idle_actions(env))
    np.testing.assert_array_equal(env.report_buffers, before)
    assert info["bytes_transmitted"] == 0


def test_recipient_mapping_is_fixed_and_deterministic():
    env = make_env()
    env.reset(seed=5)
    values = [env._decode_peer_recipient(0, x) for x in (-1.0, -0.5, 0.0, 0.5, 0.999)]
    assert values == [1, 2, 4, 5, GCS_RECIPIENT]
    assert env._decode_peer_recipient(0, 0.0) == env._decode_peer_recipient(0, 0.0)


def test_received_data_cannot_be_forwarded_again_in_same_rl_step():
    env = make_env()
    env.reset(seed=6)
    env.positions[:, :2] = np.array(
        [[500, 500], [900, 500], [4000, 4000], [4200, 4000], [4400, 4000], [4600, 4000]],
        dtype=float,
    )
    env.gcs_position = np.array([1300.0, 500.0, 0.0])
    env._refresh_links()
    env._enqueue_report(0, 0)

    actions = idle_actions(env)
    # For UAV0, recipient scalar -1 maps to UAV1.
    actions["uav_0"] = np.array([-1.0, 0.0, 0.0, 1.0, 1.0, -1.0], dtype=np.float32)
    # UAV1 tries to forward to the GCS in the same slot.
    actions["uav_1"] = np.array([-1.0, 0.0, 0.0, 1.0, 1.0, 0.999], dtype=np.float32)

    env.step(actions)
    assert env.report_buffers[0, 1] > 0
    assert env.report_delivered_bytes[0] == 0


def test_gcs_delivery_requires_valid_paper_rate_and_reports_metric():
    env = make_env()
    env.reset(seed=7)
    env.positions[:, :2] = np.array(
        [[2500, 2500], [4900, 4900], [4700, 4900], [4900, 4700], [4500, 4900], [4900, 4500]],
        dtype=float,
    )
    env.gcs_position = np.array([2500.0, 2500.0, 0.0])
    env._refresh_links()
    env._enqueue_report(0, 0)
    actions = idle_actions(env)
    actions["uav_0"] = np.array([-1.0, 0.0, 0.0, 1.0, 1.0, 0.999], dtype=np.float32)

    _, _, _, _, info = env.step(actions)
    assert info["bytes_transmitted"] > 0
    # u6 uses a 2 Mbps nominal link and a 1 MB report, so delivery spans
    # multiple 1-second macro-steps rather than teleporting the full report.
    for _ in range(40):
        if info["reports_delivered"] == 1:
            break
        _, _, _, _, info = env.step(actions)
    assert info["reports_delivered"] == 1
    assert info["mission_delivery_rate"] > 0


def test_u6_config_is_available_without_changing_legacy_paper_scenarios():
    cfg = load_config("masac", "u6")
    assert cfg["scenario"]["architecture"] == "homogeneous_peer"
    assert cfg["scenario"]["num_uavs"] == 6
    legacy = load_config("masac", "f1_m5")
    assert legacy["scenario"]["fixed_wing"] == 1
    assert legacy["scenario"]["multirotor"] == 5


def test_report_is_not_partially_created_when_buffer_cannot_hold_it():
    cfg = load_config("masac", "u6")
    cfg["scenario"]["network_backend"] = "analytical"
    cfg["scenario"]["buffer_bytes"] = 500_000
    cfg["scenario"]["report_bytes"] = 1_000_000
    env = PaperUAVEnv(cfg, seed=8)
    env._enqueue_report(0, 0)
    assert not env.report_generated[0]
    assert env.report_buffers[0, 0] == 0
    assert env.queue_bytes[0] == 0


def test_minimum_tx_power_is_a_real_transmission_not_old_zero_amount_encoding():
    env = make_env()
    env.reset(seed=9)
    env.positions[0, :2] = env.gcs_position[:2]
    env._refresh_links()
    env._enqueue_report(0, 0)
    actions = idle_actions(env)
    # Action coordinate 4 is RF power; -1 maps to the configured 0.1 W minimum.
    actions["uav_0"] = np.array([-1.0, 0.0, 0.0, 1.0, -1.0, 0.999], dtype=np.float32)

    _, _, _, _, info = env.step(actions)

    assert env._decode_tx_power_w(-1.0) == pytest.approx(0.1)
    assert info["bytes_transmitted"] > 0


def test_u6_defaults_to_uavnetsim_backend():
    cfg = load_config("masac", "u6")
    assert cfg["scenario"]["network_backend"] == "uavnetsim"


def test_peer_env_uses_configured_network_backend_and_exposes_network_metrics():
    cfg = load_config("masac", "u6")
    cfg["scenario"]["network_backend"] = "analytical"
    env = PaperUAVEnv(cfg, seed=44)
    _, info = env.reset(seed=44)
    assert env.network_backend.name == "analytical"
    assert info["network_backend"] == "analytical"
    assert info["network_attempted_bytes"] == 0
    assert info["network_delivered_bytes"] == 0
    assert info["network_byte_pdr"] == 0.0
    assert info["network_throughput_bps"] == 0.0
    assert info["network_mean_delay_s"] == 0.0
    assert info["network_phy_failures"] == 0
    assert info["network_tx_energy_j"] == 0.0


def test_peer_transmission_metrics_come_from_configured_backend():
    env = make_env(seed=17)
    env.reset(seed=17)
    env.positions[:, :2] = np.array(
        [[2500, 2500], [2600, 2500], [4700, 4700], [4800, 4700], [4700, 4800], [4800, 4800]],
        dtype=float,
    )
    env.gcs_position = np.array([2500.0, 2500.0, 0.0])
    env.obstacles[:, :2] = [4900.0, 4900.0]
    env.obstacles[:, 2] = 10.0
    env._refresh_links()
    env._enqueue_report(0, 0)
    actions = idle_actions(env)
    actions["uav_0"] = np.array([-1.0, 0.0, 0.0, 1.0, 1.0, 0.999], dtype=np.float32)

    _, _, _, _, info = env.step(actions)

    assert info["network_backend"] == "analytical"
    assert info["network_attempted_bytes"] > 0
    assert info["network_delivered_bytes"] > 0
    assert info["network_byte_pdr"] > 0.0
    assert info["network_throughput_bps"] > 0.0
    assert info["network_tx_energy_j"] > 0.0


def test_u6_research_scenario_uses_edge_gcs_and_longer_horizon_only_for_u6():
    cfg = load_config("masac", "u6")
    env = PaperUAVEnv({**cfg, "scenario": {**cfg["scenario"], "network_backend": "analytical"}}, seed=44)
    assert env.max_steps == 600
    np.testing.assert_allclose(env.gcs_position, [0.0, 2500.0, 0.0])

    legacy = load_config("masac", "f1_m5")
    legacy_env = PaperUAVEnv(legacy, seed=44)
    assert legacy_env.max_steps == 50


def test_u6_launch_zone_initialization_uses_safe_common_gcs_launch_pads():
    env = make_env(seed=44)
    env.reset(seed=44)
    radius = float(env.scenario["launch_radius_m"])
    min_sep = float(env.scenario["initial_min_separation_m"])

    radial = np.linalg.norm(env.positions[:, :2] - env.gcs_position[:2][None, :], axis=1)
    assert np.all(radial <= radius + 1e-9)
    assert np.all(env.positions[:, 0] >= env.gcs_position[0] - 1e-9)
    for i in range(env.n_agents):
        for j in range(i + 1, env.n_agents):
            assert np.linalg.norm(env.positions[i, :2] - env.positions[j, :2]) >= min_sep - 1e-9

    env2 = make_env(seed=44)
    env2.reset(seed=44)
    np.testing.assert_allclose(env.positions, env2.positions)



def test_peer_observation_hides_live_remote_state_and_uses_stale_cache():
    env = make_env(seed=21)
    env.reset(seed=21)
    # Put UAV0/UAV1 in contact and all other peers far away.
    env.positions[:, :2] = np.array(
        [[300, 2500], [600, 2500], [3000, 3000], [3500, 3500], [4000, 4000], [4500, 4500]],
        dtype=float,
    )
    env.obstacles[:, :2] = [4900.0, 4900.0]
    env.obstacles[:, 2] = 10.0
    env._refresh_links()
    first = env._observations()["uav_0"].copy()
    # Neighbor UAV1 occupies the first six-value neighbor slot.
    slot = first[9:15]
    assert slot[-1] > 0.0
    cached_x = float(slot[1])

    # Break contact and change UAV1's ground-truth position dramatically.
    env.positions[1, :2] = [4900.0, 100.0]
    env.step_count += 1
    env._refresh_links()
    second = env._observations()["uav_0"]
    stale_slot = second[9:15]
    assert stale_slot[-1] < slot[-1]
    assert stale_slot[-1] >= 0.0
    # Cached x remains the last exchanged value, not the new global truth.
    assert stale_slot[1] == pytest.approx(cached_x)
    assert stale_slot[1] != pytest.approx(env.positions[1, 0] / env.area_size_m)


def test_target_discovery_is_local_knowledge_until_report_reaches_peer():
    env = make_env(seed=22)
    env.reset(seed=22)
    env.targets[:] = [4900.0, 4900.0]
    env.targets[0] = [500.0, 500.0]
    env.positions[:, :2] = [3000.0, 3000.0]
    env.positions[0, :2] = [500.0, 500.0]
    env.positions[1, :2] = [900.0, 500.0]
    env.obstacles[:, :2] = [4900.0, 4900.0]
    env.obstacles[:, 2] = 10.0
    env._refresh_links()

    env.step(idle_actions(env))
    assert env.target_found[0]
    assert env.target_known_by_agent[0, 0]
    assert not env.target_known_by_agent[1, 0]

    actions = idle_actions(env)
    actions["uav_0"] = np.array([-1.0, 0.0, 0.0, 1.0, 1.0, -1.0], dtype=np.float32)
    env.step(actions)
    assert env.target_known_by_agent[1, 0]


def test_detected_report_retries_after_buffer_space_becomes_available():
    cfg = load_config("masac", "u6")
    cfg["scenario"]["network_backend"] = "analytical"
    cfg["scenario"]["buffer_bytes"] = 1_000_000
    cfg["scenario"]["report_bytes"] = 1_000_000
    env = PaperUAVEnv(cfg, seed=23)
    env.reset(seed=23)
    # Fill UAV0's only report slot with target 1.
    env._enqueue_report(0, 1)
    env.targets[:] = [4900.0, 4900.0]
    env.targets[0] = env.positions[0, :2]
    env.step(idle_actions(env))
    assert env.target_found[0]
    assert not env.report_generated[0]
    assert env.pending_report_source[0] == 0

    # Free the buffer and verify the pending report is created on a later step.
    env.report_buffers[1, 0] = 0
    env.queue_bytes[0] = 0
    env.step(idle_actions(env))
    assert env.report_generated[0]
    assert env.report_buffers[0, 0] == env.peer_report_bytes


def test_bad_selected_link_is_counted_as_network_attempt_not_filtered_by_env():
    env = make_env(seed=24)
    env.reset(seed=24)
    env.positions[:, :2] = np.array(
        [[100, 100], [4900, 4900], [3000, 3000], [3200, 3200], [3400, 3400], [3600, 3600]],
        dtype=float,
    )
    env.obstacles[:, :2] = [2500.0, 2500.0]
    env.obstacles[:, 2] = 10.0
    env._refresh_links()
    env._enqueue_report(0, 0)
    actions = idle_actions(env)
    # sender 0 -> UAV1 (recipient scalar -1) across an invalid/far link.
    actions["uav_0"] = np.array([-1.0, 0.0, 0.0, 1.0, 1.0, -1.0], dtype=np.float32)
    _, _, _, _, info = env.step(actions)
    assert info["network_attempted_bytes"] > 0
    assert info["network_delivered_bytes"] == 0


def test_peer_info_distinguishes_direct_multihop_and_disconnected_to_gcs():
    env = make_env(seed=25)
    env.reset(seed=25)
    env.positions[:, :2] = np.array(
        [[300, 2500], [1200, 2500], [2100, 2500], [4000, 4000], [4300, 4300], [4600, 4600]],
        dtype=float,
    )
    env.gcs_position = np.array([0.0, 2500.0, 0.0])
    env.obstacles[:, :2] = [4900.0, 100.0]
    env.obstacles[:, 2] = 10.0
    env._refresh_links()
    info = env._info(step_energy_j=0.0, action_saturation=0.0)
    assert info["direct_gcs_uavs"] >= 1
    assert info["multihop_gcs_uavs"] >= 1
    assert info["disconnected_gcs_uavs"] >= 1
    assert info["direct_gcs_uavs"] + info["multihop_gcs_uavs"] + info["disconnected_gcs_uavs"] == env.n_agents



def test_u6_runtime_step_override_takes_precedence_over_research_default():
    cfg = load_config("masac", "u6")
    cfg["scenario"]["network_backend"] = "analytical"
    cfg["runtime"]["episode_steps_override"] = 8
    env = PaperUAVEnv(cfg, seed=44)
    assert env.max_steps == 8


def test_generated_report_expires_after_ttl_and_frees_buffer():
    cfg = load_config("masac", "u6")
    cfg["scenario"]["network_backend"] = "analytical"
    cfg["scenario"]["report_ttl_s"] = 2.0
    env = PaperUAVEnv(cfg, seed=26)
    env.reset(seed=26)
    env._enqueue_report(0, 0)
    assert env.queue_bytes[0] == env.peer_report_bytes
    env.step_count = 2
    env._expire_reports()
    assert env.report_expired[0]
    assert env.queue_bytes[0] == 0
    assert env.report_buffers[0].sum() == 0
    assert env.expired_reports == 1


def test_neighbor_cache_respects_directed_receiver_transmitter_semantics(monkeypatch):
    env = make_env(seed=27)
    env.reset(seed=27)
    env.neighbor_cache_seen_step.fill(-1)
    adjacency = np.zeros((env.n_agents, env.n_agents), dtype=np.int8)
    # Only UAV0 -> UAV1 is usable: adjacency[receiver=1, transmitter=0] = 1.
    adjacency[1, 0] = 1
    pair_rates = adjacency.astype(float) * (float(env.paper["min_comm_rate_bps"]) + 1.0)
    snapshot = NetworkLinkSnapshot(
        pair_rates_bps=pair_rates,
        gcs_rates_bps=np.zeros(env.n_agents, dtype=float),
        adjacency=adjacency,
    )
    monkeypatch.setattr(env.network_backend, "link_snapshot", lambda *args, **kwargs: snapshot)
    env._refresh_links()
    assert env.neighbor_cache_seen_step[1, 0] == env.step_count
    assert env.neighbor_cache_seen_step[0, 1] == -1


def test_gcs_hops_do_not_use_reverse_only_peer_edge():
    env = make_env(seed=28)
    env.reset(seed=28)
    env.last_adjacency.fill(0)
    env.last_gcs_rates_bps.fill(0.0)
    rmin = float(env.paper["min_comm_rate_bps"])
    env.last_gcs_rates_bps[0] = rmin + 1.0  # UAV0 -> GCS direct

    # Reverse-only edge UAV0 -> UAV1 cannot help UAV1 forward toward UAV0/GCS.
    env.last_adjacency[1, 0] = 1
    hops = env._gcs_hops()
    assert hops[0] == 1
    assert hops[1] == -1

    # Add the usable forwarding direction UAV1 -> UAV0.
    env.last_adjacency[1, 0] = 0
    env.last_adjacency[0, 1] = 1
    hops = env._gcs_hops()
    assert hops[1] == 2


def test_network_tx_energy_reduces_sender_battery_in_u6():
    cfg = load_config("masac", "u6")
    cfg["scenario"]["network_backend"] = "analytical"
    tx_env = PaperUAVEnv(cfg, seed=29)
    idle_env = PaperUAVEnv(cfg, seed=29)
    for env in (tx_env, idle_env):
        env.reset(seed=29)
        env.positions[:, :2] = np.array(
            [[100.0, 2500.0], [3000.0, 3000.0], [3200.0, 3200.0],
             [3400.0, 3400.0], [3600.0, 3600.0], [3800.0, 3800.0]],
            dtype=float,
        )
        env.gcs_position = np.array([0.0, 2500.0, 0.0])
        env.obstacles[:, :2] = [4900.0, 100.0]
        env.obstacles[:, 2] = 10.0
        env._refresh_links()
        env._enqueue_report(0, 0)

    tx_actions = idle_actions(tx_env)
    tx_actions["uav_0"] = np.array([-1.0, 0.0, 0.0, 1.0, 1.0, 0.999], dtype=np.float32)
    tx_env.step(tx_actions)
    idle_env.step(idle_actions(idle_env))

    assert tx_env.last_network_result.tx_energy_j > 0.0
    assert tx_env.last_energy_by_agent_j[0] > idle_env.last_energy_by_agent_j[0]
    assert tx_env.battery_pct[0] < idle_env.battery_pct[0]


def test_uavnetsim_ack_energy_reduces_receiver_battery():
    import importlib.util

    if importlib.util.find_spec("simpy") is None or importlib.util.find_spec("phy.channel") is None:
        pytest.skip("pinned UavNetSim optional dependency is not installed")

    cfg = load_config("masac", "u6")
    tx_env = PaperUAVEnv(cfg, seed=93)
    idle_env = PaperUAVEnv(cfg, seed=93)
    for env in (tx_env, idle_env):
        env.reset(seed=93)
        env.positions[:, :2] = np.array(
            [[100.0, 100.0], [150.0, 100.0], [4500.0, 4500.0],
             [4600.0, 4500.0], [4500.0, 4600.0], [4600.0, 4600.0]],
            dtype=float,
        )
        env.gcs_position = np.array([200.0, 100.0, 0.0])
        env.targets[:] = [4000.0, 4000.0]
        env.obstacles[:, :2] = [4900.0, 4900.0]
        env.obstacles[:, 2] = 10.0
        env._refresh_links()
        env._enqueue_report(0, 0)

    tx_actions = idle_actions(tx_env)
    # Sender 0: gate on, minimum RF power, first recipient bin -> UAV 1.
    tx_actions["uav_0"] = np.array([-1.0, 0.0, 0.0, 1.0, -1.0, -1.0], dtype=np.float32)
    tx_env.step(tx_actions)
    idle_env.step(idle_actions(idle_env))

    assert tx_env.last_network_result.delivered_bytes > 0
    assert tx_env.last_network_result.node_tx_energy_j[1] > 0.0
    assert tx_env.last_energy_by_agent_j[1] > idle_env.last_energy_by_agent_j[1]
    assert tx_env.battery_pct[1] < idle_env.battery_pct[1]


def test_u6_reset_seed_is_forwarded_to_uavnetsim_backend():
    import importlib.util

    if importlib.util.find_spec("simpy") is None or importlib.util.find_spec("phy.channel") is None:
        pytest.skip("pinned UavNetSim optional dependency is not installed")

    cfg = load_config("masac", "u6")
    env = PaperUAVEnv(cfg, seed=44)
    env.reset(seed=101)
    assert env.network_backend.seed == 101
    env.reset(seed=202)
    assert env.network_backend.seed == 202
