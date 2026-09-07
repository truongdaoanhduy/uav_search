import numpy as np

from uav_search.config import load_config
from uav_search.envs.paper_env import PaperUAVEnv, GCS_RECIPIENT


def make_env(seed=44):
    return PaperUAVEnv(load_config("masac", "u6"), seed=seed)


def idle_actions(env):
    return {a: np.array([-1.0, 0.0, -1.0, -1.0, -1.0], dtype=np.float32) for a in env.agents}


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
    assert env.max_steps == 50
    assert env.action_dim == 5
    assert env.action_space.shape == (5,)
    assert np.allclose(env.positions[:, 2], env.assumed["multirotor_altitude_m"])
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
    actions["uav_0"] = np.array([-1.0, 0.0, 1.0, 1.0, -1.0], dtype=np.float32)
    # UAV1 tries to forward to the GCS in the same slot.
    actions["uav_1"] = np.array([-1.0, 0.0, 1.0, 1.0, 0.999], dtype=np.float32)

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
    actions["uav_0"] = np.array([-1.0, 0.0, 1.0, 1.0, 0.999], dtype=np.float32)

    _, _, _, _, info = env.step(actions)

    assert info["bytes_transmitted"] > 0
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
    cfg["scenario"]["buffer_bytes"] = 500_000
    cfg["scenario"]["report_bytes"] = 1_000_000
    env = PaperUAVEnv(cfg, seed=8)
    env._enqueue_report(0, 0)
    assert not env.report_generated[0]
    assert env.report_buffers[0, 0] == 0
    assert env.queue_bytes[0] == 0


def test_zero_tx_amount_cannot_farm_positive_communication_reward():
    env = make_env()
    env.reset(seed=9)
    env.positions[0, :2] = env.gcs_position[:2]
    env._refresh_links()
    env._enqueue_report(0, 0)
    actions = idle_actions(env)
    actions["uav_0"] = np.array([-1.0, 0.0, 1.0, -1.0, 0.999], dtype=np.float32)

    _, _, _, _, info = env.step(actions)

    assert info["bytes_transmitted"] == 0
    assert info["reward_components_sum"]["communication"] == 0.0
