import numpy as np
import pytest

from uav_search.config import load_config
from uav_search.envs.paper_env import PaperUAVEnv


def make_env(seed=44):
    cfg = load_config("masac", "u6")
    cfg["scenario"]["network_backend"] = "analytical"
    return PaperUAVEnv(cfg, seed=seed)


def idle_actions(env):
    return {
        agent: np.array([-1.0, 0.0, 0.0, -1.0, -1.0, -1.0], dtype=np.float32)
        for agent in env.agents
    }


def put_target_in_uav_cell(env, agent_idx=0, target_idx=0):
    cell = float(env.scenario["sensing_grid_cell_m"])
    x = np.floor(env.positions[agent_idx, 0] / cell) * cell + 0.5 * cell
    y = np.floor(env.positions[agent_idx, 1] / cell) * cell + 0.5 * cell
    env.targets[target_idx] = [x, y]
    return env._target_grid_cell(target_idx)


def test_u6_initializes_independent_half_probability_belief_maps():
    env = make_env(seed=1)
    env.reset(seed=1)

    assert env.belief_maps.shape == (env.n_agents, 50, 50)
    np.testing.assert_allclose(env.belief_maps, 0.5)
    assert env.last_sensor_positive.shape == (env.n_agents, env.n_targets)
    assert not env.last_sensor_positive.any()


def test_one_close_scan_does_not_use_old_dfound_shortcut():
    env = make_env(seed=2)
    env.reset(seed=2)
    env.positions[0, 2] = 50.0
    put_target_in_uav_cell(env, 0, 0)
    env.positions[0, :2] = env.targets[0]
    env.assumed["target_found_m"] = 10_000.0  # would force the legacy rule to succeed

    env.step(idle_actions(env))

    assert not env.target_found[0]
    assert not env.report_generated[0]


def test_same_seed_reproduces_sensor_samples_and_belief_updates():
    a = make_env(seed=77)
    b = make_env(seed=77)
    a.reset(seed=77)
    b.reset(seed=77)
    for env in (a, b):
        env.positions[:, 2] = 150.0
        env.positions[:, :2] = [2500.0, 2500.0]
        env.targets[:] = [4950.0, 4950.0]
        env.targets[0] = [2550.0, 2550.0]

    a.step(idle_actions(a))
    b.step(idle_actions(b))

    np.testing.assert_array_equal(a.last_sensor_positive, b.last_sensor_positive)
    np.testing.assert_allclose(a.belief_maps, b.belief_maps)
    np.testing.assert_allclose(a.last_information_gain_by_agent, b.last_information_gain_by_agent)


def test_confirmation_threshold_creates_exactly_one_report_source():
    env = make_env(seed=3)
    env.reset(seed=3)
    target_cell = put_target_in_uav_cell(env, 0, 0)
    cy, cx = target_cell
    env.belief_maps[:, cy, cx] = 0.1
    env.belief_maps[0, cy, cx] = float(env.scenario["target_confirmation_threshold"])
    env.fine_positive_cells_by_agent[0, cy, cx] = True
    env.fine_target_evidence_by_agent[0, 0] = True

    env._confirm_peer_targets()
    env._retry_pending_reports()
    first_queue = int(env.queue_bytes.sum())
    env._confirm_peer_targets()
    env._retry_pending_reports()

    assert env.target_found[0]
    assert env.pending_report_source[0] == 0
    assert env.report_generated[0]
    assert first_queue == env.peer_report_bytes
    assert int(env.queue_bytes.sum()) == first_queue
    assert env.last_new_targets_by_agent[0] == 1


def test_unsensed_target_distance_is_not_exposed_even_inside_legacy_detect_radius():
    env = make_env(seed=4)
    env.reset(seed=4)
    env.positions[0, :2] = [1000.0, 1000.0]
    env.positions[0, 2] = 50.0
    env.targets[0] = [1300.0, 1000.0]  # < legacy 700m, outside low-level 1-cell FOV
    env.last_sensor_positive.fill(False)
    env.target_known_by_agent.fill(False)

    assert env._target_observation_distance(0, 0) == 0.0


def test_u6_continuous_sensing_uses_actual_altitude_between_reference_levels():
    env = make_env(seed=45)
    env.reset(seed=45)
    env.positions[0, :2] = [2500.0, 2500.0]
    env.positions[0, 2] = 75.0

    profile = env._sensing_profile(0)

    assert profile.pd == pytest.approx(0.85)
    assert profile.pf == pytest.approx(0.15)
    assert profile.fov_radius_m == pytest.approx(75.0)
    assert len(env._sensing_cells(0)) == 4

    env.positions[0, 2] = 125.0
    profile = env._sensing_profile(0)
    assert profile.pd == pytest.approx(0.75)
    assert profile.pf == pytest.approx(0.25)
    assert profile.fov_radius_m == pytest.approx(125.0)
    assert len(env._sensing_cells(0)) == 12

    env.positions[0, 2] = 150.0
    assert len(env._sensing_cells(0)) == 16


def test_peer_observation_appends_fixed_nine_cell_belief_patch():
    env = make_env(seed=5)
    env.reset(seed=5)
    env.positions[0, :2] = [2500.0, 2500.0]
    obs = env._observations()

    assert obs["uav_0"].shape == (env.obs_dim,)
    patch = obs["uav_0"][-9:]
    assert patch.shape == (9,)
    assert np.all((patch >= 0.0) & (patch <= 1.0))


def test_u6_info_exposes_scalar_altitude_and_belief_sensing_diagnostics():
    env = make_env(seed=88)
    _, info0 = env.reset(seed=88)

    expected_keys = {
        "mean_altitude_m",
        "min_altitude_m",
        "max_altitude_m",
        "mean_fov_radius_m",
        "mean_detection_probability",
        "mean_false_alarm_probability",
        "mean_belief_entropy",
        "mean_target_posterior",
        "scanned_cells_step",
        "scanned_cells_total",
        "positive_sensor_observations_step",
        "positive_sensor_observations_total",
        "information_gain_step",
        "information_gain_total",
        "targets_confirmed_step",
        "targets_confirmed_total",
    }
    assert expected_keys.issubset(info0)
    assert info0["mean_belief_entropy"] == pytest.approx(1.0)
    assert info0["mean_target_posterior"] == pytest.approx(0.5)
    assert info0["scanned_cells_total"] == 0
    assert info0["targets_confirmed_total"] == 0

    _, _, _, _, info1 = env.step(idle_actions(env))
    for key in expected_keys:
        assert np.isfinite(float(info1[key]))
    assert info1["scanned_cells_step"] > 0
    assert info1["scanned_cells_total"] >= info1["scanned_cells_step"]
    assert info1["information_gain_total"] >= info1["information_gain_step"] >= 0.0


def test_minimum_uncertainty_fusion_updates_only_successful_sync_receiver():
    env = make_env(seed=90)
    env.reset(seed=90)
    y, x = 10, 10
    env.belief_maps[:, y, x] = [0.60, 0.90, 0.70, 0.55, 0.65, 0.80]
    sender_map = env.belief_maps[1].copy()

    env._fuse_received_peer_belief(receiver=0, sender_belief=sender_map)

    assert env.belief_maps[0, y, x] == pytest.approx(0.90)
    np.testing.assert_allclose(env.belief_maps[1:, y, x], [0.90, 0.70, 0.55, 0.65, 0.80])


def test_empty_high_posterior_cell_is_recorded_as_false_confirmation():
    env = make_env(seed=91)
    env.reset(seed=91)
    target_cells = {env._target_grid_cell(k) for k in range(env.n_targets)}
    empty_cell = next((y, x) for y in range(env.peer_sensing_grid_n) for x in range(env.peer_sensing_grid_n)
                      if (y, x) not in target_cells)
    y, x = empty_cell
    env.belief_maps[:, y, x] = 0.5
    env.belief_maps[2, y, x] = env.peer_target_confirmation_threshold + 1e-4
    env.fine_positive_cells_by_agent[2, y, x] = True

    env._confirm_peer_targets()

    assert env.confirmed_cells[y, x]
    assert env.false_confirmed_cells[y, x]
    assert env.total_false_confirmations == 1
    assert not env.target_found.any()


def test_true_target_confirmation_is_driven_by_confirmed_cell_and_tracks_source():
    env = make_env(seed=92)
    env.reset(seed=92)
    y, x = put_target_in_uav_cell(env, agent_idx=0, target_idx=0)
    env.belief_maps[:, y, x] = 0.5
    env.belief_maps[3, y, x] = env.peer_target_confirmation_threshold + 1e-4
    env.fine_positive_cells_by_agent[3, y, x] = True
    env.fine_target_evidence_by_agent[3, 0] = True

    env._confirm_peer_targets()

    assert env.confirmed_cells[y, x]
    assert not env.false_confirmed_cells[y, x]
    assert env.target_found[0]
    assert env.pending_report_source[0] == 3
    assert env.target_known_by_agent[3, 0]


def test_inactive_uav_belief_cannot_drive_fusion_or_target_confirmation():
    env = make_env(seed=93)
    env.reset(seed=93)
    y, x = put_target_in_uav_cell(env, agent_idx=0, target_idx=0)
    env.belief_maps[:, y, x] = 0.5
    env.belief_maps[0, y, x] = env.peer_target_confirmation_threshold + 1e-4
    env.fine_positive_cells_by_agent[0, y, x] = True
    env.fine_target_evidence_by_agent[0, 0] = True
    env.uav_active[0] = False
    env.battery_pct[0] = 0.0

    env._confirm_peer_targets()

    assert not env.target_found[0]
    assert env.belief_source_map[y, x] != 0
    assert env.belief_maps[0, y, x] == pytest.approx(env.peer_target_confirmation_threshold + 1e-4)
    np.testing.assert_allclose(env.belief_maps[1:, y, x], 0.5)


def test_no_target_or_cell_confirmation_occurs_when_all_uavs_are_inactive():
    env = make_env(seed=94)
    env.reset(seed=94)
    y, x = put_target_in_uav_cell(env, agent_idx=0, target_idx=0)
    env.belief_maps[:, y, x] = env.peer_target_confirmation_threshold + 1e-4
    env.uav_active[:] = False
    env.battery_pct[:] = 0.0

    env._confirm_peer_targets()

    assert not env.confirmed_cells[y, x]
    assert not env.target_found[0]
    assert env.belief_source_map[y, x] == -1
