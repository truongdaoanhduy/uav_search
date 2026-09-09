import numpy as np

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


def test_peer_observation_appends_fixed_nine_cell_belief_patch():
    env = make_env(seed=5)
    env.reset(seed=5)
    env.positions[0, :2] = [2500.0, 2500.0]
    obs = env._observations()
    base_without_patch = 9 + (env.n_agents - 1) * 6 + env.n_targets + 5

    assert env.obs_dim == base_without_patch + 9
    patch = obs["uav_0"][-9:]
    profile = env._sensing_profile(0)
    assert np.count_nonzero(patch) == profile.fov_cells
