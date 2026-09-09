import numpy as np

from uav_search.config import load_config
from uav_search.envs.paper_env import PaperUAVEnv


def make_u6(seed=44):
    cfg = load_config("masac", "u6")
    cfg["scenario"]["network_backend"] = "analytical"
    return PaperUAVEnv(cfg, seed=seed)


def idle_actions(env):
    return {
        agent: np.array([-1.0, 0.0, 0.0, -1.0, -1.0, -1.0], dtype=np.float32)
        for agent in env.agents
    }


def test_u6_uses_six_dimensional_action_but_paper_scenario_stays_two_dimensional():
    u6 = make_u6()
    paper = PaperUAVEnv(load_config("masac", "f1_m5"), seed=44)

    assert u6.action_dim == 6
    assert u6.action_space.shape == (6,)
    assert paper.action_dim == 2
    assert paper.action_space.shape == (2,)


def test_u6_initial_altitudes_are_seeded_reference_levels():
    a = make_u6(seed=44)
    b = make_u6(seed=44)
    a.reset(seed=44)
    b.reset(seed=44)

    levels = np.asarray(a.scenario["altitude_levels_m"], dtype=float)
    assert set(np.unique(a.positions[:, 2])).issubset(set(levels.tolist()))
    np.testing.assert_allclose(a.positions[:, 2], b.positions[:, 2])
    assert a.velocities.shape == (a.n_agents, 3)


def test_positive_vertical_action_changes_altitude_and_z_velocity():
    env = make_u6(seed=1)
    env.reset(seed=1)
    env.positions[0, 2] = 100.0
    env.velocities[0] = 0.0
    actions = idle_actions(env)
    actions["uav_0"] = np.array([-1.0, 0.0, 1.0, -1.0, -1.0, -1.0], dtype=np.float32)

    before_xy = env.positions[0, :2].copy()
    env.step(actions)

    np.testing.assert_allclose(env.positions[0, :2], before_xy, atol=1e-9)
    assert env.positions[0, 2] > 100.0
    assert env.velocities[0, 2] > 0.0


def test_vertical_motion_is_clipped_to_configured_altitude_bounds():
    env = make_u6(seed=2)
    env.reset(seed=2)
    env.positions[0, 2] = float(env.scenario["altitude_max_m"])
    env.velocities[0] = 0.0
    actions = idle_actions(env)
    actions["uav_0"] = np.array([-1.0, 0.0, 1.0, -1.0, -1.0, -1.0], dtype=np.float32)

    _, _, _, _, info = env.step(actions)

    assert env.positions[0, 2] == float(env.scenario["altitude_max_m"])
    assert info["boundary_hits"] >= 1


def test_peer_motion_rejects_candidate_that_would_violate_safe_distance():
    env = make_u6(seed=93)
    env.reset(seed=93)
    env.obstacles[:, :2] = [4900.0, 4900.0]
    env.obstacles[:, 2] = 1.0
    env.positions[:] = np.array([
        [1000.0, 1000.0, 100.0],
        [1160.0, 1000.0, 100.0],
        [2500.0, 2500.0, 100.0],
        [3000.0, 3000.0, 100.0],
        [3500.0, 3500.0, 100.0],
        [4000.0, 4000.0, 100.0],
    ])
    env.velocities[:] = 0.0
    env.velocities[0, 0] = 10.0
    env.velocities[1, 0] = -10.0
    before = env.positions.copy()

    env.step(idle_actions(env))

    np.testing.assert_allclose(env.positions[0], before[0])
    np.testing.assert_allclose(env.positions[1], before[1])
    np.testing.assert_allclose(env.velocities[0], 0.0)
    np.testing.assert_allclose(env.velocities[1], 0.0)
    assert np.linalg.norm(env.positions[0] - env.positions[1]) >= env.safety_distance_m


def test_peer_safety_filter_rechecks_after_pair_rollback():
    env = make_u6(seed=94)
    env.reset(seed=94)
    env.obstacles[:, :2] = [4900.0, 4900.0]
    env.obstacles[:, 2] = 1.0
    env.positions[:] = np.array([
        [1000.0, 1000.0, 100.0],
        [1150.0, 1000.0, 100.0],
        [1300.0, 1000.0, 100.0],
        [2500.0, 2500.0, 100.0],
        [3500.0, 3500.0, 100.0],
        [4500.0, 4500.0, 100.0],
    ])
    env.velocities[:] = 0.0
    env.velocities[0, 0] = 10.0
    env.velocities[1, 0] = -10.0
    env.velocities[2, 0] = -10.0

    env.step(idle_actions(env))

    for i in range(env.n_agents):
        for j in range(i + 1, env.n_agents):
            assert np.linalg.norm(env.positions[i] - env.positions[j]) > env.safety_distance_m
