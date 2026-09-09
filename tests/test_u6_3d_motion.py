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
