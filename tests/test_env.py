import numpy as np

from uav_search.config import load_config
from uav_search.envs.models import communication_rate_bps, multirotor_power_w
from uav_search.envs.paper_env import PaperUAVEnv


def make_env(scenario="f1_m5", seed=7):
    return PaperUAVEnv(load_config("masac", scenario), seed=seed)


def test_seeded_reset_is_reproducible_and_shapes_match():
    a = make_env(seed=42)
    b = make_env(seed=42)
    oa, ia = a.reset(seed=42)
    ob, ib = b.reset(seed=42)
    assert a.n_agents == 6
    assert len(oa) == 6
    assert a.action_dim == 2
    assert a.obs_dim > 20
    for agent in a.agents:
        assert oa[agent].shape == (a.obs_dim,)
        assert np.allclose(oa[agent], ob[agent])
    assert np.allclose(a.positions, b.positions)
    assert np.allclose(a.targets, b.targets)
    assert np.allclose(a.obstacles, b.obstacles)
    assert ia["targets_total"] == 10


def test_step_returns_finite_parallel_outputs_and_metrics():
    env = make_env()
    obs, _ = env.reset(seed=1)
    actions = {agent: np.zeros(2, dtype=np.float32) for agent in env.agents}
    nxt, rewards, terminated, truncated, info = env.step(actions)
    assert set(nxt) == set(env.agents)
    assert all(np.isfinite(v).all() for v in nxt.values())
    assert all(np.isfinite(v) for v in rewards.values())
    assert all(v is False for v in terminated.values())
    assert all(v is False for v in truncated.values())
    assert info["mean_comm_rate_mbps"] >= 0
    assert info["energy_used_j"] >= 0
    assert 0 <= info["search_rate"] <= 1
    assert info["broken_links"] >= 0
    assert np.all(env.positions[:, :2] >= 0)
    assert np.all(env.positions[:, :2] <= env.area_size_m)


def test_target_is_confirmed_when_multirotor_enters_found_radius():
    env = make_env()
    env.reset(seed=3)
    rotor_idx = env.multirotor_indices[0]
    env.assumed["search_reward_coeff"] = 1000.0
    env.targets[:] = np.array([4900.0, 4900.0])
    env.targets[0] = np.array([500.0, 500.0])
    env.positions[rotor_idx, :2] = env.targets[0]
    env.positions[env.fixed_indices[0], :2] = np.array([500.0, 1000.0])
    for k, idx in enumerate(env.multirotor_indices[1:], start=1):
        env.positions[idx, :2] = np.array([2500.0 + 250.0 * k, 2500.0])
    env.obstacles[:, :2] = np.array([4500.0, 4500.0])
    env.obstacles[:, 2] = 10.0
    env._refresh_links()
    actions = {agent: np.zeros(2, dtype=np.float32) for agent in env.agents}
    _, rewards, _, _, info = env.step(actions)
    assert env.target_found[0]
    assert info["targets_found"] >= 1
    # The paper's Eq. (24) awards zeta when a multi-rotor confirms a target.
    assert rewards[env.agents[rotor_idx]] > 500.0


def test_episode_truncates_at_paper_length():
    env = make_env()
    env.reset(seed=2)
    actions = {agent: np.zeros(2, dtype=np.float32) for agent in env.agents}
    for _ in range(env.max_steps):
        _, _, _, truncated, _ = env.step(actions)
    assert all(truncated.values())


def test_physical_models_are_monotonic_and_positive():
    cfg = load_config("masac", "f1_m5")
    near = communication_rate_bps(200.0, 140.0, cfg)
    far = communication_rate_bps(4000.0, 140.0, cfg)
    assert near > far > 0
    p0 = multirotor_power_w(0.0, 0.0, cfg)
    pfast = multirotor_power_w(10.0, 4.0, cfg)
    assert pfast > p0 > 0


def test_interference_reduces_a2a_rate():
    cfg = load_config("masac", "f1_m5")
    clean = communication_rate_bps(500.0, 140.0, cfg, interference_power_w=0.0)
    interfered = communication_rate_bps(500.0, 140.0, cfg, interference_power_w=1e-10)
    assert interfered < clean


def test_one_leader_star_assignment_and_fixed_wing_mesh():
    env = make_env(seed=1)
    assert np.all(env.rotor_leaders == env.fixed_indices[0])

    cfg = load_config("masac", "f1_m5")
    cfg["scenario"]["fixed_wing"] = 2
    cfg["scenario"]["multirotor"] = 4
    env2 = PaperUAVEnv(cfg, seed=1)
    env2.positions[:] = np.array([
        [1000.0, 1000.0, env2.assumed["fixed_altitude_m"]],
        [1100.0, 1000.0, env2.assumed["fixed_altitude_m"]],
        [4000.0, 4000.0, env2.assumed["multirotor_altitude_m"]],
        [4200.0, 4000.0, env2.assumed["multirotor_altitude_m"]],
        [4000.0, 4200.0, env2.assumed["multirotor_altitude_m"]],
        [4200.0, 4200.0, env2.assumed["multirotor_altitude_m"]],
    ])
    env2._refresh_links()
    assert env2.last_adjacency[0, 1] == 1
    assert env2.last_adjacency[1, 0] == 1
    assert set(env2.rotor_leaders.tolist()).issubset(set(env2.fixed_indices))


def test_rotor_adjacency_only_uses_its_formation_leader():
    cfg = load_config("masac", "f1_m5")
    cfg["scenario"]["fixed_wing"] = 2
    cfg["scenario"]["multirotor"] = 2
    env = PaperUAVEnv(cfg, seed=2)
    # With two leaders and two rotors, deterministic balanced assignment is fixed_0, fixed_1.
    assert env.rotor_leaders.tolist() == [0, 1]
    for local_rotor, rotor_idx in enumerate(env.multirotor_indices):
        leader = env.rotor_leaders[local_rotor]
        other_leader = 1 - leader
        # Even if the other leader is physically near, it is not a formation edge.
        env.positions[rotor_idx, :2] = env.positions[other_leader, :2]
    env._refresh_links()
    for local_rotor, rotor_idx in enumerate(env.multirotor_indices):
        leader = env.rotor_leaders[local_rotor]
        for fi in env.fixed_indices:
            if fi != leader:
                assert env.last_adjacency[rotor_idx, fi] == 0
                assert env.last_adjacency[fi, rotor_idx] == 0
