import numpy as np

from uav_search.config import load_config
from uav_search.envs.models import (
    circle_collision,
    communication_rate_bps,
    multirotor_power_w,
)
from uav_search.envs.paper_env import PaperUAVEnv


def make_env(scenario="f1_m5", seed=7):
    return PaperUAVEnv(load_config("masac", scenario), seed=seed)


def test_seeded_reset_is_reproducible_and_shapes_match():
    a = make_env(seed=42)
    b = make_env(seed=42)
    oa, ia = a.reset(seed=42)
    ob, _ib = b.reset(seed=42)
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
    _obs, _ = env.reset(seed=1)
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


def test_targets_and_building_centers_are_sampled_without_paper_margin():
    env = make_env(seed=11)
    seen_edge = False
    for seed in range(100):
        env.reset(seed=seed)
        if (
            np.any(env.targets < 300.0)
            or np.any(env.targets > 4700.0)
            or np.any(env.obstacles[:, :2] < 300.0)
            or np.any(env.obstacles[:, :2] > 4700.0)
        ):
            seen_edge = True
            break
    assert seen_edge


def test_uav_initialization_keeps_documented_assumed_margin():
    env = make_env(seed=8)
    env.reset(seed=8)
    margin = env.assumed["uav_init_margin_m"]
    assert np.all(env.positions[:, :2] >= margin)
    assert np.all(env.positions[:, :2] <= env.area_size_m - margin)


def test_speed_constraints_follow_paper_table_i():
    env = make_env(seed=2)
    actions = {agent: np.array([1.0, 1.0], dtype=np.float32) for agent in env.agents}
    for _ in range(20):
        env.step(actions)
    fixed_speeds = np.linalg.norm(env.velocities[env.fixed_indices], axis=1)
    rotor_speeds = np.linalg.norm(env.velocities[env.multirotor_indices], axis=1)
    assert np.all((fixed_speeds >= 10.0) & (fixed_speeds <= 40.0))
    assert np.all(rotor_speeds <= 10.0 + 1e-9)


def test_obstacle_domain_is_hard_constraint():
    env = make_env(seed=3)
    rotor = env.multirotor_indices[0]
    env.positions[rotor, :2] = [980.0, 1000.0]
    env.velocities[rotor] = 0.0
    env.obstacles[:, :2] = [4500.0, 4500.0]
    env.obstacles[:, 2] = 10.0
    env.obstacles[0] = [1008.0, 1000.0, 20.0]
    env._refresh_links()
    actions = {a: np.zeros(2, dtype=np.float32) for a in env.agents}
    actions[env.agents[rotor]] = np.array([1.0, 0.0], dtype=np.float32)
    _, _, _, _, info = env.step(actions)
    assert info["obstacle_hits"] >= 1
    assert not circle_collision(env.positions[rotor, :2], env.obstacles[0])


def test_energy_reward_matches_paper_eq22():
    env = make_env(seed=20)
    rotor = env.multirotor_indices[0]
    env.battery_pct[rotor] = 50.0
    assert env._energy_reward(rotor) == env.paper["energy_reward_scale"] * 50.0
    env.battery_pct[rotor] = env.paper["safe_battery_pct"]
    assert env._energy_reward(rotor) == 0.0


def test_search_reward_matches_paper_eq24_in_environment_distance_units():
    env = make_env(seed=21)
    rotor = env.multirotor_indices[0]
    env.target_found[:] = True
    env.target_found[0] = False
    env.positions[rotor, :2] = [1000.0, 1000.0]
    env.targets[0] = [1200.0, 1000.0]
    zeta = env.assumed["search_reward_coeff"]
    delta = env.assumed["reward_distance_epsilon_m"]
    assert np.isclose(env._task_reward(rotor, fixed=False), zeta / (200.0 + delta))
    env.positions[rotor, :2] = env.targets[0]
    assert env._task_reward(rotor, fixed=False) == zeta


def test_safety_reward_matches_paper_eq23_in_environment_distance_units():
    env = make_env(seed=22)
    rotor = env.multirotor_indices[0]
    other = env.multirotor_indices[1]
    env.positions[:, :2] = [4000.0, 4000.0]
    env.positions[rotor, :2] = [1000.0, 1000.0]
    env.positions[other, :2] = [1050.0, 1000.0]
    eta = env.assumed["safety_reward_coeff"]
    delta = env.assumed["reward_distance_epsilon_m"]
    reward, close = env._safety_reward(rotor)
    assert close == 1
    assert np.isclose(reward, -eta / (50.0 + delta))


def test_communication_reward_matches_paper_eq21_and_formation_leader_distance():
    env = make_env(seed=23)
    rotor = env.multirotor_indices[0]
    local = 0
    leader = int(env.rotor_leaders[local])
    env.positions[rotor] = [1000.0, 1000.0, env.assumed["multirotor_altitude_m"]]
    env.positions[leader] = [1400.0, 1000.0, env.assumed["fixed_altitude_m"]]
    env.last_rates_bps[local] = 0.5 * (
        env.paper["min_comm_rate_bps"] + env.assumed["comm_rate_max_bps"]
    )
    d = float(np.linalg.norm(env.positions[rotor] - env.positions[leader]))
    expected = env.assumed["comm_reward_max"] / (d + env.assumed["reward_distance_epsilon_m"])
    assert np.isclose(env._communication_reward(rotor), expected)
    env.last_rates_bps[local] = 0.5 * env.paper["min_comm_rate_bps"]
    assert env._communication_reward(rotor) == -env.assumed["comm_reward_max"]
    env.last_rates_bps[local] = 2.0 * env.assumed["comm_rate_max_bps"]
    assert env._communication_reward(rotor) == env.assumed["comm_reward_max"]


def test_a2a_rate_uses_reference_backed_tx_power_not_pcom():
    cfg = load_config("masac", "f1_m5")
    rate_default = communication_rate_bps(500.0, 140.0, cfg)
    cfg["reference_backed"]["tx_power_w"] = 0.5
    rate_lower_tx = communication_rate_bps(500.0, 140.0, cfg)
    assert rate_lower_tx < rate_default

    # P_com remains the root-paper explicit 5 W term in the rotor energy model.
    p0 = multirotor_power_w(0.0, 0.0, cfg)
    assert np.isclose(p0, cfg["assumed"]["hover_power_w"] + 5.0)


def test_observation_layout_matches_paper_eq17_to_eq20_union():
    env = make_env(seed=30)
    # Shared padded baseline layout contains only the union of paper Eqs. (17)-(20):
    # own {p(3), v(3), e, n, psi}, each other UAV {d_ij, p_j(3), e_j, n_j},
    # and one sensed distance d_i,k per target. Obstacle geometry is not an Actor input.
    expected = 9 + (env.n_agents - 1) * 6 + env.n_targets
    assert env.obs_dim == expected
    obs = env._observations()
    assert all(v.shape == (expected,) for v in obs.values())


def test_type_specific_self_state_masks_fields_not_in_paper_state():
    env = make_env(seed=31)
    fixed = env.fixed_indices[0]
    rotor = env.multirotor_indices[0]
    env.battery_pct[fixed] = 17.0
    env.battery_pct[rotor] = 42.0
    env.headings[fixed] = np.pi / 2
    env.headings[rotor] = -np.pi / 3
    env.last_adjacency.fill(0)
    leader = int(env.rotor_leaders[0])
    env.last_adjacency[rotor, leader] = 1
    env.last_adjacency[leader, rotor] = 1

    obs = env._observations()
    fixed_self = obs[env.agents[fixed]][:9]
    rotor_self = obs[env.agents[rotor]][:9]

    # Eq. (19): fixed-wing self-state is {p, v, psi}; no energy/network fields.
    assert fixed_self[6] == 0.0
    assert fixed_self[7] == 0.0
    assert np.isclose(fixed_self[8], 0.5)

    # Eq. (17): multi-rotor self-state is {p, v, e, n}; no heading field.
    assert np.isclose(rotor_self[6], 0.42)
    assert rotor_self[7] == 1.0
    assert rotor_self[8] == 0.0


def test_neighbor_features_follow_eq18_and_eq20_without_extra_type_flags():
    env = make_env(seed=32)
    fixed = env.fixed_indices[0]
    rotor = env.multirotor_indices[0]
    env.positions[fixed] = [1000.0, 1200.0, env.assumed["fixed_altitude_m"]]
    env.positions[rotor] = [1300.0, 1600.0, env.assumed["multirotor_altitude_m"]]
    env.battery_pct[rotor] = 55.0
    env.last_adjacency.fill(0)
    env.last_adjacency[fixed, rotor] = 1
    env.last_adjacency[rotor, fixed] = 1

    obs = env._observations()
    area = env.area_size_m

    # fixed_0's first neighbor is rotor_0 in f1_m5. Eq. (20) has p_j and n_j,
    # but not d_i,j or e_j, so their padded slots must be zero.
    fixed_neighbor = obs[env.agents[fixed]][9:15]
    assert fixed_neighbor[0] == 0.0
    assert np.allclose(fixed_neighbor[1:4], env.positions[rotor] / area)
    assert fixed_neighbor[4] == 0.0
    assert fixed_neighbor[5] == 1.0

    # rotor_0's first neighbor is fixed_0. Eq. (18) includes d_i,j, p_j, e_j, n_j.
    rotor_neighbor = obs[env.agents[rotor]][9:15]
    expected_d = np.linalg.norm(env.positions[fixed] - env.positions[rotor]) / area
    assert np.isclose(rotor_neighbor[0], expected_d)
    assert np.allclose(rotor_neighbor[1:4], env.positions[fixed] / area)
    # Fixed-wing energy is not modeled by the paper, so e_j is masked for a fixed neighbor.
    assert rotor_neighbor[4] == 0.0
    assert rotor_neighbor[5] == 1.0


def test_target_distance_is_visible_only_inside_type_sensing_range_and_unfound():
    env = make_env(seed=33)
    fixed = env.fixed_indices[0]
    rotor = env.multirotor_indices[0]
    env.positions[fixed, :2] = [1000.0, 1000.0]
    env.positions[rotor, :2] = [1000.0, 1000.0]
    env.targets[:] = [4900.0, 4900.0]
    env.targets[0] = [1100.0, 1000.0]   # 100 m: visible to both
    env.targets[1] = [2000.0, 1000.0]   # 1000 m: outside rotor 700, inside fixed 1500
    env.target_found[:] = False

    target_start = 9 + (env.n_agents - 1) * 6
    obs = env._observations()
    fixed_targets = obs[env.agents[fixed]][target_start:target_start + env.n_targets]
    rotor_targets = obs[env.agents[rotor]][target_start:target_start + env.n_targets]

    assert np.isclose(rotor_targets[0], 100.0 / env.area_size_m)
    assert rotor_targets[1] == 0.0
    assert np.isclose(fixed_targets[0], 100.0 / env.area_size_m)
    assert np.isclose(fixed_targets[1], 1000.0 / env.area_size_m)

    # Section IV-D explicitly masks targets after discovery.
    env.target_found[0] = True
    obs = env._observations()
    assert obs[env.agents[rotor]][target_start] == 0.0
    assert obs[env.agents[fixed]][target_start] == 0.0


def test_actor_observation_does_not_leak_obstacle_geometry():
    env = make_env(seed=34)
    before = env._observations()
    env.obstacles[:, :2] = [10.0, 10.0]
    env.obstacles[:, 2] = 999.0
    after = env._observations()
    for agent in env.agents:
        assert np.allclose(before[agent], after[agent])
