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
    assert ia["targets_total"] == 5


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
    env.positions[rotor_idx, :2] = env.targets[0]
    actions = {agent: np.zeros(2, dtype=np.float32) for agent in env.agents}
    _, _, _, _, info = env.step(actions)
    assert env.target_found[0]
    assert info["targets_found"] >= 1


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
    assert near > cfg["paper"]["min_comm_rate_bps"]
    p0 = multirotor_power_w(0.0, 0.0, cfg)
    pfast = multirotor_power_w(10.0, 4.0, cfg)
    assert pfast > p0 > 0
