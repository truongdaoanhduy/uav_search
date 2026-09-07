from copy import deepcopy

import numpy as np
import pytest

from uav_search.algorithms.factory import make_algorithm
from uav_search.config import load_config
from uav_search.envs.paper_env import PaperUAVEnv


@pytest.mark.parametrize("name", ["maddpg", "matd3", "masac"])
def test_algorithms_support_homogeneous_u6_five_dimensional_action(name):
    cfg = deepcopy(load_config(name, "u6"))
    cfg["runtime"]["batch_size"] = 4
    cfg["runtime"]["replay_size"] = 32
    cfg["runtime"]["hidden_sizes"] = [16, 16]
    env = PaperUAVEnv(cfg, seed=44)
    obs_dict, _ = env.reset(seed=44)
    obs = np.stack([obs_dict[a] for a in env.agents])
    algo = make_algorithm(name, env, cfg, device="cpu", seed=44)

    action = algo.act(obs, explore=False)
    assert action.shape == (6, 5)
    assert np.isfinite(action).all()

    rng = np.random.default_rng(44)
    for _ in range(8):
        o = rng.normal(size=(env.n_agents, env.obs_dim)).astype(np.float32)
        a = rng.uniform(-1, 1, size=(env.n_agents, env.action_dim)).astype(np.float32)
        r = rng.normal(size=env.n_agents).astype(np.float32)
        no = rng.normal(size=(env.n_agents, env.obs_dim)).astype(np.float32)
        d = np.zeros(env.n_agents, dtype=np.float32)
        algo.store(o, a, r, no, d)

    metrics = algo.update()
    if name == "matd3":
        metrics = algo.update()
    assert metrics
    assert all(np.isfinite(v) for k, v in metrics.items() if isinstance(v, (int, float)))
