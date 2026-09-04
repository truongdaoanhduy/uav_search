from copy import deepcopy

import numpy as np
import pytest

from uav_search.algorithms.factory import make_algorithm
from uav_search.config import load_config
from uav_search.envs.paper_env import PaperUAVEnv


@pytest.mark.parametrize("name", ["maddpg", "matd3", "masac"])
def test_algorithm_action_update_and_checkpoint_roundtrip(name, tmp_path):
    cfg = deepcopy(load_config(name, "f1_m5"))
    cfg["runtime"]["batch_size"] = 8
    cfg["runtime"]["replay_size"] = 64
    cfg["runtime"]["hidden_sizes"] = [32, 32]
    env = PaperUAVEnv(cfg, seed=5)
    obs_dict, _ = env.reset(seed=5)
    obs = np.stack([obs_dict[a] for a in env.agents])
    algo = make_algorithm(name, env, cfg, device="cpu", seed=5)

    action = algo.act(obs, explore=False)
    assert action.shape == (env.n_agents, 2)
    assert np.isfinite(action).all()
    assert np.max(np.abs(action)) <= 1.0 + 1e-6

    rng = np.random.default_rng(9)
    for _ in range(12):
        o = rng.normal(size=(env.n_agents, env.obs_dim)).astype(np.float32)
        a = rng.uniform(-1, 1, size=(env.n_agents, 2)).astype(np.float32)
        r = rng.normal(size=env.n_agents).astype(np.float32)
        no = rng.normal(size=(env.n_agents, env.obs_dim)).astype(np.float32)
        d = np.zeros(env.n_agents, dtype=np.float32)
        algo.store(o, a, r, no, d)

    metrics = algo.update()
    if name == "matd3":
        metrics = algo.update()  # second update triggers delayed actor
    assert metrics
    for key, value in metrics.items():
        if key.endswith("loss"):
            assert np.isfinite(value), (key, value)

    ckpt = tmp_path / f"{name}.pt"
    algo.save(ckpt)
    clone = make_algorithm(name, env, cfg, device="cpu", seed=99)
    clone.load(ckpt)
    a1 = algo.act(obs, explore=False)
    a2 = clone.act(obs, explore=False)
    assert np.allclose(a1, a2, atol=1e-6)
