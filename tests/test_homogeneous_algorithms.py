from copy import deepcopy

import numpy as np
import pytest

from uav_search.algorithms.factory import make_algorithm
from uav_search.config import load_config
from uav_search.envs.paper_env import PaperUAVEnv


@pytest.mark.parametrize("name", ["maddpg", "matd3", "masac"])
def test_algorithms_support_homogeneous_u6_six_dimensional_action(name):
    cfg = deepcopy(load_config(name, "u6"))
    cfg["scenario"]["network_backend"] = "analytical"
    cfg["runtime"]["batch_size"] = 4
    cfg["runtime"]["replay_size"] = 32
    cfg["runtime"]["hidden_sizes"] = [16, 16]
    env = PaperUAVEnv(cfg, seed=44)
    obs_dict, _ = env.reset(seed=44)
    obs = np.stack([obs_dict[a] for a in env.agents])
    algo = make_algorithm(name, env, cfg, device="cpu", seed=44)

    action = algo.act(obs, explore=False)
    assert action.shape == (6, 6)
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



def test_u6_matd3_target_smoothing_does_not_perturb_gate_or_recipient():
    cfg = deepcopy(load_config("matd3", "u6"))
    cfg["scenario"]["network_backend"] = "analytical"
    cfg["runtime"]["hidden_sizes"] = [16, 16]
    env = PaperUAVEnv(cfg, seed=44)
    algo = make_algorithm("matd3", env, cfg, device="cpu", seed=44)
    np.testing.assert_array_equal(
        algo.target_smoothing_mask.detach().cpu().numpy().reshape(-1),
        np.asarray([1.0, 1.0, 1.0, 0.0, 1.0, 0.0], dtype=np.float32),
    )
