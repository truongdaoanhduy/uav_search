from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
import torch

from uav_search.actions import (
    canonicalize_peer_action_numpy,
    canonicalize_peer_action_torch,
    peer_recipient_bin_centers,
)
from uav_search.algorithms.factory import make_algorithm
from uav_search.config import load_config
from uav_search.envs.paper_env import PaperUAVEnv


def make_env(algorithm: str = "maddpg", scenario: str = "u6") -> PaperUAVEnv:
    cfg = deepcopy(load_config(algorithm, scenario))
    cfg["scenario"]["network_backend"] = "analytical"
    env = PaperUAVEnv(cfg, seed=314)
    env.reset(seed=314)
    env.obstacles[:, :2] = np.asarray([4900.0, 4900.0])
    env.obstacles[:, 2] = 1.0
    env.positions[:, :2] = np.asarray([[500.0 + 650.0 * i, 2500.0] for i in range(env.n_agents)])
    env.positions[:, 2] = 100.0
    env.velocities.fill(0.0)
    env._refresh_links()
    return env


def neutral_actions(env: PaperUAVEnv) -> dict[str, np.ndarray]:
    a = np.zeros((env.n_agents, env.action_dim), dtype=np.float32)
    a[:, 3] = -1.0
    a[:, 4] = -1.0
    a[:, 5] = -1.0
    return {name: a[i] for i, name in enumerate(env.agents)}


def test_zero_cartesian_motion_action_is_really_zero_acceleration() -> None:
    env = make_env()
    before = env.positions.copy()
    env.step(neutral_actions(env))
    np.testing.assert_allclose(env.positions, before, atol=1e-9)
    np.testing.assert_allclose(env.velocities, 0.0, atol=1e-9)


def test_cartesian_x_action_has_no_hidden_azimuth_or_y_acceleration() -> None:
    env = make_env()
    actions = neutral_actions(env)
    actions["uav_0"][0] = 0.5
    before = env.positions[0].copy()
    env.step(actions)
    assert env.positions[0, 0] > before[0]
    assert env.positions[0, 1] == pytest.approx(before[1])
    assert env.positions[0, 2] == pytest.approx(before[2])
    assert env.velocities[0, 0] > 0.0
    assert env.velocities[0, 1] == pytest.approx(0.0)


def test_peer_action_canonicalization_matches_environment_discrete_semantics() -> None:
    raw = np.asarray([[0.1, -0.2, 0.3, 0.01, -0.4, 0.0]], dtype=np.float32)
    canonical = canonicalize_peer_action_numpy(raw, n_agents=6)
    assert canonical[0, 3] == 1.0
    assert canonical[0, 5] in peer_recipient_bin_centers(6)
    np.testing.assert_allclose(canonical[0, [0, 1, 2, 4]], raw[0, [0, 1, 2, 4]])


def test_straight_through_discrete_canonicalization_has_hard_forward_and_gradient() -> None:
    raw = torch.tensor([[0.1, -0.2, 0.3, 0.01, -0.4, 0.0]], requires_grad=True)
    canonical = canonicalize_peer_action_torch(raw, n_agents=6, straight_through=True)
    assert float(canonical[0, 3].detach()) == 1.0
    centers = torch.as_tensor(peer_recipient_bin_centers(6), dtype=canonical.dtype)
    assert torch.any(torch.isclose(canonical[0, 5].detach(), centers))
    canonical[0, 3].backward(retain_graph=True)
    assert raw.grad is not None
    assert raw.grad[0, 3].item() == pytest.approx(1.0)


def test_peer_replay_stores_canonical_gate_and_recipient() -> None:
    env = make_env("maddpg")
    cfg = deepcopy(load_config("maddpg", "u6"))
    cfg["scenario"]["network_backend"] = "analytical"
    cfg["runtime"]["batch_size"] = 1
    cfg["runtime"]["replay_size"] = 4
    cfg["runtime"]["hidden_sizes"] = [8, 8]
    algo = make_algorithm("maddpg", env, cfg, device="cpu", seed=314)
    obs = np.zeros((env.n_agents, env.obs_dim), dtype=np.float32)
    actions = np.zeros((env.n_agents, env.action_dim), dtype=np.float32)
    actions[:, 3] = 0.2
    actions[:, 5] = 0.13
    algo.store(obs, actions, np.zeros(env.n_agents, dtype=np.float32), obs, np.zeros(env.n_agents, dtype=np.float32))
    stored = algo.replay.actions[0].numpy()
    assert np.all(np.isin(stored[:, 3], [-1.0, 1.0]))
    centers = peer_recipient_bin_centers(env.n_agents)
    assert all(any(np.isclose(v, c) for c in centers) for v in stored[:, 5])


def test_peer_masac_entropy_dimension_counts_only_continuous_controls() -> None:
    env = make_env("masac")
    cfg = deepcopy(load_config("masac", "u6"))
    cfg["scenario"]["network_backend"] = "analytical"
    cfg["runtime"]["hidden_sizes"] = [8, 8]
    algo = make_algorithm("masac", env, cfg, device="cpu", seed=314)
    assert algo.target_entropy == -4.0


def test_peer_action_saturation_ignores_hard_gate_recipient_and_inactive_power() -> None:
    env = make_env()
    actions = neutral_actions(env)
    _obs, _rewards, _terminated, _truncated, info = env.step(actions)
    assert info["action_saturation"] == pytest.approx(0.0)
