from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
import torch

from uav_search.algorithms.common import ReplayBuffer
from uav_search.config import load_config
from uav_search.envs.paper_env import PaperUAVEnv


def make_u6_env(*, max_steps: int | None = None, battery_capacity_j: float | None = None) -> PaperUAVEnv:
    cfg = deepcopy(load_config("maddpg", "u6"))
    cfg["scenario"]["network_backend"] = "analytical"
    if max_steps is not None:
        cfg["runtime"]["episode_steps_override"] = int(max_steps)
    if battery_capacity_j is not None:
        cfg["scenario"]["battery_capacity_j"] = float(battery_capacity_j)
    env = PaperUAVEnv(cfg, seed=44)
    env.reset(seed=44)
    return env


def idle_actions(env: PaperUAVEnv) -> dict[str, np.ndarray]:
    actions = np.zeros((env.n_agents, env.action_dim), dtype=np.float32)
    # Cartesian acceleration: a zero motion vector is physically neutral.
    actions[:, 3] = -1.0
    return {name: actions[i] for i, name in enumerate(env.agents)}


def test_terminal_transition_keeps_reward_for_uav_that_depletes_during_step() -> None:
    env = make_u6_env(battery_capacity_j=1.0)

    _obs, rewards, terminated, truncated, info = env.step(idle_actions(env))

    assert all(terminated.values())
    assert not any(truncated.values())
    assert info["termination_reason"] == "all_uavs_depleted"
    # The final action still caused propulsion energy and therefore owns the
    # transition reward. Only later post-terminal samples should be invalid.
    assert all(rewards[name] < 0.0 for name in env.agents)
    assert info["reward_components_sum"]["energy"] < 0.0


def test_peer_mission_horizon_is_terminal_not_external_truncation() -> None:
    env = make_u6_env(max_steps=1)

    _obs, _rewards, terminated, truncated, info = env.step(idle_actions(env))

    assert info["termination_reason"] == "horizon"
    assert all(terminated.values())
    assert not any(truncated.values())


def test_inactive_uav_is_not_counted_as_safety_violation_or_network_disconnection() -> None:
    env = make_u6_env()
    env.positions[:] = np.asarray(
        [[500.0 + 1000.0 * i, 2500.0, 100.0] for i in range(env.n_agents)],
        dtype=np.float64,
    )
    env.positions[1] = env.positions[0] + np.asarray([1.0, 0.0, 0.0])
    env.uav_active[1] = False
    env.battery_pct[1] = 0.0
    env._refresh_links()

    _obs, _rewards, _terminated, _truncated, info = env.step(idle_actions(env))

    assert info["safety_distance_violations"] == 0
    assert 1 not in env.episode_safety_violation_uavs
    assert (
        info["direct_gcs_uavs"]
        + info["multihop_gcs_uavs"]
        + info["disconnected_gcs_uavs"]
        == info["active_uavs"]
    )


def test_replay_buffer_preserves_per_agent_transition_validity() -> None:
    rb = ReplayBuffer(4, n_agents=2, obs_dim=3, action_dim=2, seed=7)
    obs = np.zeros((2, 3), dtype=np.float32)
    actions = np.zeros((2, 2), dtype=np.float32)
    rewards = np.zeros(2, dtype=np.float32)
    next_obs = np.ones((2, 3), dtype=np.float32)
    dones = np.asarray([0.0, 1.0], dtype=np.float32)
    valids = np.asarray([1.0, 0.0], dtype=np.float32)

    rb.add(obs, actions, rewards, next_obs, dones, valids=valids)
    batch = rb.sample(1, torch.device("cpu"))

    np.testing.assert_array_equal(batch.valids.numpy()[0], valids)


@pytest.mark.parametrize("algorithm", ["maddpg", "matd3", "masac"])
def test_invalid_agent_samples_do_not_update_its_actor_or_critic(algorithm: str) -> None:
    from uav_search.algorithms.factory import make_algorithm

    cfg = deepcopy(load_config(algorithm, "u6"))
    cfg["scenario"]["network_backend"] = "analytical"
    cfg["runtime"]["batch_size"] = 4
    cfg["runtime"]["replay_size"] = 32
    cfg["runtime"]["hidden_sizes"] = [8, 8]
    if algorithm == "matd3":
        cfg["algorithm"]["policy_delay"] = 1
    env = PaperUAVEnv(cfg, seed=71)
    algo = make_algorithm(algorithm, env, cfg, device="cpu", seed=71)
    rng = np.random.default_rng(71)

    for _ in range(4):
        obs = rng.normal(size=(env.n_agents, env.obs_dim)).astype(np.float32)
        actions = rng.uniform(-1.0, 1.0, size=(env.n_agents, env.action_dim)).astype(np.float32)
        rewards = rng.normal(size=env.n_agents).astype(np.float32)
        next_obs = rng.normal(size=(env.n_agents, env.obs_dim)).astype(np.float32)
        dones = np.zeros(env.n_agents, dtype=np.float32)
        valids = np.ones(env.n_agents, dtype=np.float32)
        valids[0] = 0.0
        algo.store(obs, actions, rewards, next_obs, dones, valids=valids)

    def snapshot(module):
        return [p.detach().clone() for p in module.parameters()]

    actor0_before = snapshot(algo.actors[0])
    actor1_before = snapshot(algo.actors[1])
    if algorithm == "masac":
        critic0 = algo.critics1[0]
        critic1 = algo.critics1[1]
    else:
        critic0 = algo.critics[0]
        critic1 = algo.critics[1]
    critic0_before = snapshot(critic0)
    critic1_before = snapshot(critic1)

    metrics = algo.update()
    assert metrics

    assert all(torch.equal(before, after) for before, after in zip(actor0_before, algo.actors[0].parameters()))
    assert all(torch.equal(before, after) for before, after in zip(critic0_before, critic0.parameters()))
    assert any(not torch.equal(before, after) for before, after in zip(actor1_before, algo.actors[1].parameters()))
    assert any(not torch.equal(before, after) for before, after in zip(critic1_before, critic1.parameters()))


def test_runner_transition_valid_mask_tracks_activity_before_step() -> None:
    from uav_search.runner.train import _replay_valid_mask

    env = make_u6_env()
    env.uav_active[2] = False

    mask = _replay_valid_mask(env)

    np.testing.assert_array_equal(
        mask,
        np.asarray([1.0, 1.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float32),
    )
