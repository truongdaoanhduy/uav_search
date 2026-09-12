from copy import deepcopy

import numpy as np
import pytest
import torch

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
    for required in ["q_mean", "target_q_mean", "td_error_abs_mean"]:
        assert required in metrics, (name, required, metrics)
        assert np.isfinite(metrics[required]), (name, required, metrics[required])
    for key, value in metrics.items():
        if key.endswith("loss"):
            assert np.isfinite(value), (key, value)

    assert "amp_scaler" in algo.checkpoint()
    ckpt = tmp_path / f"{name}.pt"
    algo.save(ckpt)
    clone = make_algorithm(name, env, cfg, device="cpu", seed=99)
    clone.load(ckpt)
    a1 = algo.act(obs, explore=False)
    a2 = clone.act(obs, explore=False)
    assert np.allclose(a1, a2, atol=1e-6)


@pytest.mark.parametrize("name", ["maddpg", "matd3", "masac"])
def test_action_inference_calls_each_per_agent_actor_once(name):
    cfg = deepcopy(load_config(name, "f1_m5"))
    cfg["runtime"]["hidden_sizes"] = [16, 16]
    env = PaperUAVEnv(cfg, seed=7)
    obs_dict, _ = env.reset(seed=7)
    obs = np.stack([obs_dict[a] for a in env.agents])
    algo = make_algorithm(name, env, cfg, device="cpu", seed=7)

    calls = [0 for _ in range(env.n_agents)]
    hooks = []
    for i, actor in enumerate(algo.actors):
        hooks.append(actor.register_forward_hook(
            lambda module, args, output, i=i: calls.__setitem__(i, calls[i] + 1)
        ))
    try:
        algo.act(obs, explore=False)
    finally:
        for hook in hooks:
            hook.remove()

    assert calls == [1 for _ in range(env.n_agents)]


@pytest.mark.parametrize("name", ["maddpg", "matd3", "masac"])
def test_cpu_runtime_profile_keeps_amp_disabled(name):
    cfg = deepcopy(load_config(name, "f1_m5"))
    cfg["runtime"]["amp_mode"] = "auto"
    cfg["runtime"]["deterministic"] = False
    env = PaperUAVEnv(cfg, seed=13)
    algo = make_algorithm(name, env, cfg, device="cpu", seed=13)
    assert algo.runtime_profile.device.type == "cpu"
    assert algo.runtime_profile.amp_enabled is False
    assert algo.scaler.is_enabled() is False


class _ConstantCritic(torch.nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.value = float(value)

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return torch.full(
            (obs.shape[0], 1), self.value, dtype=obs.dtype, device=obs.device
        )


@pytest.mark.parametrize("name", ["matd3", "masac"])
def test_twin_critic_q_gap_metric_measures_q_disagreement_not_loss_gap(name, monkeypatch):
    cfg = deepcopy(load_config(name, "f1_m5"))
    cfg["runtime"]["batch_size"] = 1
    cfg["runtime"]["replay_size"] = 4
    cfg["runtime"]["hidden_sizes"] = [8, 8]
    if name == "matd3":
        cfg["algorithm"]["policy_delay"] = 2
    env = PaperUAVEnv(cfg, seed=17)
    algo = make_algorithm(name, env, cfg, device="cpu", seed=17)

    obs = np.zeros((env.n_agents, env.obs_dim), dtype=np.float32)
    actions = np.zeros((env.n_agents, env.action_dim), dtype=np.float32)
    rewards = np.zeros(env.n_agents, dtype=np.float32)
    dones = np.zeros(env.n_agents, dtype=np.float32)
    algo.store(obs, actions, rewards, obs, dones)

    # Make Q1=+1 and Q2=-1 while both critics have the same MSE to target Q=0.
    # A true Q-gap metric is therefore exactly 2, whereas a loss-gap metric is 0.
    for i in range(env.n_agents):
        if name == "matd3":
            algo.critics[i] = _ConstantCritic(1.0)
            algo.critics2[i] = _ConstantCritic(-1.0)
            algo.target_critics[i] = _ConstantCritic(0.0)
            algo.target_critics2[i] = _ConstantCritic(0.0)
        else:
            algo.critics1[i] = _ConstantCritic(1.0)
            algo.critics2[i] = _ConstantCritic(-1.0)
            algo.target_critics1[i] = _ConstantCritic(0.0)
            algo.target_critics2[i] = _ConstantCritic(0.0)

    monkeypatch.setattr(algo, "optimizer_step", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(algo, "finish_optimizer_steps", lambda: None)
    if name == "masac":
        def zero_sample(sample_obs, **_kwargs):
            batch = sample_obs.shape[0]
            return (
                torch.zeros(
                    (batch, env.n_agents, env.action_dim),
                    dtype=sample_obs.dtype,
                    device=sample_obs.device,
                ),
                torch.zeros(
                    (batch, env.n_agents, 1),
                    dtype=sample_obs.dtype,
                    device=sample_obs.device,
                ),
            )

        monkeypatch.setattr(algo, "_sample_actions", zero_sample)

    metrics = algo.update()
    assert metrics["critic1_loss"] == pytest.approx(1.0)
    assert metrics["critic2_loss"] == pytest.approx(1.0)
    assert metrics["q_gap_abs_mean"] == pytest.approx(2.0)
