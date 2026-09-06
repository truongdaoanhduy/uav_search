from copy import deepcopy
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from uav_search.algorithms.factory import make_algorithm
from uav_search.config import PAPER_SCENARIOS, load_config
from uav_search.envs.models import communication_rate_bps
from uav_search.envs.paper_env import PaperUAVEnv


ROOT = Path(__file__).resolve().parents[1]


def test_all_six_paper_scenarios_are_available_with_figure_counts():
    assert PAPER_SCENARIOS == (
        "f1_m5",
        "f1_m9",
        "f2_m10",
        "f2_m18",
        "f4_m12",
        "f8_m24",
    )
    expected = {
        "f1_m5": (1, 5, 10),
        "f1_m9": (1, 9, 10),
        "f2_m10": (2, 10, 10),
        "f2_m18": (2, 18, 10),
        "f4_m12": (4, 12, 20),
        "f8_m24": (8, 24, 20),
    }
    for name, (fixed, rotor, targets) in expected.items():
        cfg = load_config("maddpg", name)
        assert cfg["scenario"]["fixed_wing"] == fixed
        assert cfg["scenario"]["multirotor"] == rotor
        assert cfg["scenario"]["targets"] == targets


def test_a2a_transmit_power_uses_ref39_value_and_is_not_confused_with_pcom():
    cfg = load_config("masac", "f1_m5")
    assert cfg["paper"]["communication_power_w"] == 5.0
    assert cfg["reference_backed"]["tx_power_w"] == 10.0  # 40 dBm in target-paper ref. [39]
    assert "tx_power_w" not in cfg["assumed"]
    rate_10w = communication_rate_bps(500.0, 140.0, cfg)
    cfg2 = deepcopy(cfg)
    cfg2["reference_backed"]["tx_power_w"] = 1.0
    assert communication_rate_bps(500.0, 140.0, cfg2) < rate_10w


def test_train_cli_exposes_all_six_paper_scenarios():
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "train.py"), "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    for scenario in PAPER_SCENARIOS:
        assert scenario in proc.stdout


def test_maddpg_uses_one_actor_and_one_centralized_critic_per_uav():
    cfg = deepcopy(load_config("maddpg", "f1_m5"))
    cfg["runtime"]["replay_size"] = 16
    cfg["runtime"]["batch_size"] = 4
    cfg["runtime"]["hidden_sizes"] = [16, 16]
    env = PaperUAVEnv(cfg, seed=1)
    algo = make_algorithm("maddpg", env, cfg, device="cpu", seed=1)
    assert len(algo.actors) == env.n_agents
    assert len(algo.critics) == env.n_agents
    assert len(algo.target_actors) == env.n_agents
    assert len(algo.target_critics) == env.n_agents


def test_matd3_uses_twin_centralized_critics_per_uav():
    cfg = deepcopy(load_config("matd3", "f1_m5"))
    cfg["runtime"]["replay_size"] = 16
    cfg["runtime"]["batch_size"] = 4
    cfg["runtime"]["hidden_sizes"] = [16, 16]
    env = PaperUAVEnv(cfg, seed=2)
    algo = make_algorithm("matd3", env, cfg, device="cpu", seed=2)
    assert len(algo.critics1) == env.n_agents
    assert len(algo.critics2) == env.n_agents
    assert len(algo.target_critics1) == env.n_agents
    assert len(algo.target_critics2) == env.n_agents


def test_masac_ref44_autotunes_independent_temperature_per_actor():
    cfg = deepcopy(load_config("masac", "f1_m5"))
    cfg["runtime"]["replay_size"] = 32
    cfg["runtime"]["batch_size"] = 4
    cfg["runtime"]["hidden_sizes"] = [16, 16]
    env = PaperUAVEnv(cfg, seed=3)
    algo = make_algorithm("masac", env, cfg, device="cpu", seed=3)
    assert len(algo.actors) == env.n_agents
    assert len(algo.critics1) == env.n_agents
    assert len(algo.critics2) == env.n_agents
    assert algo.log_alpha.shape == (env.n_agents,)
    assert torch.allclose(algo.alpha.detach().cpu(), torch.full((env.n_agents,), 0.01), atol=1e-6)
    assert algo.target_entropy == -float(env.action_dim)

    rng = np.random.default_rng(4)
    for _ in range(8):
        o = rng.normal(size=(env.n_agents, env.obs_dim)).astype(np.float32)
        a = rng.uniform(-1, 1, size=(env.n_agents, env.action_dim)).astype(np.float32)
        r = rng.normal(size=env.n_agents).astype(np.float32)
        no = rng.normal(size=(env.n_agents, env.obs_dim)).astype(np.float32)
        d = np.zeros(env.n_agents, dtype=np.float32)
        algo.store(o, a, r, no, d)
    before = algo.alpha.detach().clone()
    metrics = algo.update()
    after = algo.alpha.detach().clone()
    assert metrics
    assert "alpha_loss" in metrics
    assert not torch.allclose(before, after)


def test_large_scenario_can_construct_with_memory_budgeted_replay():
    cfg = deepcopy(load_config("maddpg", "f8_m24"))
    cfg["runtime"]["replay_memory_budget_mb"] = 8
    cfg["runtime"]["hidden_sizes"] = [8, 8]
    env = PaperUAVEnv(cfg, seed=5)
    algo = make_algorithm("maddpg", env, cfg, device="cpu", seed=5)
    assert algo.replay.capacity < cfg["runtime"]["replay_size"]
    assert algo.replay.estimated_bytes <= 8 * 1024 * 1024
