from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from uav_search.runner.train import train_experiment


@pytest.mark.parametrize("algorithm", ["maddpg", "matd3", "masac"])
def test_two_episode_training_pipeline_creates_artifacts(algorithm, tmp_path):
    run_dir = train_experiment(
        algorithm=algorithm,
        scenario="f1_m5",
        episodes=2,
        steps=4,
        seed=11,
        device="cpu",
        output_root=tmp_path,
        runtime_overrides={
            "batch_size": 4,
            "replay_size": 64,
            "hidden_sizes": [16, 16],
            "warmup_steps": 0,
            "update_after": 4,
            "checkpoint_every_episodes": 1,
        },
    )
    run_dir = Path(run_dir)
    assert (run_dir / "config.yaml").exists()
    assert (run_dir / "checkpoints" / "final.pt").exists()
    assert (run_dir / "checkpoints" / "latest.pt").exists()
    assert (run_dir / "logs" / "errors.log").exists()
    ep = pd.read_csv(run_dir / "metrics" / "episodes.csv")
    perf = pd.read_csv(run_dir / "metrics" / "performance.csv")
    assert len(ep) == 2
    assert len(perf) == 2
    for col in ["episode_sec", "env_steps_per_sec", "updates_per_sec", "wall_time_sec"]:
        assert col in ep.columns
        assert col in perf.columns
        assert (ep[col] >= 0).all()
        assert (perf[col] >= 0).all()
    for col in ["q_mean", "target_q_mean", "td_error_abs_mean"]:
        assert col in ep.columns
        assert ep[col].map(lambda value: pd.notna(value)).all()
    for col in [
        "avg_battery_pct", "depleted_uavs", "safety_distance_violation_uavs", "obstacle_hit_uavs",
        "boundary_hit_uavs", "broken_link_uavs", "fixed_return_mean", "rotor_return_mean",
    ]:
        assert col in ep.columns
    assert not any(str(col).startswith("agent/") for col in ep.columns)
    for name in ["reward.png", "search_rate.png", "energy.png", "broken_link.png", "trajectory.png"]:
        p = run_dir / "plots" / name
        assert p.exists() and p.stat().st_size > 0
    assert (run_dir / "rollouts" / "final_trajectory.npz").exists()


def test_wandb_update_metrics_are_throttled_to_every_100_updates(monkeypatch, tmp_path):
    import uav_search.runner.train as train_module

    instances = []
    class FakeWandbLogger:
        def __init__(self, *args, **kwargs):
            self.logs = []
            self.status = "fake"
            self.project_url = "fake://project"
            self.run_url = None
            self.mode = "fake"
            instances.append(self)
        def log(self, payload, step=None): self.logs.append(dict(payload))
        def log_low_episode(self, payload): pass
        def log_error(self, payload): pass
        def log_model(self, *args, **kwargs): pass
        def log_image(self, *args, **kwargs): pass
        def log_run_artifact(self, *args, **kwargs): pass
        def update_summary(self, *args, **kwargs): pass
        def finish(self): pass

    monkeypatch.setattr(train_module, "WandbLogger", FakeWandbLogger)
    train_module.train_experiment(
        algorithm="maddpg", scenario="f1_m5", episodes=1, steps=205, seed=5,
        device="cpu", output_root=tmp_path, amp_mode="off",
        runtime_overrides={
            "batch_size": 2, "replay_size": 512, "hidden_sizes": [8, 8],
            "warmup_steps": 0, "update_after": 2, "checkpoint_every_episodes": 0,
        },
    )
    update_numbers = [
        int(payload["update/update"])
        for payload in instances[0].logs
        if "update/update" in payload
    ]
    assert update_numbers == [100, 200]
    episode_payloads = [payload for payload in instances[0].logs if "paper/episode" in payload]
    assert len(episode_payloads) == 1
    episode_payload = episode_payloads[0]
    assert "swarm/avg_battery_pct" in episode_payload
    assert "swarm/safety_distance_violation_uavs" in episode_payload
    assert "group/fixed_return_mean" in episode_payload
    assert "group/rotor_return_mean" in episode_payload
    assert not any("/agent/" in key or "worst_agent" in key for key in episode_payload)


def test_deterministic_rollout_stops_on_terminated(monkeypatch):
    import uav_search.runner.train as train_module

    class FakeEnv:
        def __init__(self, cfg, seed):
            self.agents = ["uav_0", "uav_1"]
            self.n_agents = 2
            self.max_steps = 5
            self.total_energy_used_j = 0.0
            self.step_calls = 0

        def reset(self, seed=None):
            obs = {agent: np.zeros(3, dtype=np.float32) for agent in self.agents}
            return obs, {}

        def step(self, action_dict):
            self.step_calls += 1
            obs = {agent: np.zeros(3, dtype=np.float32) for agent in self.agents}
            rewards = {agent: 1.0 for agent in self.agents}
            terminated = {agent: True for agent in self.agents}
            truncated = {agent: False for agent in self.agents}
            info = {"mean_comm_rate_mbps": 0.0, "action_saturation": 0.0}
            return obs, rewards, terminated, truncated, info

    class FakeAlgo:
        def act(self, obs, explore=False):
            return np.zeros((2, 1), dtype=np.float32)

    monkeypatch.setattr(train_module, "PaperUAVEnv", FakeEnv)
    env, metrics = train_module.deterministic_rollout(FakeAlgo(), {}, seed=44)

    assert env.step_calls == 1
    assert metrics["return_sum"] == pytest.approx(2.0)


def test_replay_terminal_mask_bootstraps_across_time_limit_truncation():
    import uav_search.runner.train as train_module

    agents = ["uav_0", "uav_1"]
    terminated = {"uav_0": False, "uav_1": True}
    truncated = {"uav_0": True, "uav_1": False}

    mask = train_module._replay_terminal_mask(terminated, agents)

    # Time-limit truncation ends rollout collection but must still bootstrap Q.
    np.testing.assert_array_equal(mask, np.array([0.0, 1.0], dtype=np.float32))
