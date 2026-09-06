import json

import numpy as np
import pandas as pd

from uav_search.runner.logging import RunLogger
from uav_search.runner.visualize import plot_training_curves, plot_trajectory


def test_run_logger_writes_metrics_low_episode_and_error(tmp_path):
    logger = RunLogger(tmp_path, low_window=3)
    logger.log_episode({
        "episode": 1, "return_mean": 10.0, "search_rate": 0.6, "targets_found": 3,
        "energy_j": 1000.0, "mean_broken_link_s": 0.2, "max_broken_link_s": 0.5,
        "collisions": 0, "obstacle_hits": 0, "min_battery_pct": 99.0,
        "action_saturation": 0.1, "mean_comm_rate_mbps": 2.0, "critic_loss": 1.0, "actor_loss": -1.0,
    })
    low_payload = logger.log_episode({
        "episode": 2, "return_mean": -20.0, "search_rate": 0.0, "targets_found": 0,
        "energy_j": 5000.0, "mean_broken_link_s": 10.0, "max_broken_link_s": 20.0,
        "collisions": 3, "obstacle_hits": 2, "min_battery_pct": 8.0,
        "action_saturation": 0.95, "mean_comm_rate_mbps": 0.2, "critic_loss": 1e6, "actor_loss": 100.0,
    })
    logger.log_update({"update": 1, "critic_loss": 2.0, "actor_loss": -0.5})
    try:
        raise RuntimeError("boom")
    except RuntimeError as exc:
        logger.log_error(exc, {"episode": 2, "step": 7})

    ep = pd.read_csv(tmp_path / "metrics" / "episodes.csv")
    up = pd.read_csv(tmp_path / "metrics" / "updates.csv")
    assert len(ep) == 2 and len(up) == 1
    lines = (tmp_path / "logs" / "low_episodes.jsonl").read_text().strip().splitlines()
    assert lines
    low_csv = pd.read_csv(tmp_path / "logs" / "low_metric_episodes.csv")
    assert len(low_csv) >= 1
    low = json.loads(lines[-1])
    assert "search_low" in low["signals"]
    assert "link_unstable" in low["signals"]
    assert "collision_high" in low["signals"]
    assert "action_saturated" in low["signals"]
    assert low_payload is not None
    assert low_payload["severity"] == "critical"
    assert "search_low" in low_csv.iloc[-1]["signals"]
    assert "boom" in (tmp_path / "logs" / "errors.log").read_text()


def test_visualizers_create_expected_pngs(tmp_path):
    metrics = tmp_path / "episodes.csv"
    pd.DataFrame([
        {"episode": 1, "return_mean": 1.0, "return_sum": 6.0, "search_rate": 0.2, "energy_j": 100.0, "energy_consumption_pct": 10.0, "mean_broken_link_s": 1.0, "fixed_return_mean": 2.0, "rotor_return_mean": 0.8, "rotor_return_min": 0.3, "reward_task_sum": 3.0, "reward_communication_sum": -1.0, "reward_energy_sum": 2.0, "reward_safety_sum": -0.5},
        {"episode": 2, "return_mean": 2.0, "return_sum": 12.0, "search_rate": 0.4, "energy_j": 120.0, "energy_consumption_pct": 12.0, "mean_broken_link_s": 0.5, "fixed_return_mean": 3.0, "rotor_return_mean": 1.8, "rotor_return_min": 1.1, "reward_task_sum": 5.0, "reward_communication_sum": 0.0, "reward_energy_sum": 2.1, "reward_safety_sum": -0.2},
    ]).to_csv(metrics, index=False)
    out = tmp_path / "plots"
    paths = plot_training_curves(metrics, out)
    assert {p.name for p in paths} == {"reward.png", "search_rate.png", "energy.png", "broken_link.png", "group_returns.png", "reward_components.png"}
    assert all(p.exists() and p.stat().st_size > 0 for p in paths)

    trajectory = np.zeros((3, 2, 3), dtype=float)
    trajectory[:, 0, :2] = [[0, 0], [10, 10], [20, 20]]
    trajectory[:, 1, :2] = [[20, 0], [20, 10], [20, 20]]
    p = plot_trajectory(
        trajectory, np.array([[100, 100]]), np.array([[50, 50, 10]]),
        ["fixed_0", "rotor_0"], [0, 1], out / "trajectory.png", area_size_m=5000,
    )
    assert p.exists() and p.stat().st_size > 0


def test_warmup_low_search_alone_is_not_reported_as_an_anomaly(tmp_path):
    logger = RunLogger(tmp_path, low_window=5)
    payload = logger.log_episode({
        "episode": 1,
        "phase": "warmup",
        "return_mean": 0.0,
        "search_rate": 0.0,
        "mean_broken_link_s": 0.0,
        "mean_comm_rate_mbps": 5.0,
        "collisions": 0,
        "obstacle_hits": 0,
        "min_battery_pct": 100.0,
        "action_saturation": 0.0,
        "critic_loss": 0.0,
    })
    assert payload is None
