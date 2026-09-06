from pathlib import Path

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
    for name in ["reward.png", "search_rate.png", "energy.png", "broken_link.png", "trajectory.png"]:
        p = run_dir / "plots" / name
        assert p.exists() and p.stat().st_size > 0
    assert (run_dir / "rollouts" / "final_trajectory.npz").exists()
