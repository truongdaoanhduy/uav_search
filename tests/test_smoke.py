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
        },
    )
    run_dir = Path(run_dir)
    assert (run_dir / "config.yaml").exists()
    assert (run_dir / "checkpoints" / "final.pt").exists()
    assert (run_dir / "logs" / "errors.log").exists()
    ep = pd.read_csv(run_dir / "metrics" / "episodes.csv")
    assert len(ep) == 2
    for name in ["reward.png", "search_rate.png", "energy.png", "broken_link.png", "trajectory.png"]:
        p = run_dir / "plots" / name
        assert p.exists() and p.stat().st_size > 0
    assert (run_dir / "rollouts" / "final_trajectory.npz").exists()
