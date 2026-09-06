from pathlib import Path

import pandas as pd

from uav_search.runner.visualize import plot_paper_comparison


def _make_run(root: Path, algorithm: str, scenario: str, rewards: list[float], targets: float, energy: float, broken: float) -> Path:
    run = root / algorithm / scenario / "run"
    (run / "metrics").mkdir(parents=True, exist_ok=True)
    (run / "evaluation_5000").mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"episode": list(range(1, len(rewards) + 1)), "return_sum": rewards}).to_csv(
        run / "metrics" / "episodes.csv", index=False
    )
    import json
    (run / "evaluation_5000" / "evaluation_summary.json").write_text(json.dumps({
        "mean_targets_found": targets,
        "mean_energy_consumption_pct": energy,
        "mean_mean_broken_link_s": broken,
    }))
    return run


def test_paper_comparison_creates_fig7_and_fig10_to_fig12(tmp_path):
    runs = {}
    for ai, algorithm in enumerate(("masac", "matd3", "maddpg")):
        for si, scenario in enumerate(("f1_m5", "f1_m9")):
            runs[(algorithm, scenario)] = _make_run(
                tmp_path, algorithm, scenario,
                rewards=[1 + ai + si, 2 + ai + si, 3 + ai + si],
                targets=5 + ai + si,
                energy=40 + ai + si,
                broken=2 + ai + si,
            )
    out = tmp_path / "paper_figures"
    paths = plot_paper_comparison(runs, out, evaluation_dir_name="evaluation_5000")
    assert {p.name for p in paths} == {
        "fig07_training_reward.png",
        "fig10_average_targets_found.png",
        "fig11_average_energy_consumption.png",
        "fig12_average_broken_link_duration.png",
    }
    assert all(p.exists() and p.stat().st_size > 0 for p in paths)
