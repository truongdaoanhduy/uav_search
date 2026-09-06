#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from uav_search.config import ACTIVE_SCENARIOS, PAPER_ALGORITHMS, PAPER_SCENARIOS
from uav_search.runner.evaluate import evaluate_checkpoint
from uav_search.runner.train import train_experiment
from uav_search.runner.visualize import plot_paper_comparison
from uav_search.runner.wandb_logger import WandbLogger


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Train MADDPG, MATD3 and MASAC for the current Fig. 7 reproduction scope "
            "(f1_m5/f1_m9), then evaluate and build the paper comparison figures."
        )
    )
    p.add_argument("--algorithms", nargs="+", choices=PAPER_ALGORITHMS, default=list(PAPER_ALGORITHMS))
    p.add_argument(
        "--scenarios", nargs="+", choices=PAPER_SCENARIOS, default=list(ACTIVE_SCENARIOS),
        help="Current default scope is f1_m5/f1_m9; larger paper scenarios remain available for later phases.",
    )
    p.add_argument("--episodes", type=int, default=10, help="Development default; pass 50000 for the paper protocol")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument(
        "--eval-episodes", type=int, default=None,
        help="Evaluation cases per run. Default: 5000 for 50k-episode paper runs, otherwise 10 for development.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--amp", choices=["auto", "on", "off"], default="auto")
    p.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True, help="Deterministic same-seed paper runs by default; use --no-deterministic for maximum speed")
    p.add_argument("--output-root", default="runs")
    p.add_argument("--wandb-api-key", default=None, help="W&B API key passed directly to this process")
    p.add_argument(
        "--wandb-mode",
        choices=["auto", "online", "offline", "disabled"],
        default="auto",
    )
    p.add_argument("--local-only", action="store_true")
    p.add_argument("--progress-every", type=int, default=100, help="Print progress every 100 episodes by default")
    return p


def _resolved_eval_cases(episodes: int, requested: int | None) -> int:
    if requested is not None:
        if int(requested) < 1:
            raise ValueError("eval-episodes must be >= 1")
        return int(requested)
    return 5000 if int(episodes) >= 50000 else 10


def main() -> None:
    a = build_parser().parse_args()
    eval_cases = _resolved_eval_cases(a.episodes, a.eval_episodes)
    run_dirs: dict[tuple[str, str], Path] = {}
    eval_rows: list[dict[str, object]] = []

    for algorithm in a.algorithms:
        for scenario in a.scenarios:
            path = Path(train_experiment(
                algorithm=algorithm,
                scenario=scenario,
                episodes=a.episodes,
                steps=a.steps,
                seed=a.seed,
                device=a.device,
                output_root=a.output_root,
                wandb=False if a.local_only else None,
                wandb_api_key=a.wandb_api_key,
                deterministic=a.deterministic,
                amp_mode=a.amp,
                wandb_mode="disabled" if a.local_only else a.wandb_mode,
                progress_every=a.progress_every,
            ))
            run_dirs[(algorithm, scenario)] = path
            print(f"{algorithm}/{scenario}: {path}")

            eval_dir = path / f"evaluation_{eval_cases}"
            summary = evaluate_checkpoint(
                path / "checkpoints" / "final.pt",
                episodes=eval_cases,
                device=a.device,
                output_dir=eval_dir,
            )
            eval_rows.append({"algorithm": algorithm, "scenario": scenario, **summary})
            print(f"evaluation {algorithm}/{scenario}: {eval_cases} cases -> {eval_dir}")

    if not run_dirs:
        return

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    comparison_dir = Path(a.output_root) / "paper_comparison" / stamp
    paper_figure_dir = comparison_dir / "paper_figures"
    evaluation_dir_name = f"evaluation_{eval_cases}"
    figure_paths = plot_paper_comparison(run_dirs, paper_figure_dir, evaluation_dir_name=evaluation_dir_name)

    evaluations_copy = comparison_dir / "evaluations"
    for (algorithm, scenario), run_dir in run_dirs.items():
        src = run_dir / evaluation_dir_name
        dst = evaluations_copy / algorithm / scenario
        if src.exists():
            shutil.copytree(src, dst, dirs_exist_ok=True)

    metrics_dir = comparison_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    if eval_rows:
        columns = sorted({key for row in eval_rows for key in row})
        with (metrics_dir / "evaluation_summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader(); writer.writerows(eval_rows)
    (comparison_dir / "summary.json").write_text(json.dumps({
        "algorithms": list(a.algorithms),
        "scenarios": list(a.scenarios),
        "training_episodes": int(a.episodes),
        "evaluation_cases": eval_cases,
        "paper_figures": [path.name for path in figure_paths],
        "runs": {f"{alg}/{scenario}": str(path) for (alg, scenario), path in run_dirs.items()},
    }, indent=2), encoding="utf-8")

    wb = WandbLogger(
        run_name=f"paper-comparison-{stamp}",
        config={
            "algorithms": list(a.algorithms),
            "scenarios": list(a.scenarios),
            "training_episodes": int(a.episodes),
            "evaluation_cases": eval_cases,
        },
        run_dir=comparison_dir,
        api_key=a.wandb_api_key,
        mode="disabled" if a.local_only else a.wandb_mode,
        enabled=False if a.local_only else None,
    )
    for path in figure_paths:
        wb.log_image(path, f"paper_figures/{path.stem}")
    for row in eval_rows:
        algorithm = str(row["algorithm"]); scenario = str(row["scenario"])
        wb.log({
            f"paper_eval/{algorithm}/{scenario}/targets_found": float(row.get("mean_targets_found", 0.0)),
            f"paper_eval/{algorithm}/{scenario}/energy_consumption_pct": float(row.get("mean_energy_consumption_pct", 0.0)),
            f"paper_eval/{algorithm}/{scenario}/broken_link_duration_s": float(row.get("mean_mean_broken_link_s", 0.0)),
        })
    wb.log_run_artifact(f"paper-comparison-{stamp}")
    wb.finish()
    print(f"Paper comparison directory: {comparison_dir}")


if __name__ == "__main__":
    main()
