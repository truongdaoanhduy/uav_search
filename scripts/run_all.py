#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os

from uav_search.runner.train import train_experiment


def main() -> None:
    p = argparse.ArgumentParser(description="Train MASAC, MATD3 and MADDPG on both primary paper scenarios.")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--amp", choices=["auto", "on", "off"], default="auto")
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--output-root", default="runs")
    p.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "uav-search-paper-baselines"))
    p.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"))
    p.add_argument(
        "--wandb-mode",
        choices=["auto", "online", "offline", "disabled"],
        default="auto",
    )
    p.add_argument("--local-only", action="store_true")
    a = p.parse_args()

    for algorithm in ("masac", "matd3", "maddpg"):
        for scenario in ("f1_m5", "f1_m9"):
            path = train_experiment(
                algorithm=algorithm,
                scenario=scenario,
                episodes=a.episodes,
                steps=a.steps,
                seed=a.seed,
                device=a.device,
                output_root=a.output_root,
                wandb=False if a.local_only else None,
                wandb_project=a.wandb_project,
                wandb_entity=a.wandb_entity,
                deterministic=a.deterministic,
                amp_mode=a.amp,
                wandb_mode="disabled" if a.local_only else a.wandb_mode,
            )
            print(f"{algorithm}/{scenario}: {path}")


if __name__ == "__main__":
    main()
