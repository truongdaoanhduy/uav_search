#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
    p.add_argument("--wandb-api-key", default=None, help="W&B API key passed directly to this process")
    p.add_argument(
        "--wandb-mode",
        choices=["auto", "online", "offline", "disabled"],
        default="auto",
    )
    p.add_argument("--local-only", action="store_true")
    p.add_argument("--progress-every", type=int, default=10)
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
                wandb_api_key=a.wandb_api_key,
                deterministic=a.deterministic,
                amp_mode=a.amp,
                wandb_mode="disabled" if a.local_only else a.wandb_mode,
                progress_every=a.progress_every,
            )
            print(f"{algorithm}/{scenario}: {path}")


if __name__ == "__main__":
    main()
