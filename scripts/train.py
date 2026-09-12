#!/usr/bin/env python3
from __future__ import annotations

import argparse

from uav_search.config import ALL_SCENARIOS, PAPER_ALGORITHMS
from uav_search.runner.train import train_experiment


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train a UAV-search MARL baseline on a configured research or reproduction scenario.")
    p.add_argument("--algorithm", choices=PAPER_ALGORITHMS, required=True)
    p.add_argument("--scenario", choices=ALL_SCENARIOS, required=True)
    p.add_argument("--episodes", type=int, default=None, help="Override training episodes; paper reference is 50,000")
    p.add_argument("--steps", type=int, default=None, help="Development-only horizon override; omit to use the selected scenario config")
    p.add_argument("--seed", type=int, default=44)
    p.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    p.add_argument("--amp", choices=["auto", "on", "off"], default="auto", help="Mixed precision policy for CUDA")
    p.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True, help="Deterministic same-seed paper runs by default; use --no-deterministic for maximum speed")
    p.add_argument("--output-root", default="runs", help="Local mirror/fallback directory")
    p.add_argument("--wandb-api-key", default=None, help="W&B API key passed directly to this process")
    p.add_argument(
        "--wandb-mode",
        choices=["auto", "online", "offline", "disabled"],
        default="auto",
        help="auto: online when --wandb-api-key is provided, otherwise local",
    )
    p.add_argument("--local-only", action="store_true", help="Never connect to W&B")
    p.add_argument("--run-name", default=None)
    p.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print training progress every N episodes (default: every 100 episodes); first/final always print",
    )
    return p


def main() -> None:
    a = build_parser().parse_args()
    run = train_experiment(
        algorithm=a.algorithm,
        scenario=a.scenario,
        episodes=a.episodes,
        steps=a.steps,
        seed=a.seed,
        device=a.device,
        output_root=a.output_root,
        wandb=False if a.local_only else None,
        wandb_api_key=a.wandb_api_key,
        run_name=a.run_name,
        deterministic=a.deterministic,
        amp_mode=a.amp,
        wandb_mode="disabled" if a.local_only else a.wandb_mode,
        progress_every=a.progress_every,
    )
    print(f"Run directory: {run}")


if __name__ == "__main__":
    main()
