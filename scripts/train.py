#!/usr/bin/env python3
from __future__ import annotations

import argparse
from uav_search.runner.train import train_experiment


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train one heterogeneous-UAV paper baseline.")
    p.add_argument("--algorithm", choices=["masac", "matd3", "maddpg"], required=True)
    p.add_argument("--scenario", choices=["f1_m5", "f1_m9"], required=True)
    p.add_argument("--episodes", type=int, default=None, help="Override training episodes; paper reference is 50,000")
    p.add_argument("--steps", type=int, default=None, help="Development-only override; paper value is 50 steps/episode")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    p.add_argument("--amp", choices=["auto", "on", "off"], default="auto", help="Mixed precision policy for CUDA")
    p.add_argument("--deterministic", action="store_true", help="Use deterministic PyTorch kernels when available (slower)")
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
        default=10,
        help="Print training progress every N episodes; first/final always print",
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
