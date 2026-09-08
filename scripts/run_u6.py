#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from uav_search.config import PAPER_ALGORITHMS
from uav_search.runner.train import train_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train the three requested continuous-action CTDE baselines "
            "(MASAC, MATD3, MADDPG) on the homogeneous peer-u6 research scenario."
        )
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=PAPER_ALGORITHMS,
        default=list(PAPER_ALGORITHMS),
    )
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Override the u6 scenario horizon; otherwise configs/scenarios/u6.yaml is used.",
    )
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", choices=["auto", "on", "off"], default="auto")
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-root", default="runs")
    parser.add_argument("--wandb-api-key", default=None)
    parser.add_argument(
        "--wandb-mode",
        choices=["auto", "online", "offline", "disabled"],
        default="auto",
    )
    parser.add_argument("--local-only", action="store_true")
    parser.add_argument("--progress-every", type=int, default=10)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.episodes < 1:
        raise ValueError("episodes must be >= 1")
    if args.steps is not None and args.steps < 1:
        raise ValueError("steps must be >= 1")

    for algorithm in args.algorithms:
        run_dir = Path(
            train_experiment(
                algorithm=algorithm,
                scenario="u6",
                episodes=args.episodes,
                steps=args.steps,
                seed=args.seed,
                device=args.device,
                output_root=args.output_root,
                wandb=False if args.local_only else None,
                wandb_api_key=args.wandb_api_key,
                deterministic=args.deterministic,
                amp_mode=args.amp,
                wandb_mode="disabled" if args.local_only else args.wandb_mode,
                progress_every=args.progress_every,
            )
        )
        print(f"{algorithm}/u6: {run_dir}")


if __name__ == "__main__":
    main()
