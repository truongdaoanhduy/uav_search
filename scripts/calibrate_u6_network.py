#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from uav_search.runner.network_calibration import (
    run_calibration_episode,
    summarize_calibration,
    write_calibration_outputs,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Non-learning multi-seed topology/network calibration for u6 using real UavNetSim.")
    parser.add_argument(
        "--ranges", nargs="+", type=float, default=[1000.0, 1500.0, 2000.0, 2500.0],
        help="Reference peer/GCS contact ranges in meters at the configured reference power (0.1 W in u6).",
    )
    parser.add_argument("--seed-start", type=int, default=44)
    parser.add_argument("--num-seeds", type=int, default=50)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument(
        "--tx-power-w", type=float, default=0.1,
        help="Fixed transmit power used by the optional --traffic policy; topology-only fractions use the reference-power link snapshot.",
    )
    parser.add_argument("--traffic", action="store_true", help="Enable natural TargetReport traffic; topology-only is the default.")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/calibration/u6-network"))
    args = parser.parse_args()

    rows = []
    for contact_range in args.ranges:
        for seed in range(args.seed_start, args.seed_start + args.num_seeds):
            row = run_calibration_episode(
                contact_range_m=contact_range,
                seed=seed,
                steps=args.steps,
                tx_power_w=args.tx_power_w,
                traffic=args.traffic,
            )
            rows.append(row)
            pdr_text = "n/a" if row["byte_pdr"] is None else f"{row['byte_pdr']:.3f}"
            print(
                f"range={contact_range:.0f}m seed={seed} steps={row['executed_steps']} "
                f"direct={row['direct_fraction']:.3f} multi={row['multihop_fraction']:.3f} "
                f"disc={row['disconnected_fraction']:.3f} pdr={pdr_text}"
            )

    summary = summarize_calibration(rows)
    csv_path, json_path = write_calibration_outputs(rows, summary, args.output_dir)
    print("\nSummary")
    for row in summary:
        pdr_text = "n/a" if row["byte_pdr"] is None else f"{row['byte_pdr']:.3f}"
        print(
            f"range={row['contact_range_m']:.0f}m n={row['episodes']} "
            f"direct={row['direct_fraction']:.3f} multi={row['multihop_fraction']:.3f} "
            f"disc={row['disconnected_fraction']:.3f} hops={row['mean_gcs_hops']:.2f} "
            f"degree={row['mean_neighbor_degree']:.2f} pdr={pdr_text}"
        )
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")


if __name__ == "__main__":
    main()
