#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from uav_search.runtime import system_report


def main() -> None:
    p = argparse.ArgumentParser(description="Show the backend that UAV Search will use.")
    p.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    a = p.parse_args()
    report = system_report(a.device)
    print(json.dumps(report, indent=2, default=str))
    print()
    if report["device"] == "cpu":
        print("Recommendation: CPU mode is ready. Use --device cpu or --device auto.")
    else:
        print("Recommendation: CUDA detected. Use --device auto --amp auto for the fast default.")
        print("Use --deterministic only when strict reproducibility matters more than speed.")


if __name__ == "__main__":
    main()
