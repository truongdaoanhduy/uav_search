#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from uav_search.config import load_config
from uav_search.dependencies import require_pinned_uavnetsim, uavnetsim_status
from uav_search.runtime import system_report


def main() -> None:
    p = argparse.ArgumentParser(description="Verify compute and U6/U9 network runtime readiness.")
    p.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    p.add_argument(
        "--require-uavnetsim",
        action="store_true",
        help="Exit non-zero unless the exact pinned lightweight UavNetSim runtime is installed.",
    )
    a = p.parse_args()
    report = system_report(a.device)
    report.update(uavnetsim_status())
    report["research_scenarios"] = {
        scenario: {
            "backend": load_config("masac", scenario)["scenario"]["network_backend"],
            "loadable": True,
        }
        for scenario in ("u6", "u9")
    }
    print(json.dumps(report, indent=2, default=str))
    print()
    print("Deterministic paper runs are the default; use --no-deterministic only for speed-first experiments.")
    if report["device"] == "cpu":
        print("Recommendation: CPU mode is ready. Use --device cpu or --device auto.")
    else:
        print("Recommendation: CUDA detected. Use --device auto --amp auto; deterministic mode stays enabled by default.")
    if report["lightweight_runtime_ready"]:
        print("UavNetSim: pinned lightweight MAC/PHY runtime is ready for u6/u9.")
    else:
        print("UavNetSim: NOT ready for default u6/u9. Run scripts/install_uavnetsim.sh.")
    if a.require_uavnetsim:
        require_pinned_uavnetsim()


if __name__ == "__main__":
    main()
