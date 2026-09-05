from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import torch

from uav_search.algorithms.factory import make_algorithm
from uav_search.envs.paper_env import PaperUAVEnv
from .train import deterministic_rollout, resolve_device
from .visualize import plot_trajectory


def evaluate_checkpoint(checkpoint: str | Path, episodes: int = 5000, device: str = "auto", output_dir: str | Path | None = None) -> dict[str, Any]:
    checkpoint = Path(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = payload["config"]
    algorithm = payload["algorithm"]
    device = resolve_device(device)
    env = PaperUAVEnv(cfg, seed=0)
    algo = make_algorithm(algorithm, env, cfg, device=device, seed=0)
    algo.load_checkpoint(payload)
    rows = []
    final_env = None
    for ep in range(int(episodes)):
        final_env, metrics = deterministic_rollout(algo, cfg, seed=200_000 + ep)
        rows.append({"episode": ep + 1, **metrics})
    numeric = [k for k, v in rows[0].items() if k != "episode" and isinstance(v, (int, float))]
    summary = {f"mean_{k}": sum(float(r[k]) for r in rows) / len(rows) for k in numeric}
    summary.update({"algorithm": algorithm, "episodes": int(episodes)})
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        with (out / "evaluation.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        with (out / "evaluation_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        assert final_env is not None
        plot_trajectory(
            final_env.trajectory, final_env.targets, final_env.obstacles, final_env.agents,
            final_env.agent_types, out / "trajectory.png", final_env.area_size_m,
        )
    return summary
