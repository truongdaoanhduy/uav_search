from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle


def _curve(df: pd.DataFrame, x: str, y: str, ylabel: str, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(df[x], df[y], linewidth=1.4)
    if len(df) >= 5:
        window = min(20, max(3, len(df) // 10))
        smooth = df[y].rolling(window=window, min_periods=1).mean()
        ax.plot(df[x], smooth, linewidth=2.0, label=f"rolling mean ({window})")
        ax.legend()
    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_training_curves(metrics_csv: str | Path, output_dir: str | Path) -> list[Path]:
    df = pd.read_csv(metrics_csv)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    specs = [
        ("return_mean", "Mean episode return", "reward.png"),
        ("search_rate", "Target search rate", "search_rate.png"),
        ("energy_j", "Multi-rotor energy (J)", "energy.png"),
        ("mean_broken_link_s", "Mean broken-link duration (s)", "broken_link.png"),
    ]
    return [_curve(df, "episode", col, label, out / name) for col, label, name in specs if col in df.columns]


def plot_trajectory(
    trajectory: np.ndarray,
    targets: np.ndarray,
    obstacles: np.ndarray,
    agent_names: Iterable[str],
    agent_types: Iterable[int],
    output_path: str | Path,
    area_size_m: float,
) -> Path:
    trajectory = np.asarray(trajectory)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 7))
    for circle in np.asarray(obstacles):
        ax.add_patch(Circle((circle[0], circle[1]), circle[2], fill=False, linewidth=1.0, alpha=0.6))
    if len(targets):
        ax.scatter(targets[:, 0], targets[:, 1], marker="*", s=90, label="Targets")
    for i, (name, typ) in enumerate(zip(agent_names, agent_types)):
        marker = "^" if int(typ) == 0 else "o"
        ax.plot(trajectory[:, i, 0], trajectory[:, i, 1], linewidth=1.1, label=name)
        ax.scatter(trajectory[-1, i, 0], trajectory[-1, i, 1], marker=marker, s=35)
    ax.set_xlim(0, area_size_m)
    ax.set_ylim(0, area_size_m)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Final evaluation trajectory")
    if trajectory.shape[1] <= 10:
        ax.legend(fontsize=7, loc="best")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out
