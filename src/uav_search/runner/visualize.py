from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle


def _curve(df: pd.DataFrame, x: str, y: str, ylabel: str, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(df[x], df[y], linewidth=1.1, alpha=0.45, label="raw")
    if len(df) >= 5:
        window = min(500, max(3, len(df) // 20))
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


def _multi_curve(df: pd.DataFrame, columns: list[tuple[str, str]], ylabel: str, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for column, label in columns:
        if column in df.columns:
            ax.plot(df["episode"], df[column], linewidth=1.5, label=label)
    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_training_curves(metrics_csv: str | Path, output_dir: str | Path) -> list[Path]:
    df = pd.read_csv(metrics_csv)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    reward_col = "return_sum" if "return_sum" in df.columns else "return_mean"
    energy_col = "energy_consumption_pct" if "energy_consumption_pct" in df.columns else "energy_j"
    energy_label = "Average multi-rotor battery consumption (%)" if energy_col == "energy_consumption_pct" else "Multi-rotor energy (J)"
    specs = [
        (reward_col, "Team episode return", "reward.png"),
        ("search_rate", "Target search rate", "search_rate.png"),
        (energy_col, energy_label, "energy.png"),
        ("mean_broken_link_s", "Mean broken-link duration (s)", "broken_link.png"),
    ]
    paths = [_curve(df, "episode", col, label, out / name) for col, label, name in specs if col in df.columns]
    group_cols = [("fixed_return_mean", "fixed-wing"), ("rotor_return_mean", "rotor mean"), ("rotor_return_min", "rotor worst")]
    if any(col in df.columns for col, _ in group_cols):
        paths.append(_multi_curve(df, group_cols, "Episode return", out / "group_returns.png"))
    reward_cols = [
        ("reward_task_sum", "task"),
        ("reward_communication_sum", "communication"),
        ("reward_energy_sum", "energy"),
        ("reward_safety_sum", "safety"),
    ]
    if any(col in df.columns for col, _ in reward_cols):
        paths.append(_multi_curve(df, reward_cols, "Accumulated reward component", out / "reward_components.png"))
    return paths


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


def plot_simulation_scenario(
    positions: np.ndarray,
    targets: np.ndarray,
    obstacles: np.ndarray,
    agent_names: Iterable[str],
    agent_types: Iterable[int],
    rotor_leaders: Iterable[int],
    output_path: str | Path,
    area_size_m: float,
) -> Path:
    """Fig. 6-style software scenario sanity plot (not a physical experiment)."""
    positions = np.asarray(positions)
    names = list(agent_names)
    types = list(agent_types)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 7))
    for circle in np.asarray(obstacles):
        ax.add_patch(Circle((circle[0], circle[1]), circle[2], fill=False, linewidth=1.2, alpha=0.7))
    if len(targets):
        ax.scatter(targets[:, 0], targets[:, 1], marker="*", s=85, label="Target")
    fixed_indices = [i for i, typ in enumerate(types) if int(typ) == 0]
    rotor_indices = [i for i, typ in enumerate(types) if int(typ) == 1]
    for local, rotor_idx in enumerate(rotor_indices):
        leaders = list(rotor_leaders)
        if local < len(leaders) and int(leaders[local]) >= 0:
            leader = int(leaders[local])
            ax.plot(
                [positions[leader, 0], positions[rotor_idx, 0]],
                [positions[leader, 1], positions[rotor_idx, 1]],
                linestyle="--", linewidth=0.8, alpha=0.6,
            )
    if fixed_indices:
        ax.scatter(positions[fixed_indices, 0], positions[fixed_indices, 1], marker="^", s=90, label="Relay fixed-wing UAV")
    if rotor_indices:
        ax.scatter(positions[rotor_indices, 0], positions[rotor_indices, 1], marker="o", s=45, label="Search multi-rotor UAV")
    ax.set_xlim(0, area_size_m); ax.set_ylim(0, area_size_m)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title("Fig. 6-style post-disaster simulation scenario")
    ax.grid(alpha=0.2); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=180); plt.close(fig)
    return out


def _load_eval_summary(run_dir: Path, evaluation_dir_name: str) -> dict[str, float]:
    path = run_dir / evaluation_dir_name / "evaluation_summary.json"
    if not path.exists():
        # Development fallback: one deterministic rollout written by train.py.
        fallback = run_dir / "evaluation.json"
        if not fallback.exists():
            raise FileNotFoundError(f"Missing evaluation summary for {run_dir}")
        return json.loads(fallback.read_text(encoding="utf-8"))
    return json.loads(path.read_text(encoding="utf-8"))


def plot_paper_comparison(
    run_dirs: Mapping[tuple[str, str], str | Path],
    output_dir: str | Path,
    evaluation_dir_name: str = "evaluation_5000",
) -> list[Path]:
    """Recreate the software-result plots relevant to current Fig. 7 scope."""
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    algorithms = [a for a in ("masac", "matd3", "maddpg") if any(k[0] == a for k in run_dirs)]
    scenarios = [s for s in ("f1_m5", "f1_m9") if any(k[1] == s for k in run_dirs)]

    fig, axes = plt.subplots(1, len(scenarios), figsize=(6.5 * len(scenarios), 4.2), squeeze=False)
    for ax, scenario in zip(axes[0], scenarios):
        for algorithm in algorithms:
            key = (algorithm, scenario)
            if key not in run_dirs:
                continue
            df = pd.read_csv(Path(run_dirs[key]) / "metrics" / "episodes.csv")
            y = "return_sum" if "return_sum" in df.columns else "return_mean"
            window = min(500, max(1, len(df) // 50))
            values = df[y].rolling(window=window, min_periods=1).mean()
            ax.plot(df["episode"], values, linewidth=1.5, label=algorithm.upper())
        ax.set_title("(1,5)" if scenario == "f1_m5" else "(1,9)")
        ax.set_xlabel("Episode"); ax.set_ylabel("Reward Value"); ax.grid(alpha=0.25); ax.legend(fontsize=8)
    fig.tight_layout()
    fig7 = out / "fig07_training_reward.png"; fig.savefig(fig7, dpi=180); plt.close(fig)

    summaries = {(a, s): _load_eval_summary(Path(path), evaluation_dir_name) for (a, s), path in run_dirs.items()}

    def grouped_bars(metric: str, ylabel: str, filename: str) -> Path:
        fig, ax = plt.subplots(figsize=(7.4, 4.4))
        x = np.arange(len(scenarios), dtype=float)
        width = 0.8 / max(1, len(algorithms))
        for j, algorithm in enumerate(algorithms):
            values = [float(summaries[(algorithm, scenario)].get(metric, 0.0)) for scenario in scenarios]
            offset = (j - (len(algorithms) - 1) / 2) * width
            ax.bar(x + offset, values, width=width, label=algorithm.upper())
        ax.set_xticks(x, ["(1,5)", "(1,9)"][: len(scenarios)])
        ax.set_xlabel("(Number of Fixed-Wing UAVs, Number of Multi-Rotor UAVs)")
        ax.set_ylabel(ylabel); ax.grid(axis="y", alpha=0.25); ax.legend(fontsize=8)
        fig.tight_layout(); path = out / filename; fig.savefig(path, dpi=180); plt.close(fig); return path

    return [
        fig7,
        grouped_bars("mean_targets_found", "Average Number of Targets Found", "fig10_average_targets_found.png"),
        grouped_bars("mean_energy_consumption_pct", "Average Energy Consumption (%)", "fig11_average_energy_consumption.png"),
        grouped_bars("mean_mean_broken_link_s", "Average Broken Link Duration (s)", "fig12_average_broken_link_duration.png"),
    ]
