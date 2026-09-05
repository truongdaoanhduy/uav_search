from __future__ import annotations

from typing import Any


def should_report_episode(episode: int, total_episodes: int, every: int) -> bool:
    """Return True for the first, periodic, and final episode."""
    if every < 1:
        raise ValueError("progress interval must be >= 1")
    return episode == 1 or episode == total_episodes or episode % every == 0


def format_episode_progress(metrics: dict[str, Any], total_episodes: int) -> str:
    """Format one compact, flush-friendly terminal progress line."""
    episode = int(metrics.get("episode", 0))
    progress_pct = 100.0 * episode / max(int(total_episodes), 1)
    return (
        f"[train] episode {episode}/{int(total_episodes)} ({progress_pct:.2f}%)"
        f" | return={float(metrics.get('return_mean', 0.0)):.3f}"
        f" | search={100.0 * float(metrics.get('search_rate', 0.0)):.1f}%"
        f" | targets={int(metrics.get('targets_found', 0))}"
        f" | {float(metrics.get('episode_sec', 0.0)):.2f}s"
        f" | {float(metrics.get('env_steps_per_sec', 0.0)):.1f} step/s"
    )


def format_startup_summary(
    *,
    algorithm: str,
    scenario: str,
    episodes: int,
    steps: int,
    device: str,
    device_name: str,
    amp_enabled: bool,
    deterministic: bool,
    tracking_status: str,
    project_url: str,
    run_url: str | None,
    run_dir: str,
) -> str:
    """Describe the resolved runtime before a long training run starts."""
    lines = [
        "=" * 72,
        "UAV SEARCH TRAINING",
        f"Algorithm={algorithm.upper()} | Scenario={scenario} | Episodes={episodes} | Steps={steps}",
        (
            f"Device={device} | Hardware={device_name} | "
            f"AMP={'on' if amp_enabled else 'off'} | "
            f"Deterministic={'on' if deterministic else 'off'}"
        ),
        f"W&B={tracking_status} | Project={project_url}",
    ]
    if run_url:
        lines.append(f"W&B run={run_url}")
    lines.extend([f"Local run={run_dir}", "=" * 72])
    return "\n".join(lines)
