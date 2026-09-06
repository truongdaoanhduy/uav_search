from __future__ import annotations

import math
from typing import Any


def diagnose_episode(metrics: dict[str, Any]) -> dict[str, Any]:
    """Classify an episode using swarm/group aggregates rather than per-UAV telemetry.

    Diagnostic interpretation is not part of the paper reward or environment.
    It intentionally stays coarse so W&B remains readable at 50k episodes.
    """
    critic_loss = float(metrics.get("critic_loss", 0.0))
    if not math.isfinite(critic_loss) or abs(critic_loss) >= 1e5:
        return {
            "failure_scope": "swarm",
            "primary_cause": "rl_training_instability",
            "secondary_cause": "critic_unstable",
        }

    causes: list[str] = []
    failure_scope = "swarm"

    broken_uavs = int(metrics.get("broken_link_uavs", 0))
    comm = float(metrics.get("mean_comm_rate_mbps", 99.0))
    broken_time = float(metrics.get("mean_broken_link_s", 0.0))
    if broken_uavs > 0 or comm < 1.0 or broken_time >= 5.0:
        causes.append("communication")
        failure_scope = "rotor_group"

    safety_uavs = (
        int(metrics.get("collided_uavs", 0))
        + int(metrics.get("obstacle_hit_uavs", 0))
        + int(metrics.get("boundary_hit_uavs", 0))
    )
    if safety_uavs > 0 or float(metrics.get("reward_safety_sum", 0.0)) < 0.0:
        causes.append("safety")

    if int(metrics.get("depleted_uavs", 0)) > 0 or float(metrics.get("avg_battery_pct", 100.0)) <= 15.0:
        causes.append("energy")
        if not causes[:-1]:
            failure_scope = "rotor_group"

    if float(metrics.get("action_saturation", 0.0)) >= 0.8:
        causes.append("control_saturation")

    if float(metrics.get("search_rate", 1.0)) <= 0.2 and float(metrics.get("reward_task_sum", 0.0)) <= 0.0:
        causes.append("search")
        if failure_scope == "swarm":
            fixed_return = float(metrics.get("fixed_return_mean", 0.0))
            rotor_return = float(metrics.get("rotor_return_mean", 0.0))
            scale = max(abs(fixed_return), abs(rotor_return), 1.0)
            if rotor_return < fixed_return - 0.25 * scale:
                failure_scope = "rotor_group"
            elif fixed_return < rotor_return - 0.25 * scale:
                failure_scope = "fixed_wing"

    if not causes:
        causes.append("underperformance")

    return {
        "failure_scope": failure_scope,
        "primary_cause": causes[0],
        "secondary_cause": causes[1] if len(causes) > 1 else "none",
    }
