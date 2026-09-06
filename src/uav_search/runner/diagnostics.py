from __future__ import annotations

import math
from typing import Any

import numpy as np


def _agent_returns(metrics: dict[str, Any]) -> dict[str, float]:
    prefix = "agent/"
    suffix = "/return"
    out: dict[str, float] = {}
    for key, value in metrics.items():
        if key.startswith(prefix) and key.endswith(suffix):
            name = key[len(prefix) : -len(suffix)]
            try:
                out[name] = float(value)
            except (TypeError, ValueError):
                continue
    return out


def diagnose_episode(metrics: dict[str, Any]) -> dict[str, Any]:
    """Infer where an under-performing episode most likely failed.

    This is intentionally diagnostic rather than part of the paper's reward or
    environment. It only interprets already logged state/reward/training signals.
    """
    returns = _agent_returns(metrics)
    if not returns:
        return {"failure_scope": "unknown", "worst_agent": "unknown", "primary_cause": "unknown"}

    worst_agent = min(returns, key=returns.get)
    critic_loss = float(metrics.get("critic_loss", 0.0))
    if not math.isfinite(critic_loss) or abs(critic_loss) >= 1e5:
        return {
            "failure_scope": "swarm",
            "worst_agent": worst_agent,
            "primary_cause": "rl_training_instability",
            "secondary_cause": "critic_unstable",
        }

    rotor_returns = {k: v for k, v in returns.items() if k.startswith("rotor_")}
    fixed_returns = {k: v for k, v in returns.items() if k.startswith("fixed_")}
    failure_scope = "single_agent"
    if rotor_returns:
        values = np.asarray(list(rotor_returns.values()), dtype=float)
        median = float(np.median(values))
        threshold = max(1.0, 0.25 * max(abs(median), 1.0))
        bad = [name for name, value in rotor_returns.items() if value < median - threshold]
        if len(bad) >= max(2, int(math.ceil(len(rotor_returns) / 2))):
            failure_scope = "rotor_group"
        elif worst_agent.startswith("fixed_") and fixed_returns:
            failure_scope = "fixed_wing"
    if returns and all(v < 0 for v in returns.values()):
        failure_scope = "swarm"

    prefix = f"agent/{worst_agent}/"
    comm = float(metrics.get(prefix + "comm_rate_mbps", 99.0))
    broken = float(metrics.get(prefix + "broken_link_s", 0.0))
    battery = float(metrics.get(prefix + "battery_pct", 100.0))
    safety = float(metrics.get(prefix + "safety_reward", 0.0))
    task = float(metrics.get(prefix + "task_reward", 0.0))
    saturation = float(metrics.get(prefix + "action_saturation", 0.0))

    causes: list[str] = []
    if worst_agent.startswith("rotor_") and (comm < 1.0 or broken >= 5.0):
        causes.append("communication")
    if int(metrics.get("collisions", 0)) > 0 or int(metrics.get("obstacle_hits", 0)) > 0 or safety < 0:
        causes.append("safety")
    if worst_agent.startswith("rotor_") and battery <= 15.0:
        causes.append("energy")
    if saturation >= 0.8:
        causes.append("control_saturation")
    if float(metrics.get("search_rate", 1.0)) <= 0.2 and task <= 0.0:
        causes.append("search")
    if not causes:
        causes.append("underperformance")

    return {
        "failure_scope": failure_scope,
        "worst_agent": worst_agent,
        "worst_agent_return": float(returns[worst_agent]),
        "primary_cause": causes[0],
        "secondary_cause": causes[1] if len(causes) > 1 else "none",
    }
