from __future__ import annotations

import csv
import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from uav_search.config import load_config
from uav_search.envs.paper_env import PaperUAVEnv


def tx_power_to_action(power_w: float, minimum_w: float, maximum_w: float) -> float:
    """Map a physical transmit-power value back into the u6 continuous action slot."""
    minimum = float(minimum_w)
    maximum = float(maximum_w)
    if maximum <= minimum:
        return -1.0
    clipped = float(np.clip(float(power_w), minimum, maximum))
    return float(2.0 * (clipped - minimum) / (maximum - minimum) - 1.0)


class RandomWaypointCalibrationPolicy:
    """Deterministic-by-seed non-learning policy used only to exercise u6 topology."""

    def __init__(self, env: PaperUAVEnv, seed: int, tx_power_w: float, traffic: bool = False):
        self.rng = np.random.default_rng(int(seed))
        self.area = float(env.area_size_m)
        self.waypoints = self.rng.uniform(0.05 * self.area, 0.95 * self.area, size=(env.n_agents, 2))
        self.traffic = bool(traffic)
        self.tx_power_action = tx_power_to_action(
            tx_power_w,
            float(env.scenario["tx_power_min_w"]),
            float(env.scenario["tx_power_max_w"]),
        )

    def actions(self, env: PaperUAVEnv) -> dict[str, np.ndarray]:
        actions: dict[str, np.ndarray] = {}
        for idx, agent in enumerate(env.agents):
            if not bool(env.uav_active[idx]):
                actions[agent] = np.array([-1.0, 0.0, -1.0, self.tx_power_action, -1.0], dtype=np.float32)
                continue
            delta = self.waypoints[idx] - env.positions[idx, :2]
            if float(np.linalg.norm(delta)) < 150.0:
                self.waypoints[idx] = self.rng.uniform(0.05 * self.area, 0.95 * self.area, size=2)
                delta = self.waypoints[idx] - env.positions[idx, :2]
            angle = math.atan2(float(delta[1]), float(delta[0]))
            direction_action = float(np.clip(angle / math.pi, -1.0, 1.0))
            recipient_action = float(self.rng.uniform(-1.0, 1.0))
            actions[agent] = np.array(
                [1.0, direction_action, 1.0 if self.traffic else -1.0, self.tx_power_action, recipient_action],
                dtype=np.float32,
            )
        return actions


def run_calibration_episode(
    *,
    contact_range_m: float,
    seed: int,
    steps: int = 600,
    tx_power_w: float = 0.1,
    traffic: bool = False,
) -> dict[str, Any]:
    """Run one non-learning u6 episode with the real UavNetSim backend."""
    cfg = deepcopy(load_config("masac", "u6"))
    cfg["scenario"]["network_backend"] = "uavnetsim"
    cfg["scenario"]["peer_contact_range_m"] = float(contact_range_m)
    cfg["scenario"]["gcs_contact_range_m"] = float(contact_range_m)
    cfg["runtime"]["episode_steps_override"] = int(steps)

    env = PaperUAVEnv(cfg, seed=int(seed))
    env.reset(seed=int(seed))
    policy = RandomWaypointCalibrationPolicy(
        env, seed=int(seed), tx_power_w=float(tx_power_w), traffic=bool(traffic)
    )

    direct = multihop = disconnected = 0
    connected_hop_sum = 0.0
    connected_hop_count = 0
    neighbor_degree_sum = 0.0
    neighbor_degree_samples = 0
    attempted = delivered = 0
    delay_weighted_sum = 0.0
    phy_failures = 0
    tx_energy_j = 0.0
    info: dict[str, Any] = {}
    executed = 0

    for _ in range(int(steps)):
        _, _, terminated, truncated, info = env.step(policy.actions(env))
        executed += 1
        direct += int(info["direct_gcs_uavs"])
        multihop += int(info["multihop_gcs_uavs"])
        disconnected += int(info["disconnected_gcs_uavs"])

        hops = env._gcs_hops()
        positive = hops[hops > 0]
        connected_hop_sum += float(positive.sum())
        connected_hop_count += int(positive.size)
        neighbor_degree_sum += float(env.last_adjacency.sum())
        neighbor_degree_samples += int(env.n_agents)

        step_attempted = int(info["network_attempted_bytes"])
        step_delivered = int(info["network_delivered_bytes"])
        attempted += step_attempted
        delivered += step_delivered
        delay_weighted_sum += float(info["network_mean_delay_s"]) * step_delivered
        phy_failures += int(info["network_phy_failures"])
        tx_energy_j += float(info["network_tx_energy_j"])

        if all(terminated.values()) or all(truncated.values()):
            break

    node_steps = max(1, direct + multihop + disconnected)
    return {
        "contact_range_m": float(contact_range_m),
        "seed": int(seed),
        "requested_steps": int(steps),
        "executed_steps": int(executed),
        "tx_power_w": float(tx_power_w),
        "traffic_enabled": bool(traffic),
        "network_backend": str(info.get("network_backend", env.network_backend.name)),
        "direct_node_steps": int(direct),
        "multihop_node_steps": int(multihop),
        "disconnected_node_steps": int(disconnected),
        "direct_fraction": float(direct / node_steps),
        "multihop_fraction": float(multihop / node_steps),
        "disconnected_fraction": float(disconnected / node_steps),
        "connected_hop_sum": float(connected_hop_sum),
        "connected_hop_count": int(connected_hop_count),
        "mean_gcs_hops": float(connected_hop_sum / connected_hop_count) if connected_hop_count else 0.0,
        "neighbor_degree_sum": float(neighbor_degree_sum),
        "neighbor_degree_samples": int(neighbor_degree_samples),
        "mean_neighbor_degree": float(neighbor_degree_sum / neighbor_degree_samples) if neighbor_degree_samples else 0.0,
        "attempted_bytes": int(attempted),
        "delivered_bytes": int(delivered),
        "traffic_metrics_available": bool(attempted > 0),
        "byte_pdr": float(delivered / attempted) if attempted else None,
        "throughput_bps": (
            float(delivered * 8.0 / max(executed * env.dt, 1e-12)) if attempted else None
        ),
        "delay_weighted_sum": float(delay_weighted_sum),
        "mean_delay_s": float(delay_weighted_sum / delivered) if delivered else None,
        "phy_failures": int(phy_failures),
        "network_tx_energy_j": float(tx_energy_j),
        "network_sim_time_s": float(info.get("network_sim_time_s", env.network_backend.simulation_time_s)),
        "mean_broken_link_s": float(info.get("mean_broken_link_s", 0.0)),
        "targets_found": int(info.get("targets_found", 0)),
        "reports_delivered": int(info.get("reports_delivered", 0)),
        "termination_reason": str(info.get("termination_reason", "")),
    }


def summarize_calibration(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[float, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(float(row["contact_range_m"]), []).append(row)

    summary: list[dict[str, Any]] = []
    for contact_range, group in sorted(groups.items()):
        direct = sum(int(r["direct_node_steps"]) for r in group)
        multihop = sum(int(r["multihop_node_steps"]) for r in group)
        disconnected = sum(int(r["disconnected_node_steps"]) for r in group)
        node_steps = max(1, direct + multihop + disconnected)
        attempted = sum(int(r["attempted_bytes"]) for r in group)
        delivered = sum(int(r["delivered_bytes"]) for r in group)
        hop_sum = sum(float(r["connected_hop_sum"]) for r in group)
        hop_count = sum(int(r["connected_hop_count"]) for r in group)
        degree_sum = sum(float(r["neighbor_degree_sum"]) for r in group)
        degree_count = sum(int(r["neighbor_degree_samples"]) for r in group)
        delay_sum = sum(float(r["delay_weighted_sum"]) for r in group)
        total_time = sum(float(r["network_sim_time_s"]) for r in group)
        summary.append({
            "contact_range_m": float(contact_range),
            "episodes": int(len(group)),
            "direct_fraction": float(direct / node_steps),
            "multihop_fraction": float(multihop / node_steps),
            "disconnected_fraction": float(disconnected / node_steps),
            "mean_gcs_hops": float(hop_sum / hop_count) if hop_count else 0.0,
            "mean_neighbor_degree": float(degree_sum / degree_count) if degree_count else 0.0,
            "attempted_bytes": int(attempted),
            "delivered_bytes": int(delivered),
            "traffic_metrics_available": bool(attempted > 0),
            "byte_pdr": float(delivered / attempted) if attempted else None,
            "throughput_bps": (
                float(delivered * 8.0 / total_time) if attempted and total_time > 0 else None
            ),
            "mean_delay_s": float(delay_sum / delivered) if delivered else None,
            "phy_failures": int(sum(int(r["phy_failures"]) for r in group)),
            "network_tx_energy_j": float(sum(float(r["network_tx_energy_j"]) for r in group)),
            "mean_broken_link_s": float(np.mean([float(r["mean_broken_link_s"]) for r in group])),
        })
    return summary


def write_calibration_outputs(
    rows: list[dict[str, Any]],
    summary: list[dict[str, Any]],
    output_dir: str | Path,
) -> tuple[Path, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "episodes.csv"
    json_path = output / "summary.json"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    else:
        csv_path.write_text("", encoding="utf-8")
    json_path.write_text(json.dumps({"episodes": rows, "summary": summary}, indent=2), encoding="utf-8")
    return csv_path, json_path
