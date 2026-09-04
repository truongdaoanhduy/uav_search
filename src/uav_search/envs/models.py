from __future__ import annotations

import math
from typing import Any

import numpy as np

C = 299_792_458.0


def communication_rate_bps(horizontal_distance_m: float, vertical_distance_m: float, cfg: dict[str, Any]) -> float:
    """Paper Eq. (3)-(7): probabilistic LoS/NLoS path loss -> Shannon A2A rate."""
    p = cfg["paper"]
    a = cfg["assumed"]
    h = max(float(horizontal_distance_m), 1.0)
    z = abs(float(vertical_distance_m))
    d = max(math.hypot(h, z), 1.0)
    theta_deg = math.degrees(math.atan2(z, h))
    p_los = 1.0 / (1.0 + a["los_a"] * math.exp(-a["los_b"] * (theta_deg - a["los_a"])))
    fspl_db = 20.0 * math.log10(4.0 * math.pi * d * a["carrier_hz"] / C)
    avg_loss_db = fspl_db + p_los * a["los_extra_loss_db"] + (1.0 - p_los) * a["nlos_extra_loss_db"]
    path_gain = 10.0 ** (-avg_loss_db / 10.0)
    snr = p["communication_power_w"] * path_gain / max(a["noise_power_w"], 1e-20)
    return float(a["bandwidth_hz"] * math.log2(1.0 + max(snr, 0.0)))


def multirotor_power_w(speed_mps: float, accel_mps2: float, cfg: dict[str, Any]) -> float:
    """Paper Eq. (13) plus constant communication power."""
    p = cfg["paper"]
    a = cfg["assumed"]
    v = max(float(speed_mps), 0.0)
    acc = abs(float(accel_mps2))
    return float(
        a["hover_power_w"]
        + a["blade_drag_coeff"] * v**2
        + a["frame_drag_coeff"] * acc * v**3
        + p["communication_power_w"]
    )


def circle_collision(point_xy: np.ndarray, circle_xyr: np.ndarray) -> bool:
    return bool(np.linalg.norm(point_xy - circle_xyr[:2]) <= circle_xyr[2])
