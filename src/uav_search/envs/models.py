from __future__ import annotations

import math
from typing import Any

import numpy as np

C = 299_792_458.0


def _channel_value(cfg: dict[str, Any], key: str) -> float:
    """Return a channel constant with explicit provenance preference."""
    if key in cfg.get("reference_backed", {}):
        return float(cfg["reference_backed"][key])
    return float(cfg["assumed"][key])


def path_gain_linear(
    horizontal_distance_m: float,
    vertical_distance_m: float,
    cfg: dict[str, Any],
    *,
    force_los: bool = False,
) -> float:
    """Paper Eqs. (3)-(5): probabilistic LoS/NLoS average path loss as linear gain."""
    h = max(abs(float(horizontal_distance_m)), 0.0)
    z = abs(float(vertical_distance_m))
    d = max(math.hypot(h, z), 1.0)
    if force_los:
        p_los = 1.0
    else:
        theta_deg = math.degrees(math.atan2(z, max(h, 1e-12)))
        a = _channel_value(cfg, "los_a")
        b = _channel_value(cfg, "los_b")
        p_los = 1.0 / (1.0 + a * math.exp(-b * (theta_deg - a)))
    carrier_hz = _channel_value(cfg, "carrier_hz")
    fspl_db = 20.0 * math.log10(4.0 * math.pi * d * carrier_hz / C)
    los_db = fspl_db + _channel_value(cfg, "los_extra_loss_db")
    nlos_db = fspl_db + _channel_value(cfg, "nlos_extra_loss_db")
    avg_loss_db = p_los * los_db + (1.0 - p_los) * nlos_db
    return float(10.0 ** (-avg_loss_db / 10.0))


def communication_sinr(
    horizontal_distance_m: float,
    vertical_distance_m: float,
    cfg: dict[str, Any],
    *,
    interference_power_w: float = 0.0,
    force_los: bool = False,
    tx_power_w: float | None = None,
) -> float:
    """Paper Eq. (6): A2A SINR using received signal, interference, and Gaussian noise."""
    gain = path_gain_linear(horizontal_distance_m, vertical_distance_m, cfg, force_los=force_los)
    # Eq. (6) uses transmit power P_tx,u. The root paper does not publish it,
    # so the numerical fallback comes from root-paper ref. [39] (40 dBm = 10 W).
    # Table I's P_com=5 W remains a separate communication-energy term.
    power_w = _channel_value(cfg, "tx_power_w") if tx_power_w is None else max(float(tx_power_w), 0.0)
    signal_w = power_w * gain
    noise_w = max(_channel_value(cfg, "noise_power_w"), 1e-20)
    denominator = max(float(interference_power_w), 0.0) + noise_w
    return float(signal_w / denominator)


def communication_rate_bps(
    horizontal_distance_m: float,
    vertical_distance_m: float,
    cfg: dict[str, Any],
    *,
    interference_power_w: float = 0.0,
    force_los: bool = False,
    tx_power_w: float | None = None,
) -> float:
    """Paper Eq. (7): Shannon A2A rate computed from the paper's SINR model."""
    sinr = communication_sinr(
        horizontal_distance_m,
        vertical_distance_m,
        cfg,
        interference_power_w=interference_power_w,
        force_los=force_los,
        tx_power_w=tx_power_w,
    )
    bandwidth_hz = _channel_value(cfg, "bandwidth_hz")
    return float(bandwidth_hz * math.log2(1.0 + max(sinr, 0.0)))


def multirotor_power_w(
    speed_mps: float,
    accel_mps2: float,
    cfg: dict[str, Any],
    *,
    include_communication_power: bool = True,
) -> float:
    """Paper Eqs. (13)-(14) with optional separation of radio energy.

    Paper-faithful scenarios retain the root paper's constant ``P_com`` term.
    The peer/DTN research scenario disables that constant and accounts radio
    transmission energy through the selected network backend instead.
    """
    p = cfg["paper"]
    a = cfg["assumed"]
    v = max(float(speed_mps), 0.0)
    acc = abs(float(accel_mps2))
    return float(
        a["hover_power_w"]
        + a["blade_drag_coeff"] * v**2
        + a["frame_drag_coeff"] * acc * v**3
        + (p["communication_power_w"] if include_communication_power else 0.0)
    )


def circle_collision(point_xy: np.ndarray, circle_xyr: np.ndarray) -> bool:
    return bool(np.linalg.norm(point_xy - circle_xyr[:2]) <= circle_xyr[2])


def segment_circle_collision(start_xy: np.ndarray, end_xy: np.ndarray, circle_xyr: np.ndarray) -> bool:
    """Return whether a planar movement segment intersects a circular obstacle."""
    start = np.asarray(start_xy, dtype=np.float64)[:2]
    end = np.asarray(end_xy, dtype=np.float64)[:2]
    circle = np.asarray(circle_xyr, dtype=np.float64)
    delta = end - start
    denom = float(np.dot(delta, delta))
    if denom <= 1e-12:
        return circle_collision(start, circle)
    t = float(np.clip(np.dot(circle[:2] - start, delta) / denom, 0.0, 1.0))
    closest = start + t * delta
    return bool(np.linalg.norm(closest - circle[:2]) <= circle[2])
segment_circle_intersects = segment_circle_collision
