from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class SensingProfile:
    level_index: int
    reference_altitude_m: float
    fov_cells: int
    pd: float
    pf: float


@dataclass(frozen=True)
class ContinuousSensingProfile:
    altitude_m: float
    fov_radius_m: float
    pd: float
    pf: float


def _validate_profiles(
    levels_m: Sequence[float],
    fov_sizes: Sequence[int],
    pd_values: Sequence[float],
    pf_values: Sequence[float],
) -> None:
    n = len(levels_m)
    if n == 0 or len(fov_sizes) != n or len(pd_values) != n or len(pf_values) != n:
        raise ValueError("sensing profile arrays must be non-empty and have equal length")
    levels = np.asarray(levels_m, dtype=np.float64)
    if np.any(~np.isfinite(levels)) or np.any(np.diff(levels) <= 0.0):
        raise ValueError("sensing altitude levels must be finite and strictly increasing")
    for size in fov_sizes:
        if int(size) not in (1, 5, 9):
            raise ValueError("supported sensing FOV sizes are 1, 5, and 9 cells")
    for pd, pf in zip(pd_values, pf_values):
        if not (0.0 < float(pd) < 1.0 and 0.0 < float(pf) < 1.0):
            raise ValueError("Pd and Pf must lie strictly between zero and one")
        if float(pd) <= float(pf):
            raise ValueError("Pd must exceed Pf for an informative sensor")


def profile_for_altitude(
    z_m: float,
    levels_m: Sequence[float],
    fov_sizes: Sequence[int],
    pd_values: Sequence[float],
    pf_values: Sequence[float],
) -> SensingProfile:
    """Return the configured low/mid/high sensing profile nearest to ``z_m``."""
    _validate_profiles(levels_m, fov_sizes, pd_values, pf_values)
    levels = np.asarray(levels_m, dtype=np.float64)
    idx = int(np.argmin(np.abs(levels - float(z_m))))
    return SensingProfile(
        level_index=idx,
        reference_altitude_m=float(levels[idx]),
        fov_cells=int(fov_sizes[idx]),
        pd=float(pd_values[idx]),
        pf=float(pf_values[idx]),
    )


def _validate_continuous_anchors(
    levels_m: Sequence[float],
    pd_values: Sequence[float],
    pf_values: Sequence[float],
) -> np.ndarray:
    n = len(levels_m)
    if n < 2 or len(pd_values) != n or len(pf_values) != n:
        raise ValueError("continuous sensing anchors must contain at least two aligned altitude/Pd/Pf values")
    levels = np.asarray(levels_m, dtype=np.float64)
    pd = np.asarray(pd_values, dtype=np.float64)
    pf = np.asarray(pf_values, dtype=np.float64)
    if np.any(~np.isfinite(levels)) or np.any(np.diff(levels) <= 0.0):
        raise ValueError("continuous sensing altitude anchors must be finite and strictly increasing")
    if np.any(~np.isfinite(pd)) or np.any(~np.isfinite(pf)):
        raise ValueError("continuous sensing probability anchors must be finite")
    if np.any((pd <= 0.0) | (pd >= 1.0)) or np.any((pf <= 0.0) | (pf >= 1.0)):
        raise ValueError("Pd and Pf anchors must lie strictly between zero and one")
    if np.any(pd <= pf):
        raise ValueError("Pd must exceed Pf at every continuous sensing anchor")
    if np.any(np.diff(pd) > 1e-12):
        raise ValueError("Pd anchors must be monotonically non-increasing with altitude")
    if np.any(np.diff(pf) < -1e-12):
        raise ValueError("Pf anchors must be monotonically non-decreasing with altitude")
    return levels


def continuous_profile_for_altitude(
    z_m: float,
    levels_m: Sequence[float],
    pd_values: Sequence[float],
    pf_values: Sequence[float],
    *,
    full_fov_deg: float,
) -> ContinuousSensingProfile:
    """Return a continuous altitude-aware camera/sensor profile.

    The ground-footprint radius follows Hu et al.'s nadir-camera geometry,
    ``r = h * tan(FOV/2)``. Detection and false-alarm probabilities are
    piecewise-linearly interpolated through the configured Liu et al. anchor
    profiles so the published low/mid/high values are preserved exactly while
    intermediate altitudes no longer snap to a nearest level.
    """
    levels = _validate_continuous_anchors(levels_m, pd_values, pf_values)
    z = float(z_m)
    if not math.isfinite(z):
        raise ValueError("altitude must be finite")
    if z < float(levels[0]) - 1e-9 or z > float(levels[-1]) + 1e-9:
        raise ValueError(
            f"altitude {z} m is outside continuous sensing calibration range "
            f"[{float(levels[0])}, {float(levels[-1])}] m"
        )
    full_fov = float(full_fov_deg)
    if not math.isfinite(full_fov) or not (0.0 < full_fov < 180.0):
        raise ValueError("full_fov_deg must lie strictly between 0 and 180 degrees")

    # Clamp tiny floating-point boundary excursions after validating the model
    # domain, then interpolate exactly through the published anchor values.
    z_interp = float(np.clip(z, levels[0], levels[-1]))
    pd = float(np.interp(z_interp, levels, np.asarray(pd_values, dtype=np.float64)))
    pf = float(np.interp(z_interp, levels, np.asarray(pf_values, dtype=np.float64)))
    half_angle_rad = math.radians(full_fov * 0.5)
    radius_m = float(z_interp * math.tan(half_angle_rad))
    return ContinuousSensingProfile(
        altitude_m=z_interp,
        fov_radius_m=radius_m,
        pd=pd,
        pf=pf,
    )


def continuous_fov_offsets(
    fov_radius_m: float, grid_cell_m: float
) -> tuple[tuple[int, int], ...]:
    """Legacy center-offset rasterization helper for diagnostics/discrete use.

    Homogeneous-peer sensing uses :func:`continuous_fov_cells` instead because
    the physical footprint must be evaluated at the UAV's exact continuous XY
    position rather than implicitly snapping the vehicle to a grid-cell center.
    """
    radius = float(fov_radius_m)
    cell = float(grid_cell_m)
    if not math.isfinite(radius) or radius < 0.0:
        raise ValueError("fov_radius_m must be finite and non-negative")
    if not math.isfinite(cell) or cell <= 0.0:
        raise ValueError("grid_cell_m must be finite and positive")
    max_offset = int(math.ceil(radius / cell))
    tolerance = max(1e-9, 1e-12 * max(radius, cell))
    offsets = [
        (dy, dx)
        for dy in range(-max_offset, max_offset + 1)
        for dx in range(-max_offset, max_offset + 1)
        if math.hypot(dx * cell, dy * cell) <= radius + tolerance
    ]
    if (0, 0) not in offsets:
        offsets.append((0, 0))
    return tuple(offsets)


def continuous_fov_cells(
    center_xy: np.ndarray,
    fov_radius_m: float,
    grid_cell_m: float,
    grid_n: int,
) -> tuple[tuple[int, int], ...]:
    """Return grid cells intersected by a circular footprint at an exact XY pose.

    A cell is included when the footprint overlaps any non-zero part of its
    square.  Unlike offset rasterization, this remains correct when the UAV is
    close to a cell boundary instead of implicitly snapping it to a cell center.
    """
    center = np.asarray(center_xy, dtype=np.float64)
    radius = float(fov_radius_m)
    cell = float(grid_cell_m)
    size = int(grid_n)
    if center.shape != (2,) or np.any(~np.isfinite(center)):
        raise ValueError("center_xy must contain two finite coordinates")
    if not math.isfinite(radius) or radius < 0.0:
        raise ValueError("fov_radius_m must be finite and non-negative")
    if not math.isfinite(cell) or cell <= 0.0:
        raise ValueError("grid_cell_m must be finite and positive")
    if size <= 0:
        raise ValueError("grid_n must be positive")

    min_x = max(0, int(math.floor((center[0] - radius) / cell)))
    max_x = min(size - 1, int(math.floor((center[0] + radius) / cell)))
    min_y = max(0, int(math.floor((center[1] - radius) / cell)))
    max_y = min(size - 1, int(math.floor((center[1] + radius) / cell)))
    tolerance = max(1e-9, 1e-12 * max(radius, cell))
    cells: list[tuple[int, int]] = []
    for y in range(min_y, max_y + 1):
        y0, y1 = y * cell, (y + 1) * cell
        closest_y = float(np.clip(center[1], y0, y1))
        for x in range(min_x, max_x + 1):
            x0, x1 = x * cell, (x + 1) * cell
            closest_x = float(np.clip(center[0], x0, x1))
            distance = math.hypot(center[0] - closest_x, center[1] - closest_y)
            # Require positive-area overlap. A cell that only touches the circular
            # footprint tangentially has zero covered area and must not generate a
            # sensor measurement. Preserve the containing cell for the degenerate
            # zero-radius case.
            if (radius <= tolerance and distance <= tolerance) or distance < radius - tolerance:
                cells.append((y, x))
    return tuple(cells)


def fov_offsets(size: int) -> tuple[tuple[int, int], ...]:
    """Return a fixed local-grid footprint for the paper's 1/5/9-cell FOV sizes."""
    size = int(size)
    if size == 1:
        return ((0, 0),)
    if size == 5:
        return ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))
    if size == 9:
        return tuple((dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1))
    raise ValueError("supported sensing FOV sizes are 1, 5, and 9 cells")


def bayes_update(prior: float, measurement: bool, pd: float, pf: float) -> float:
    """Bayesian occupancy update using detection and false-alarm probabilities."""
    p = float(np.clip(prior, 0.0, 1.0))
    pd = float(pd)
    pf = float(pf)
    if not (0.0 < pd < 1.0 and 0.0 < pf < 1.0):
        raise ValueError("Pd and Pf must lie strictly between zero and one")
    if measurement:
        numerator = pd * p
        denominator = numerator + pf * (1.0 - p)
    else:
        numerator = (1.0 - pd) * p
        denominator = numerator + (1.0 - pf) * (1.0 - p)
    if denominator <= 0.0:
        return p
    return float(np.clip(numerator / denominator, 0.0, 1.0))


def binary_entropy(probability: float) -> float:
    """Binary Shannon entropy in bits, stable at p=0 and p=1."""
    p = float(np.clip(probability, 0.0, 1.0))
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return float(-p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p))
