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
