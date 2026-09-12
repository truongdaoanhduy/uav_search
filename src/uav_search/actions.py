from __future__ import annotations

import numpy as np
import torch

PEER_ACTION_DIM = 6
PEER_ACTION_NAMES = (
    "acceleration_x",
    "acceleration_y",
    "acceleration_z",
    "transmit_gate",
    "transmit_power",
    "recipient",
)
PEER_CONTINUOUS_INDICES = (0, 1, 2, 4)
PEER_DISCRETE_INDICES = (3, 5)
PEER_ACTION_SCHEMA = "peer_cartesian_hybrid_v2"
LEGACY_ACTION_SCHEMA = "legacy_paper_action_v1"
CHECKPOINT_FORMAT_VERSION = 2


def peer_recipient_bin_centers(n_agents: int) -> np.ndarray:
    """Return scalar centers for the n_agents recipient choices (n-1 peers + GCS)."""
    n = int(n_agents)
    if n < 1:
        raise ValueError("n_agents must be >= 1")
    return np.asarray([2.0 * ((i + 0.5) / n) - 1.0 for i in range(n)], dtype=np.float32)


def peer_recipient_bin_index(code: float, n_agents: int) -> int:
    n = int(n_agents)
    if n < 1:
        raise ValueError("n_agents must be >= 1")
    x = float(np.clip((float(code) + 1.0) * 0.5, 0.0, 1.0 - 1e-12))
    return min(int(x * n), n - 1)


def canonicalize_peer_action_numpy(actions: np.ndarray, n_agents: int) -> np.ndarray:
    """Map hybrid Box coordinates to the exact action values executed by the env.

    Continuous Cartesian acceleration and RF power remain unchanged. The TX gate
    is represented as {-1,+1}; the recipient coordinate is snapped to the center
    of its deterministic destination bin. Keeping replay on this canonical action
    manifold avoids training critics on multiple raw scalars that execute the same
    discrete action.
    """
    arr = np.asarray(actions, dtype=np.float32)
    if arr.shape[-1] != PEER_ACTION_DIM:
        raise ValueError(f"peer action must end in {PEER_ACTION_DIM} coordinates, got {arr.shape}")
    out = np.clip(arr, -1.0, 1.0).copy()
    out[..., 3] = np.where(out[..., 3] > 0.0, 1.0, -1.0)
    n = int(n_agents)
    if n < 1:
        raise ValueError("n_agents must be >= 1")
    x = np.clip((out[..., 5] + 1.0) * 0.5, 0.0, 1.0 - 1e-7)
    idx = np.minimum((x * n).astype(np.int64), n - 1)
    centers = peer_recipient_bin_centers(n)
    out[..., 5] = centers[idx]
    # Power and recipient are conditional parameters of the transmit action.  When
    # the hard gate is OFF they have no physical effect, so collapse every such
    # Box point to one semantic no-transmit action for replay/critic training.
    gate_off = out[..., 3] < 0.0
    out[..., 4] = np.where(gate_off, -1.0, out[..., 4])
    out[..., 5] = np.where(gate_off, centers[0], out[..., 5])
    return out


def canonicalize_peer_action_torch(
    actions: torch.Tensor,
    n_agents: int,
    *,
    straight_through: bool,
) -> torch.Tensor:
    """Torch equivalent with optional straight-through gradients for discrete dims."""
    if actions.shape[-1] != PEER_ACTION_DIM:
        raise ValueError(f"peer action must end in {PEER_ACTION_DIM} coordinates, got {tuple(actions.shape)}")
    n = int(n_agents)
    if n < 1:
        raise ValueError("n_agents must be >= 1")
    clipped = actions.clamp(-1.0, 1.0)
    hard_gate = torch.where(clipped[..., 3] > 0.0, torch.ones_like(clipped[..., 3]), -torch.ones_like(clipped[..., 3]))
    x = ((clipped[..., 5] + 1.0) * 0.5).clamp(0.0, 1.0 - 1e-7)
    idx = torch.floor(x * n).long().clamp(0, n - 1)
    centers = torch.linspace(
        -1.0 + 1.0 / n,
        1.0 - 1.0 / n,
        n,
        dtype=clipped.dtype,
        device=clipped.device,
    )
    hard_recipient = centers[idx]
    gate_on = hard_gate > 0.0
    if straight_through:
        gate = clipped[..., 3] + (hard_gate - clipped[..., 3]).detach()
        recipient_on = clipped[..., 5] + (hard_recipient - clipped[..., 5]).detach()
    else:
        gate = hard_gate
        recipient_on = hard_recipient
    neutral_power = torch.full_like(clipped[..., 4], -1.0)
    neutral_recipient = torch.full_like(clipped[..., 5], float(-1.0 + 1.0 / n))
    power = torch.where(gate_on, clipped[..., 4], neutral_power)
    recipient = torch.where(gate_on, recipient_on, neutral_recipient)
    out = clipped.clone()
    out[..., 3] = gate
    out[..., 4] = power
    out[..., 5] = recipient
    return out


def neutral_peer_action(*, recipient_code: float = -1.0) -> np.ndarray:
    """Neutral no-acceleration, TX-off action useful for tests and scripted probes."""
    return np.asarray([0.0, 0.0, 0.0, -1.0, -1.0, recipient_code], dtype=np.float32)


def peer_action_saturation_fraction(actions: np.ndarray, threshold: float = 0.95) -> float:
    """Saturation of meaningful continuous controls in the hybrid peer action.

    Cartesian acceleration always counts. RF power counts only when TX is on;
    the hard gate and quantized recipient are discrete by design and therefore
    must not be diagnosed as continuous-control saturation.
    """
    arr = np.asarray(actions)
    if arr.ndim != 2 or arr.shape[1] != PEER_ACTION_DIM:
        raise ValueError(f"expected (n_agents, {PEER_ACTION_DIM}) peer actions, got {arr.shape}")
    motion = np.abs(arr[:, :3]) > float(threshold)
    gate_on = arr[:, 3] > 0.0
    power = (np.abs(arr[:, 4]) > float(threshold)) & gate_on
    saturated = int(np.count_nonzero(motion)) + int(np.count_nonzero(power))
    denominator = 3 * arr.shape[0] + int(np.count_nonzero(gate_on))
    return float(saturated / denominator) if denominator > 0 else 0.0
