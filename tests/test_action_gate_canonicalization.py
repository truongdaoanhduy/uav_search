import numpy as np
import pytest
import torch

from uav_search.actions import (
    canonicalize_peer_action_numpy,
    canonicalize_peer_action_torch,
    peer_recipient_bin_centers,
)


def test_gate_off_neutralizes_power_and_recipient_in_replay_semantics() -> None:
    raw = np.asarray(
        [
            [0.1, -0.2, 0.3, -0.4, 0.9, 0.91],
            [0.1, -0.2, 0.3, -0.4, -0.1, -0.77],
        ],
        dtype=np.float32,
    )

    canonical = canonicalize_peer_action_numpy(raw, n_agents=6)

    assert np.all(canonical[:, 3] == -1.0)
    assert np.all(canonical[:, 4] == -1.0)
    neutral_recipient = peer_recipient_bin_centers(6)[0]
    assert np.allclose(canonical[:, 5], neutral_recipient)
    np.testing.assert_allclose(canonical[0], canonical[1])


def test_gate_on_keeps_power_and_canonical_recipient() -> None:
    raw = np.asarray([[0.0, 0.0, 0.0, 0.2, 0.37, 0.33]], dtype=np.float32)
    canonical = canonicalize_peer_action_numpy(raw, n_agents=6)

    assert canonical[0, 3] == 1.0
    assert canonical[0, 4] == pytest.approx(0.37)
    assert canonical[0, 5] in peer_recipient_bin_centers(6)


def test_gate_off_straight_through_blocks_irrelevant_power_recipient_gradients() -> None:
    raw = torch.tensor(
        [[0.1, -0.2, 0.3, -0.4, 0.9, 0.91]],
        dtype=torch.float32,
        requires_grad=True,
    )
    canonical = canonicalize_peer_action_torch(raw, n_agents=6, straight_through=True)

    loss = canonical[0, 3] + canonical[0, 4] + canonical[0, 5]
    loss.backward()

    assert raw.grad is not None
    assert raw.grad[0, 3].item() == pytest.approx(1.0)
    assert raw.grad[0, 4].item() == pytest.approx(0.0)
    assert raw.grad[0, 5].item() == pytest.approx(0.0)
