from __future__ import annotations

import numpy as np

from uav_search.runner.termination import episode_finished
from uav_search.runner.train import _replay_terminal_mask


def test_episode_finished_handles_mixed_terminated_and_truncated_agents() -> None:
    terminated = {"uav_0": True, "uav_1": False}
    truncated = {"uav_0": False, "uav_1": True}

    assert episode_finished(terminated, truncated)


def test_episode_finished_requires_every_agent_to_be_done() -> None:
    terminated = {"uav_0": True, "uav_1": False}
    truncated = {"uav_0": False, "uav_1": False}

    assert not episode_finished(terminated, truncated)


def test_truncation_does_not_cut_replay_bootstrap() -> None:
    agents = ["uav_0", "uav_1"]
    terminated = {"uav_0": False, "uav_1": True}

    mask = _replay_terminal_mask(terminated, agents)

    np.testing.assert_array_equal(mask, np.array([0.0, 1.0], dtype=np.float32))
