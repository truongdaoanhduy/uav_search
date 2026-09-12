from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from uav_search.config import load_config
from uav_search.envs.network_backends import TransmissionIntent, UavNetSimBackend


def test_uavnetsim_pdr_uses_mac_admitted_bytes_not_unpacketized_offered_bytes() -> None:
    if importlib.util.find_spec("simpy") is None or importlib.util.find_spec("phy.channel") is None:
        pytest.skip("pinned UavNetSim optional dependency is not installed")

    cfg = load_config("masac", "u6")
    backend = UavNetSimBackend(cfg, seed=44)
    positions = np.array(
        [
            [100.0, 100.0, 60.0],
            [150.0, 100.0, 60.0],
            [4500.0, 4500.0, 60.0],
            [4600.0, 4500.0, 60.0],
            [4500.0, 4600.0, 60.0],
            [4600.0, 4600.0, 60.0],
        ],
        dtype=float,
    )

    result = backend.transmit(
        [TransmissionIntent(sender=0, recipient=1, requested_bytes=250_000)],
        positions,
        np.array([200.0, 100.0, 0.0], dtype=float),
        np.empty((0, 3), dtype=float),
        dt_s=1.0,
        step_index=0,
    )

    # The closed 1 s slot cannot packetize all 250 kB, but every packet that was
    # actually admitted to MAC is ACKed. PDR therefore must be 1.0; the lower
    # application offered-load ratio is a separate metric.
    assert result.attempted_bytes == 250_000
    assert 0 < result.admitted_bytes < result.attempted_bytes
    assert result.delivered_bytes == result.admitted_bytes
    assert result.byte_pdr == pytest.approx(1.0)
    assert result.offered_delivery_ratio == pytest.approx(
        result.delivered_bytes / result.attempted_bytes
    )


def test_episode_network_metrics_separate_mac_pdr_from_offered_service_ratio() -> None:
    from uav_search.runner.train import _network_episode_metrics_from_info

    metrics = _network_episode_metrics_from_info(
        {
            "total_network_attempted_bytes": 250_000,
            "total_network_admitted_bytes": 200_000,
            "total_network_delivered_bytes": 200_000,
        },
        elapsed_s=1.0,
    )

    assert metrics["network_byte_pdr"] == pytest.approx(1.0)
    assert metrics["network_offered_delivery_ratio"] == pytest.approx(0.8)
    assert metrics["network_throughput_bps"] == pytest.approx(1_600_000.0)


def test_peer_info_separates_potential_link_rate_from_selected_tx_rate() -> None:
    from copy import deepcopy

    from uav_search.envs.paper_env import PaperUAVEnv

    cfg = deepcopy(load_config("masac", "u6"))
    cfg["scenario"]["network_backend"] = "analytical"
    env = PaperUAVEnv(cfg, seed=72)
    env.reset(seed=72)
    env.positions[:] = np.asarray(
        [[env.gcs_position[0] + 50.0 * i, env.gcs_position[1], 100.0] for i in range(env.n_agents)],
        dtype=np.float64,
    )
    env._refresh_links()
    actions = np.zeros((env.n_agents, env.action_dim), dtype=np.float32)
    actions[:, 3] = -1.0

    _obs, _rewards, _terminated, _truncated, info = env.step(
        {name: actions[i] for i, name in enumerate(env.agents)}
    )

    assert info["mean_potential_comm_rate_mbps"] > 0.0
    assert info["mean_selected_tx_rate_mbps"] == pytest.approx(0.0)
    # Historical key remains a potential-connectivity diagnostic for dashboards.
    assert info["mean_comm_rate_mbps"] == pytest.approx(info["mean_potential_comm_rate_mbps"])
