import numpy as np
import pytest

from uav_search.config import load_config
from uav_search.envs.network_backends import (
    AnalyticalNetworkBackend,
    TransmissionIntent,
    UavNetSimBackend,
    create_network_backend,
)


def test_backend_factory_builds_analytical_backend():
    cfg = load_config("masac", "u6")
    backend = create_network_backend("analytical", cfg, seed=44)
    assert isinstance(backend, AnalyticalNetworkBackend)
    assert backend.name == "analytical"


def test_backend_factory_rejects_unknown_backend():
    cfg = load_config("masac", "u6")
    with pytest.raises(ValueError, match="Unsupported network backend"):
        create_network_backend("does-not-exist", cfg, seed=44)


def test_analytical_link_snapshot_matches_existing_peer_channel_model():
    from uav_search.envs.paper_env import PaperUAVEnv

    cfg = load_config("masac", "u6")
    cfg["scenario"]["network_backend"] = "analytical"
    env = PaperUAVEnv(cfg, seed=11)
    env.reset(seed=11)
    backend = create_network_backend("analytical", cfg, seed=11)

    snapshot = backend.link_snapshot(env.positions, env.gcs_position, env.obstacles)

    np.testing.assert_allclose(snapshot.pair_rates_bps, env.last_pair_rates_bps)
    np.testing.assert_allclose(snapshot.gcs_rates_bps, env.last_gcs_rates_bps)
    np.testing.assert_array_equal(snapshot.adjacency, env.last_adjacency)


def test_uavnetsim_dependency_error_points_to_pinned_installer(monkeypatch):
    cfg = load_config("masac", "u6")

    def missing(_name):
        raise ModuleNotFoundError("missing UavNetSim test dependency")

    monkeypatch.setattr("importlib.import_module", missing)
    with pytest.raises(RuntimeError, match="scripts/install_uavnetsim.sh"):
        UavNetSimBackend(cfg, seed=44)


def test_uavnetsim_a2a_link_and_csma_delivery_when_dependency_is_available():
    import importlib.util

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
    gcs = np.array([200.0, 100.0, 0.0], dtype=float)
    obstacles = np.empty((0, 3), dtype=float)

    snapshot = backend.link_snapshot(positions, gcs, obstacles)
    assert snapshot.pair_rates_bps[1, 0] > 0
    assert snapshot.adjacency[1, 0] == 1

    result = backend.transmit(
        [TransmissionIntent(sender=0, recipient=1, requested_bytes=1000)],
        positions,
        gcs,
        obstacles,
        dt_s=1.0,
        step_index=0,
    )
    assert result.attempted_bytes == 1000
    assert result.delivered_bytes == 1000
    assert result.byte_pdr == pytest.approx(1.0)
    assert result.throughput_bps == pytest.approx(8000.0)
    assert result.mean_delay_s > 0.0
    assert result.tx_energy_j > 0.0
    assert result.outcomes[0].recipient == 1


def test_analytical_transmit_preserves_requested_recipient_and_reports_metrics():
    cfg = load_config("masac", "u6")
    backend = create_network_backend("analytical", cfg, seed=44)
    positions = np.array(
        [[100.0, 100.0, 60.0], [101.0, 100.0, 60.0], [4500.0, 4500.0, 60.0],
         [4600.0, 4500.0, 60.0], [4500.0, 4600.0, 60.0], [4600.0, 4600.0, 60.0]],
        dtype=float,
    )
    gcs = np.array([200.0, 100.0, 0.0], dtype=float)
    result = backend.transmit(
        [TransmissionIntent(sender=0, recipient=1, requested_bytes=1000)],
        positions,
        gcs,
        np.empty((0, 3), dtype=float),
        dt_s=1.0,
        step_index=0,
    )
    assert result.outcomes[0].recipient == 1
    assert result.attempted_bytes == 1000
    assert result.delivered_bytes == 1000
    assert result.byte_pdr == pytest.approx(1.0)


def test_uavnetsim_full_slot_request_accounts_for_csma_and_header_overhead():
    import importlib.util

    if importlib.util.find_spec("simpy") is None or importlib.util.find_spec("phy.channel") is None:
        pytest.skip("pinned UavNetSim optional dependency is not installed")

    cfg = load_config("masac", "u6")
    backend = UavNetSimBackend(cfg, seed=44)
    positions = np.array(
        [[100.0, 100.0, 60.0], [150.0, 100.0, 60.0], [4500.0, 4500.0, 60.0],
         [4600.0, 4500.0, 60.0], [4500.0, 4600.0, 60.0], [4600.0, 4600.0, 60.0]],
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
    assert result.attempted_bytes == 250_000
    assert 0 < result.delivered_bytes <= 250_000
    assert 0.0 < result.byte_pdr <= 1.0
