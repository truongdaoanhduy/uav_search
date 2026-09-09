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


def test_uavnetsim_native_ack_is_enabled_and_policy_recipient_is_preserved():
    import importlib.util

    if importlib.util.find_spec("simpy") is None or importlib.util.find_spec("phy.channel") is None:
        pytest.skip("pinned UavNetSim optional dependency is not installed")

    cfg = load_config("masac", "u6")
    backend = UavNetSimBackend(cfg, seed=61)
    positions = np.array(
        [[100.0, 100.0, 60.0], [150.0, 100.0, 60.0], [4500.0, 4500.0, 60.0],
         [4600.0, 4500.0, 60.0], [4500.0, 4600.0, 60.0], [4600.0, 4600.0, 60.0]],
        dtype=float,
    )
    gcs = np.array([200.0, 100.0, 0.0], dtype=float)
    backend.reset_episode(positions, gcs, np.empty((0, 3), dtype=float))

    assert backend._episode is not None
    assert all(node.mac_protocol.enable_ack for node in backend._episode.drones)

    result = backend.transmit(
        [TransmissionIntent(sender=0, recipient=1, requested_bytes=1000, tx_power_w=0.1)],
        positions,
        gcs,
        np.empty((0, 3), dtype=float),
        dt_s=1.0,
        step_index=0,
    )
    assert result.outcomes[0].recipient == 1
    assert result.delivered_bytes == 1000
    # Native data TX charges the sender and native ACK TX charges the receiver.
    assert result.node_tx_energy_j[0] > 0.0
    assert result.node_tx_energy_j[1] > 0.0
    assert result.tx_energy_j == pytest.approx(sum(result.node_tx_energy_j.values()))
    ack_events = [
        data for event_type, _time_us, data in backend._episode.event_bus.events
        if event_type == "packet_ack_received"
    ]
    assert ack_events
    assert all(int(event["sender"]) == 0 and int(event["ack_from"]) == 1 for event in ack_events)


def test_uavnetsim_native_arq_retries_after_ack_timeout():
    import importlib.util

    if importlib.util.find_spec("simpy") is None or importlib.util.find_spec("phy.channel") is None:
        pytest.skip("pinned UavNetSim optional dependency is not installed")

    cfg = load_config("masac", "u6")
    backend = UavNetSimBackend(cfg, seed=62)
    positions = np.array(
        [[100.0, 100.0, 60.0], [150.0, 100.0, 60.0], [4500.0, 4500.0, 60.0],
         [4600.0, 4500.0, 60.0], [4500.0, 4600.0, 60.0], [4600.0, 4600.0, 60.0]],
        dtype=float,
    )
    gcs = np.array([200.0, 100.0, 0.0], dtype=float)
    backend.reset_episode(positions, gcs, np.empty((0, 3), dtype=float))

    result = backend.transmit(
        [TransmissionIntent(sender=0, recipient=1, requested_bytes=1000, tx_power_w=1e-12)],
        positions,
        gcs,
        np.empty((0, 3), dtype=float),
        dt_s=1.0,
        step_index=0,
    )
    assert result.delivered_bytes == 0
    timeout_events = [
        data for event_type, _time_us, data in backend._episode.event_bus.events
        if event_type == "packet_ack_timeout"
    ]
    tx_events = [
        data for event_type, _time_us, data in backend._episode.event_bus.events
        if event_type == "packet_tx_started" and data.get("packet_type") == "DataPacket"
    ]
    assert len(timeout_events) == 5
    assert len(tx_events) == 5
    assert len({int(event["packet_id"]) for event in tx_events}) == 1



def test_u6_contact_range_caps_analytical_peer_and_gcs_links():
    cfg = load_config("masac", "u6")
    backend = create_network_backend("analytical", cfg, seed=44)
    positions = np.array(
        [[100.0, 100.0, 60.0], [2300.0, 100.0, 60.0], [3000.0, 3000.0, 60.0],
         [3200.0, 3200.0, 60.0], [3400.0, 3400.0, 60.0], [3600.0, 3600.0, 60.0]],
        dtype=float,
    )
    gcs = np.array([0.0, 100.0, 0.0], dtype=float)
    snapshot = backend.link_snapshot(positions, gcs, np.empty((0, 3), dtype=float))
    assert np.linalg.norm(positions[1] - positions[0]) > cfg["scenario"]["peer_contact_range_m"]
    assert snapshot.pair_rates_bps[1, 0] == 0.0
    assert snapshot.adjacency[1, 0] == 0
    assert snapshot.gcs_rates_bps[0] > 0.0
    assert snapshot.gcs_rates_bps[1] == 0.0


def test_uavnetsim_same_seed_reproduces_csma_ack_and_energy():
    import importlib.util

    if importlib.util.find_spec("simpy") is None or importlib.util.find_spec("phy.channel") is None:
        pytest.skip("pinned UavNetSim optional dependency is not installed")

    cfg = load_config("masac", "u6")
    positions = np.array(
        [[100.0, 100.0, 60.0], [150.0, 100.0, 60.0], [100.0, 150.0, 60.0],
         [4600.0, 4500.0, 60.0], [4500.0, 4600.0, 60.0], [4600.0, 4600.0, 60.0]],
        dtype=float,
    )
    gcs = np.array([200.0, 100.0, 0.0], dtype=float)
    obstacles = np.empty((0, 3), dtype=float)
    intents = [
        TransmissionIntent(sender=0, recipient=1, requested_bytes=1000, tx_power_w=0.1),
        TransmissionIntent(sender=2, recipient=1, requested_bytes=1000, tx_power_w=0.1),
    ]

    results = []
    traces = []
    for _ in range(2):
        backend = UavNetSimBackend(cfg, seed=77)
        backend.reset_episode(positions, gcs, obstacles)
        result = backend.transmit(intents, positions, gcs, obstacles, dt_s=1.0, step_index=0)
        results.append(result)
        traces.append([
            (kind, round(float(time_us), 9), tuple(sorted(data.items())))
            for kind, time_us, data in backend._episode.event_bus.events
            if kind in {"packet_tx_started", "packet_ack_received", "packet_ack_timeout"}
        ])

    assert [(o.sender, o.recipient, o.delivered_bytes, o.delay_s) for o in results[0].outcomes] == \
           [(o.sender, o.recipient, o.delivered_bytes, o.delay_s) for o in results[1].outcomes]
    assert results[0].node_tx_energy_j == pytest.approx(results[1].node_tx_energy_j)
    assert traces[0] == traces[1]


def test_uavnetsim_inactive_receiver_cannot_ack_or_complete_hop():
    import importlib.util

    if importlib.util.find_spec("simpy") is None or importlib.util.find_spec("phy.channel") is None:
        pytest.skip("pinned UavNetSim optional dependency is not installed")

    cfg = load_config("masac", "u6")
    backend = UavNetSimBackend(cfg, seed=79)
    positions = np.array(
        [[100.0, 100.0, 60.0], [150.0, 100.0, 60.0], [4500.0, 4500.0, 60.0],
         [4600.0, 4500.0, 60.0], [4500.0, 4600.0, 60.0], [4600.0, 4600.0, 60.0]],
        dtype=float,
    )
    gcs = np.array([200.0, 100.0, 0.0], dtype=float)
    obstacles = np.empty((0, 3), dtype=float)
    backend.reset_episode(positions, gcs, obstacles)
    backend.set_node_active(np.array([True, False, True, True, True, True]))

    result = backend.transmit(
        [TransmissionIntent(sender=0, recipient=1, requested_bytes=1000, tx_power_w=0.1)],
        positions, gcs, obstacles, dt_s=1.0, step_index=0,
    )
    assert result.delivered_bytes == 0
    assert 1 not in result.node_tx_energy_j
    assert not any(
        event_type == "packet_ack_received" and int(data.get("ack_from", -1)) == 1
        for event_type, _time_us, data in backend._episode.event_bus.events
    )
