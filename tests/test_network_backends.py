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

    snapshot = backend.link_snapshot(
        env.positions,
        env.gcs_position,
        env.obstacles,
        tx_power_w=env.peer_tx_power_max_w,
    )

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


def test_uavnetsim_reports_only_the_longest_in_order_delivered_prefix():
    import importlib.util

    if importlib.util.find_spec("simpy") is None or importlib.util.find_spec("phy.channel") is None:
        pytest.skip("pinned UavNetSim optional dependency is not installed")

    cfg = load_config("masac", "u6")
    backend = UavNetSimBackend(cfg, seed=1)
    positions = np.array(
        [[100.0, 100.0, 100.0], [120.0, 100.0, 100.0],
         [4500.0, 4500.0, 100.0], [4600.0, 4500.0, 100.0],
         [4500.0, 4600.0, 100.0], [4600.0, 4600.0, 100.0]],
        dtype=float,
    )
    result = backend.transmit(
        [TransmissionIntent(sender=1, recipient=0, requested_bytes=250_000, tx_power_w=0.1)],
        positions,
        np.array([0.0, 100.0, 0.0], dtype=float),
        np.empty((0, 3), dtype=float),
        dt_s=1.0,
        step_index=0,
    )

    outcome = result.outcomes[0]
    delivered_prefix = getattr(outcome, "delivered_prefix_bytes", outcome.delivered_bytes)
    assert 0 < delivered_prefix < outcome.delivered_bytes
    assert delivered_prefix == 1024


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


def test_uavnetsim_a2g_gain_improves_with_elevation_at_equal_3d_distance():
    cfg = load_config("masac", "u6")
    backend = UavNetSimBackend(cfg, seed=81)
    gcs = np.array([0.0, 0.0, 0.0], dtype=float)
    distance = 1000.0
    low_z, high_z = 50.0, 150.0
    low = np.array([np.sqrt(distance**2 - low_z**2), 0.0, low_z], dtype=float)
    high = np.array([np.sqrt(distance**2 - high_z**2), 0.0, high_z], dtype=float)

    assert np.linalg.norm(low - gcs) == pytest.approx(distance)
    assert np.linalg.norm(high - gcs) == pytest.approx(distance)
    assert backend._a2g_gain(high, gcs) > backend._a2g_gain(low, gcs)


def test_uavnetsim_a2a_gain_still_delegates_to_native_model(monkeypatch):
    from uav_search.envs.network_backends import _PaperAirspace

    cfg = load_config("masac", "u6")
    backend = UavNetSimBackend(cfg, seed=82)
    sentinel = 0.123456
    monkeypatch.setattr(backend._modules["a2a"], "path_gain", lambda *args, **kwargs: sentinel)
    airspace = _PaperAirspace(np.empty((0, 3), dtype=float))

    gain = backend._gain(
        np.array([0.0, 0.0, 100.0]),
        np.array([100.0, 0.0, 100.0]),
        airspace,
        gcs=False,
    )
    assert gain == pytest.approx(sentinel)


def test_persistent_uavnetsim_channel_uses_a2g_estimate_for_gcs_pairs():
    cfg = load_config("masac", "u6")
    backend = UavNetSimBackend(cfg, seed=83)
    positions = np.array(
        [[100.0, 100.0, 60.0], [150.0, 100.0, 60.0], [4500.0, 4500.0, 60.0],
         [4600.0, 4500.0, 60.0], [4500.0, 4600.0, 60.0], [4600.0, 4600.0, 60.0]],
        dtype=float,
    )
    gcs = np.array([200.0, 100.0, 0.0], dtype=float)
    obstacles = np.empty((0, 3), dtype=float)
    backend.reset_episode(positions, gcs, obstacles)
    gcs_id = len(positions)

    with backend._with_radio_parameters():
        gcs_estimate = backend._episode.channel._estimates([0], gcs_id)[(0, gcs_id)]
        peer_estimate = backend._episode.channel._estimates([0], 1)[(0, 1)]

        assert gcs_estimate.nominal == pytest.approx(backend._a2g_gain(positions[0], gcs))
        assert peer_estimate.nominal == pytest.approx(
            backend._gain(positions[0], positions[1], backend._episode.airspace, gcs=False)
        )


def test_uavnetsim_snapshot_power_exposes_link_feasible_only_at_high_power():
    cfg = load_config("masac", "u6")
    backend = UavNetSimBackend(cfg, seed=84)
    positions = np.array(
        [[100.0, 100.0, 60.0], [700.0, 100.0, 60.0], [4500.0, 4500.0, 60.0],
         [4600.0, 4500.0, 60.0], [4500.0, 4600.0, 60.0], [4600.0, 4600.0, 60.0]],
        dtype=float,
    )
    gcs = np.array([0.0, 2500.0, 0.0], dtype=float)
    obstacles = np.array([[400.0, 100.0, 50.0]], dtype=float)

    low = backend.link_snapshot(positions, gcs, obstacles, tx_power_w=0.1)
    high = backend.link_snapshot(positions, gcs, obstacles, tx_power_w=0.4)

    assert low.pair_rates_bps[1, 0] == 0.0
    assert low.adjacency[1, 0] == 0
    assert high.pair_rates_bps[1, 0] > 0.0
    assert high.adjacency[1, 0] == 1


def test_uavnetsim_a2g_data_and_ack_complete_gcs_hop():
    from uav_search.envs.network_backends import GCS_NODE

    cfg = load_config("masac", "u6")
    backend = UavNetSimBackend(cfg, seed=85)
    positions = np.array(
        [[100.0, 100.0, 100.0], [4500.0, 4500.0, 100.0], [4600.0, 4500.0, 100.0],
         [4500.0, 4600.0, 100.0], [4600.0, 4600.0, 100.0], [4400.0, 4600.0, 100.0]],
        dtype=float,
    )
    gcs = np.array([0.0, 100.0, 0.0], dtype=float)
    obstacles = np.empty((0, 3), dtype=float)

    result = backend.transmit(
        [TransmissionIntent(sender=0, recipient=GCS_NODE, requested_bytes=1000, tx_power_w=0.1)],
        positions,
        gcs,
        obstacles,
        dt_s=1.0,
        step_index=0,
    )

    assert result.delivered_bytes == 1000
    assert result.outcomes[0].recipient == GCS_NODE
    assert result.node_tx_energy_j[0] > 0.0
    assert any(
        event_type == "packet_ack_received" and int(data.get("sender", -1)) == 0
        for event_type, _time_us, data in backend._episode.event_bus.events
    )
