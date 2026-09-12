from __future__ import annotations

import importlib.util
from copy import deepcopy

import numpy as np
import pytest

from uav_search.config import load_config
from uav_search.envs.network_backends import (
    GCS_NODE,
    AnalyticalNetworkBackend,
    NetworkStepResult,
    TransmissionIntent,
    TransmissionOutcome,
    UavNetSimBackend,
)
from uav_search.envs.paper_env import PaperUAVEnv
from uav_search.envs.sensing import (
    belief_probability_floor,
    decode_belief_probabilities,
    encode_belief_probabilities,
)


def test_belief_codec_preserves_neutrality_and_avoids_absorbing_endpoints() -> None:
    probabilities = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float64)

    encoded = encode_belief_probabilities(probabilities, levels=256)
    decoded = decode_belief_probabilities(encoded, levels=256)

    assert encoded.dtype == np.uint8
    assert encoded.nbytes == probabilities.size
    assert decoded[0] > 0.0
    assert decoded[2] == 0.5
    assert decoded[-1] < 1.0
    assert decoded[0] == pytest.approx(1.0 - decoded[-1])
    assert decoded[1] == pytest.approx(1.0 - decoded[-2])
    assert np.all(np.diff(decoded) > 0.0)
    assert decoded[0] == pytest.approx(belief_probability_floor(256))


def test_belief_codec_rejects_invalid_levels_and_unused_even_level_code() -> None:
    with pytest.raises(ValueError, match=r"\[3, 256\]"):
        encode_belief_probabilities(np.array([0.5]), levels=2)
    with pytest.raises(ValueError, match="unused|range"):
        decode_belief_probabilities(np.array([255], dtype=np.uint8), levels=256)


def recipient_code(env: PaperUAVEnv, sender: int, recipient: int) -> float:
    candidates = [idx for idx in range(env.n_agents) if idx != sender] + [GCS_NODE]
    bin_index = candidates.index(recipient)
    return 2.0 * ((bin_index + 0.5) / len(candidates)) - 1.0


def make_u6_env(*, seed: int = 0) -> PaperUAVEnv:
    cfg = deepcopy(load_config("masac", "u6"))
    env = PaperUAVEnv(cfg, seed=seed)
    env.reset(seed=seed)
    return env


def test_peer_fan_in_reserves_receiver_report_capacity_once_per_slot(monkeypatch) -> None:
    env = make_u6_env(seed=906)
    assert env._enqueue_report(0, 0)
    assert env._enqueue_report(1, 1)
    receiver = 2
    free_bytes = 1_000
    occupied = env.peer_buffer_bytes - free_bytes
    env.report_buffers[2, receiver] = occupied
    env.queue_bytes[receiver] = occupied
    captured_intents: list[TransmissionIntent] = []

    def full_ack(intents, *_args, **_kwargs):
        captured_intents.extend(intents)
        outcomes = [
            TransmissionOutcome(
                sender=intent.sender,
                recipient=intent.recipient,
                requested_bytes=intent.requested_bytes,
                delivered_bytes=intent.requested_bytes,
                rate_bps=2_000_000.0,
                distance_m=100.0,
                delivered_prefix_bytes=intent.requested_bytes,
            )
            for intent in intents
        ]
        attempted = sum(intent.requested_bytes for intent in intents)
        return NetworkStepResult(
            outcomes=outcomes,
            attempted_bytes=attempted,
            delivered_bytes=attempted,
            byte_pdr=1.0,
        )

    monkeypatch.setattr(env.network_backend, "transmit", full_ack)
    actions = np.zeros((env.n_agents, env.action_dim), dtype=np.float64)
    actions[:, 3] = -1.0
    actions[:, 4] = -1.0
    actions[:, 5] = -1.0
    for sender in (0, 1):
        actions[sender, 3] = 1.0
        actions[sender, 4] = 1.0
        actions[sender, 5] = recipient_code(env, sender, receiver)

    env._peer_transmit(actions)

    admitted_report_bytes = sum(
        intent.requested_bytes - env.peer_sync_bytes for intent in captured_intents
    )
    assert admitted_report_bytes == free_bytes
    assert env.queue_bytes[receiver] == env.peer_buffer_bytes
    assert env.last_peer_syncs == 2


def test_initial_confirmation_accepts_accumulated_posterior_and_direct_fine_evidence() -> None:
    """A later Bayes update must not erase valid fine-observation provenance."""
    env = make_u6_env(seed=901)
    target_idx = 0
    y, x = env._target_grid_cell(target_idx)
    env.uav_active[:] = False
    env.uav_active[0] = True
    env.belief_maps[0, y, x] = env.peer_target_confirmation_threshold + 1e-4
    env.fine_positive_cells_by_agent[0, y, x] = True
    env.fine_target_evidence_by_agent[0, target_idx] = True
    env.last_sensor_positive[0, target_idx] = False

    env._confirm_peer_targets()

    assert env.target_found[target_idx]
    assert env.target_directly_confirmed_by_agent[0, target_idx]
    assert env.pending_report_source[target_idx] == 0


def test_analytical_lone_sender_is_not_jammed_by_idle_uav_geometry() -> None:
    """Only simultaneous transmission intents may contribute interference."""
    cfg = load_config("masac", "u6")
    backend = AnalyticalNetworkBackend(cfg, seed=902)
    gcs = np.array([0.0, 2500.0, 0.0])
    idle_far = np.array(
        [
            [100.0, 2500.0, 100.0],
            [4900.0, 4900.0, 100.0],
            [4800.0, 4900.0, 100.0],
            [4700.0, 4900.0, 100.0],
            [4600.0, 4900.0, 100.0],
            [4500.0, 4900.0, 100.0],
        ]
    )
    idle_near = idle_far.copy()
    idle_near[1:] = np.array(
        [
            [101.0, 2500.0, 100.0],
            [102.0, 2500.0, 100.0],
            [103.0, 2500.0, 100.0],
            [104.0, 2500.0, 100.0],
            [105.0, 2500.0, 100.0],
        ]
    )
    intents = [TransmissionIntent(0, GCS_NODE, 1000, 0.4)]

    far = backend.transmit(intents, idle_far, gcs, None, dt_s=1.0, step_index=0)
    near = backend.transmit(intents, idle_near, gcs, None, dt_s=1.0, step_index=0)

    assert far.outcomes[0].rate_bps == pytest.approx(near.outcomes[0].rate_bps)
    assert far.delivered_bytes == near.delivered_bytes == 1000


@pytest.mark.skipif(
    importlib.util.find_spec("simpy") is None or importlib.util.find_spec("phy.channel") is None,
    reason="pinned UavNetSim optional dependency is not installed",
)
def test_uavnetsim_large_request_commits_a_contiguous_closed_slot_prefix() -> None:
    """Parallel packet admission must not reduce a good 250 kB slot to one 1 KiB prefix."""
    cfg = load_config("masac", "u6")
    backend = UavNetSimBackend(cfg, seed=1)
    positions = np.array(
        [
            [100.0, 100.0, 100.0],
            [120.0, 100.0, 100.0],
            [4500.0, 4500.0, 100.0],
            [4600.0, 4500.0, 100.0],
            [4500.0, 4600.0, 100.0],
            [4600.0, 4600.0, 100.0],
        ]
    )

    result = backend.transmit(
        [TransmissionIntent(1, 0, 250_000, 0.1)],
        positions,
        np.array([0.0, 100.0, 0.0]),
        np.empty((0, 3)),
        dt_s=1.0,
        step_index=0,
    )

    outcome = result.outcomes[0]
    assert outcome.delivered_prefix_bytes == outcome.delivered_bytes
    assert outcome.delivered_bytes > 100_000


@pytest.mark.skipif(
    importlib.util.find_spec("simpy") is None or importlib.util.find_spec("phy.channel") is None,
    reason="pinned UavNetSim optional dependency is not installed",
)
def test_uavnetsim_closed_slot_has_no_late_packet_work_in_next_idle_action() -> None:
    """An action from slot zero must not transmit or ACK after that slot returns."""
    cfg = load_config("masac", "u6")
    backend = UavNetSimBackend(cfg, seed=1)
    positions = np.array(
        [
            [100.0, 100.0, 100.0],
            [120.0, 100.0, 100.0],
            [4500.0, 4500.0, 100.0],
            [4600.0, 4500.0, 100.0],
            [4500.0, 4600.0, 100.0],
            [4600.0, 4600.0, 100.0],
        ]
    )
    gcs = np.array([0.0, 100.0, 0.0])
    obstacles = np.empty((0, 3))

    backend.transmit(
        [TransmissionIntent(1, 0, 250_000, 0.1)],
        positions,
        gcs,
        obstacles,
        dt_s=1.0,
        step_index=0,
    )
    backend.transmit([], positions, gcs, obstacles, dt_s=1.0, step_index=1)

    stale_work = [
        (kind, time_us, data)
        for kind, time_us, data in backend._episode.event_bus.events
        if kind in {"packet_tx_started", "packet_ack_received", "packet_tx_energy"}
        and float(time_us) > 1_000_000.0
    ]
    assert stale_work == []
    for node in backend._episode.drones:
        assert node.mac_process_dict == {}
        assert node.mac_process_finish == {}
        assert node.mac_protocol.wait_ack_process_dict == {}
        assert node.mac_protocol.wait_ack_process_finish == {}


def test_gcs_progress_reward_is_proportional_to_newly_committed_bytes() -> None:
    """A one-byte ACK must not earn the same shaping reward as a full slot."""
    env = make_u6_env(seed=903)
    idx = 0
    env.last_tx_active[idx] = True
    env.last_selected_recipient[idx] = GCS_NODE
    env.last_tx_success[idx] = True
    env.last_selected_tx_rate_bps[idx] = env.peer_comm_reward_rate_bps
    env.last_report_bytes_attempted_by_agent[idx] = 250_000

    env.last_report_bytes_transmitted_by_agent[idx] = 1
    one_byte_reward = env._communication_reward(idx)
    env.last_report_bytes_transmitted_by_agent[idx] = 250_000
    quarter_report_reward = env._communication_reward(idx)

    rc = float(env.assumed["comm_reward_max"])
    assert one_byte_reward == pytest.approx(rc / env.peer_report_bytes)
    assert quarter_report_reward == pytest.approx(0.25 * rc)


def test_shield_correction_is_visible_to_reward_and_counted_once_per_uav() -> None:
    """Safe realized state must not hide an unsafe nominal policy action."""
    env = make_u6_env(seed=904)
    previous = np.array(
        [
            [500.0, 2500.0, 100.0],
            [700.0, 2500.0, 100.0],
            [1500.0, 500.0, 100.0],
            [2000.0, 500.0, 100.0],
            [2500.0, 500.0, 100.0],
            [3000.0, 500.0, 100.0],
        ]
    )
    env.positions[:] = previous
    env.positions[0] = np.array([550.0, 2500.0, 100.0])
    env.positions[1] = np.array([650.0, 2500.0, 100.0])
    env.velocities[:] = env.positions - previous
    env.uav_active[:] = True

    env._apply_peer_safety_projection(previous)
    safety_reward, close_count = env._safety_reward(0)

    assert np.linalg.norm(env.positions[0] - env.positions[1]) > env.safety_distance_m
    assert close_count == 0
    assert safety_reward < 0.0
    assert env.last_safety_filter_interventions == 2


def test_analytical_simultaneous_sender_contributes_interference() -> None:
    """Removing idle-node interference must not remove real concurrent interference."""
    cfg = load_config("masac", "u6")
    backend = AnalyticalNetworkBackend(cfg, seed=905)
    gcs = np.array([0.0, 2500.0, 0.0])
    positions = np.array(
        [
            [100.0, 2500.0, 100.0],
            [101.0, 2500.0, 100.0],
            [4500.0, 4500.0, 100.0],
            [4600.0, 4500.0, 100.0],
            [4500.0, 4600.0, 100.0],
            [4600.0, 4600.0, 100.0],
        ]
    )
    lone = backend.transmit(
        [TransmissionIntent(0, GCS_NODE, 1000, 0.4)],
        positions,
        gcs,
        None,
        dt_s=1.0,
        step_index=0,
    )
    concurrent = backend.transmit(
        [
            TransmissionIntent(0, GCS_NODE, 1000, 0.4),
            TransmissionIntent(1, GCS_NODE, 1000, 0.4),
        ],
        positions,
        gcs,
        None,
        dt_s=1.0,
        step_index=0,
    )

    assert concurrent.outcomes[0].rate_bps < lone.outcomes[0].rate_bps


def test_radio_depletion_refreshes_topology_before_next_observation(monkeypatch) -> None:
    """A UAV that dies from radio energy must disappear from next-state links immediately."""
    cfg = deepcopy(load_config("masac", "u6"))
    cfg["scenario"]["network_backend"] = "analytical"
    env = PaperUAVEnv(cfg, seed=907)
    env.reset(seed=907)
    env.positions[:] = np.array(
        [[400.0 + 200.0 * i, 2500.0, 100.0] for i in range(env.n_agents)],
        dtype=np.float64,
    )
    env._refresh_links()
    assert np.any(env.max_power_pair_rates_bps[0, 1:] > 0.0)

    def drain_sender_zero(_intents, *_args, **_kwargs):
        return NetworkStepResult(
            tx_energy_j=300_000.0,
            node_tx_energy_j={0: 300_000.0},
        )

    monkeypatch.setattr(env.network_backend, "transmit", drain_sender_zero)
    action_array = np.zeros((env.n_agents, env.action_dim), dtype=np.float32)
    action_array[:, 0] = -1.0
    action_array[:, 3] = -1.0
    action_array[0, 3] = 1.0
    action_array[0, 5] = recipient_code(env, 0, 1)
    action_dict = {agent: action_array[i] for i, agent in enumerate(env.agents)}

    _obs, _reward, terminated, _truncated, info = env.step(action_dict)

    assert terminated["uav_0"] is True
    assert env.uav_active[0] is np.False_
    assert np.all(env.min_power_pair_rates_bps[0, :] == 0.0)
    assert np.all(env.min_power_pair_rates_bps[:, 0] == 0.0)
    assert np.all(env.max_power_pair_rates_bps[0, :] == 0.0)
    assert np.all(env.max_power_pair_rates_bps[:, 0] == 0.0)
    assert env.min_power_gcs_rates_bps[0] == 0.0
    assert env.max_power_gcs_rates_bps[0] == 0.0
    assert info["active_uavs"] == env.n_agents - 1


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_env_rejects_nonfinite_actions_before_state_mutation(bad_value: float) -> None:
    """Invalid policy outputs must fail fast instead of poisoning rewards/replay."""
    cfg = deepcopy(load_config("masac", "u6"))
    cfg["scenario"]["network_backend"] = "analytical"
    env = PaperUAVEnv(cfg, seed=908)
    env.reset(seed=908)
    positions_before = env.positions.copy()
    velocities_before = env.velocities.copy()
    battery_before = env.battery_pct.copy()
    step_before = env.step_count

    actions = {
        agent: np.zeros(env.action_dim, dtype=np.float32) for agent in env.agents
    }
    actions[env.agents[0]][0] = bad_value

    with pytest.raises(ValueError, match="finite"):
        env.step(actions)

    np.testing.assert_array_equal(env.positions, positions_before)
    np.testing.assert_array_equal(env.velocities, velocities_before)
    np.testing.assert_array_equal(env.battery_pct, battery_before)
    assert env.step_count == step_before
