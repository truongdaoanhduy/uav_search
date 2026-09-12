from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from uav_search.config import ACTIVE_SCENARIOS, RESEARCH_SCENARIOS, load_config
from uav_search.envs.network_backends import NetworkStepResult, TransmissionOutcome
from uav_search.envs.paper_env import PaperUAVEnv
from uav_search.envs.sensing import (
    decode_belief_probabilities,
    encode_belief_probabilities,
)


def make_env(scenario: str = "u6", seed: int = 101) -> PaperUAVEnv:
    cfg = deepcopy(load_config("masac", scenario))
    cfg["scenario"]["network_backend"] = "analytical"
    return PaperUAVEnv(cfg, seed=seed)


def recipient_code(env: PaperUAVEnv, sender: int, recipient: int) -> float:
    candidates = [j for j in range(env.n_agents) if j != sender] + [-1]
    idx = candidates.index(recipient)
    return 2.0 * ((idx + 0.5) / len(candidates)) - 1.0


def peer_sync_actions(env: PaperUAVEnv, sender: int, recipient: int) -> np.ndarray:
    act = np.zeros((env.n_agents, env.action_dim), dtype=np.float64)
    act[:, 0] = -1.0
    act[:, 3] = -1.0
    act[:, 4] = -1.0
    act[:, 5] = -1.0
    act[sender, 3] = 1.0
    act[sender, 4] = 1.0
    act[sender, 5] = recipient_code(env, sender, recipient)
    return act


def place_single_good_peer_link(env: PaperUAVEnv) -> None:
    env.positions[:] = np.array([4500.0, 4500.0, 100.0])
    env.positions[0] = [500.0, 2500.0, 100.0]
    env.positions[1] = [600.0, 2500.0, 100.0]
    env.gcs_position = np.array([0.0, 2500.0, 0.0])
    env._refresh_links()


def quantized_belief(env: PaperUAVEnv, value: float) -> float:
    code = encode_belief_probabilities(
        np.array([value]),
        env.peer_sync_quantization_levels,
    )
    return float(
        decode_belief_probabilities(
            code,
            env.peer_sync_quantization_levels,
        )[0]
    )


def test_active_research_scope_is_only_u6_and_u9():
    assert RESEARCH_SCENARIOS == ("u6", "u9")
    assert ACTIVE_SCENARIOS == RESEARCH_SCENARIOS


def test_u9_is_same_research_architecture_with_nine_homogeneous_uavs():
    env = make_env("u9", seed=102)
    obs, _ = env.reset(seed=102)

    assert env.peer_mode
    assert env.n_fixed == 0
    assert env.n_rotor == 9
    assert env.n_agents == 9
    assert env.action_dim == 6
    assert len(obs) == 9
    assert env.scenario["name"] == "u9"

    min_sep = float(env.scenario["initial_min_separation_m"])
    for i in range(env.n_agents):
        for j in range(i + 1, env.n_agents):
            assert np.linalg.norm(env.positions[i, :2] - env.positions[j, :2]) >= min_sep - 1e-9


@pytest.mark.parametrize("scenario", ["u6", "u9"])
def test_peer_targets_start_in_unique_binary_belief_cells(scenario):
    env = make_env(scenario=scenario, seed=35)
    cells = [env._target_grid_cell(k) for k in range(env.n_targets)]
    assert len(set(cells)) == env.n_targets


def test_topology_snapshot_does_not_refresh_actor_neighbor_cache_without_packet():
    env = make_env(seed=103)
    env.neighbor_cache_seen_step.fill(-1)
    place_single_good_peer_link(env)

    assert env.last_adjacency[1, 0] == 1
    assert env.neighbor_cache_seen_step[1, 0] == -1


def test_local_beliefs_do_not_fuse_globally_without_communication():
    env = make_env(seed=104)
    y, x = 10, 10
    env.belief_maps[:, y, x] = 0.5
    env.belief_maps[0, y, x] = 0.90

    env._confirm_peer_targets()

    assert env.belief_maps[0, y, x] == pytest.approx(0.90)
    np.testing.assert_allclose(env.belief_maps[1:, y, x], 0.5)


def test_successful_peer_sync_refreshes_only_receiver_cache_and_fuses_receiver_belief():
    env = make_env(seed=105)
    place_single_good_peer_link(env)
    env.neighbor_cache_seen_step.fill(-1)
    y, x = 11, 11
    env.belief_maps[:, y, x] = 0.5
    env.belief_maps[0, y, x] = 0.90
    env.belief_maps[1, y, x] = 0.60

    env._peer_transmit(peer_sync_actions(env, sender=0, recipient=1))

    assert env.last_tx_success[0]
    assert env.neighbor_cache_seen_step[1, 0] == env.step_count
    np.testing.assert_allclose(env.neighbor_cache_positions[1, 0], env.positions[0])
    assert env.neighbor_cache_battery[1, 0] == pytest.approx(env.battery_pct[0])
    assert env.belief_maps[1, y, x] == pytest.approx(quantized_belief(env, 0.90))
    assert env.belief_maps[2, y, x] == pytest.approx(0.50)


def test_successful_peer_sync_is_fully_fresh_in_returned_observation():
    env = make_env(seed=110)
    env.positions[0] = [500.0, 2500.0, 100.0]
    env.positions[1] = [700.0, 2500.0, 100.0]
    env._refresh_links()
    actions = peer_sync_actions(env, sender=0, recipient=1)

    obs, _, _, _, _ = env.step({name: actions[i] for i, name in enumerate(env.agents)})

    # For receiver UAV1, sender UAV0 is the first six-value peer slot.
    assert env.last_tx_success[0]
    assert obs["uav_1"][14] == pytest.approx(1.0)


def test_expired_peer_cache_masks_stale_state_from_actor_observation():
    env = make_env(seed=111)
    env.neighbor_cache_seen_step[1, 0] = 0
    env.neighbor_cache_positions[1, 0] = [1234.0, 2345.0, 100.0]
    env.neighbor_cache_battery[1, 0] = 77.0
    # State t=6 is five completed unsynchronized transitions after slot 0.
    env.step_count = env.peer_neighbor_cache_ttl_steps + 1

    slot = env._observations()["uav_1"][9:15]

    np.testing.assert_array_equal(slot, np.zeros(6, dtype=np.float32))


def test_control_only_peer_sync_cannot_farm_communication_reward():
    env = make_env(seed=106)
    place_single_good_peer_link(env)

    env._peer_transmit(peer_sync_actions(env, sender=0, recipient=1))

    assert env.last_tx_success[0]
    assert env.last_bytes_transmitted == 0
    assert env._communication_reward(0) == pytest.approx(
        -env.peer_communication_attempt_penalty
    )


def test_partial_peer_sync_is_not_reported_to_actor_as_success(monkeypatch):
    env = make_env(seed=114)
    place_single_good_peer_link(env)
    env.neighbor_cache_seen_step.fill(-1)

    def partial_transmit(intents, *args, **kwargs):
        intent = intents[0]
        delivered = env.peer_sync_bytes - 1
        return NetworkStepResult(
            outcomes=[TransmissionOutcome(
                sender=intent.sender, recipient=intent.recipient,
                requested_bytes=intent.requested_bytes, delivered_bytes=delivered,
                rate_bps=1_000_000.0, distance_m=100.0,
            )],
            attempted_bytes=intent.requested_bytes,
            delivered_bytes=delivered,
            byte_pdr=delivered / intent.requested_bytes,
        )

    monkeypatch.setattr(env.network_backend, "transmit", partial_transmit)
    env._peer_transmit(peer_sync_actions(env, sender=0, recipient=1))

    assert not env.last_tx_success[0]
    assert env.neighbor_cache_seen_step[1, 0] == -1
    assert env.last_peer_syncs == 0


def test_out_of_order_packet_acks_do_not_commit_past_first_gap(monkeypatch):
    env = make_env(seed=115)
    place_single_good_peer_link(env)
    env.neighbor_cache_seen_step.fill(-1)
    assert env._enqueue_report(0, 0)
    sender_before = int(env.report_buffers[0, 0])

    def out_of_order_transmit(intents, *args, **kwargs):
        intent = intents[0]
        # Link-level ACKs include later report packets, but one byte in the
        # leading synchronization bundle is still missing.
        delivered = env.peer_sync_bytes + 1024
        return NetworkStepResult(
            outcomes=[SimpleNamespace(
                sender=intent.sender,
                recipient=intent.recipient,
                requested_bytes=intent.requested_bytes,
                delivered_bytes=delivered,
                delivered_prefix_bytes=env.peer_sync_bytes - 1,
                rate_bps=1_000_000.0,
                distance_m=100.0,
                delay_s=0.1,
                tx_energy_j=0.0,
            )],
            attempted_bytes=intent.requested_bytes,
            delivered_bytes=delivered,
            byte_pdr=delivered / intent.requested_bytes,
        )

    monkeypatch.setattr(env.network_backend, "transmit", out_of_order_transmit)
    env._peer_transmit(peer_sync_actions(env, sender=0, recipient=1))

    assert not env.last_tx_success[0]
    assert env.neighbor_cache_seen_step[1, 0] == -1
    assert env.last_peer_syncs == 0
    assert env.last_bytes_transmitted == 0
    assert env.report_buffers[0, 0] == sender_before
    assert env.report_buffers[0, 1] == 0


def test_failed_peer_sync_does_not_refresh_cache_or_belief():
    env = make_env(seed=107)
    env.neighbor_cache_seen_step.fill(-1)
    env.positions[0] = [100.0, 100.0, 100.0]
    env.positions[1] = [4900.0, 4900.0, 100.0]
    env._refresh_links()
    y, x = 12, 12
    env.belief_maps[:, y, x] = 0.5
    env.belief_maps[0, y, x] = 0.90

    env._peer_transmit(peer_sync_actions(env, sender=0, recipient=1))

    assert not env.last_tx_success[0]
    assert env.neighbor_cache_seen_step[1, 0] == -1
    assert env.belief_maps[1, y, x] == pytest.approx(0.5)


def test_independent_local_confirmation_updates_agent_knowledge_after_global_confirmation():
    env = make_env(seed=112)
    y, x = env._target_grid_cell(0)
    env.confirmed_cells[y, x] = True
    env.target_found[0] = True
    env.target_known_by_agent[0, 0] = True
    env.positions[1, :2] = env.targets[0]
    env.positions[1, 2] = env.peer_altitude_levels_m[0]
    env.belief_maps[1, y, x] = 0.95

    class AlwaysPositive:
        @staticmethod
        def random():
            return 0.0

    env.rng = AlwaysPositive()
    for _ in range(20):
        env._perform_peer_target_detection()
        if env.belief_maps[1, y, x] >= env.peer_target_confirmation_threshold:
            break

    assert env.belief_maps[1, y, x] >= env.peer_target_confirmation_threshold
    assert env.target_known_by_agent[1, 0]


def test_independent_detector_can_replace_depleted_pending_report_source():
    env = make_env(seed=113)
    env.peer_buffer_bytes = env.peer_report_bytes
    env._enqueue_report(0, 1)  # fill UAV0 buffer
    y, x = env._target_grid_cell(0)
    env.confirmed_cells[y, x] = True
    env.target_found[0] = True
    env.target_known_by_agent[0, 0] = True
    env.pending_report_source[0] = 0
    env.positions[1, :2] = env.targets[0]
    env.positions[1, 2] = env.peer_altitude_levels_m[0]
    env.belief_maps[1, y, x] = 0.95

    class AlwaysPositive:
        @staticmethod
        def random():
            return 0.0

    env.rng = AlwaysPositive()
    for _ in range(20):
        env._perform_peer_target_detection()
        if env.target_known_by_agent[1, 0]:
            break
    assert env.target_known_by_agent[1, 0]

    env.uav_active[0] = False
    env.battery_pct[0] = 0.0
    env.report_buffers[1, 0] = 0
    env.queue_bytes[0] = 0
    env._retry_pending_reports()

    assert env.report_generated[0]
    assert env.pending_report_source[0] == 1
    assert env.report_buffers[0, 1] == env.peer_report_bytes


def test_pending_report_is_not_generated_into_a_depleted_detector_buffer():
    env = make_env(seed=109)
    env.peer_buffer_bytes = env.peer_report_bytes
    env._enqueue_report(0, 1)
    env.target_found[0] = True
    env.target_known_by_agent[0, 0] = True
    env.pending_report_source[0] = 0
    assert not env._enqueue_report(0, 0)

    env.uav_active[0] = False
    env.battery_pct[0] = 0.0
    env.report_buffers[1, 0] = 0
    env.queue_bytes[0] = 0
    env._retry_pending_reports()

    assert not env.report_generated[0]
    assert env.report_buffers[0, 0] == 0


def test_peer_belief_sync_is_at_most_one_hop_per_macro_step():
    env = make_env(seed=108)
    env.positions[:] = np.array([4500.0, 4500.0, 100.0])
    env.positions[0] = [500.0, 2500.0, 100.0]
    env.positions[1] = [600.0, 2500.0, 100.0]
    env.positions[2] = [700.0, 2500.0, 100.0]
    env._refresh_links()
    y, x = 13, 13
    env.belief_maps[:, y, x] = 0.5
    env.belief_maps[0, y, x] = 0.90

    act = np.zeros((env.n_agents, env.action_dim), dtype=np.float64)
    act[:, 0] = -1.0
    act[:, 3:] = -1.0
    for sender, recipient in ((0, 1), (1, 2)):
        act[sender, 3] = 1.0
        act[sender, 4] = 1.0
        act[sender, 5] = recipient_code(env, sender, recipient)

    env._peer_transmit(act)

    assert env.last_tx_success[0]
    assert env.last_tx_success[1]
    assert env.belief_maps[1, y, x] == pytest.approx(quantized_belief(env, 0.90))
    # UAV1's slot-start map was still 0.5, so UAV2 cannot receive UAV0's belief
    # through UAV1 until a later macro-step. It only receives UAV1's quantized
    # slot-start 0.5 belief, not UAV0's 0.9 belief.
    assert env.belief_maps[2, y, x] == pytest.approx(quantized_belief(env, 0.50))

    env._peer_transmit(peer_sync_actions(env, sender=1, recipient=2))
    assert env.last_tx_success[1]
    assert env.belief_maps[2, y, x] == pytest.approx(quantized_belief(env, 0.90))
