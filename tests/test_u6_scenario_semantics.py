from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from uav_search.config import load_config
from uav_search.envs.network_backends import NetworkStepResult
from uav_search.envs.paper_env import PaperUAVEnv
from uav_search.envs.sensing import (
    belief_probability_floor,
    decode_belief_probabilities,
    encode_belief_probabilities,
)


class FixedRandom:
    def __init__(self, value: float):
        self.value = float(value)

    def random(self) -> float:
        return self.value


def make_env(seed: int = 44) -> PaperUAVEnv:
    cfg = deepcopy(load_config("masac", "u6"))
    cfg["scenario"]["network_backend"] = "analytical"
    return PaperUAVEnv(cfg, seed=seed)


def isolate_agent(env: PaperUAVEnv, idx: int = 0) -> None:
    env.uav_active[:] = False
    env.uav_active[idx] = True


def test_continuous_footprint_scans_adjacent_cell_crossing_subcell_boundary() -> None:
    env = make_env()
    env.positions[0] = np.asarray([99.0, 50.0, 50.0])
    assert (0, 1) in env._sensing_cells(0)


def test_target_outside_physical_footprint_is_not_treated_as_occupied() -> None:
    env = make_env()
    isolate_agent(env)
    env.positions[0] = np.asarray([99.0, 99.0, 50.0])
    env.targets[0] = np.asarray([0.0, 0.0])
    for target_idx in range(1, env.n_targets):
        env.targets[target_idx] = np.asarray([1000.0 + 100.0 * target_idx, 1000.0])
    y, x = env._target_grid_cell(0)
    env.belief_maps[0, y, x] = 0.5
    env.rng = FixedRandom(0.5)

    env._perform_peer_target_detection()

    assert env.belief_maps[0, y, x] < 0.5
    assert not env.last_sensor_positive[0, 0]


def test_peer_observation_is_invariant_to_unconfirmed_ground_truth_target_position() -> None:
    env = make_env()
    env.last_sensor_positive[0, 0] = True
    env.targets[0] = np.asarray([100.0, 100.0])
    before = env._observations()["uav_0"].copy()

    env.targets[0] = np.asarray([4900.0, 4900.0])
    after = env._observations()["uav_0"].copy()

    np.testing.assert_array_equal(before, after)


def test_high_altitude_posterior_cannot_finalize_target_without_fine_evidence() -> None:
    env = make_env()
    isolate_agent(env)
    env.positions[0, 2] = env.peer_altitude_max_m
    target_idx = 0
    y, x = env._target_grid_cell(target_idx)
    env.belief_maps[0, y, x] = env.peer_target_confirmation_threshold + 1e-4

    env._confirm_peer_targets()

    assert not env.target_found[target_idx]
    assert not env.confirmed_cells[y, x]


def test_cognitive_information_reward_is_signed_potential_change() -> None:
    env = make_env()
    isolate_agent(env)
    env.positions[0] = np.asarray([50.0, 50.0, 50.0])
    env.targets[:] = np.asarray(
        [[1000.0 + 200.0 * i, 1000.0] for i in range(env.n_targets)],
        dtype=np.float64,
    )
    y, x = env._xy_grid_cell(env.positions[0, :2])
    env.belief_maps[0, y, x] = 0.5

    env.rng = FixedRandom(0.0)
    env._perform_peer_target_detection()
    positive_gain = float(env.last_information_gain_by_agent[0])
    assert positive_gain > 0.0

    env.rng = FixedRandom(0.99)
    env._perform_peer_target_detection()
    negative_gain = float(env.last_information_gain_by_agent[0])

    assert negative_gain == pytest.approx(-positive_gain)
    assert env.belief_maps[0, y, x] == pytest.approx(0.5)


def recipient_code(env: PaperUAVEnv, sender: int, recipient: int) -> float:
    candidates = [j for j in range(env.n_agents) if j != sender] + [-1]
    index = candidates.index(recipient)
    return 2.0 * ((index + 0.5) / len(candidates)) - 1.0


def idle_actions(env: PaperUAVEnv) -> dict[str, np.ndarray]:
    return {
        name: np.asarray([-1.0, 0.0, 0.0, -1.0, -1.0, -1.0], dtype=np.float32)
        for name in env.agents
    }


def tx_to_gcs_actions(env: PaperUAVEnv, sender: int = 0) -> np.ndarray:
    actions = np.zeros((env.n_agents, env.action_dim), dtype=np.float64)
    actions[:, 0] = -1.0
    actions[:, 3] = -1.0
    actions[:, 4] = -1.0
    actions[:, 5] = -1.0
    actions[sender, 3] = 1.0
    actions[sender, 4] = 1.0
    actions[sender, 5] = recipient_code(env, sender, -1)
    return actions


def set_safe_static_geometry(env: PaperUAVEnv) -> None:
    env.positions[:, :2] = np.asarray(
        [[100.0 + 500.0 * i, 2500.0] for i in range(env.n_agents)],
        dtype=np.float64,
    )
    env.positions[:, 2] = env.peer_altitude_min_m
    env.velocities.fill(0.0)
    env.obstacles[:, :2] = np.asarray([4900.0, 4900.0])
    env.obstacles[:, 2] = 1.0
    env._refresh_links()


def mark_fine_confirmation(
    env: PaperUAVEnv, target_idx: int, agent_idx: int, *, step: int
) -> None:
    y, x = env._target_grid_cell(target_idx)
    env.step_count = step
    env.belief_maps[agent_idx, y, x] = env.peer_target_confirmation_threshold + 1e-4
    env.fine_positive_cells_by_agent[agent_idx, y, x] = True
    env.fine_target_evidence_by_agent[agent_idx, target_idx] = True
    env._confirm_peer_targets()


def test_report_lifetime_starts_at_fine_confirmation_while_buffer_is_full() -> None:
    env = make_env()
    env.peer_buffer_bytes = env.peer_report_bytes
    assert env._enqueue_report(0, 1)
    mark_fine_confirmation(env, target_idx=0, agent_idx=0, step=17)

    env._retry_pending_reports()

    assert env.target_found[0]
    assert not env.report_generated[0]
    assert env.report_created_step[0] == 17


def test_report_scheduler_sends_oldest_deadline_before_target_index() -> None:
    env = make_env()
    env.positions[0] = env.gcs_position + np.asarray([0.0, 0.0, 50.0])
    env._refresh_links()
    assert env._enqueue_report(0, 0)
    assert env._enqueue_report(0, 9)
    env.report_created_step[0] = 90
    env.report_created_step[9] = 10
    env.step_count = 100
    before_zero = int(env.report_buffers[0, 0])
    before_nine = int(env.report_buffers[9, 0])

    env._peer_transmit(tx_to_gcs_actions(env, sender=0))

    assert env.report_buffers[9, 0] < before_nine
    assert env.report_buffers[0, 0] == before_zero


def test_depleted_recipient_is_recorded_as_failed_attempt_without_rf_energy(monkeypatch) -> None:
    env = make_env(seed=86)
    assert env._enqueue_report(0, 0)
    env.uav_active[1] = False
    captured_intents = []

    def no_network_work(intents, *_args, **_kwargs):
        captured_intents.extend(intents)
        return NetworkStepResult()

    monkeypatch.setattr(env.network_backend, "transmit", no_network_work)
    actions = np.zeros((env.n_agents, env.action_dim), dtype=np.float64)
    actions[:, 0] = -1.0
    actions[:, 3] = -1.0
    actions[:, 4] = -1.0
    actions[:, 5] = -1.0
    actions[0, 3] = 1.0
    actions[0, 4] = 1.0
    actions[0, 5] = recipient_code(env, 0, 1)

    env._peer_transmit(actions)

    nominal_slot_bytes = int(
        float(env.scenario["uavnetsim_bit_rate_bps"]) * env.dt / 8.0
    )
    expected_report_attempt = min(
        env.peer_report_bytes,
        nominal_slot_bytes - env.peer_sync_bytes,
    )
    assert captured_intents == []
    assert env.last_tx_active[0]
    assert not env.last_tx_success[0]
    assert env.last_selected_recipient[0] == 1
    assert env.last_report_bytes_attempted_by_agent[0] == expected_report_attempt
    assert env.last_network_result.tx_energy_j == 0.0
    expected_reward = -float(env.assumed["comm_reward_max"])
    expected_reward -= env.peer_communication_attempt_penalty
    assert env._communication_reward(0) == pytest.approx(expected_reward)

    actions[0, 3] = -1.0
    env._peer_transmit(actions)
    assert not env.last_tx_active[0]
    assert env._communication_reward(0) == 0.0


def test_failed_report_attempt_receives_communication_penalty() -> None:
    env = make_env()
    env.positions[0] = np.asarray([4900.0, 4900.0, 50.0])
    env.gcs_position = np.asarray([0.0, 0.0, 0.0])
    env._refresh_links()
    assert env._enqueue_report(0, 0)

    env._peer_transmit(tx_to_gcs_actions(env, sender=0))

    assert env.last_tx_active[0]
    assert env.last_report_bytes_transmitted_by_agent[0] == 0
    assert env._communication_reward(0) == pytest.approx(
        -float(env.assumed["comm_reward_max"])
    )


def test_report_expiry_is_nonterminal_and_penalized_for_team() -> None:
    cfg = deepcopy(load_config("masac", "u6"))
    cfg["scenario"]["network_backend"] = "analytical"
    cfg["scenario"]["report_ttl_s"] = 1.0
    cfg["scenario"]["energy_cost_per_j"] = 0.0
    cfg["scenario"]["sensing_cognitive_reward_weight"] = 0.0
    env = PaperUAVEnv(cfg, seed=44)
    set_safe_static_geometry(env)
    assert env._enqueue_report(0, 0)
    env.step_count = 1
    env.rng = FixedRandom(0.99)

    _, rewards, terminated, truncated, info = env.step(idle_actions(env))

    assert not any(terminated.values())
    assert not any(truncated.values())
    assert info["termination_reason"] == "running"
    assert env.expired_reports == 1
    assert all(value <= -env.peer_delivery_reward for value in rewards.values())


def test_depleted_agent_gets_zero_reward_and_individual_termination() -> None:
    cfg = deepcopy(load_config("masac", "u6"))
    cfg["scenario"]["network_backend"] = "analytical"
    cfg["scenario"]["energy_cost_per_j"] = 0.0
    cfg["scenario"]["sensing_cognitive_reward_weight"] = 0.0
    env = PaperUAVEnv(cfg, seed=89)
    set_safe_static_geometry(env)
    env.battery_pct[0] = 0.0
    env.targets[:] = np.asarray(
        [[4000.0, 4000.0 - 200.0 * i] for i in range(env.n_targets)],
        dtype=np.float64,
    )
    env.targets[0] = env.positions[1, :2]
    y, x = env._target_grid_cell(0)
    env.belief_maps[1, y, x] = env.peer_target_confirmation_threshold + 1e-4
    env.fine_positive_cells_by_agent[1, y, x] = True
    env.fine_target_evidence_by_agent[1, 0] = True
    env.rng = FixedRandom(0.0)

    _, rewards, terminated, truncated, _ = env.step(idle_actions(env))

    assert rewards["uav_0"] == 0.0
    assert rewards["uav_1"] > 0.0
    assert terminated["uav_0"]
    assert not terminated["uav_1"]
    assert not truncated["uav_0"]


def test_horizon_mixes_dead_termination_with_live_truncation() -> None:
    cfg = deepcopy(load_config("masac", "u6"))
    cfg["scenario"]["network_backend"] = "analytical"
    cfg["runtime"]["episode_steps_override"] = 1
    cfg["scenario"]["energy_cost_per_j"] = 0.0
    cfg["scenario"]["sensing_cognitive_reward_weight"] = 0.0
    cfg["scenario"]["sensing_target_reward_weight"] = 0.0
    env = PaperUAVEnv(cfg, seed=90)
    set_safe_static_geometry(env)
    env.battery_pct[0] = 0.0
    env.rng = FixedRandom(0.99)

    _, _, terminated, truncated, info = env.step(idle_actions(env))

    assert terminated["uav_0"]
    assert not truncated["uav_0"]
    assert all(not terminated[name] for name in env.agents[1:])
    assert all(truncated[name] for name in env.agents[1:])
    assert info["termination_reason"] == "horizon"


def test_false_confirmation_has_one_symmetric_team_penalty_and_consumes_evidence() -> None:
    cfg = deepcopy(load_config("masac", "u6"))
    cfg["scenario"]["network_backend"] = "analytical"
    cfg["scenario"]["energy_cost_per_j"] = 0.0
    cfg["scenario"]["sensing_cognitive_reward_weight"] = 0.0
    env = PaperUAVEnv(cfg, seed=87)
    set_safe_static_geometry(env)
    env.targets[:] = np.asarray(
        [[4000.0, 4000.0 - 200.0 * i] for i in range(env.n_targets)],
        dtype=np.float64,
    )
    detector = 0
    y, x = env._xy_grid_cell(env.positions[detector, :2])
    env.belief_maps[detector, y, x] = (
        env.peer_target_confirmation_threshold + 1e-4
    )
    env.fine_positive_cells_by_agent[detector, y, x] = True
    env.rng = FixedRandom(0.0)

    _, rewards, _, _, info = env.step(idle_actions(env))

    expected_penalty = -float(env.assumed["search_reward_coeff"])
    expected_penalty *= env.peer_sensing_target_reward_weight
    assert env.confirmed_cells[y, x]
    assert env.false_confirmed_cells[y, x]
    assert info["false_confirmations_step"] == 1
    assert all(value == pytest.approx(expected_penalty) for value in rewards.values())
    assert env.belief_maps[detector, y, x] == pytest.approx(
        belief_probability_floor(env.peer_sync_quantization_levels)
    )
    assert not env.fine_positive_cells_by_agent[detector, y, x]

    _, repeated_rewards, _, _, repeated_info = env.step(idle_actions(env))

    assert repeated_info["false_confirmations_step"] == 0
    assert all(value == pytest.approx(0.0) for value in repeated_rewards.values())


@pytest.mark.parametrize("constraint", ["boundary", "obstacle"])
def test_world_constraint_correction_is_separate_and_penalized(constraint: str) -> None:
    cfg = deepcopy(load_config("masac", "u6"))
    cfg["scenario"]["network_backend"] = "analytical"
    cfg["scenario"]["energy_cost_per_j"] = 0.0
    cfg["scenario"]["sensing_cognitive_reward_weight"] = 0.0
    cfg["scenario"]["sensing_target_reward_weight"] = 0.0
    constrained = PaperUAVEnv(cfg, seed=88)
    idle = PaperUAVEnv(cfg, seed=88)
    for env in (constrained, idle):
        set_safe_static_geometry(env)
        env.rng = FixedRandom(0.99)
        env.obstacles[:, :2] = np.asarray([4900.0, 4900.0])
        env.obstacles[:, 2] = 1.0

    action = idle_actions(constrained)
    if constraint == "boundary":
        for env in (constrained, idle):
            env.positions[0, 0] = 0.0
        action["uav_0"] = np.asarray(
            [1.0, 1.0, 0.0, -1.0, -1.0, -1.0],
            dtype=np.float32,
        )
    else:
        for env in (constrained, idle):
            env.positions[0, :2] = np.asarray([100.0, 2500.0])
            env.obstacles[0] = np.asarray([102.5, 2500.0, 1.0])
        action["uav_0"] = np.asarray(
            [1.0, 0.0, 0.0, -1.0, -1.0, -1.0],
            dtype=np.float32,
        )
    for env in (constrained, idle):
        env._refresh_links()

    _, constrained_rewards, _, _, info = constrained.step(action)
    _, idle_rewards, _, _, _ = idle.step(idle_actions(idle))

    assert np.allclose(constrained.positions[0], idle.positions[0])
    assert constrained.last_world_constraint_correction_m_by_agent[0] > 0.0
    assert constrained.last_safety_correction_m_by_agent[0] == 0.0
    assert constrained_rewards["uav_0"] < idle_rewards["uav_0"]
    assert info["world_constraint_correction_m"] > 0.0
    assert info["max_world_constraint_correction_m"] > 0.0


def test_target_confirmation_reward_is_shared_with_all_agents() -> None:
    cfg = deepcopy(load_config("masac", "u6"))
    cfg["scenario"]["network_backend"] = "analytical"
    cfg["scenario"]["energy_cost_per_j"] = 0.0
    cfg["scenario"]["sensing_cognitive_reward_weight"] = 0.0
    env = PaperUAVEnv(cfg, seed=44)
    set_safe_static_geometry(env)
    env.targets[:] = np.asarray(
        [[4000.0, 4000.0 - 200.0 * i] for i in range(env.n_targets)],
        dtype=np.float64,
    )
    env.targets[0] = env.positions[0, :2]
    y, x = env._target_grid_cell(0)
    env.belief_maps[0, y, x] = env.peer_target_confirmation_threshold
    env.rng = FixedRandom(0.5)

    _, rewards, _, _, _ = env.step(idle_actions(env))

    expected = float(env.assumed["search_reward_coeff"])
    assert env.target_found[0]
    assert all(value == pytest.approx(expected) for value in rewards.values())


def test_final_delivery_reward_is_shared_with_non_sender_agents() -> None:
    cfg = deepcopy(load_config("masac", "u6"))
    cfg["scenario"]["network_backend"] = "analytical"
    cfg["scenario"]["report_bytes"] = 10_000
    cfg["scenario"]["buffer_bytes"] = 30_000
    cfg["scenario"]["energy_cost_per_j"] = 0.0
    cfg["scenario"]["sensing_cognitive_reward_weight"] = 0.0
    env = PaperUAVEnv(cfg, seed=44)
    set_safe_static_geometry(env)
    env.positions[0] = env.gcs_position + np.asarray([0.0, 0.0, 50.0])
    env._refresh_links()
    assert env._enqueue_report(0, 0)
    env.rng = FixedRandom(0.99)
    action_dict = idle_actions(env)
    action_dict["uav_0"] = tx_to_gcs_actions(env, sender=0)[0].astype(np.float32)

    _, rewards, _, _, _ = env.step(action_dict)

    assert env.report_delivered[0]
    assert rewards["uav_1"] == pytest.approx(env.peer_delivery_reward)


@pytest.mark.parametrize("scenario", ["u6", "u9"])
def test_default_buffer_capacity_is_three_complete_reports(scenario: str) -> None:
    cfg = load_config("masac", scenario)["scenario"]
    assert cfg["buffer_bytes"] == 3 * cfg["report_bytes"]


def test_report_age_changes_local_observation_with_identical_queue_bytes() -> None:
    fresh = make_env(seed=71)
    old = make_env(seed=71)
    for env in (fresh, old):
        env.step_count = 100
        assert env._enqueue_report(0, 0)
        env.step_count = 100
    fresh.report_created_step[0] = 99
    old.report_created_step[0] = 1

    fresh_obs = fresh._observations()["uav_0"]
    old_obs = old._observations()["uav_0"]

    assert int(fresh.queue_bytes[0]) == int(old.queue_bytes[0])
    assert not np.array_equal(fresh_obs, old_obs)


def test_nearby_obstacle_changes_local_observation() -> None:
    clear = make_env(seed=72)
    blocked = make_env(seed=72)
    for env in (clear, blocked):
        env.positions[0] = np.asarray([1000.0, 1000.0, 50.0])
        env.obstacles[:, :2] = np.asarray([4900.0, 4900.0])
        env.obstacles[:, 2] = 1.0
    blocked.obstacles[0] = np.asarray([1010.0, 1000.0, 2.0])
    clear._refresh_links()
    blocked._refresh_links()

    clear_obs = clear._observations()["uav_0"]
    blocked_obs = blocked._observations()["uav_0"]

    assert not np.array_equal(clear_obs, blocked_obs)


def test_link_refresh_collects_minimum_and_maximum_power_snapshots(monkeypatch) -> None:
    from uav_search.envs.network_backends import NetworkLinkSnapshot

    env = make_env(seed=73)
    calls: list[float] = []

    def snapshot(positions, gcs_position, obstacles=None, tx_power_w=None):
        power = float(tx_power_w)
        calls.append(power)
        rate = 0.0 if power == env.peer_tx_power_min_w else 2_000_000.0
        pair = np.zeros((env.n_agents, env.n_agents), dtype=np.float64)
        pair[1, 0] = rate
        gcs = np.zeros(env.n_agents, dtype=np.float64)
        gcs[0] = rate
        adjacency = (pair > 0.0).astype(np.int8)
        return NetworkLinkSnapshot(pair, gcs, adjacency)

    monkeypatch.setattr(env.network_backend, "link_snapshot", snapshot)

    env._refresh_links()

    assert calls == [env.peer_tx_power_min_w, env.peer_tx_power_max_w]
    assert env.min_power_pair_rates_bps[1, 0] == 0.0
    assert env.max_power_pair_rates_bps[1, 0] == pytest.approx(2_000_000.0)
    assert env.min_power_gcs_rates_bps[0] == 0.0
    assert env.max_power_gcs_rates_bps[0] == pytest.approx(2_000_000.0)


def test_peer_sync_fuses_transmitted_quantized_belief_not_float64_oracle() -> None:
    env = make_env(seed=74)
    env.positions[0] = np.asarray([500.0, 2500.0, 100.0])
    env.positions[1] = np.asarray([600.0, 2500.0, 100.0])
    env._refresh_links()
    y, x = 10, 10
    env.belief_maps[:, y, x] = 0.5
    env.belief_maps[0, y, x] = 0.9
    actions = np.zeros((env.n_agents, env.action_dim), dtype=np.float64)
    actions[:, 0] = -1.0
    actions[:, 3] = -1.0
    actions[:, 4] = -1.0
    actions[:, 5] = -1.0
    actions[0, 3] = 1.0
    actions[0, 4] = 1.0
    actions[0, 5] = recipient_code(env, 0, 1)

    env._peer_transmit(actions)

    code = encode_belief_probabilities(
        np.array([0.9]),
        env.peer_sync_quantization_levels,
    )
    expected = decode_belief_probabilities(
        code,
        env.peer_sync_quantization_levels,
    )[0]
    assert env.belief_maps[1, y, x] == pytest.approx(expected)
    assert env.belief_maps[1, y, x] != pytest.approx(0.9)


def test_peer_pose_and_altitude_use_vehicle_specific_normalization() -> None:
    env = make_env(seed=75)
    env.positions[0, 2] = env.peer_altitude_max_m
    env.velocities[0, 0] = float(env.paper["multirotor_speed_max_mps"])

    obs = env._observations()["uav_0"]

    assert obs[2] == pytest.approx(1.0)
    assert obs[3] == pytest.approx(1.0)


def test_successful_sync_exposes_cached_sender_queue_to_receiver() -> None:
    env = make_env(seed=76)
    env.positions[0] = np.asarray([500.0, 2500.0, 100.0])
    env.positions[1] = np.asarray([600.0, 2500.0, 100.0])
    env._refresh_links()
    assert env._enqueue_report(0, 0)
    before = env._observations()["uav_1"].copy()
    actions = np.zeros((env.n_agents, env.action_dim), dtype=np.float64)
    actions[:, 0] = -1.0
    actions[:, 3] = -1.0
    actions[:, 4] = -1.0
    actions[:, 5] = -1.0
    actions[0, 3] = 1.0
    actions[0, 4] = 1.0
    actions[0, 5] = recipient_code(env, 0, 1)

    env._peer_transmit(actions)
    after = env._observations()["uav_1"]

    sender_slot_start = 9
    cached_queue_index = sender_slot_start + 6
    assert before[cached_queue_index] == 0.0
    assert after[cached_queue_index] > 0.0


def test_peer_primary_network_feature_is_local_contact_degree_not_duplicate_gcs_flag() -> None:
    env = make_env(seed=77)
    env.last_adjacency.fill(0)
    env.last_adjacency[1, 0] = 1
    env.last_adjacency[2, 0] = 1
    env.last_gcs_rates_bps.fill(0.0)
    env.last_gcs_rates_bps[0] = 2_000_000.0

    assert env._actor_network_state(0) == pytest.approx(2.0 / (env.n_agents - 1))


def test_info_names_safety_distance_events_without_calling_them_collisions() -> None:
    env = make_env(seed=78)
    env.positions[0] = np.asarray([500.0, 500.0, 50.0])
    env.positions[1] = np.asarray([500.0 + 0.5 * env.safety_distance_m, 500.0, 50.0])
    env.episode_safety_violation_uavs = {0, 1}
    env.safety_distance_violation_count = 1

    info = env._info(step_energy_j=0.0, action_saturation=0.0)

    assert info["safety_distance_violation_uavs"] == 2
    assert info["safety_distance_violations"] == 1
    assert "collided_uavs" not in info
    assert "collisions" not in info


def test_continuous_footprint_excludes_zero_area_tangent_neighbor_cells() -> None:
    env = make_env(seed=79)
    env.positions[0] = np.asarray([50.0, 50.0, 50.0])

    assert env._sensing_cells(0) == [(0, 0)]


def test_successful_peer_report_forward_does_not_create_positive_loop_reward() -> None:
    env = make_env(seed=81)
    env.positions[0] = np.asarray([500.0, 2500.0, 100.0])
    env.positions[1] = np.asarray([600.0, 2500.0, 100.0])
    env._refresh_links()
    assert env._enqueue_report(0, 0)

    actions = np.zeros((env.n_agents, env.action_dim), dtype=np.float64)
    actions[:, 0] = -1.0
    actions[:, 3] = -1.0
    actions[:, 4] = -1.0
    actions[:, 5] = -1.0
    actions[0, 3] = 1.0
    actions[0, 4] = 1.0
    actions[0, 5] = recipient_code(env, 0, 1)

    env._peer_transmit(actions)

    assert env.last_report_bytes_transmitted_by_agent[0] > 0
    assert env.report_buffers[0, 1] > 0
    assert env._communication_reward(0) <= 0.0


def test_peer_sync_attempt_has_explicit_overhead_penalty() -> None:
    env = make_env(seed=82)
    env.positions[0] = np.asarray([500.0, 2500.0, 100.0])
    env.positions[1] = np.asarray([600.0, 2500.0, 100.0])
    env._refresh_links()

    actions = np.zeros((env.n_agents, env.action_dim), dtype=np.float64)
    actions[:, 0] = -1.0
    actions[:, 3] = -1.0
    actions[:, 4] = -1.0
    actions[:, 5] = -1.0
    actions[0, 3] = 1.0
    actions[0, 4] = 1.0
    actions[0, 5] = recipient_code(env, 0, 1)

    env._peer_transmit(actions)

    assert env.last_tx_success[0]
    assert env.last_report_bytes_attempted_by_agent[0] == 0
    assert env._communication_reward(0) < 0.0


def test_accumulated_posterior_and_direct_fine_evidence_can_confirm_later() -> None:
    env = make_env(seed=83)
    isolate_agent(env)
    target_idx = 0
    y, x = env._target_grid_cell(target_idx)
    env.positions[0, 2] = env.peer_altitude_max_m
    env.belief_maps[0, y, x] = env.peer_target_confirmation_threshold + 1e-4
    env.fine_positive_cells_by_agent[0, y, x] = True
    env.fine_target_evidence_by_agent[0, target_idx] = True
    env.last_sensor_positive[0, target_idx] = False

    env._confirm_peer_targets()

    assert env.target_found[target_idx]
    assert env.confirmed_cells[y, x]


def test_native_gcs_rate_is_normalized_to_one_in_peer_observation() -> None:
    env = make_env(seed=84)
    env.last_gcs_rates_bps[0] = float(env.scenario["uavnetsim_bit_rate_bps"])

    obs = env._observations()["uav_0"]
    peer_extra_start = 9 + (env.n_agents - 1) * env.peer_neighbor_obs_dim + env.peer_report_obs_dim
    gcs_rate_feature = obs[peer_extra_start + 3]

    assert gcs_rate_feature == pytest.approx(1.0)
