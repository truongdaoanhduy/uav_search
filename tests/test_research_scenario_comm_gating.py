from copy import deepcopy

import numpy as np
import pytest

from uav_search.config import ACTIVE_SCENARIOS, RESEARCH_SCENARIOS, load_config
from uav_search.envs.paper_env import PaperUAVEnv


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
    assert env.belief_maps[1, y, x] == pytest.approx(0.90)
    assert env.belief_maps[2, y, x] == pytest.approx(0.50)


def test_control_only_peer_sync_cannot_farm_communication_reward():
    env = make_env(seed=106)
    place_single_good_peer_link(env)

    env._peer_transmit(peer_sync_actions(env, sender=0, recipient=1))

    assert env.last_tx_success[0]
    assert env.last_bytes_transmitted == 0
    assert env._communication_reward(0) == pytest.approx(0.0)


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
    assert env.belief_maps[1, y, x] == pytest.approx(0.90)
    # UAV1's slot-start map was still 0.5, so UAV2 cannot receive UAV0's belief
    # through UAV1 until a later macro-step.
    assert env.belief_maps[2, y, x] == pytest.approx(0.50)

    env._peer_transmit(peer_sync_actions(env, sender=1, recipient=2))
    assert env.last_tx_success[1]
    assert env.belief_maps[2, y, x] == pytest.approx(0.90)
