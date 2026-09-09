import json
from pathlib import Path

import pytest

from uav_search.config import load_config
from uav_search.envs.paper_env import PaperUAVEnv
from uav_search.runner.network_calibration import (
    RandomWaypointCalibrationPolicy,
    run_calibration_episode,
    summarize_calibration,
    tx_power_to_action,
    write_calibration_outputs,
)


def test_tx_power_to_action_maps_configured_interval():
    assert tx_power_to_action(0.1, 0.1, 0.4) == pytest.approx(-1.0)
    assert tx_power_to_action(0.25, 0.1, 0.4) == pytest.approx(0.0)
    assert tx_power_to_action(0.4, 0.1, 0.4) == pytest.approx(1.0)




def test_calibration_policy_emits_six_dimensional_u6_actions():
    cfg = load_config("masac", "u6")
    cfg["scenario"]["network_backend"] = "analytical"
    env = PaperUAVEnv(cfg, seed=44)
    env.reset(seed=44)
    policy = RandomWaypointCalibrationPolicy(env, seed=44, tx_power_w=0.1, traffic=False)

    actions = policy.actions(env)

    assert set(actions) == set(env.agents)
    assert all(action.shape == (6,) for action in actions.values())
    # Full-3D topology calibration must exercise altitude rather than freezing
    # every UAV at its seeded initial sensing level.
    assert any(abs(float(action[2])) > 1e-6 for action in actions.values())
    assert policy.waypoints.shape == (env.n_agents, 3)
    assert all(env.peer_altitude_min_m <= z <= env.peer_altitude_max_m for z in policy.waypoints[:, 2])


def test_calibration_episode_uses_real_uavnetsim_and_advances_full_macro_time():
    row = run_calibration_episode(contact_range_m=2000.0, seed=44, steps=2, tx_power_w=0.1)
    assert row["network_backend"] == "uavnetsim"
    assert row["executed_steps"] == 2
    assert row["network_sim_time_s"] == pytest.approx(2.0)
    assert row["direct_node_steps"] + row["multihop_node_steps"] + row["disconnected_node_steps"] == 12
    assert row["direct_fraction"] + row["multihop_fraction"] + row["disconnected_fraction"] == pytest.approx(1.0)
    assert row["traffic_metrics_available"] is False
    assert row["byte_pdr"] is None
    assert row["throughput_bps"] is None
    assert row["mean_delay_s"] is None


def test_summary_and_outputs_are_machine_readable(tmp_path):
    rows = [
        {
            "contact_range_m": 1000.0, "seed": 44, "executed_steps": 2,
            "direct_node_steps": 2, "multihop_node_steps": 4, "disconnected_node_steps": 6,
            "connected_hop_sum": 10.0, "connected_hop_count": 6,
            "neighbor_degree_sum": 8.0, "neighbor_degree_samples": 4,
            "attempted_bytes": 100, "delivered_bytes": 80, "delay_weighted_sum": 8.0,
            "phy_failures": 1, "network_tx_energy_j": 0.2,
            "network_sim_time_s": 2.0, "mean_broken_link_s": 1.0,
        },
        {
            "contact_range_m": 1000.0, "seed": 45, "executed_steps": 2,
            "direct_node_steps": 4, "multihop_node_steps": 2, "disconnected_node_steps": 6,
            "connected_hop_sum": 8.0, "connected_hop_count": 6,
            "neighbor_degree_sum": 12.0, "neighbor_degree_samples": 4,
            "attempted_bytes": 100, "delivered_bytes": 100, "delay_weighted_sum": 5.0,
            "phy_failures": 0, "network_tx_energy_j": 0.3,
            "network_sim_time_s": 2.0, "mean_broken_link_s": 2.0,
        },
    ]
    summary = summarize_calibration(rows)
    assert len(summary) == 1
    assert summary[0]["contact_range_m"] == pytest.approx(1000.0)
    assert summary[0]["byte_pdr"] == pytest.approx(0.9)
    assert summary[0]["direct_fraction"] == pytest.approx(0.25)
    assert summary[0]["multihop_fraction"] == pytest.approx(0.25)
    assert summary[0]["disconnected_fraction"] == pytest.approx(0.5)

    csv_path, json_path = write_calibration_outputs(rows, summary, tmp_path)
    assert csv_path.exists()
    assert json_path.exists()
    payload = json.loads(json_path.read_text())
    assert payload["summary"][0]["byte_pdr"] == pytest.approx(0.9)
