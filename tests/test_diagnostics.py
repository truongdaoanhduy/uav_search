import numpy as np

from uav_search.config import ACTIVE_SCENARIOS, load_config
from uav_search.envs.paper_env import PaperUAVEnv
from uav_search.runner.diagnostics import diagnose_episode


def test_current_research_scope_defaults_to_u6_and_u9():
    assert ACTIVE_SCENARIOS == ("u6", "u9")


def test_environment_exposes_aggregate_reward_and_swarm_diagnostics_only():
    cfg = load_config("masac", "f1_m5")
    env = PaperUAVEnv(cfg, seed=123)
    env.reset(seed=123)
    actions = {name: np.zeros(2, dtype=np.float32) for name in env.agents}
    _, rewards, _, _, info = env.step(actions)

    components = info["reward_components_sum"]
    assert set(components) == {"communication", "energy", "safety", "task", "total"}
    assert np.isclose(components["total"], sum(rewards.values()))
    for forbidden in [
        "agent_reward_components", "agent_battery_pct", "agent_action_saturation",
        "agent_broken_link_s", "agent_comm_rate_mbps", "agent_energy_consumption_pct",
        "agent_energy_used_j",
    ]:
        assert forbidden not in info

    expected_pct = 100.0 - float(np.mean(env.battery_pct[env.multirotor_indices]))
    assert np.isclose(info["energy_consumption_pct"], expected_pct)
    assert np.isclose(info["avg_battery_pct"], float(np.mean(env.battery_pct[env.multirotor_indices])))
    for key in ["collided_uavs", "obstacle_hit_uavs", "boundary_hit_uavs", "broken_link_uavs", "depleted_uavs"]:
        assert key in info
        assert isinstance(info[key], int)
        assert 0 <= info[key] <= env.n_agents


def test_diagnose_episode_uses_aggregate_swarm_signals_for_communication_failure():
    metrics = {
        "episode": 31000,
        "phase": "post_convergence_reference",
        "search_rate": 0.1,
        "critic_loss": 2.0,
        "collided_uavs": 0,
        "obstacle_hit_uavs": 0,
        "boundary_hit_uavs": 0,
        "broken_link_uavs": 2,
        "depleted_uavs": 0,
        "avg_battery_pct": 91.0,
        "mean_comm_rate_mbps": 0.2,
        "mean_broken_link_s": 12.0,
        "fixed_return_mean": 80.0,
        "rotor_return_mean": 10.0,
        "reward_task_sum": 0.0,
        "reward_safety_sum": 0.0,
    }
    diagnosis = diagnose_episode(metrics)
    assert diagnosis["failure_scope"] == "rotor_group"
    assert diagnosis["primary_cause"] == "communication"
    assert "worst_agent" not in diagnosis


def test_diagnose_episode_prioritizes_rl_instability_for_swarm_wide_collapse():
    metrics = {
        "episode": 35000,
        "phase": "post_convergence_reference",
        "search_rate": 0.0,
        "critic_loss": 1e8,
        "collided_uavs": 0,
        "obstacle_hit_uavs": 0,
        "boundary_hit_uavs": 0,
        "broken_link_uavs": 0,
        "depleted_uavs": 0,
        "avg_battery_pct": 90.0,
        "fixed_return_mean": -200.0,
        "rotor_return_mean": -300.0,
    }
    diagnosis = diagnose_episode(metrics)
    assert diagnosis["failure_scope"] == "swarm"
    assert diagnosis["primary_cause"] == "rl_training_instability"
    assert "worst_agent" not in diagnosis
