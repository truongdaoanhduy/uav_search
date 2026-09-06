import numpy as np

from uav_search.config import ACTIVE_SCENARIOS, load_config
from uav_search.envs.paper_env import PaperUAVEnv
from uav_search.runner.diagnostics import diagnose_episode


def test_current_reproduction_scope_defaults_to_fig7_small_scale_scenarios():
    assert ACTIVE_SCENARIOS == ("f1_m5", "f1_m9")


def test_environment_exposes_per_agent_reward_network_and_energy_diagnostics():
    cfg = load_config("masac", "f1_m5")
    env = PaperUAVEnv(cfg, seed=123)
    env.reset(seed=123)
    actions = {name: np.zeros(2, dtype=np.float32) for name in env.agents}
    _, rewards, _, _, info = env.step(actions)

    components = info["agent_reward_components"]
    assert set(components) == set(env.agents)
    assert set(components["rotor_0"]) == {"communication", "energy", "safety", "task", "total"}
    assert set(components["fixed_0"]) == {"communication", "energy", "safety", "task", "total"}
    for name in env.agents:
        assert np.isclose(components[name]["total"], rewards[name])

    assert set(info["agent_battery_pct"]) == set(env.agents)
    assert set(info["agent_action_saturation"]) == set(env.agents)
    assert set(info["agent_broken_link_s"]) == {f"rotor_{i}" for i in range(env.n_rotor)}
    assert set(info["agent_comm_rate_mbps"]) == {f"rotor_{i}" for i in range(env.n_rotor)}
    expected_pct = 100.0 - float(np.mean(env.battery_pct[env.multirotor_indices]))
    assert np.isclose(info["energy_consumption_pct"], expected_pct)


def test_diagnose_episode_identifies_single_rotor_communication_failure():
    metrics = {
        "episode": 31000,
        "phase": "post_convergence_reference",
        "search_rate": 0.1,
        "critic_loss": 2.0,
        "collisions": 0,
        "obstacle_hits": 0,
        "agent/fixed_0/return": 80.0,
        "agent/rotor_0/return": 75.0,
        "agent/rotor_1/return": -100.0,
        "agent/rotor_2/return": 78.0,
        "agent/rotor_3/return": 76.0,
        "agent/rotor_4/return": 77.0,
        "agent/rotor_1/comm_rate_mbps": 0.2,
        "agent/rotor_1/broken_link_s": 12.0,
        "agent/rotor_1/battery_pct": 91.0,
        "agent/rotor_1/safety_reward": 0.0,
        "agent/rotor_1/task_reward": 0.0,
        "agent/rotor_1/action_saturation": 0.1,
    }
    diagnosis = diagnose_episode(metrics)
    assert diagnosis["failure_scope"] == "single_agent"
    assert diagnosis["worst_agent"] == "rotor_1"
    assert diagnosis["primary_cause"] == "communication"


def test_diagnose_episode_prioritizes_rl_instability_for_swarm_wide_collapse():
    metrics = {
        "episode": 35000,
        "phase": "post_convergence_reference",
        "search_rate": 0.0,
        "critic_loss": 1e8,
        "agent/fixed_0/return": -200.0,
        "agent/rotor_0/return": -300.0,
        "agent/rotor_1/return": -310.0,
        "agent/rotor_2/return": -290.0,
        "agent/rotor_3/return": -305.0,
        "agent/rotor_4/return": -295.0,
    }
    diagnosis = diagnose_episode(metrics)
    assert diagnosis["failure_scope"] == "swarm"
    assert diagnosis["primary_cause"] == "rl_training_instability"
