from uav_search.config import load_config


def test_paper_explicit_constants_are_preserved():
    cfg = load_config("masac", "f1_m5")
    p = cfg["paper"]
    assert p["area_size_m"] == 5000
    assert p["episode_steps"] == 50
    assert p["fixed_wing_mass_kg"] == 10
    assert p["multirotor_mass_kg"] == 1
    assert p["communication_power_w"] == 5
    assert p["min_comm_rate_bps"] == 1_000_000
    assert p["fixed_speed_max_mps"] == 40
    assert p["fixed_speed_min_mps"] == 10
    assert p["max_accel_mps2"] == 8
    assert p["multirotor_speed_max_mps"] == 10
    assert p["safe_battery_pct"] == 10
    assert p["energy_reward_scale"] == 0.2
    assert p["training_rounds"] == 50_000
    assert p["learning_rate"] == 0.001
    assert p["gamma"] == 0.99
    assert p["soft_update_epsilon"] == 0.99


def test_scenario_counts_and_algorithm_merge():
    cfg5 = load_config("maddpg", "f1_m5")
    cfg9 = load_config("matd3", "f1_m9")
    assert cfg5["scenario"]["fixed_wing"] == 1
    assert cfg5["scenario"]["multirotor"] == 5
    assert cfg9["scenario"]["fixed_wing"] == 1
    assert cfg9["scenario"]["multirotor"] == 9
    assert cfg5["algorithm"]["name"] == "maddpg"
    assert cfg9["algorithm"]["name"] == "matd3"
    assert cfg5["algorithm"]["gamma"] == 0.99
    assert cfg5["algorithm"]["tau"] == 0.01


def test_assumed_values_are_isolated():
    cfg = load_config("masac", "f1_m5")
    assert "assumed" in cfg
    assert cfg["assumed"]["provenance"] == "ASSUMED"
    assert cfg["paper"]["provenance"] == "PAPER_EXPLICIT"
