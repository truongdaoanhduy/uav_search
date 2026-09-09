from uav_search.runner.train import _sensing_metrics_from_info


def test_sensing_metrics_from_info_keeps_episode_level_altitude_and_belief_fields():
    info = {
        "mean_altitude_m": 100.0,
        "min_altitude_m": 50.0,
        "max_altitude_m": 150.0,
        "mean_belief_entropy": 0.75,
        "mean_target_posterior": 0.6,
        "scanned_cells_total": 123,
        "positive_sensor_observations_total": 17,
        "information_gain_total": 9.25,
        "targets_confirmed_total": 3,
    }

    metrics = _sensing_metrics_from_info(info)

    for key, value in info.items():
        assert metrics[key] == value
    assert metrics["false_confirmations_total"] == 0
    assert metrics["confirmed_cells_total"] == 0


def test_sensing_metrics_from_info_has_zero_defaults_for_legacy_scenarios():
    metrics = _sensing_metrics_from_info({})
    assert metrics["mean_altitude_m"] == 0.0
    assert metrics["scanned_cells_total"] == 0
    assert metrics["information_gain_total"] == 0.0
    assert metrics["targets_confirmed_total"] == 0
    assert metrics["false_confirmations_total"] == 0
    assert metrics["confirmed_cells_total"] == 0


def test_sensing_metrics_from_info_propagates_cell_confirmation_diagnostics():
    info = {
        "false_confirmations_total": 4,
        "confirmed_cells_total": 9,
    }

    metrics = _sensing_metrics_from_info(info)

    assert metrics["false_confirmations_total"] == 4
    assert metrics["confirmed_cells_total"] == 9
