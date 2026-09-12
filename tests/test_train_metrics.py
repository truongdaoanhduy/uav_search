from typing import ClassVar

import numpy as np

import uav_search.runner.train as train_module
from uav_search.runner.train import _sensing_metrics_from_info, deterministic_rollout


def test_deterministic_rollout_reports_episode_network_metrics_not_last_step(monkeypatch):
    """A later idle slot must not erase traffic delivered earlier in the episode."""

    class FakeEnv:
        agents: ClassVar[list[str]] = ["uav_0"]
        n_agents = 1
        action_dim = 6
        max_steps = 2
        dt = 1.0
        fixed_indices: ClassVar[list[int]] = []
        multirotor_indices: ClassVar[list[int]] = [0]
        total_energy_used_j = 0.0

        def __init__(self, _cfg, seed):
            self.seed = seed
            self._step = 0

        def reset(self, seed):
            self.seed = seed
            self._step = 0
            return {"uav_0": np.zeros(1, dtype=np.float32)}, {}

        def step(self, _actions):
            self._step += 1
            last_slot_had_traffic = self._step == 1
            info = {
                "step": self._step,
                "mean_comm_rate_mbps": 0.0,
                "action_saturation": 0.0,
                "network_byte_pdr": 1.0 if last_slot_had_traffic else 0.0,
                "network_throughput_bps": 2_000_000.0 if last_slot_had_traffic else 0.0,
                "total_network_attempted_bytes": 250_000,
                "total_network_delivered_bytes": 250_000,
            }
            terminated = {"uav_0": False}
            truncated = {"uav_0": self._step == self.max_steps}
            return (
                {"uav_0": np.zeros(1, dtype=np.float32)},
                {"uav_0": 0.0},
                terminated,
                truncated,
                info,
            )

    class FakeAlgo:
        def act(self, _obs, explore):
            assert explore is False
            return np.zeros((1, 6), dtype=np.float32)

    monkeypatch.setattr(train_module, "PaperUAVEnv", FakeEnv)

    _env, metrics = deterministic_rollout(FakeAlgo(), {}, seed=44)

    assert metrics["network_byte_pdr"] == 1.0
    assert metrics["network_throughput_bps"] == 1_000_000.0


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
