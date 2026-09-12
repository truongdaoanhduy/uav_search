from copy import deepcopy

import pytest

from uav_search.algorithms.factory import make_algorithm
from uav_search.config import load_config
from uav_search.envs.paper_env import PaperUAVEnv


def test_masac_numeric_target_entropy_override_is_honored() -> None:
    cfg = deepcopy(load_config("masac", "u6"))
    cfg["scenario"]["network_backend"] = "analytical"
    cfg["runtime"]["hidden_sizes"] = [8, 8]
    cfg["algorithm"]["target_entropy"] = -2.5
    env = PaperUAVEnv(cfg, seed=301)

    algo = make_algorithm("masac", env, cfg, device="cpu", seed=301)

    assert algo.target_entropy == pytest.approx(-2.5)
