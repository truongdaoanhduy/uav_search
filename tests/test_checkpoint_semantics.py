from copy import deepcopy

import pytest

from uav_search.algorithms.factory import make_algorithm
from uav_search.config import load_config
from uav_search.envs.paper_env import PaperUAVEnv


@pytest.mark.parametrize("name", ["maddpg", "matd3", "masac"])
def test_peer_checkpoint_records_cartesian_hybrid_action_schema(name: str) -> None:
    cfg = deepcopy(load_config(name, "u6"))
    cfg["scenario"]["network_backend"] = "analytical"
    cfg["runtime"]["hidden_sizes"] = [16, 16]
    env = PaperUAVEnv(cfg, seed=201)
    algo = make_algorithm(name, env, cfg, device="cpu", seed=201)

    payload = algo.checkpoint()

    assert payload["checkpoint_format_version"] >= 2
    assert payload["action_schema"] == "peer_cartesian_hybrid_v2"
    assert payload["n_agents"] == env.n_agents
    assert payload["obs_dim"] == env.obs_dim
    assert payload["action_dim"] == env.action_dim


@pytest.mark.parametrize("name", ["maddpg", "matd3", "masac"])
def test_peer_checkpoint_rejects_pre_cartesian_shape_compatible_payload(name: str) -> None:
    cfg = deepcopy(load_config(name, "u6"))
    cfg["scenario"]["network_backend"] = "analytical"
    cfg["runtime"]["hidden_sizes"] = [16, 16]
    env = PaperUAVEnv(cfg, seed=202)
    algo = make_algorithm(name, env, cfg, device="cpu", seed=202)
    payload = algo.checkpoint()
    payload.pop("action_schema", None)
    payload.pop("checkpoint_format_version", None)

    clone = make_algorithm(name, env, cfg, device="cpu", seed=203)
    with pytest.raises(ValueError, match="action schema"):
        clone.load_checkpoint(payload)


def test_legacy_paper_checkpoint_without_schema_remains_loadable() -> None:
    cfg = deepcopy(load_config("maddpg", "f1_m5"))
    cfg["runtime"]["hidden_sizes"] = [16, 16]
    env = PaperUAVEnv(cfg, seed=204)
    algo = make_algorithm("maddpg", env, cfg, device="cpu", seed=204)
    payload = algo.checkpoint()
    payload.pop("action_schema", None)
    payload.pop("checkpoint_format_version", None)

    clone = make_algorithm("maddpg", env, cfg, device="cpu", seed=205)
    clone.load_checkpoint(payload)
