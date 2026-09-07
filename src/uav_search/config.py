from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs"

# The six software-simulation scales shown in Figs. 7-9 of the root paper.
PAPER_SCENARIOS = (
    "f1_m5",
    "f1_m9",
    "f2_m10",
    "f2_m18",
    "f4_m12",
    "f8_m24",
)
PAPER_ALGORITHMS = ("masac", "matd3", "maddpg")
RESEARCH_SCENARIOS = ("u6",)
ALL_SCENARIOS = PAPER_SCENARIOS + RESEARCH_SCENARIOS

# Current reproduction phase: Fig. 7 small-scale experiments only.
# The remaining paper scenarios stay configured for later phases.
ACTIVE_SCENARIOS = ("f1_m5", "f1_m9")


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _deep_merge(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def load_config(algorithm: str, scenario: str) -> dict[str, Any]:
    algorithm = algorithm.lower()
    if algorithm not in PAPER_ALGORITHMS:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    if scenario not in ALL_SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}")
    scenario_path = CONFIG_DIR / "scenarios" / f"{scenario}.yaml"
    algo_path = CONFIG_DIR / "algorithms" / f"{algorithm}.yaml"
    if not scenario_path.exists():
        raise ValueError(f"Scenario config is missing: {scenario_path}")
    cfg = _read_yaml(CONFIG_DIR / "paper.yaml")
    cfg = _deep_merge(cfg, _read_yaml(scenario_path))
    cfg = _deep_merge(cfg, _read_yaml(algo_path))
    return cfg
