from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs"


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
    if algorithm not in {"masac", "matd3", "maddpg"}:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    scenario_path = CONFIG_DIR / "scenarios" / f"{scenario}.yaml"
    algo_path = CONFIG_DIR / "algorithms" / f"{algorithm}.yaml"
    if not scenario_path.exists():
        raise ValueError(f"Unknown scenario: {scenario}")
    cfg = _read_yaml(CONFIG_DIR / "paper.yaml")
    cfg = _deep_merge(cfg, _read_yaml(scenario_path))
    cfg = _deep_merge(cfg, _read_yaml(algo_path))
    return cfg
