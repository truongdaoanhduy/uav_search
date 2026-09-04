from __future__ import annotations

from .maddpg import MADDPG
from .masac import MASAC
from .matd3 import MATD3


def make_algorithm(name: str, env, cfg, device: str = "cpu", seed: int = 0):
    classes = {"maddpg": MADDPG, "matd3": MATD3, "masac": MASAC}
    try:
        cls = classes[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported algorithm {name!r}; choose {sorted(classes)}") from exc
    return cls(env, cfg, device=device, seed=seed)
