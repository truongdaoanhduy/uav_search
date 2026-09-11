from __future__ import annotations

from collections.abc import Mapping


def episode_finished(
    terminated: Mapping[str, bool],
    truncated: Mapping[str, bool],
) -> bool:
    """Return whether every agent has either terminated or been truncated."""
    agents = terminated.keys() | truncated.keys()
    return bool(agents) and all(
        bool(terminated.get(agent, False) or truncated.get(agent, False))
        for agent in agents
    )
