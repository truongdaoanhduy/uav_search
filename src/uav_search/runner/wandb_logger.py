from __future__ import annotations

from typing import Any


class WandbLogger:
    """Optional W&B adapter; import happens only when explicitly enabled."""

    def __init__(self, enabled: bool, project: str, run_name: str, config: dict[str, Any]):
        self.run = None
        if not enabled:
            return
        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError("W&B requested but not installed. Run: pip install -e '.[wandb]'") from exc
        self.run = wandb.init(project=project, name=run_name, config=config)

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        if self.run is not None:
            self.run.log(metrics, step=step)

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()
