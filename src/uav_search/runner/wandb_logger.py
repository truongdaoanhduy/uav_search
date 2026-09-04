from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def tracking_mode(api_key: str | None = None, requested: str = "auto") -> str:
    """Resolve tracking mode without ever storing or printing the API key."""
    requested = str(requested).lower()
    if requested not in {"auto", "online", "offline", "disabled"}:
        raise ValueError("W&B mode must be one of: auto, online, offline, disabled")
    if requested == "disabled":
        return "local"
    if requested == "offline":
        return "offline"
    has_key = bool(api_key and api_key.strip())
    if requested == "online":
        return "online" if has_key else "local"
    return "online" if has_key else "local"


class WandbLogger:
    """Best-effort W&B mirror on top of durable local logging.

    The training loop writes local files first.  This class mirrors the same
    information to W&B when credentials are present.  Any W&B SDK/network
    exception disables further remote calls but is never allowed to terminate
    MARL training.
    """

    def __init__(
        self,
        project: str,
        run_name: str,
        config: dict[str, Any],
        run_dir: str | Path,
        api_key: str | None = None,
        mode: str = "auto",
        entity: str | None = None,
        enabled: bool | None = None,
    ):
        self.run_dir = Path(run_dir)
        self.logs_dir = self.run_dir / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.fallback_path = self.logs_dir / "tracking_fallback.log"
        if enabled is False:
            mode = "disabled"
        self.requested_mode = str(mode).lower()
        self.mode = tracking_mode(api_key=api_key, requested=self.requested_mode)
        self.active = False
        self.online = False
        self.run = None
        self._wandb = None
        self._low_rows: list[dict[str, Any]] = []
        self._error_rows: list[dict[str, Any]] = []

        if self.mode == "local":
            return

        try:
            import wandb

            self._wandb = wandb
            if self.mode == "online":
                wandb.login(key=api_key, relogin=True)
            kwargs: dict[str, Any] = {
                "project": project,
                "name": run_name,
                "config": config,
                "mode": self.mode,
                "dir": str(self.run_dir),
            }
            if entity:
                kwargs["entity"] = entity
            self.run = wandb.init(**kwargs)
            self.active = self.run is not None
            self.online = self.active and self.mode == "online"
        except BaseException as exc:
            self._fallback("init", exc)

    @property
    def status(self) -> str:
        if self.active:
            return f"wandb-{self.mode}"
        return "local"

    def _fallback(self, operation: str, exc: BaseException) -> None:
        self.active = False
        self.online = False
        self.mode = "local"
        with self.fallback_path.open("a", encoding="utf-8") as handle:
            handle.write(f"{operation}: {type(exc).__name__}: {exc}\n")

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        if not self.active or self.run is None:
            return
        try:
            self.run.log(metrics, step=step)
        except BaseException as exc:
            self._fallback("log", exc)

    def log_low_episode(self, payload: dict[str, Any]) -> None:
        row = dict(payload)
        self._low_rows.append(row)
        if not self.active or self.run is None:
            return
        try:
            self.run.log({"diagnostics/low_episode": json.dumps(row, ensure_ascii=False, default=str)})
        except BaseException as exc:
            self._fallback("low-episode", exc)

    def log_error(self, payload: dict[str, Any]) -> None:
        row = dict(payload)
        self._error_rows.append(row)
        if not self.active or self.run is None:
            return
        try:
            self.run.log({"diagnostics/error": json.dumps(row, ensure_ascii=False, default=str)})
        except BaseException as exc:
            self._fallback("error", exc)

    def log_image(self, path: str | Path, key: str) -> None:
        if not self.active or self.run is None or self._wandb is None:
            return
        file_path = Path(path)
        if not file_path.exists():
            return
        try:
            self.run.log({key: self._wandb.Image(str(file_path))})
        except BaseException as exc:
            self._fallback(f"image:{key}", exc)

    def log_model(self, path: str | Path, name: str, aliases: Iterable[str] = ()) -> None:
        if not self.active or self.run is None or self._wandb is None:
            return
        file_path = Path(path)
        if not file_path.exists():
            return
        try:
            artifact = self._wandb.Artifact(name=name, type="model")
            artifact.add_file(str(file_path), name=file_path.name)
            self.run.log_artifact(artifact, aliases=list(aliases))
        except BaseException as exc:
            self._fallback(f"model:{file_path.name}", exc)

    def log_run_artifact(self, name: str) -> None:
        if not self.active or self.run is None or self._wandb is None:
            return
        try:
            artifact = self._wandb.Artifact(name=name, type="run-output")
            for filename in ("config.yaml", "summary.json", "evaluation.json"):
                path = self.run_dir / filename
                if path.exists():
                    artifact.add_file(str(path), name=filename)
            for dirname in ("checkpoints", "metrics", "logs", "plots", "rollouts"):
                path = self.run_dir / dirname
                if path.exists():
                    artifact.add_dir(str(path))
            self.run.log_artifact(artifact, aliases=["latest"])
        except BaseException as exc:
            self._fallback("run-artifact", exc)

    def update_summary(self, values: dict[str, Any]) -> None:
        if not self.active or self.run is None:
            return
        try:
            self.run.summary.update(values)
        except BaseException as exc:
            self._fallback("summary", exc)

    @staticmethod
    def _table_rows(rows: list[dict[str, Any]]) -> tuple[list[str], list[list[Any]]]:
        if not rows:
            return [], []
        columns = sorted({key for row in rows for key in row})
        data: list[list[Any]] = []
        for row in rows:
            values: list[Any] = []
            for column in columns:
                value = row.get(column)
                if isinstance(value, (dict, list, tuple)):
                    value = json.dumps(value, ensure_ascii=False, default=str)
                values.append(value)
            data.append(values)
        return columns, data

    def _flush_table(self, key: str, rows: list[dict[str, Any]]) -> None:
        if not rows or not self.active or self.run is None or self._wandb is None:
            return
        try:
            columns, data = self._table_rows(rows)
            self.run.log({key: self._wandb.Table(columns=columns, data=data)})
        except BaseException as exc:
            self._fallback(f"table:{key}", exc)

    def finish(self) -> None:
        if not self.active or self.run is None:
            return
        self._flush_table("diagnostics/low_metric_episodes", self._low_rows)
        self._flush_table("diagnostics/errors", self._error_rows)
        if not self.active:
            return
        try:
            self.run.finish()
        except BaseException as exc:
            self._fallback("finish", exc)
