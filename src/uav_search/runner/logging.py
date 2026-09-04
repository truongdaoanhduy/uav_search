from __future__ import annotations

import csv
import json
import math
import traceback
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class RunLogger:
    """Durable local-first experiment logger.

    Local files are always written, even when W&B is online.  This guarantees
    that a network outage cannot destroy the evidence needed to diagnose or
    resume a long training job.
    """

    def __init__(self, run_dir: str | Path, low_window: int = 20):
        self.run_dir = Path(run_dir)
        self.metrics_dir = self.run_dir / "metrics"
        self.logs_dir = self.run_dir / "logs"
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.plots_dir = self.run_dir / "plots"
        for directory in (self.metrics_dir, self.logs_dir, self.checkpoint_dir, self.plots_dir):
            directory.mkdir(parents=True, exist_ok=True)

        self.episode_csv = self.metrics_dir / "episodes.csv"
        self.update_csv = self.metrics_dir / "updates.csv"
        self.performance_csv = self.metrics_dir / "performance.csv"
        self.low_path = self.logs_dir / "low_episodes.jsonl"  # backwards-compatible name
        self.low_csv_path = self.logs_dir / "low_metric_episodes.csv"
        self.error_path = self.logs_dir / "errors.log"
        self.error_jsonl_path = self.logs_dir / "errors.jsonl"
        self.low_path.touch(exist_ok=True)
        self.error_path.touch(exist_ok=True)
        self.error_jsonl_path.touch(exist_ok=True)
        self.low_window = int(low_window)
        self.returns = deque(maxlen=max(2, self.low_window))

    @staticmethod
    def _to_scalar(value: Any) -> Any:
        if hasattr(value, "item"):
            try:
                return value.item()
            except (ValueError, TypeError):
                pass
        return value

    @classmethod
    def _append_csv(cls, path: Path, row: dict[str, Any]) -> None:
        clean = {key: cls._to_scalar(value) for key, value in row.items()}
        exists = path.exists() and path.stat().st_size > 0
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(clean.keys()), extrasaction="ignore")
            if not exists:
                writer.writeheader()
            writer.writerow(clean)

    @staticmethod
    def _signals(metrics: dict[str, Any]) -> list[str]:
        signals: list[str] = []
        if float(metrics.get("search_rate", 1.0)) <= 0.2:
            signals.append("search_low")
        broken = float(metrics.get("mean_broken_link_s", 0.0))
        comm = float(metrics.get("mean_comm_rate_mbps", 99.0))
        if broken >= 5.0:
            signals.append("link_unstable")
        if comm < 1.0:
            signals.append("communication_low")
            if "link_unstable" not in signals:
                signals.append("link_unstable")
        if int(metrics.get("collisions", 0)) > 0 or int(metrics.get("obstacle_hits", 0)) > 0:
            signals.append("collision_high")
        if float(metrics.get("min_battery_pct", 100.0)) <= 15.0:
            signals.append("energy_high")
        if float(metrics.get("action_saturation", 0.0)) >= 0.8:
            signals.append("action_saturated")
        critic_loss = float(metrics.get("critic_loss", 0.0))
        if abs(critic_loss) >= 1e5 or not math.isfinite(critic_loss):
            signals.append("critic_unstable")
        return signals

    @staticmethod
    def _severity(signals: list[str], metrics: dict[str, Any]) -> str:
        critical = {"critic_unstable"}
        if critical.intersection(signals):
            return "critical"
        if int(metrics.get("collisions", 0)) >= 3 or float(metrics.get("min_battery_pct", 100.0)) <= 5.0:
            return "critical"
        if len(signals) >= 4:
            return "critical"
        if signals:
            return "warning"
        return "info"

    def log_episode(self, metrics: dict[str, Any]) -> dict[str, Any] | None:
        current_return = float(metrics.get("return_mean", 0.0))
        historical = list(self.returns)
        self._append_csv(self.episode_csv, metrics)
        signals = self._signals(metrics)
        low_by_return = False
        if len(historical) >= 3:
            sorted_hist = sorted(historical)
            qidx = max(0, int(0.25 * (len(sorted_hist) - 1)))
            low_by_return = current_return <= sorted_hist[qidx]
        if low_by_return:
            signals.append("return_low")

        payload: dict[str, Any] | None = None
        if signals:
            unique_signals = sorted(set(signals))
            payload = dict(metrics)
            payload["signals"] = unique_signals
            payload["severity"] = self._severity(unique_signals, metrics)
            payload["logged_at"] = datetime.now(timezone.utc).isoformat()
            with self.low_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False, allow_nan=False, default=str) + "\n")
            csv_payload = dict(payload)
            csv_payload["signals"] = ";".join(unique_signals)
            self._append_csv(self.low_csv_path, csv_payload)

        self.returns.append(current_return)
        return payload

    def log_update(self, metrics: dict[str, Any]) -> None:
        self._append_csv(self.update_csv, metrics)

    def log_performance(self, metrics: dict[str, Any]) -> None:
        self._append_csv(self.performance_csv, metrics)

    def log_error(self, exc: BaseException, context: dict[str, Any] | None = None) -> dict[str, Any]:
        context = context or {}
        tb = traceback.format_exc()
        if tb.strip() == "NoneType: None":
            tb = ""
        payload = {
            "logged_at": datetime.now(timezone.utc).isoformat(),
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "context": dict(context),
            "traceback": tb,
        }
        with self.error_path.open("a", encoding="utf-8") as handle:
            handle.write(f"\n[{payload['logged_at']}] {payload['exception_type']}: {payload['message']}\n")
            handle.write(f"context={json.dumps(context, ensure_ascii=False, default=str)}\n")
            if tb:
                handle.write(tb)
        with self.error_jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
        return payload
