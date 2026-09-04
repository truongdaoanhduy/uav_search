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
    def __init__(self, run_dir: str | Path, low_window: int = 20):
        self.run_dir = Path(run_dir)
        self.metrics_dir = self.run_dir / "metrics"
        self.logs_dir = self.run_dir / "logs"
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.plots_dir = self.run_dir / "plots"
        for d in (self.metrics_dir, self.logs_dir, self.checkpoint_dir, self.plots_dir):
            d.mkdir(parents=True, exist_ok=True)
        self.episode_csv = self.metrics_dir / "episodes.csv"
        self.update_csv = self.metrics_dir / "updates.csv"
        self.low_path = self.logs_dir / "low_episodes.jsonl"
        self.error_path = self.logs_dir / "errors.log"
        self.low_path.touch(exist_ok=True)
        self.error_path.touch(exist_ok=True)
        self.low_window = int(low_window)
        self.returns = deque(maxlen=max(2, self.low_window))

    @staticmethod
    def _append_csv(path: Path, row: dict[str, Any]) -> None:
        clean = {k: (float(v) if hasattr(v, "item") else v) for k, v in row.items()}
        exists = path.exists() and path.stat().st_size > 0
        with path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(clean.keys()), extrasaction="ignore")
            if not exists:
                writer.writeheader()
            writer.writerow(clean)

    def _signals(self, m: dict[str, Any]) -> list[str]:
        out: list[str] = []
        if float(m.get("search_rate", 1.0)) <= 0.2:
            out.append("search_low")
        if float(m.get("mean_broken_link_s", 0.0)) >= 5.0 or float(m.get("mean_comm_rate_mbps", 99.0)) < 1.0:
            out.append("link_unstable")
        if int(m.get("collisions", 0)) > 0 or int(m.get("obstacle_hits", 0)) > 0:
            out.append("collision_high")
        if float(m.get("min_battery_pct", 100.0)) <= 15.0:
            out.append("energy_high")
        if float(m.get("action_saturation", 0.0)) >= 0.8:
            out.append("action_saturated")
        if abs(float(m.get("critic_loss", 0.0))) >= 1e5 or not math.isfinite(float(m.get("critic_loss", 0.0))):
            out.append("critic_unstable")
        return out

    def log_episode(self, metrics: dict[str, Any]) -> None:
        current_return = float(metrics.get("return_mean", 0.0))
        historical = list(self.returns)
        self._append_csv(self.episode_csv, metrics)
        signals = self._signals(metrics)
        low_by_return = False
        if len(historical) >= 3:
            sorted_hist = sorted(historical)
            qidx = max(0, int(0.25 * (len(sorted_hist) - 1)))
            low_by_return = current_return <= sorted_hist[qidx]
        if signals or low_by_return:
            payload = dict(metrics)
            payload["signals"] = sorted(set(signals + (["return_low"] if low_by_return else [])))
            payload["logged_at"] = datetime.now(timezone.utc).isoformat()
            with self.low_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")
        self.returns.append(current_return)

    def log_update(self, metrics: dict[str, Any]) -> None:
        self._append_csv(self.update_csv, metrics)

    def log_error(self, exc: BaseException, context: dict[str, Any] | None = None) -> None:
        context = context or {}
        with self.error_path.open("a", encoding="utf-8") as f:
            f.write(f"\n[{datetime.now(timezone.utc).isoformat()}] {type(exc).__name__}: {exc}\n")
            f.write(f"context={json.dumps(context, ensure_ascii=False, default=str)}\n")
            tb = traceback.format_exc()
            if tb.strip() != "NoneType: None":
                f.write(tb)
