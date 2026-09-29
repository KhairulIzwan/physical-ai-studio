"""Decision/policy loop telemetry: latency and cadence of inference calls.

Prior to this callback, ``physicalai.runtime.events.InferenceEvent`` (emitted
by both ``SyncExecution`` and ``AsyncExecution`` on every completed inference)
was never consumed anywhere in Studio, so decision-loop cadence and inference
latency were invisible -- the same gap identified and fixed for the actuator
loop's ``get_metrics_summary()``.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

if TYPE_CHECKING:
    from physicalai.runtime import InferenceEvent

# Mirrors ActuatorLoop100Hz's _RAW_METRICS_DIR: one CSV per session, written on
# close(), so decision-loop timeseries/histograms can be reconstructed offline
# the same way the actuator loop's are.
_RAW_METRICS_DIR = Path.home() / ".local" / "share" / "physicalai" / "decision_loop_metrics"


class InferenceMetricsCallback:
    """Record per-inference latency/cadence and log a summary on session close."""

    def __init__(self) -> None:
        self._timestamps: list[float] = []
        self._latencies_s: list[float] = []

    def on_inference(self, event: InferenceEvent) -> None:
        self._timestamps.append(event.timestamp)
        self._latencies_s.append(event.latency_s)

    def get_metrics_summary(self) -> dict[str, float]:
        """Compute inference latency/cadence stats for this session so far."""
        if not self._latencies_s:
            return {}
        latency_ms = np.array(self._latencies_s) * 1000.0
        summary = {
            "inference_count": float(len(latency_ms)),
            "mean_latency_ms": float(np.mean(latency_ms)),
            "p95_latency_ms": float(np.percentile(latency_ms, 95)),
            "max_latency_ms": float(np.max(latency_ms)),
        }
        if len(self._timestamps) >= 2:
            periods_ms = np.diff(np.array(self._timestamps)) * 1000.0
            summary["mean_decision_period_ms"] = float(np.mean(periods_ms))
            summary["mean_decision_hz"] = float(1000.0 / np.mean(periods_ms))
        return summary

    def close(self) -> None:
        """Log the session's decision-loop metrics summary. Called by the runtime on teardown."""
        metrics = self.get_metrics_summary()
        if metrics:
            logger.info("Decision/policy loop inference metrics: {}", metrics)
            self._dump_raw_metrics_csv()

    def _dump_raw_metrics_csv(self) -> None:
        """Write raw per-inference timestamp/latency/period samples to CSV for offline plotting.

        Complements the summary log line -- one row per inference, so achieved
        decision-loop timeseries/histograms can be reconstructed after the
        session ends. Mirrors ActuatorLoop100Hz._dump_raw_metrics_csv().
        """
        if len(self._timestamps) < 2:
            return
        ts = np.array(self._timestamps)
        latency_ms = np.array(self._latencies_s) * 1000.0
        periods_ms = np.diff(ts) * 1000.0

        try:
            _RAW_METRICS_DIR.mkdir(parents=True, exist_ok=True)
            filename = f"decision_loop_{time.strftime('%Y-%m-%d_%H-%M-%S')}.csv"
            path = _RAW_METRICS_DIR / filename
            with path.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["inference_index", "t_since_start_s", "latency_ms", "period_ms"])
                t0 = ts[0]
                for i in range(len(ts)):
                    period = float(periods_ms[i - 1]) if i > 0 else ""
                    writer.writerow([i, ts[i] - t0, float(latency_ms[i]), period])
            logger.info("InferenceMetricsCallback: raw per-inference metrics written to {}", path)
        except OSError:
            logger.exception("InferenceMetricsCallback: failed to write raw metrics CSV")
