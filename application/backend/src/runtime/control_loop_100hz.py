"""Decoupled high-frequency actuator control loop wrapper.

Wraps a :class:`~physicalai.robot.interface.Robot` so that the caller's
regular-cadence ``send_action()`` calls (driven by :class:`RobotRuntime`'s
camera-gated loop, currently ~30 Hz) are decoupled from the actual motor bus
write rate. Instead of writing to the bus synchronously on every call,
``send_action()`` only records the latest target ("waypoint"); a dedicated
background thread interpolates between waypoints and performs the real bus
I/O at a configurable target frequency (default 100 Hz), independent of
camera/policy timing.

This addresses the root-cause finding that ``action_source.py`` timestamps
only measure the Perception/Orchestration cadence, not a dedicated actuator
control loop -- because, prior to this wrapper, no such dedicated loop
existed: ``RobotRuntime.run()`` (external ``physicalai.runtime`` library)
calls ``robot.send_action()`` synchronously in the same tick that reads
camera frames.

Bus safety: the wrapped robot's serial connection is half-duplex, so reads
(``get_observation()``, called by the main loop) and writes (performed here,
on a background thread) must never overlap. A single lock serializes all
bus access through this wrapper.

Linux notes: unlike the Windows-oriented reference implementation this is
based on, there is no ``winmm.dll`` multimedia timer here. Pacing relies on
a coarse ``time.sleep`` followed by a short busy-wait, which is sufficient
for sub-millisecond jitter on an otherwise-idle core but is not a hard
real-time guarantee without a PREEMPT_RT kernel and SCHED_FIFO scheduling
(attempted best-effort, see :meth:`ActuatorLoop100Hz._try_enable_realtime_priority`).
"""

from __future__ import annotations

import csv
import os
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

if TYPE_CHECKING:
    from physicalai.robot.interface import Robot, RobotObservation

# Raw per-tick telemetry is dumped here on disconnect for offline analysis
# (histograms/timeseries plots), since the in-memory samples are otherwise
# lost once the session ends.
_RAW_METRICS_DIR = Path.home() / ".local" / "share" / "physicalai" / "actuator_loop_metrics"


class ActuatorLoop100Hz:
    """Robot wrapper that decouples actuator bus writes onto a background thread.

    Implements the same structural interface as :class:`Robot` (duck-typed,
    per ``physicalai.robot.interface.Robot``), so it can be passed anywhere a
    ``Robot`` is expected -- in particular, as the ``robot=`` argument to
    ``RobotRuntime`` -- while the real robot continues to be driven
    underneath at a decoupled, higher frequency.
    """

    def __init__(self, robot: Robot, *, target_hz: float = 100.0) -> None:
        """Wrap ``robot`` with a decoupled high-frequency write loop.

        Args:
            robot: The real robot device to wrap. All reads and connection
                lifecycle calls pass through directly; only ``send_action``
                is intercepted.
            target_hz: Target actuator bus write frequency in Hz.
        """
        self._robot = robot
        self.target_hz = target_hz
        self.target_period_s = 1.0 / target_hz

        self._bus_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None

        self._prev_waypoint: np.ndarray | None = None
        self._next_waypoint: np.ndarray | None = None
        self._t_waypoint_start = 0.0
        self._t_waypoint_duration = 1.0 / 30.0

        # Telemetry, cleared on start().
        self._tick_timestamps: list[float] = []
        self._bus_durations_s: list[float] = []

    # -- Robot protocol: pass-through members -----------------------------

    @property
    def joint_names(self) -> list[str]:
        return self._robot.joint_names

    @property
    def device_ids(self) -> tuple[str, ...]:
        return self._robot.device_ids

    def is_connected(self) -> bool:
        return self._robot.is_connected()

    def get_observation(self) -> RobotObservation:
        """Read robot state. Serialized against actuator-thread bus writes."""
        with self._bus_lock:
            return self._robot.get_observation()

    def connect(self) -> None:
        """Connect the wrapped robot and start the background actuator thread."""
        self._robot.connect()
        self._start_loop()

    def disconnect(self) -> None:
        """Stop the background actuator thread, log its achieved metrics, then disconnect the wrapped robot."""
        self._stop_loop()
        metrics = self.get_metrics_summary()
        if metrics:
            logger.info("ActuatorLoop100Hz session metrics: {}", metrics)
            self._dump_raw_metrics_csv()
        self._robot.disconnect()

    # -- Intercepted: send_action becomes a non-blocking waypoint push ----

    def send_action(self, action: np.ndarray, *, goal_time: float = 0.1) -> None:
        """Record the latest target waypoint; the actual bus write happens on the actuator thread.

        Called by ``RobotRuntime.run()`` at its regular (camera-gated,
        currently ~30 Hz) cadence. ``goal_time`` is treated as the expected
        arrival period of the next waypoint (i.e. the interpolation window).
        """
        now = time.perf_counter()
        target = np.array(action, dtype=np.float32, copy=True)
        with self._state_lock:
            self._prev_waypoint = self._next_waypoint if self._next_waypoint is not None else target
            self._next_waypoint = target
            self._t_waypoint_start = now
            self._t_waypoint_duration = max(goal_time, 1e-4)

    # -- Background actuator thread ----------------------------------------

    def _start_loop(self) -> None:
        if self._running:
            return
        self._tick_timestamps = []
        self._bus_durations_s = []
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, name="ActuatorLoop100Hz", daemon=True)
        self._thread.start()

    def _stop_loop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.5)
            self._thread = None

    def _compute_interpolated_setpoint(self, now: float) -> np.ndarray | None:
        with self._state_lock:
            prev, nxt = self._prev_waypoint, self._next_waypoint
            t_start, duration = self._t_waypoint_start, self._t_waypoint_duration
        if nxt is None:
            return None
        if prev is None:
            return nxt
        alpha = np.clip((now - t_start) / duration, 0.0, 1.0)
        return (1.0 - alpha) * prev + alpha * nxt

    def _run_loop(self) -> None:
        """Deterministic-cadence loop: coarse sleep followed by a short busy-wait."""
        self._try_enable_realtime_priority()
        next_tick = time.perf_counter()
        while self._running:
            tick_start = time.perf_counter()
            self._tick_timestamps.append(tick_start)

            setpoint = self._compute_interpolated_setpoint(tick_start)
            if setpoint is not None:
                comm_start = time.perf_counter()
                try:
                    with self._bus_lock:
                        self._robot.send_action(setpoint, goal_time=self.target_period_s)
                except Exception:  # keep the actuator thread alive; log and continue
                    logger.exception("ActuatorLoop100Hz: send_action failed")
                self._bus_durations_s.append(time.perf_counter() - comm_start)

            next_tick += self.target_period_s
            remaining = next_tick - time.perf_counter()
            if remaining > 0.0015:
                time.sleep(remaining - 0.0012)
            while time.perf_counter() < next_tick:
                pass

    @staticmethod
    def _try_enable_realtime_priority() -> None:
        """Best-effort SCHED_FIFO request for this thread (Linux only, requires privilege).

        Silently no-ops if unsupported or unprivileged -- pacing still works
        via the coarse-sleep + busy-wait fallback, just with looser jitter
        bounds under system load.
        """
        if not hasattr(os, "sched_setscheduler"):
            return
        try:
            os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(1))
        except (PermissionError, OSError):
            logger.debug("ActuatorLoop100Hz: SCHED_FIFO unavailable, using best-effort pacing")

    def get_metrics_summary(self) -> dict[str, float]:
        """Compute achieved frequency/jitter/bus-timing stats for this run so far."""
        if len(self._tick_timestamps) < 2:
            return {}
        ts = np.array(self._tick_timestamps)
        periods_ms = np.diff(ts) * 1000.0
        jitter_ms = np.abs(periods_ms - (self.target_period_s * 1000.0))
        return {
            "mean_frequency_hz": float(1000.0 / np.mean(periods_ms)),
            "mean_period_ms": float(np.mean(periods_ms)),
            "std_period_ms": float(np.std(periods_ms)),
            "mean_jitter_ms": float(np.mean(jitter_ms)),
            "p95_jitter_ms": float(np.percentile(jitter_ms, 95)),
            "max_jitter_ms": float(np.max(jitter_ms)),
            "mean_bus_comm_ms": (float(np.mean(self._bus_durations_s) * 1000.0) if self._bus_durations_s else 0.0),
            "total_ticks": float(len(periods_ms)),
        }

    def _dump_raw_metrics_csv(self) -> None:
        """Write raw per-tick period/jitter/bus-duration samples to CSV for offline plotting.

        Complements the summary log line -- one row per tick, so achieved
        timeseries/histograms can be reconstructed after the session ends.
        """
        ts = np.array(self._tick_timestamps)
        if len(ts) < 2:
            return
        periods_ms = np.diff(ts) * 1000.0
        jitter_ms = np.abs(periods_ms - (self.target_period_s * 1000.0))
        bus_ms = np.array(self._bus_durations_s) * 1000.0

        try:
            _RAW_METRICS_DIR.mkdir(parents=True, exist_ok=True)
            filename = f"actuator_loop_{time.strftime('%Y-%m-%d_%H-%M-%S')}.csv"
            path = _RAW_METRICS_DIR / filename
            with path.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["tick_index", "t_since_start_s", "period_ms", "jitter_ms", "bus_comm_ms"])
                t0 = ts[0]
                for i, period in enumerate(periods_ms):
                    bus = float(bus_ms[i]) if i < len(bus_ms) else ""
                    writer.writerow([i, ts[i + 1] - t0, period, jitter_ms[i], bus])
            logger.info("ActuatorLoop100Hz: raw per-tick metrics written to {}", path)
        except OSError:
            logger.exception("ActuatorLoop100Hz: failed to write raw metrics CSV")
