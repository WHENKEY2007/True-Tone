"""
Temporal confidence aggregation for live AI voice detection.

Chunk classifiers are noisy in real calls. This module turns raw per-window
scores into a session-aware probability using rolling memory, EMA smoothing,
trend analysis, uncertainty, and hysteresis.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import time

import numpy as np


@dataclass(frozen=True)
class TemporalDecision:
    raw_probability: float
    stabilized_probability: float
    rolling_probability: float
    ema_probability: float
    trend: float
    uncertainty: float
    anomaly_score: float
    state: str
    speech_window_count: int
    synthetic_streak: int
    real_streak: int
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True)
class _TemporalPoint:
    probability: float
    behavior_score: float
    cadence_consistency: float
    is_speech: bool
    timestamp: float


class TemporalConfidenceAggregator:
    """Session-level score stabilizer with hysteresis and confidence memory."""

    def __init__(
        self,
        memory_seconds: float = 30.0,
        ema_alpha: float = 0.35,
        enter_synthetic: float = 0.65,
        exit_synthetic: float = 0.52,
        enter_real: float = 0.32,
        exit_real: float = 0.46,
        min_speech_windows: int = 2,
    ):
        self.memory_seconds = memory_seconds
        self.ema_alpha = ema_alpha
        self.enter_synthetic = enter_synthetic
        self.exit_synthetic = exit_synthetic
        self.enter_real = enter_real
        self.exit_real = exit_real
        self.min_speech_windows = min_speech_windows

        self._points: deque[_TemporalPoint] = deque()
        self._ema: float | None = None
        self._state = "insufficient_speech"
        self._synthetic_streak = 0
        self._real_streak = 0

    def reset(self) -> None:
        self._points.clear()
        self._ema = None
        self._state = "insufficient_speech"
        self._synthetic_streak = 0
        self._real_streak = 0

    def update(
        self,
        probability: float,
        *,
        is_speech: bool,
        behavior_score: float = 0.0,
        cadence_consistency: float = 0.0,
        timestamp: float | None = None,
    ) -> TemporalDecision:
        now = time.time() if timestamp is None else timestamp
        probability = float(np.clip(probability, 0.0, 1.0))
        behavior_score = float(np.clip(behavior_score, 0.0, 1.0))
        cadence_consistency = float(np.clip(cadence_consistency, 0.0, 1.0))

        if is_speech:
            self._points.append(
                _TemporalPoint(
                    probability=probability,
                    behavior_score=behavior_score,
                    cadence_consistency=cadence_consistency,
                    is_speech=True,
                    timestamp=now,
                )
            )
            self._ema = probability if self._ema is None else (
                self.ema_alpha * probability + (1.0 - self.ema_alpha) * self._ema
            )
        elif self._ema is not None:
            # Confidence decay during silence prevents old suspicion from
            # hanging forever while still avoiding instant resets.
            self._ema *= 0.96

        self._expire_old(now)
        speech_points = [p for p in self._points if p.is_speech]
        probs = np.array([p.probability for p in speech_points], dtype=np.float32)

        if probs.size == 0:
            rolling = 0.0
            trend = 0.0
            uncertainty = 1.0
            anomaly = 0.0
        else:
            weights = np.linspace(0.55, 1.0, probs.size, dtype=np.float32)
            rolling = float(np.average(probs, weights=weights))
            trend = self._compute_trend(probs)
            uncertainty = float(np.clip(np.std(probs) * 1.7, 0.0, 1.0))
            behavior = float(np.mean([p.behavior_score for p in speech_points]))
            cadence = float(np.mean([p.cadence_consistency for p in speech_points]))
            anomaly = float(np.clip(0.65 * behavior + 0.35 * cadence, 0.0, 1.0))

        ema = float(self._ema if self._ema is not None else 0.0)
        stabilized = float(np.clip(0.48 * ema + 0.34 * rolling + 0.10 * anomaly + 0.08 * max(trend, 0.0), 0.0, 1.0))

        if probs.size < self.min_speech_windows:
            self._state = "insufficient_speech"
        else:
            self._update_state(stabilized)

        return TemporalDecision(
            raw_probability=probability,
            stabilized_probability=stabilized if probs.size >= self.min_speech_windows else probability,
            rolling_probability=rolling,
            ema_probability=ema,
            trend=trend,
            uncertainty=uncertainty,
            anomaly_score=anomaly,
            state=self._state,
            speech_window_count=int(probs.size),
            synthetic_streak=self._synthetic_streak,
            real_streak=self._real_streak,
            timestamp=now,
        )

    def _expire_old(self, now: float) -> None:
        cutoff = now - self.memory_seconds
        while self._points and self._points[0].timestamp < cutoff:
            self._points.popleft()

    def _compute_trend(self, probs: np.ndarray) -> float:
        if probs.size < 3:
            return 0.0
        y = probs[-min(8, probs.size) :]
        x = np.arange(y.size, dtype=np.float32)
        slope = float(np.polyfit(x, y, 1)[0])
        return float(np.clip(slope * 4.0, -1.0, 1.0))

    def _update_state(self, stabilized: float) -> None:
        if stabilized >= self.enter_synthetic:
            self._synthetic_streak += 1
            self._real_streak = 0
        elif stabilized <= self.enter_real:
            self._real_streak += 1
            self._synthetic_streak = 0
        else:
            self._synthetic_streak = max(0, self._synthetic_streak - 1)
            self._real_streak = max(0, self._real_streak - 1)

        if self._state == "likely_synthetic":
            if stabilized <= self.exit_synthetic and self._synthetic_streak == 0:
                self._state = "suspicious"
        elif self._state == "likely_real":
            if stabilized >= self.exit_real and self._real_streak == 0:
                self._state = "suspicious"
        elif self._synthetic_streak >= 2:
            self._state = "likely_synthetic"
        elif self._real_streak >= 2:
            self._state = "likely_real"
        else:
            self._state = "suspicious"
