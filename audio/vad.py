from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SpeechGateResult:
    samples: np.ndarray
    is_speech: bool
    rms: float
    peak: float


class EnergySpeechGate:
    """Small CPU-only speech gate for live Day 3 wiring."""

    def __init__(self, min_rms: float = 0.002, min_peak: float = 0.01):
        self.min_rms = min_rms
        self.min_peak = min_peak

    def process(self, samples: np.ndarray) -> SpeechGateResult:
        audio = np.asarray(samples, dtype=np.float32).reshape(-1)
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        rms = float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0
        is_speech = rms >= self.min_rms and peak >= self.min_peak
        return SpeechGateResult(samples=audio, is_speech=is_speech, rms=rms, peak=peak)
