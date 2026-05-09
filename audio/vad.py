"""
Voice Activity Detection module for True-Tone.

Provides multiple speech gate implementations:
  - EnergySpeechGate: Fast CPU-only gate based on RMS/peak thresholds.
  - SileroSpeechGate: More accurate neural VAD using Silero VAD model.
  - HybridSpeechGate: Chains energy gate (fast pre-filter) with Silero
    (accurate confirmation) for the best of both worlds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class SpeechGateResult:
    """Result from any speech gate's process() method."""

    samples: np.ndarray
    is_speech: bool
    rms: float
    peak: float


def _compute_energy(audio: np.ndarray) -> tuple[float, float]:
    """Compute RMS and peak amplitude for audio samples."""
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        return 0.0, 0.0
    rms = float(np.sqrt(np.mean(np.square(audio))))
    peak = float(np.max(np.abs(audio)))
    return rms, peak


class EnergySpeechGate:
    """Fast CPU-only speech gate using RMS and peak amplitude thresholds.

    This is the primary gate used in the live pipeline. It's extremely
    cheap to compute and filters out obvious silence/noise before audio
    reaches the AI detector.

    Args:
        min_rms: Minimum RMS energy to classify as speech.
        min_peak: Minimum peak amplitude to classify as speech.
    """

    def __init__(self, min_rms: float = 0.002, min_peak: float = 0.01):
        self.min_rms = min_rms
        self.min_peak = min_peak

    def process(self, samples: np.ndarray) -> SpeechGateResult:
        audio = np.asarray(samples, dtype=np.float32).reshape(-1)
        rms, peak = _compute_energy(audio)
        is_speech = rms >= self.min_rms and peak >= self.min_peak
        return SpeechGateResult(samples=audio, is_speech=is_speech, rms=rms, peak=peak)


class SileroSpeechGate:
    """Neural speech gate using Silero VAD for higher accuracy.

    Silero VAD is a small ONNX/PyTorch model that's far more accurate
    than energy thresholds for real speech detection, especially in
    noisy environments.

    Args:
        threshold: Speech probability threshold (0.0–1.0).
        sample_rate: Expected input sample rate (must be 8000 or 16000).
        min_speech_ratio: Minimum fraction of frames classified as
            speech within the chunk for the overall chunk to be
            considered speech.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        sample_rate: int = 16000,
        min_speech_ratio: float = 0.3,
    ):
        self.threshold = threshold
        self.sample_rate = sample_rate
        self.min_speech_ratio = min_speech_ratio
        self._model: Any = None
        self._load_model()

    def _load_model(self) -> None:
        """Load Silero VAD model from torch hub (cached after first run)."""
        try:
            self._model, self._utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                trust_repo=True,
            )
            self._model.eval()
            print("Silero VAD model loaded successfully.")
        except Exception as exc:
            print(f"Warning: Could not load Silero VAD: {exc}")
            print("Falling back to energy-based speech detection.")
            self._model = None

    def process(self, samples: np.ndarray) -> SpeechGateResult:
        audio = np.asarray(samples, dtype=np.float32).reshape(-1)
        rms, peak = _compute_energy(audio)

        if self._model is None:
            # Fallback to energy-based detection if Silero unavailable
            is_speech = rms >= 0.002 and peak >= 0.01
            return SpeechGateResult(samples=audio, is_speech=is_speech, rms=rms, peak=peak)

        # Silero VAD operates on 16 kHz audio in 512-sample windows
        tensor = torch.from_numpy(audio)
        window_size = 512
        speech_frames = 0
        total_frames = 0

        self._model.reset_states()

        for start in range(0, len(tensor), window_size):
            chunk = tensor[start : start + window_size]
            if len(chunk) < window_size:
                chunk = torch.nn.functional.pad(chunk, (0, window_size - len(chunk)))

            with torch.no_grad():
                speech_prob = self._model(chunk.unsqueeze(0), self.sample_rate).item()

            total_frames += 1
            if speech_prob >= self.threshold:
                speech_frames += 1

        speech_ratio = speech_frames / max(total_frames, 1)
        is_speech = speech_ratio >= self.min_speech_ratio

        return SpeechGateResult(samples=audio, is_speech=is_speech, rms=rms, peak=peak)


class HybridSpeechGate:
    """Two-stage speech gate: energy pre-filter → Silero confirmation.

    Uses the cheap energy gate as a fast pre-filter to skip obvious
    silence, then runs Silero VAD on chunks that pass the energy test
    for more accurate speech detection.

    Args:
        min_rms: Energy gate minimum RMS threshold.
        min_peak: Energy gate minimum peak threshold.
        silero_threshold: Silero speech probability threshold.
        sample_rate: Expected input sample rate.
    """

    def __init__(
        self,
        min_rms: float = 0.002,
        min_peak: float = 0.01,
        silero_threshold: float = 0.5,
        sample_rate: int = 16000,
    ):
        self._energy_gate = EnergySpeechGate(min_rms=min_rms, min_peak=min_peak)
        self._silero_gate = SileroSpeechGate(
            threshold=silero_threshold, sample_rate=sample_rate
        )

    def process(self, samples: np.ndarray) -> SpeechGateResult:
        # Stage 1: Fast energy check
        energy_result = self._energy_gate.process(samples)
        if not energy_result.is_speech:
            return energy_result

        # Stage 2: Silero VAD confirmation
        return self._silero_gate.process(samples)
