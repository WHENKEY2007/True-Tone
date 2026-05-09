"""
Audio preprocessing module for True-Tone.

Handles stereo-to-mono conversion, resampling to 16 kHz, volume
normalization, chunk padding/trimming, and consistent chunk shaping
before audio reaches the AI detection engine.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import resample_poly

TARGET_SAMPLE_RATE = 16000
DEFAULT_CHUNK_SECONDS = 3.0


@dataclass(frozen=True)
class PreprocessedChunk:
    """Result of preprocessing a raw audio chunk."""

    samples: np.ndarray          # float32 mono waveform at TARGET_SAMPLE_RATE
    sample_rate: int             # always TARGET_SAMPLE_RATE after processing
    rms: float                   # root-mean-square energy level
    peak: float                  # peak absolute amplitude
    duration_seconds: float      # chunk duration in seconds
    was_padded: bool             # True if silence padding was added
    was_trimmed: bool            # True if the chunk was trimmed


def to_mono(samples: np.ndarray) -> np.ndarray:
    """Convert stereo (or multi-channel) audio to mono by averaging channels."""
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim == 2:
        return np.mean(samples, axis=1)
    return samples.reshape(-1)


def resample(samples: np.ndarray, source_rate: int, target_rate: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    """Resample audio to the target sample rate using polyphase filtering."""
    if source_rate == target_rate:
        return samples.astype(np.float32)
    gcd = np.gcd(source_rate, target_rate)
    up = target_rate // gcd
    down = source_rate // gcd
    return resample_poly(samples, up, down).astype(np.float32)


def normalize(samples: np.ndarray) -> np.ndarray:
    """Peak-normalize audio to the [-1.0, 1.0] range.

    If the audio is silent (all zeros), it is returned unchanged.
    """
    samples = np.asarray(samples, dtype=np.float32)
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    if peak == 0.0:
        return samples
    return samples / max(peak, 1.0)


def pad_or_trim(samples: np.ndarray, target_length: int) -> tuple[np.ndarray, bool, bool]:
    """Ensure audio is exactly ``target_length`` samples.

    Short chunks are zero-padded at the end. Long chunks are trimmed.

    Returns:
        A tuple of (audio, was_padded, was_trimmed).
    """
    samples = np.asarray(samples, dtype=np.float32).reshape(-1)
    if samples.size == target_length:
        return samples, False, False
    if samples.size > target_length:
        return samples[:target_length], False, True
    padded = np.zeros(target_length, dtype=np.float32)
    padded[: samples.size] = samples
    return padded, True, False


def compute_stats(samples: np.ndarray) -> tuple[float, float]:
    """Compute RMS and peak amplitude for an audio chunk.

    Returns:
        A tuple of (rms, peak).
    """
    audio = np.asarray(samples, dtype=np.float32).reshape(-1)
    rms = float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    return rms, peak


def preprocess_chunk(
    samples: np.ndarray,
    source_rate: int,
    target_rate: int = TARGET_SAMPLE_RATE,
    chunk_seconds: float = DEFAULT_CHUNK_SECONDS,
    do_normalize: bool = True,
    enforce_length: bool = True,
) -> PreprocessedChunk:
    """Full preprocessing pipeline for a single raw audio chunk.

    Steps:
        1. Convert to mono.
        2. Resample to ``target_rate``.
        3. Optionally peak-normalize.
        4. Pad or trim to exact ``chunk_seconds`` length.
        5. Compute energy statistics.

    Args:
        samples: Raw audio array (any shape/dtype).
        source_rate: Original sample rate of the input.
        target_rate: Desired output sample rate.
        chunk_seconds: Expected chunk duration (for pad/trim).
        do_normalize: Whether to apply peak normalization.
        enforce_length: Whether to pad/trim to exact chunk size.

    Returns:
        A ``PreprocessedChunk`` ready for the speech gate and detector.
    """
    audio = to_mono(samples)
    audio = resample(audio, source_rate, target_rate)
    if do_normalize:
        audio = normalize(audio)

    was_padded = False
    was_trimmed = False
    if enforce_length:
        target_length = int(target_rate * chunk_seconds)
        audio, was_padded, was_trimmed = pad_or_trim(audio, target_length)

    rms, peak = compute_stats(audio)
    duration = len(audio) / target_rate if target_rate > 0 else 0.0

    return PreprocessedChunk(
        samples=audio,
        sample_rate=target_rate,
        rms=rms,
        peak=peak,
        duration_seconds=duration,
        was_padded=was_padded,
        was_trimmed=was_trimmed,
    )


class AudioPreprocessor:
    """Stateful audio preprocessor with configurable parameters.

    Wraps the functional ``preprocess_chunk`` for use in the pipeline where
    the same settings are applied to every chunk.
    """

    def __init__(
        self,
        source_rate: int = 48000,
        target_rate: int = TARGET_SAMPLE_RATE,
        chunk_seconds: float = DEFAULT_CHUNK_SECONDS,
        do_normalize: bool = True,
        enforce_length: bool = True,
    ):
        self.source_rate = source_rate
        self.target_rate = target_rate
        self.chunk_seconds = chunk_seconds
        self.do_normalize = do_normalize
        self.enforce_length = enforce_length

    def process(self, samples: np.ndarray) -> PreprocessedChunk:
        """Preprocess a raw audio chunk.

        Args:
            samples: Raw audio samples from the capture device.

        Returns:
            A ``PreprocessedChunk`` with consistent shape and sample rate.
        """
        return preprocess_chunk(
            samples,
            source_rate=self.source_rate,
            target_rate=self.target_rate,
            chunk_seconds=self.chunk_seconds,
            do_normalize=self.do_normalize,
            enforce_length=self.enforce_length,
        )
