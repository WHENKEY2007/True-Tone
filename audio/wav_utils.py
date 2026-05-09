from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

TARGET_SAMPLE_RATE = 16000


@dataclass(frozen=True)
class AudioData:
    samples: np.ndarray
    sample_rate: int
    path: Path | None = None


def _to_mono(samples: np.ndarray) -> np.ndarray:
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim == 2:
        return np.mean(samples, axis=1)
    return samples.reshape(-1)


def _normalize(samples: np.ndarray) -> np.ndarray:
    samples = np.asarray(samples, dtype=np.float32)
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    if peak == 0.0:
        return samples
    return samples / max(peak, 1.0)


def _resample(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return samples.astype(np.float32)
    gcd = np.gcd(source_rate, target_rate)
    up = target_rate // gcd
    down = source_rate // gcd
    return resample_poly(samples, up, down).astype(np.float32)


def load_wav(
    filename: str | Path,
    target_sample_rate: int = TARGET_SAMPLE_RATE,
    normalize: bool = True,
) -> AudioData:
    """Load a WAV/audio file as mono float32 samples at the target sample rate."""
    path = Path(filename)
    samples, sample_rate = sf.read(path, always_2d=False, dtype="float32")
    samples = _to_mono(samples)
    samples = _resample(samples, sample_rate, target_sample_rate)
    if normalize:
        samples = _normalize(samples)
    return AudioData(samples=samples.astype(np.float32), sample_rate=target_sample_rate, path=path)


def save_wav(filename: str | Path, samples: np.ndarray, sample_rate: int = TARGET_SAMPLE_RATE) -> Path:
    """Save mono float audio to a WAV file."""
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.asarray(samples, dtype=np.float32), sample_rate)
    return path


def chunk_audio(
    samples: np.ndarray,
    sample_rate: int,
    chunk_seconds: float = 3.0,
    hop_seconds: float | None = None,
) -> list[np.ndarray]:
    """Split audio into fixed-size chunks, optionally using overlapping hops."""
    chunk_size = int(sample_rate * chunk_seconds)
    if chunk_size <= 0:
        raise ValueError("chunk_seconds must be positive")
    hop_size = int(sample_rate * (chunk_seconds if hop_seconds is None else hop_seconds))
    if hop_size <= 0:
        raise ValueError("hop_seconds must be positive")

    samples = np.asarray(samples, dtype=np.float32).reshape(-1)
    if samples.size == 0:
        return [np.zeros(chunk_size, dtype=np.float32)]

    if hop_size >= chunk_size:
        starts = list(range(0, samples.size, hop_size))
    elif samples.size <= chunk_size:
        starts = [0]
    else:
        starts = list(range(0, samples.size - chunk_size + 1, hop_size))
        final_full_start = samples.size - chunk_size
        if starts[-1] != final_full_start:
            starts.append(final_full_start)

    chunks = []
    for start in starts:
        chunk = samples[start : start + chunk_size]
        if chunk.size < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - chunk.size))
        chunks.append(chunk.astype(np.float32))
    return chunks


class WAVUtils:
    """Backwards-compatible wrapper around module-level WAV helpers."""

    @staticmethod
    def write_wav(filename, audio_data, sample_rate=TARGET_SAMPLE_RATE, channels=1):
        del channels
        return save_wav(filename, audio_data, sample_rate)

    @staticmethod
    def read_wav(filename):
        audio = load_wav(filename)
        return audio.samples, audio.sample_rate, 1

    @staticmethod
    def get_wav_info(filename):
        path = Path(filename)
        info = sf.info(path)
        return {
            "path": str(path),
            "sample_rate": info.samplerate,
            "channels": info.channels,
            "duration": info.duration,
            "frames": info.frames,
            "format": info.format,
            "subtype": info.subtype,
        }
