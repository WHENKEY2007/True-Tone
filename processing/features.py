"""
Handcrafted audio and behavioral features for live synthetic-speech detection.

These features are intentionally cheap enough for rolling 3-second windows.
They complement neural detectors by measuring stability, cadence, pitch drift,
pause structure, entropy, and harmonic consistency: signals that often survive
when conferencing codecs blur model-specific artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import find_peaks


EPSILON = 1e-8


@dataclass(frozen=True)
class AudioFeatureVector:
    """Feature summary for a short speech window."""

    rms: float
    peak: float
    zero_crossing_rate: float
    spectral_entropy: float
    spectral_centroid_hz: float
    spectral_bandwidth_hz: float
    spectral_flatness: float
    high_frequency_ratio: float
    temporal_energy_variance: float
    temporal_energy_cv: float
    pause_ratio: float
    pause_count: int
    mean_pause_seconds: float
    pause_duration_cv: float
    rhythm_irregularity: float
    cadence_consistency: float
    pitch_mean_hz: float
    pitch_std_hz: float
    pitch_drift_hz_per_second: float
    jitter_local: float
    shimmer_local: float
    harmonic_to_noise_db: float
    breathiness_score: float
    synthetic_behavior_score: float
    details: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, float]:
        values = {
            "rms": self.rms,
            "peak": self.peak,
            "zero_crossing_rate": self.zero_crossing_rate,
            "spectral_entropy": self.spectral_entropy,
            "spectral_centroid_hz": self.spectral_centroid_hz,
            "spectral_bandwidth_hz": self.spectral_bandwidth_hz,
            "spectral_flatness": self.spectral_flatness,
            "high_frequency_ratio": self.high_frequency_ratio,
            "temporal_energy_variance": self.temporal_energy_variance,
            "temporal_energy_cv": self.temporal_energy_cv,
            "pause_ratio": self.pause_ratio,
            "pause_count": float(self.pause_count),
            "mean_pause_seconds": self.mean_pause_seconds,
            "pause_duration_cv": self.pause_duration_cv,
            "rhythm_irregularity": self.rhythm_irregularity,
            "cadence_consistency": self.cadence_consistency,
            "pitch_mean_hz": self.pitch_mean_hz,
            "pitch_std_hz": self.pitch_std_hz,
            "pitch_drift_hz_per_second": self.pitch_drift_hz_per_second,
            "jitter_local": self.jitter_local,
            "shimmer_local": self.shimmer_local,
            "harmonic_to_noise_db": self.harmonic_to_noise_db,
            "breathiness_score": self.breathiness_score,
            "synthetic_behavior_score": self.synthetic_behavior_score,
        }
        values.update(self.details)
        return values


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _frame_audio(samples: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    if samples.size < frame_length:
        padded = np.zeros(frame_length, dtype=np.float32)
        padded[: samples.size] = samples
        return padded.reshape(1, frame_length)

    starts = np.arange(0, samples.size - frame_length + 1, hop_length)
    if starts.size == 0:
        starts = np.array([0])
    return np.stack([samples[start : start + frame_length] for start in starts]).astype(np.float32)


def _run_lengths(mask: np.ndarray) -> list[int]:
    if mask.size == 0:
        return []
    lengths: list[int] = []
    current = 0
    for item in mask:
        if bool(item):
            current += 1
        elif current:
            lengths.append(current)
            current = 0
    if current:
        lengths.append(current)
    return lengths


def _spectral_features(samples: np.ndarray, sample_rate: int) -> dict[str, float]:
    windowed = samples * np.hanning(samples.size)
    spectrum = np.abs(np.fft.rfft(windowed))
    power = np.square(spectrum)
    freqs = np.fft.rfftfreq(samples.size, 1.0 / sample_rate)
    total_power = float(np.sum(power)) + EPSILON
    prob = power / total_power

    entropy = -float(np.sum(prob * np.log2(prob + EPSILON)))
    entropy /= float(np.log2(prob.size + EPSILON))
    centroid = float(np.sum(freqs * power) / total_power)
    bandwidth = float(np.sqrt(np.sum(np.square(freqs - centroid) * power) / total_power))
    flatness = float(np.exp(np.mean(np.log(power + EPSILON))) / (np.mean(power) + EPSILON))
    high_ratio = float(np.sum(power[freqs >= 4000.0]) / total_power)

    return {
        "spectral_entropy": _clip01(entropy),
        "spectral_centroid_hz": centroid,
        "spectral_bandwidth_hz": bandwidth,
        "spectral_flatness": _clip01(flatness),
        "high_frequency_ratio": _clip01(high_ratio),
    }


def _estimate_pitch_yin(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    try:
        import librosa

        f0 = librosa.yin(
            samples.astype(np.float32),
            fmin=60.0,
            fmax=450.0,
            sr=sample_rate,
            frame_length=1024,
            hop_length=160,
        )
        f0 = np.asarray(f0, dtype=np.float32)
        f0[~np.isfinite(f0)] = 0.0
        return f0[(f0 >= 60.0) & (f0 <= 450.0)]
    except Exception:
        return np.zeros(0, dtype=np.float32)


def _estimate_hnr_db(samples: np.ndarray, sample_rate: int) -> float:
    frames = _frame_audio(samples, frame_length=int(0.04 * sample_rate), hop_length=int(0.02 * sample_rate))
    ratios = []
    min_lag = max(1, int(sample_rate / 450.0))
    max_lag = max(min_lag + 1, int(sample_rate / 60.0))

    for frame in frames:
        frame = frame - float(np.mean(frame))
        energy = float(np.dot(frame, frame))
        if energy < EPSILON:
            continue
        corr = np.correlate(frame, frame, mode="full")[frame.size - 1 :]
        corr = corr / (corr[0] + EPSILON)
        if corr.size <= min_lag:
            continue
        peak = float(np.max(corr[min_lag : min(max_lag, corr.size)]))
        peak = np.clip(peak, 0.0, 0.999)
        ratios.append(10.0 * np.log10((peak + EPSILON) / (1.0 - peak + EPSILON)))

    if not ratios:
        return 0.0
    return float(np.median(ratios))


def extract_audio_features(samples: np.ndarray, sample_rate: int = 16000) -> AudioFeatureVector:
    """Extract low-latency acoustic and behavioral features from one window."""
    audio = np.asarray(samples, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        audio = np.zeros(int(sample_rate * 3.0), dtype=np.float32)

    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0
    duration = audio.size / sample_rate if sample_rate > 0 else 0.0

    if peak > 1.0:
        audio = audio / peak

    frame_length = int(0.025 * sample_rate)
    hop_length = int(0.010 * sample_rate)
    frames = _frame_audio(audio, frame_length=frame_length, hop_length=hop_length)
    frame_rms = np.sqrt(np.mean(np.square(frames), axis=1) + EPSILON)
    energy_mean = float(np.mean(frame_rms))
    energy_std = float(np.std(frame_rms))
    energy_cv = energy_std / (energy_mean + EPSILON)
    energy_var = float(np.var(frame_rms))

    adaptive_floor = max(0.002, float(np.percentile(frame_rms, 30)) * 0.8)
    pause_mask = frame_rms < adaptive_floor
    pause_lengths = _run_lengths(pause_mask)
    pause_seconds = [length * hop_length / sample_rate for length in pause_lengths if length * hop_length / sample_rate >= 0.08]
    pause_ratio = float(np.mean(pause_mask)) if pause_mask.size else 0.0
    pause_count = len(pause_seconds)
    mean_pause = float(np.mean(pause_seconds)) if pause_seconds else 0.0
    pause_cv = float(np.std(pause_seconds) / (mean_pause + EPSILON)) if pause_seconds else 0.0

    peaks, _ = find_peaks(frame_rms, distance=max(2, int(0.12 * sample_rate / hop_length)))
    if peaks.size >= 3:
        intervals = np.diff(peaks) * hop_length / sample_rate
        rhythm_irregularity = float(np.std(intervals) / (np.mean(intervals) + EPSILON))
    else:
        rhythm_irregularity = 0.0
    cadence_consistency = 1.0 - _clip01(rhythm_irregularity / 0.75)

    zcr = float(np.mean(audio[:-1] * audio[1:] < 0.0)) if audio.size > 1 else 0.0
    spec = _spectral_features(audio, sample_rate)

    f0 = _estimate_pitch_yin(audio, sample_rate)
    if f0.size >= 3:
        pitch_mean = float(np.mean(f0))
        pitch_std = float(np.std(f0))
        pitch_drift = float((f0[-1] - f0[0]) / max(duration, EPSILON))
        jitter = float(np.mean(np.abs(np.diff(f0))) / (pitch_mean + EPSILON))
    else:
        pitch_mean = 0.0
        pitch_std = 0.0
        pitch_drift = 0.0
        jitter = 0.0

    voiced_energy = frame_rms[frame_rms >= adaptive_floor]
    if voiced_energy.size >= 3:
        shimmer = float(np.mean(np.abs(np.diff(voiced_energy))) / (np.mean(voiced_energy) + EPSILON))
    else:
        shimmer = 0.0

    hnr_db = _estimate_hnr_db(audio, sample_rate)

    low_energy_frames = frames[pause_mask] if np.any(pause_mask) else np.zeros((0, frame_length), dtype=np.float32)
    if low_energy_frames.size:
        breath_spec = np.mean(np.abs(np.fft.rfft(low_energy_frames * np.hanning(frame_length), axis=1)), axis=0)
        freqs = np.fft.rfftfreq(frame_length, 1.0 / sample_rate)
        breathiness = float(np.sum(breath_spec[(freqs >= 1000.0) & (freqs <= 6000.0)]) / (np.sum(breath_spec) + EPSILON))
        breathiness *= min(1.0, pause_ratio * 3.0)
    else:
        breathiness = 0.0

    smooth_energy = 1.0 - _clip01(energy_cv / 1.0)
    low_pitch_variation = 1.0 - _clip01(pitch_std / 85.0) if pitch_mean > 0.0 else 0.35
    low_jitter = 1.0 - _clip01(jitter / 0.045) if pitch_mean > 0.0 else 0.35
    low_shimmer = 1.0 - _clip01(shimmer / 0.22)
    low_pause_naturalness = 1.0 - _clip01((pause_ratio + min(pause_count / 5.0, 1.0)) / 2.0)
    harmonic_consistency = _clip01((hnr_db + 5.0) / 25.0)
    low_breath = 1.0 - _clip01(breathiness / 0.35)
    behavior_score = _clip01(
        0.18 * smooth_energy
        + 0.16 * low_pitch_variation
        + 0.14 * low_jitter
        + 0.12 * low_shimmer
        + 0.14 * cadence_consistency
        + 0.12 * low_pause_naturalness
        + 0.08 * harmonic_consistency
        + 0.06 * low_breath
    )

    return AudioFeatureVector(
        rms=rms,
        peak=peak,
        zero_crossing_rate=zcr,
        spectral_entropy=spec["spectral_entropy"],
        spectral_centroid_hz=spec["spectral_centroid_hz"],
        spectral_bandwidth_hz=spec["spectral_bandwidth_hz"],
        spectral_flatness=spec["spectral_flatness"],
        high_frequency_ratio=spec["high_frequency_ratio"],
        temporal_energy_variance=energy_var,
        temporal_energy_cv=energy_cv,
        pause_ratio=_clip01(pause_ratio),
        pause_count=pause_count,
        mean_pause_seconds=mean_pause,
        pause_duration_cv=pause_cv,
        rhythm_irregularity=rhythm_irregularity,
        cadence_consistency=_clip01(cadence_consistency),
        pitch_mean_hz=pitch_mean,
        pitch_std_hz=pitch_std,
        pitch_drift_hz_per_second=pitch_drift,
        jitter_local=jitter,
        shimmer_local=shimmer,
        harmonic_to_noise_db=hnr_db,
        breathiness_score=_clip01(breathiness),
        synthetic_behavior_score=behavior_score,
        details={
            "smooth_energy_score": _clip01(smooth_energy),
            "low_pitch_variation_score": _clip01(low_pitch_variation),
            "low_jitter_score": _clip01(low_jitter),
            "low_shimmer_score": _clip01(low_shimmer),
            "low_pause_naturalness_score": _clip01(low_pause_naturalness),
            "harmonic_consistency_score": _clip01(harmonic_consistency),
            "low_breath_score": _clip01(low_breath),
        },
    )
