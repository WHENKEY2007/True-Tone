"""
Generate synthetic test audio files for True-Tone testing.

Creates a set of controlled test WAV files with known properties:
  - Pure silence
  - White noise
  - Sine tones at various frequencies
  - Mixed sine tones (simulating speech-like harmonics)

These are useful for verifying the audio pipeline, VAD, and
preprocessing without needing real microphone input.

Usage:
    python tools/generate_test_audio.py
    python tools/generate_test_audio.py --output-dir test_audio/ --duration 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from audio.wav_utils import save_wav

SAMPLE_RATE = 16000


def generate_silence(duration: float) -> np.ndarray:
    """Generate pure silence."""
    return np.zeros(int(SAMPLE_RATE * duration), dtype=np.float32)


def generate_white_noise(duration: float, amplitude: float = 0.3) -> np.ndarray:
    """Generate white noise."""
    samples = int(SAMPLE_RATE * duration)
    return (np.random.randn(samples) * amplitude).astype(np.float32)


def generate_sine(duration: float, frequency: float = 440.0, amplitude: float = 0.5) -> np.ndarray:
    """Generate a pure sine wave."""
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    return (np.sin(2 * np.pi * frequency * t) * amplitude).astype(np.float32)


def generate_speech_like(duration: float, base_freq: float = 150.0) -> np.ndarray:
    """Generate speech-like harmonics (fundamental + overtones).

    Creates a rough approximation of vocal harmonics by combining
    the fundamental frequency with several overtones and adding
    amplitude modulation.
    """
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)

    # Fundamental + harmonics
    signal = np.zeros_like(t)
    for harmonic in range(1, 6):
        amp = 0.4 / harmonic
        signal += amp * np.sin(2 * np.pi * base_freq * harmonic * t)

    # Amplitude modulation (simulates syllable rhythm)
    mod = 0.5 + 0.5 * np.sin(2 * np.pi * 3.0 * t)
    signal *= mod

    # Normalize
    peak = np.max(np.abs(signal))
    if peak > 0:
        signal = signal / peak * 0.7

    return signal.astype(np.float32)


def generate_chirp(duration: float, f_start: float = 100.0, f_end: float = 4000.0) -> np.ndarray:
    """Generate a frequency sweep (chirp)."""
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    phase = 2 * np.pi * (f_start * t + (f_end - f_start) / (2 * duration) * t ** 2)
    return (np.sin(phase) * 0.5).astype(np.float32)


def generate_click_track(duration: float, bpm: float = 120.0) -> np.ndarray:
    """Generate a click track (impulses at regular intervals)."""
    samples = int(SAMPLE_RATE * duration)
    interval = int(SAMPLE_RATE * 60.0 / bpm)
    signal = np.zeros(samples, dtype=np.float32)
    for i in range(0, samples, interval):
        end = min(i + 200, samples)
        t = np.arange(end - i)
        signal[i:end] = 0.8 * np.exp(-t / 30.0) * np.sin(2 * np.pi * 1000 * t / SAMPLE_RATE)
    return signal


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic test audio files.")
    parser.add_argument(
        "--output-dir", default="test_audio",
        help="Directory to save generated files (default: test_audio/).",
    )
    parser.add_argument(
        "--duration", type=float, default=3.0,
        help="Duration in seconds for each test file (default: 3.0).",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dur = args.duration

    test_files = [
        ("silence.wav", generate_silence(dur)),
        ("white_noise.wav", generate_white_noise(dur)),
        ("sine_440hz.wav", generate_sine(dur, 440.0)),
        ("sine_150hz.wav", generate_sine(dur, 150.0)),
        ("sine_1000hz.wav", generate_sine(dur, 1000.0)),
        ("human_speech_like_150hz.wav", generate_speech_like(dur, 150.0)),
        ("human_speech_like_200hz.wav", generate_speech_like(dur, 200.0)),
        ("chirp_100_4000hz.wav", generate_chirp(dur)),
        ("click_track_120bpm.wav", generate_click_track(dur)),
        ("low_volume_noise.wav", generate_white_noise(dur, amplitude=0.001)),
    ]

    print(f"Generating {len(test_files)} test files in {output_dir}/")
    print()

    for name, audio in test_files:
        path = save_wav(output_dir / name, audio)
        rms = float(np.sqrt(np.mean(np.square(audio))))
        peak = float(np.max(np.abs(audio)))
        print(f"  [OK] {name:<35} samples={len(audio):>6}  rms={rms:.5f}  peak={peak:.5f}")

    print(f"\nDone. Files saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
