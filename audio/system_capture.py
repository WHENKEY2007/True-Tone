from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np
import soundcard as sc
from scipy.io.wavfile import write
from scipy.signal import resample_poly

SAMPLE_RATE = 16000
CHUNK_DURATION = 3
OUTPUT_DIR = Path("test_system_chunks")


def list_system_capture_devices() -> None:
    """Print speaker devices that support loopback recording through soundcard."""
    print("Available speaker loopback devices:")
    for index, speaker in enumerate(sc.all_speakers()):
        default_marker = "*" if speaker.name == sc.default_speaker().name else " "
        print(f"{default_marker} {index}: {speaker.name}")


def _to_mono(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2 and audio.shape[1] > 1:
        return np.mean(audio, axis=1)
    return np.squeeze(audio)


def _to_int16(audio: np.ndarray) -> np.ndarray:
    audio = np.squeeze(audio)
    audio = np.clip(audio, -1.0, 1.0)
    return (audio * np.iinfo(np.int16).max).astype(np.int16)


def _volume_stats(audio: np.ndarray) -> tuple[float, float]:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    rms = float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    return rms, peak


def _resample_to_target(audio: np.ndarray, source_rate: int) -> np.ndarray:
    if source_rate == SAMPLE_RATE:
        return audio.astype(np.float32)
    gcd = np.gcd(source_rate, SAMPLE_RATE)
    up = SAMPLE_RATE // gcd
    down = source_rate // gcd
    return resample_poly(audio, up, down, axis=0).astype(np.float32)


def _speaker_by_index(device: int | None):
    speakers = sc.all_speakers()
    if not speakers:
        raise RuntimeError("No speaker devices found for system audio capture.")

    if device is None:
        return sc.default_speaker()

    if device < 0 or device >= len(speakers):
        raise ValueError(f"Invalid speaker device index {device}. Run with --list-devices.")

    return speakers[device]


class SystemAudioCapture:
    """Capture speaker output as 16 kHz mono float32 chunks."""

    def __init__(
        self,
        device: int | None = None,
        chunk_duration: int = CHUNK_DURATION,
        capture_rate: int = 48000,
    ):
        self.device = device
        self.speaker = _speaker_by_index(device)
        self.chunk_duration = chunk_duration
        self.capture_rate = capture_rate

    def start(self) -> None:
        print(f"Using system speaker loopback: {self.speaker.name}")

    def stop(self) -> None:
        pass

    def read(self) -> np.ndarray:
        frames = int(self.chunk_duration * self.capture_rate)
        with sc.get_microphone(
            id=str(self.speaker.name),
            include_loopback=True,
        ).recorder(samplerate=self.capture_rate) as recorder:
            audio = recorder.record(numframes=frames)

        mono = _to_mono(audio)
        return _resample_to_target(mono, self.capture_rate)


def record_chunk(chunk_number: int, capture: SystemAudioCapture) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Recording system chunk {chunk_number} for {capture.chunk_duration}s...")
    audio = capture.read()
    rms, peak = _volume_stats(audio)
    print(f"Volume: rms={rms:.6f}, peak={peak:.6f}")

    if peak < 0.001:
        print("Warning: system audio is nearly silent. Play audio through your speakers and check output volume.")

    device_label = "default" if capture.device is None else f"device_{capture.device}"
    filename = OUTPUT_DIR / f"system_chunk_{chunk_number:03d}_{device_label}_{time.time_ns()}.wav"
    write(str(filename), SAMPLE_RATE, _to_int16(audio))
    print(f"Saved: {filename}")
    return filename


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture system/speaker audio into 3-second WAV chunks.")
    parser.add_argument("--chunks", type=int, default=3, help="Number of chunks to record.")
    parser.add_argument("--device", type=int, default=None, help="Speaker index from --list-devices.")
    parser.add_argument("--list-devices", action="store_true", help="List speaker loopback devices and exit.")
    args = parser.parse_args()

    if args.list_devices:
        list_system_capture_devices()
        return

    capture = SystemAudioCapture(device=args.device)
    capture.start()
    try:
        for chunk_number in range(1, args.chunks + 1):
            record_chunk(chunk_number, capture)
    finally:
        capture.stop()


if __name__ == "__main__":
    main()
