from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np
import sounddevice as sd
from scipy.signal import resample_poly
from scipy.io.wavfile import write

SAMPLE_RATE = 16000
CHUNK_DURATION = 3
OUTPUT_DIR = Path("test_chunks")


def list_input_devices() -> None:
    """Print available input devices with their sounddevice indexes."""
    print("Available input devices:")
    for index, device in enumerate(sd.query_devices()):
        if device["max_input_channels"] > 0:
            marker = "*" if index == sd.default.device[0] else " "
            print(
                f"{marker} {index}: {device['name']} "
                f"({device['max_input_channels']} channels, "
                f"default_samplerate={int(device['default_samplerate'])})"
            )


def _to_int16(audio: np.ndarray) -> np.ndarray:
    audio = np.squeeze(audio)
    if audio.dtype.kind == "f":
        audio = np.clip(audio, -1.0, 1.0)
        audio = (audio * np.iinfo(np.int16).max).astype(np.int16)
    return audio


def _volume_stats(audio: np.ndarray) -> tuple[float, float]:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    rms = float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    return rms, peak


def _capture_sample_rate(device: int | None) -> int:
    device_id = sd.default.device[0] if device is None else device
    return int(sd.query_devices(device_id)["default_samplerate"])


def _resample_to_target(audio: np.ndarray, source_rate: int) -> np.ndarray:
    if source_rate == SAMPLE_RATE:
        return audio.astype(np.float32)
    gcd = np.gcd(source_rate, SAMPLE_RATE)
    up = SAMPLE_RATE // gcd
    down = source_rate // gcd
    return resample_poly(audio, up, down, axis=0).astype(np.float32)


class MicrophoneCapture:
    """Capture microphone input as 16 kHz mono float32 chunks."""

    def __init__(self, device: int | None = None, chunk_duration: int = CHUNK_DURATION):
        self.device = device
        self.chunk_duration = chunk_duration
        self.capture_rate = _capture_sample_rate(device)

    def start(self) -> None:
        if self.device is not None:
            device_info = sd.query_devices(self.device)
            print(f"Using input device {self.device}: {device_info['name']}")
        else:
            print(f"Using default input device: {sd.query_devices(sd.default.device[0])['name']}")

    def stop(self) -> None:
        pass

    def read(self) -> np.ndarray:
        audio = sd.rec(
            int(self.chunk_duration * self.capture_rate),
            samplerate=self.capture_rate,
            channels=1,
            dtype="float32",
            device=self.device,
        )
        sd.wait()
        return np.squeeze(_resample_to_target(audio, self.capture_rate)).astype(np.float32)


def record_chunk(chunk_number: int, device: int | None = None) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    capture_rate = _capture_sample_rate(device)

    print(f"Recording chunk {chunk_number} for {CHUNK_DURATION}s at {capture_rate} Hz...")
    audio = sd.rec(
        int(CHUNK_DURATION * capture_rate),
        samplerate=capture_rate,
        channels=1,
        dtype="float32",
        device=device,
    )
    sd.wait()

    rms, peak = _volume_stats(audio)
    print(f"Volume: rms={rms:.6f}, peak={peak:.6f}")

    if peak < 0.001:
        print("Warning: input is nearly silent. Check microphone selection, mute state, and Windows mic permissions.")

    audio = _resample_to_target(audio, capture_rate)
    device_label = "default" if device is None else f"device_{device}"
    filename = OUTPUT_DIR / f"chunk_{chunk_number:03d}_{device_label}_{time.time_ns()}.wav"
    write(str(filename), SAMPLE_RATE, _to_int16(audio))
    print(f"Saved: {filename}")
    return filename


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture microphone audio into 3-second WAV chunks.")
    parser.add_argument("--chunks", type=int, default=3, help="Number of chunks to record.")
    parser.add_argument("--device", type=int, default=None, help="Input device index from --list-devices.")
    parser.add_argument("--list-devices", action="store_true", help="List input devices and exit.")
    args = parser.parse_args()

    if args.list_devices:
        list_input_devices()
        return

    if args.device is not None:
        device_info = sd.query_devices(args.device)
        print(f"Using input device {args.device}: {device_info['name']}")
    else:
        print(f"Using default input device: {sd.query_devices(sd.default.device[0])['name']}")

    for chunk_number in range(1, args.chunks + 1):
        record_chunk(chunk_number, args.device)


if __name__ == "__main__":
    main()
