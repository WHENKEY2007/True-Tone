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


def _apply_gain(audio: np.ndarray, gain_db: float) -> np.ndarray:
    """Apply gain in decibels to an audio signal."""
    if gain_db == 0.0:
        return audio
    factor = 10.0 ** (gain_db / 20.0)
    return np.clip(audio * factor, -1.0, 1.0).astype(np.float32)


class MicrophoneCapture:
    """Capture microphone input as 16 kHz mono float32 chunks.

    Supports optional overlapping windows for smoother real-time
    detection and a gain parameter for boosting quiet microphones.

    Args:
        device: Sounddevice input device index (None = default).
        chunk_duration: Duration of each emitted chunk in seconds.
        overlap_duration: Overlap with the previous chunk in seconds.
            Set to 0 for non-overlapping chunks.
        gain_db: Gain boost in decibels (0 = no change).
    """

    def __init__(
        self,
        device: int | None = None,
        chunk_duration: int = CHUNK_DURATION,
        overlap_duration: float = 0.0,
        gain_db: float = 0.0,
    ):
        self.device = device
        self.chunk_duration = chunk_duration
        self.overlap_duration = min(overlap_duration, chunk_duration - 0.1)
        self.gain_db = gain_db
        self.capture_rate = _capture_sample_rate(device)

        # Internal ring buffer for overlapping windows
        self._chunk_samples = int(SAMPLE_RATE * self.chunk_duration)
        self._overlap_samples = int(SAMPLE_RATE * self.overlap_duration)
        self._step_samples = self._chunk_samples - self._overlap_samples
        self._ring_buffer: np.ndarray | None = None

    def start(self) -> None:
        if self.device is not None:
            device_info = sd.query_devices(self.device)
            print(f"Using input device {self.device}: {device_info['name']}")
        else:
            print(f"Using default input device: {sd.query_devices(sd.default.device[0])['name']}")
        if self.overlap_duration > 0:
            print(f"Overlap: {self.overlap_duration:.1f}s ({self._overlap_samples} samples)")
        if self.gain_db != 0.0:
            print(f"Gain: {self.gain_db:+.1f} dB")
        self._ring_buffer = None

    def stop(self) -> None:
        self._ring_buffer = None

    def read(self) -> np.ndarray:
        """Read one chunk from the microphone.

        When overlap_duration > 0, the chunk re-uses the tail of the
        previous capture to create a sliding window effect.
        """
        if self.overlap_duration <= 0 or self._ring_buffer is None:
            # Full capture (no overlap, or first chunk)
            frames = int(self.chunk_duration * self.capture_rate)
            audio = sd.rec(frames, samplerate=self.capture_rate,
                           channels=1, dtype="float32", device=self.device)
            sd.wait()
            resampled = np.squeeze(_resample_to_target(audio, self.capture_rate)).astype(np.float32)
            resampled = _apply_gain(resampled, self.gain_db)
            # Pad/trim to exact chunk size
            resampled = _pad_or_trim(resampled, self._chunk_samples)
            self._ring_buffer = resampled
            return resampled

        # Overlapping capture: only record the new (step) portion
        step_frames_native = int(
            (self._step_samples / SAMPLE_RATE) * self.capture_rate
        )
        audio = sd.rec(step_frames_native, samplerate=self.capture_rate,
                       channels=1, dtype="float32", device=self.device)
        sd.wait()
        new_part = np.squeeze(_resample_to_target(audio, self.capture_rate)).astype(np.float32)
        new_part = _apply_gain(new_part, self.gain_db)
        new_part = _pad_or_trim(new_part, self._step_samples)

        # Concatenate tail of previous buffer + new audio
        overlap_part = self._ring_buffer[-self._overlap_samples:]
        chunk = np.concatenate([overlap_part, new_part])
        chunk = _pad_or_trim(chunk, self._chunk_samples)
        self._ring_buffer = chunk
        return chunk


def _pad_or_trim(audio: np.ndarray, target_length: int) -> np.ndarray:
    """Ensure audio is exactly target_length samples."""
    if audio.size == target_length:
        return audio
    if audio.size > target_length:
        return audio[:target_length]
    return np.pad(audio, (0, target_length - audio.size))


def record_chunk(chunk_number: int, device: int | None = None, gain_db: float = 0.0) -> Path:
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
    audio = _apply_gain(np.squeeze(audio), gain_db)

    device_label = "default" if device is None else f"device_{device}"
    filename = OUTPUT_DIR / f"chunk_{chunk_number:03d}_{device_label}_{time.time_ns()}.wav"
    write(str(filename), SAMPLE_RATE, _to_int16(audio))
    print(f"Saved: {filename}")
    return filename


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture microphone audio into 3-second WAV chunks.")
    parser.add_argument("--chunks", type=int, default=3, help="Number of chunks to record.")
    parser.add_argument("--device", type=int, default=None, help="Input device index from --list-devices.")
    parser.add_argument("--overlap", type=float, default=0.0, help="Overlap in seconds between chunks (e.g. 1.5).")
    parser.add_argument("--gain", type=float, default=0.0, help="Gain boost in dB for quiet microphones.")
    parser.add_argument("--list-devices", action="store_true", help="List input devices and exit.")
    args = parser.parse_args()

    if args.list_devices:
        list_input_devices()
        return

    capture = MicrophoneCapture(
        device=args.device,
        overlap_duration=args.overlap,
        gain_db=args.gain,
    )
    capture.start()
    try:
        for chunk_number in range(1, args.chunks + 1):
            chunk = capture.read()
            rms, peak = _volume_stats(chunk)
            print(
                f"Chunk {chunk_number}: shape={chunk.shape}, "
                f"rms={rms:.6f}, peak={peak:.6f}"
            )
            # Also save to disk
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            device_label = "default" if args.device is None else f"device_{args.device}"
            filename = OUTPUT_DIR / f"chunk_{chunk_number:03d}_{device_label}_{time.time_ns()}.wav"
            write(str(filename), SAMPLE_RATE, _to_int16(chunk))
            print(f"Saved: {filename}")
    finally:
        capture.stop()


if __name__ == "__main__":
    main()
