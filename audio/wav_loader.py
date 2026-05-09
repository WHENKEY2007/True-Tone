"""
Fallback WAV loader for True-Tone.

Provides a simple interface to load WAV files and replay them as fixed-size
chunks, matching the same contract as the live microphone capture module.
This is the demo-safety fallback described in the architecture blueprint.

Usage (CLI test mode):
    python -m audio.wav_loader path/to/audio.wav
    python -m audio.wav_loader path/to/audio.wav --chunk-seconds 5
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterator

import numpy as np

from audio.wav_utils import AudioData, chunk_audio, load_wav, TARGET_SAMPLE_RATE


def get_audio_chunks_from_wav_file(
    path: str | Path,
    chunk_seconds: float = 3.0,
    overlap_seconds: float = 0.0,
    target_sample_rate: int = TARGET_SAMPLE_RATE,
) -> list[np.ndarray]:
    """Load a WAV file and return it as a list of fixed-size float32 chunks.

    Each chunk has shape ``(sample_rate * chunk_seconds,)`` and dtype
    ``float32``, exactly matching the output contract of
    ``MicrophoneCapture.read()``.

    Args:
        path: Path to the WAV/audio file.
        chunk_seconds: Duration of each chunk in seconds.
        target_sample_rate: Target sample rate (default 16 kHz).

    Returns:
        A list of numpy arrays, one per chunk.
    """
    audio = load_wav(path, target_sample_rate=target_sample_rate)
    hop_seconds = max(0.01, chunk_seconds - overlap_seconds)
    return chunk_audio(audio.samples, audio.sample_rate, chunk_seconds, hop_seconds)


def iter_audio_chunks_from_wav_file(
    path: str | Path,
    chunk_seconds: float = 3.0,
    overlap_seconds: float = 0.0,
    target_sample_rate: int = TARGET_SAMPLE_RATE,
) -> Iterator[np.ndarray]:
    """Iterate over fixed-size chunks from a WAV file (lazy generator).

    Useful when processing large files without loading all chunks into
    memory at once.

    Args:
        path: Path to the WAV/audio file.
        chunk_seconds: Duration of each chunk in seconds.
        target_sample_rate: Target sample rate (default 16 kHz).

    Yields:
        numpy arrays of shape ``(sample_rate * chunk_seconds,)``.
    """
    for chunk in get_audio_chunks_from_wav_file(
        path, chunk_seconds, overlap_seconds, target_sample_rate
    ):
        yield chunk


class WAVFileReplay:
    """Replay a WAV file as if it were a live microphone source.

    Implements the same ``start() / stop() / read()`` interface as
    ``MicrophoneCapture`` so it can be used as a drop-in replacement
    in the pipeline.
    """

    def __init__(
        self,
        path: str | Path,
        chunk_seconds: float = 3.0,
        overlap_seconds: float = 0.0,
        target_sample_rate: int = TARGET_SAMPLE_RATE,
        loop: bool = True,
    ):
        self.path = Path(path)
        self.chunk_seconds = chunk_seconds
        self.overlap_seconds = min(overlap_seconds, chunk_seconds - 0.01)
        self.target_sample_rate = target_sample_rate
        self.loop = loop
        self._chunks: list[np.ndarray] = []
        self._position: int = 0

    def start(self) -> None:
        """Load and prepare the WAV file for replay."""
        self._chunks = get_audio_chunks_from_wav_file(
            self.path, self.chunk_seconds, self.overlap_seconds, self.target_sample_rate
        )
        self._position = 0
        print(f"WAVFileReplay: loaded {len(self._chunks)} chunks from {self.path}")

    def stop(self) -> None:
        """Reset replay state."""
        self._position = 0

    def read(self) -> np.ndarray:
        """Return the next chunk, looping if configured.

        Returns:
            float32 mono audio chunk matching the live capture contract.

        Raises:
            StopIteration: If ``loop=False`` and all chunks have been read.
        """
        if not self._chunks:
            raise RuntimeError("Call start() before read().")

        if self._position >= len(self._chunks):
            if self.loop:
                self._position = 0
            else:
                raise StopIteration("All chunks have been replayed.")

        chunk = self._chunks[self._position]
        self._position += 1
        return chunk


def main() -> None:
    """CLI test mode: load a WAV file, chunk it, and print diagnostics."""
    parser = argparse.ArgumentParser(
        description="Load a WAV/audio file and split into fixed-size chunks."
    )
    parser.add_argument("audio_file", help="Path to a WAV/audio file.")
    parser.add_argument(
        "--chunk-seconds", type=float, default=3.0,
        help="Duration of each chunk in seconds (default: 3.0).",
    )
    parser.add_argument(
        "--overlap-seconds", type=float, default=0.0,
        help="Overlap between chunks in seconds.",
    )
    parser.add_argument(
        "--replay", action="store_true",
        help="Test the WAVFileReplay interface (simulates live capture).",
    )
    args = parser.parse_args()

    path = Path(args.audio_file)
    if not path.exists():
        print(f"Error: {path} does not exist.")
        return

    if args.replay:
        print(f"Testing WAVFileReplay for {path}...")
        replay = WAVFileReplay(
            path,
            chunk_seconds=args.chunk_seconds,
            overlap_seconds=args.overlap_seconds,
            loop=False,
        )
        replay.start()
        chunk_idx = 0
        try:
            while True:
                chunk = replay.read()
                chunk_idx += 1
                rms = float(np.sqrt(np.mean(np.square(chunk)))) if chunk.size else 0.0
                peak = float(np.max(np.abs(chunk))) if chunk.size else 0.0
                print(
                    f"  Chunk {chunk_idx}: shape={chunk.shape}, "
                    f"dtype={chunk.dtype}, rms={rms:.5f}, peak={peak:.5f}"
                )
        except StopIteration:
            print(f"\nReplay finished: {chunk_idx} chunks emitted.")
        replay.stop()
    else:
        print(f"Loading {path}...")
        chunks = get_audio_chunks_from_wav_file(
            path,
            chunk_seconds=args.chunk_seconds,
            overlap_seconds=args.overlap_seconds,
        )
        print(f"Produced {len(chunks)} chunk(s) of {args.chunk_seconds}s each.\n")
        for i, chunk in enumerate(chunks, 1):
            rms = float(np.sqrt(np.mean(np.square(chunk)))) if chunk.size else 0.0
            peak = float(np.max(np.abs(chunk))) if chunk.size else 0.0
            print(
                f"  Chunk {i}: shape={chunk.shape}, "
                f"dtype={chunk.dtype}, rms={rms:.5f}, peak={peak:.5f}"
            )


if __name__ == "__main__":
    main()
