from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Full, Queue
import threading
import time

import numpy as np

from audio.mic_capture import CHUNK_DURATION, MicrophoneCapture, SAMPLE_RATE
from audio.system_capture import SystemAudioCapture
from audio.vad import EnergySpeechGate
from audio.wav_utils import chunk_audio, load_wav
from inference.detector import AudioDeepfakeDetector
from pipeline.temporal import TemporalConfidenceAggregator


@dataclass(frozen=True)
class AudioChunk:
    index: int
    samples: np.ndarray
    captured_at: float


@dataclass(frozen=True)
class PipelineScore:
    index: int
    ai_probability: float
    raw_probability: float
    rms: float
    peak: float
    is_speech: bool
    latency_seconds: float
    decision_state: str = "insufficient_speech"
    uncertainty: float = 1.0


@dataclass(frozen=True)
class WorkerError:
    worker: str
    error: Exception


class FileReplayCapture:
    """Replay a WAV/audio file as fixed-size chunks for Day 3 integration checks."""

    def __init__(
        self,
        path: str | Path,
        chunk_duration: float = CHUNK_DURATION,
        overlap_duration: float = 0.0,
        loop: bool = False,
    ):
        audio = load_wav(path, target_sample_rate=SAMPLE_RATE)
        self.path = Path(path)
        hop_seconds = max(0.01, chunk_duration - overlap_duration)
        self.chunks = chunk_audio(audio.samples, SAMPLE_RATE, chunk_duration, hop_seconds)
        self.position = 0
        self.loop = loop

    def start(self) -> None:
        print(f"Replaying audio file: {self.path}")

    def stop(self) -> None:
        pass

    def read(self) -> np.ndarray:
        if self.position >= len(self.chunks):
            if self.loop:
                self.position = 0
            else:
                raise StopIteration("All replay chunks have been emitted.")
        chunk = self.chunks[self.position]
        self.position += 1
        return chunk


class LiveDetectionPipeline:
    """Threaded capture -> speech gate -> detector pipeline."""

    def __init__(
        self,
        capture,
        detector: AudioDeepfakeDetector,
        speech_gate: EnergySpeechGate,
        max_chunks: int | None = None,
    ):
        self.capture = capture
        self.detector = detector
        self.speech_gate = speech_gate
        self.max_chunks = max_chunks
        self.audio_queue: Queue[AudioChunk | None] = Queue(maxsize=3)
        self.score_queue: Queue[PipelineScore | WorkerError | None] = Queue()
        self.stop_event = threading.Event()
        self.temporal_aggregator = TemporalConfidenceAggregator()

    def run_terminal(self) -> None:
        capture_thread = threading.Thread(target=self._capture_loop, name="capture", daemon=True)
        inference_thread = threading.Thread(target=self._inference_loop, name="inference", daemon=True)

        self.capture.start()
        capture_thread.start()
        inference_thread.start()

        try:
            while True:
                score = self.score_queue.get()
                if score is None:
                    break
                if isinstance(score, WorkerError):
                    self.stop_event.set()
                    raise RuntimeError(f"{score.worker} worker failed") from score.error
                label = "speech" if score.is_speech else "silence"
                print(
                    f"chunk={score.index:03d} "
                    f"ai_probability={score.ai_probability:.4f} "
                    f"raw={score.raw_probability:.4f} "
                    f"state={score.decision_state} "
                    f"uncertainty={score.uncertainty:.3f} "
                    f"rms={score.rms:.5f} peak={score.peak:.5f} "
                    f"{label} latency={score.latency_seconds:.2f}s",
                    flush=True,
                )
        except KeyboardInterrupt:
            print("Stopping live pipeline...")
            self.stop_event.set()
        finally:
            self.capture.stop()
            capture_thread.join(timeout=2)
            inference_thread.join(timeout=2)

    def _capture_loop(self) -> None:
        try:
            chunk_index = 1
            while not self.stop_event.is_set():
                if self.max_chunks is not None and chunk_index > self.max_chunks:
                    break
                samples = self.capture.read()
                self._put_audio(AudioChunk(chunk_index, samples, time.time()))
                chunk_index += 1
        except StopIteration:
            pass
        except Exception as exc:
            self._put_worker_error("capture", exc)
        finally:
            self._put_audio(None)

    def _inference_loop(self) -> None:
        try:
            while not self.stop_event.is_set():
                try:
                    chunk = self.audio_queue.get(timeout=0.2)
                except Empty:
                    continue
                if chunk is None:
                    break

                gate = self.speech_gate.process(chunk.samples)
                if gate.is_speech:
                    raw_probability, _, _ = self.detector.predict_samples(gate.samples)
                    feature_summary = self._latest_feature_summary()
                    behavior_score = float(feature_summary.get("synthetic_behavior_score", 0.0))
                    cadence_consistency = float(feature_summary.get("cadence_consistency", 0.0))
                else:
                    raw_probability = 0.0
                    behavior_score = 0.0
                    cadence_consistency = 0.0

                decision = self.temporal_aggregator.update(
                    raw_probability,
                    is_speech=gate.is_speech,
                    behavior_score=behavior_score,
                    cadence_consistency=cadence_consistency,
                    timestamp=chunk.captured_at,
                )

                self.score_queue.put(
                    PipelineScore(
                        index=chunk.index,
                        ai_probability=decision.stabilized_probability,
                        raw_probability=raw_probability,
                        rms=gate.rms,
                        peak=gate.peak,
                        is_speech=gate.is_speech,
                        latency_seconds=time.time() - chunk.captured_at,
                        decision_state=decision.state,
                        uncertainty=decision.uncertainty,
                    )
                )
        except Exception as exc:
            self._put_worker_error("inference", exc)
        finally:
            self.score_queue.put(None)

    def _put_audio(self, item: AudioChunk | None) -> None:
        while not self.stop_event.is_set():
            try:
                self.audio_queue.put(item, timeout=0.2)
                return
            except Full:
                continue

    def _put_worker_error(self, worker: str, error: Exception) -> None:
        self.stop_event.set()
        self.score_queue.put(WorkerError(worker, error))

    def _latest_feature_summary(self) -> dict[str, float]:
        summaries = getattr(self.detector, "last_feature_summaries", None)
        if not summaries:
            return {}
        for summary in reversed(summaries):
            if summary:
                return summary
        return {}


def _build_capture(args):
    if args.source_file:
        return FileReplayCapture(
            args.source_file,
            chunk_duration=args.chunk_seconds,
            overlap_duration=getattr(args, "overlap", 0.0),
            loop=getattr(args, "loop_file", False),
        )
    if args.source == "system":
        return SystemAudioCapture(
            device=args.device,
            chunk_duration=args.chunk_seconds,
            overlap_duration=getattr(args, "overlap", 0.0),
        )
    return MicrophoneCapture(
        device=args.device,
        chunk_duration=args.chunk_seconds,
        overlap_duration=getattr(args, "overlap", 0.0),
        gain_db=getattr(args, "gain", 0.0),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run live AI-voice detection in the terminal.")
    parser.add_argument("--source", choices=("mic", "system"), default="mic", help="Live audio source.")
    parser.add_argument("--source-file", default=None, help="Replay a WAV/audio file instead of live capture.")
    parser.add_argument("--loop-file", action="store_true", help="Loop --source-file replay instead of stopping at EOF.")
    parser.add_argument("--device", type=int, default=None, help="Input device index for the selected live source.")
    parser.add_argument("--chunks", type=int, default=None, help="Stop after this many chunks.")
    parser.add_argument("--chunk-seconds", type=int, default=CHUNK_DURATION, help="Seconds per live audio chunk.")
    parser.add_argument("--overlap", type=float, default=2.0, help="Overlap in seconds between chunks (default: 2.0 for 1s stride).")
    parser.add_argument("--gain", type=float, default=0.0, help="Gain boost in dB for quiet microphones.")
    parser.add_argument("--model-id", default=None, help="Hugging Face model id.")
    parser.add_argument("--local-files-only", action="store_true", help="Use only cached model files.")
    parser.add_argument("--min-rms", type=float, default=0.002, help="Minimum RMS for speech/inference.")
    parser.add_argument("--min-peak", type=float, default=0.01, help="Minimum peak amplitude for speech/inference.")
    args = parser.parse_args()

    capture = _build_capture(args)
    detector = AudioDeepfakeDetector(
        model_id=args.model_id,
        chunk_seconds=args.chunk_seconds,
        local_files_only=args.local_files_only,
        min_rms=args.min_rms,
    )
    speech_gate = EnergySpeechGate(min_rms=args.min_rms, min_peak=args.min_peak)
    pipeline = LiveDetectionPipeline(capture, detector, speech_gate, max_chunks=args.chunks)
    pipeline.run_terminal()


if __name__ == "__main__":
    main()
