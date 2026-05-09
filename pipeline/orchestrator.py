"""
Pipeline orchestrator for True-Tone.

Connects audio capture → preprocessing → speech gate → AI detector
using Python threading and queues for asynchronous, non-blocking
data flow. This is the backend that the Streamlit UI reads from.
"""

from __future__ import annotations

import enum
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass, field
from queue import Empty, Full, Queue
from typing import Any

import numpy as np
import ctypes


def _initialize_com():
    """Initialize Windows COM for the current thread."""
    if hasattr(ctypes, "windll"):
        try:
            # 2 = COINIT_APARTMENTTHREADED, 0 = COINIT_MULTITHREADED
            # Most audio drivers prefer apartment threaded.
            ctypes.windll.ole32.CoInitializeEx(None, 2)
        except Exception:
            pass


def _uninitialize_com():
    """Uninitialize Windows COM for the current thread."""
    if hasattr(ctypes, "windll"):
        try:
            ctypes.windll.ole32.CoUninitialize()
        except Exception:
            pass



# ── Data classes ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class ScoreRecord:
    """Single inference result emitted by the pipeline."""

    index: int
    ai_probability: float
    rms: float
    peak: float
    is_speech: bool
    latency_seconds: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class PipelineSnapshot:
    """Thread-safe snapshot of current pipeline state for the UI.

    The dashboard can call ``orchestrator.snapshot()`` to get a
    consistent view of all data it needs without holding any locks.
    """

    state: str
    smoothed_probability: float
    latest_score: ScoreRecord | None
    recent_scores: list[ScoreRecord]
    total_chunks_processed: int
    error_message: str | None


class PipelineState(enum.Enum):
    """Current state of the pipeline."""

    IDLE = "idle"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"


# ── Orchestrator ─────────────────────────────────────────────────────


class PipelineOrchestrator:
    """Thread-safe orchestrator that wires capture → gate → detector.

    The orchestrator manages two background threads:

    * **capture thread** — reads audio from the capture source and
      enqueues chunks.
    * **inference thread** — dequeues chunks, runs the speech gate and
      AI detector, and publishes ``ScoreRecord`` results.

    The Streamlit UI (or any consumer) polls ``snapshot()`` or
    ``get_latest_scores()`` to retrieve results without blocking.

    Args:
        capture: Any object implementing ``start() / stop() / read()``.
        detector: An ``AudioDeepfakeDetector`` instance.
        speech_gate: An ``EnergySpeechGate`` instance.
        history_length: How many recent scores to keep for the UI.
        score_smoothing_window: Number of recent scores to average for
            the smoothed probability (reduces UI flicker).
        max_capture_retries: How many consecutive capture failures to
            tolerate before entering the ERROR state.
    """

    def __init__(
        self,
        capture: Any,
        detector: Any,
        speech_gate: Any,
        history_length: int = 100,
        score_smoothing_window: int = 5,
        max_capture_retries: int = 3,
    ):
        self.capture = capture
        self.detector = detector
        self.speech_gate = speech_gate
        self.history_length = history_length
        self.score_smoothing_window = score_smoothing_window
        self.max_capture_retries = max_capture_retries

        self._state = PipelineState.IDLE
        self._state_lock = threading.Lock()
        self._stop_event = threading.Event()

        self._audio_queue: Queue[np.ndarray | None] = Queue(maxsize=4)
        self._scores: deque[ScoreRecord] = deque(maxlen=history_length)
        self._scores_lock = threading.Lock()
        self._latest_waveform: np.ndarray | None = None
        self._waveform_lock = threading.Lock()

        self._capture_thread: threading.Thread | None = None
        self._inference_thread: threading.Thread | None = None
        self._error: str | None = None
        self._total_chunks = 0

    # ── Public API ───────────────────────────────────────────────────

    @property
    def state(self) -> PipelineState:
        with self._state_lock:
            return self._state

    @property
    def last_error(self) -> str | None:
        return self._error

    def start(self) -> None:
        """Start the capture and inference threads."""
        with self._state_lock:
            if self._state == PipelineState.RUNNING:
                return
            self._state = PipelineState.RUNNING

        self._stop_event.clear()
        self._error = None
        self._total_chunks = 0

        # Clear stale data
        with self._scores_lock:
            self._scores.clear()
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except Empty:
                break

        self.capture.start()

        self._capture_thread = threading.Thread(
            target=self._capture_loop, name="tt-capture", daemon=True
        )
        self._inference_thread = threading.Thread(
            target=self._inference_loop, name="tt-inference", daemon=True
        )
        self._capture_thread.start()
        self._inference_thread.start()

    def stop(self) -> None:
        """Signal both threads to stop and wait for them to finish."""
        with self._state_lock:
            if self._state not in (PipelineState.RUNNING, PipelineState.ERROR):
                return
            self._state = PipelineState.STOPPING
        self._stop_event.set()

        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=5)
        if self._inference_thread and self._inference_thread.is_alive():
            self._inference_thread.join(timeout=5)

        try:
            self.capture.stop()
        except Exception:
            pass

        with self._state_lock:
            self._state = PipelineState.IDLE

    def get_latest_scores(self, n: int | None = None) -> list[ScoreRecord]:
        """Return the most recent *n* scores (thread-safe)."""
        with self._scores_lock:
            if n is None:
                return list(self._scores)
            return list(self._scores)[-n:]

    def get_smoothed_probability(self) -> float:
        """Return the moving-average AI probability over the smoothing window."""
        with self._scores_lock:
            recent = [
                s.ai_probability
                for s in list(self._scores)[-self.score_smoothing_window :]
                if s.is_speech
            ]
        if not recent:
            return 0.0
        return float(np.mean(recent))

    def get_latest_waveform(self) -> np.ndarray | None:
        """Return the most recent raw audio waveform for visualization."""
        with self._waveform_lock:
            return self._latest_waveform

    def snapshot(self) -> PipelineSnapshot:
        """Return a consistent snapshot of the pipeline state for the UI.

        This is the recommended way for the Streamlit dashboard to
        read pipeline data — one call, no lock contention.
        """
        with self._scores_lock:
            scores_list = list(self._scores)

        recent_speech = [
            s.ai_probability
            for s in scores_list[-self.score_smoothing_window :]
            if s.is_speech
        ]
        smoothed = float(np.mean(recent_speech)) if recent_speech else 0.0

        return PipelineSnapshot(
            state=self.state.value,
            smoothed_probability=smoothed,
            latest_score=scores_list[-1] if scores_list else None,
            recent_scores=scores_list[-50:],
            total_chunks_processed=self._total_chunks,
            error_message=self._error,
        )

    # ── Streamlit session-state helper ───────────────────────────────

    def to_streamlit_state(self) -> dict[str, Any]:
        """Return a dict suitable for ``st.session_state`` updates.

        Keys match the names the dashboard expects.
        """
        snap = self.snapshot()
        waveform = self.get_latest_waveform()
        return {
            "pipeline_state": snap.state,
            "smoothed_probability": snap.smoothed_probability,
            "latest_score": snap.latest_score,
            "recent_scores": snap.recent_scores,
            "total_chunks": snap.total_chunks_processed,
            "error_message": snap.error_message,
            "latest_waveform": waveform,
        }

    # ── Background threads ───────────────────────────────────────────

    def _capture_loop(self) -> None:
        _initialize_com()
        consecutive_errors = 0
        chunk_index = 0

        try:
            while not self._stop_event.is_set():
                try:
                    samples = self.capture.read()
                    consecutive_errors = 0  # reset on success
                    chunk_index += 1

                    # Store waveform for UI visualization
                    with self._waveform_lock:
                        self._latest_waveform = samples.copy()

                    try:
                        self._audio_queue.put(samples, timeout=0.3)
                    except Full:
                        # Drop chunk if inference is too slow
                        continue

                except StopIteration:
                    # File replay finished
                    break

                except Exception as exc:
                    consecutive_errors += 1
                    print(
                        f"[capture] Error #{consecutive_errors}: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    if consecutive_errors >= self.max_capture_retries:
                        self._set_error(
                            f"Capture failed {consecutive_errors} times: {exc}"
                        )
                        break
                    # Brief pause before retry
                    time.sleep(0.5)

            # Sentinel to unblock the inference thread
            try:
                self._audio_queue.put(None, timeout=1.0)
            except Full:
                pass
        finally:
            _uninitialize_com()


    def _inference_loop(self) -> None:
        _initialize_com()
        chunk_index = 0
        try:
            while not self._stop_event.is_set():
                try:
                    samples = self._audio_queue.get(timeout=0.3)
                except Empty:
                    continue
                if samples is None:
                    break

                chunk_index += 1
                self._total_chunks = chunk_index
                t0 = time.time()

                gate = self.speech_gate.process(samples)
                if gate.is_speech:
                    ai_prob, _, _ = self.detector.predict_samples(gate.samples)
                else:
                    ai_prob = 0.0

                record = ScoreRecord(
                    index=chunk_index,
                    ai_probability=ai_prob,
                    rms=gate.rms,
                    peak=gate.peak,
                    is_speech=gate.is_speech,
                    latency_seconds=time.time() - t0,
                )
                with self._scores_lock:
                    self._scores.append(record)

        except Exception as exc:
            self._set_error(f"Inference failed: {exc}\n{traceback.format_exc()}")
        finally:
            _uninitialize_com()


    def _set_error(self, message: str) -> None:
        self._error = message
        with self._state_lock:
            self._state = PipelineState.ERROR
        self._stop_event.set()
