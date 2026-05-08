from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import tempfile
import time

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".cache" / "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from audio.mic_capture import CHUNK_DURATION, MicrophoneCapture
from audio.system_capture import SystemAudioCapture
from audio.vad import EnergySpeechGate
from inference.detector import DEFAULT_MODEL_ID, AudioDeepfakeDetector
from live_pipeline import FileReplayCapture, PipelineScore


ALERT_THRESHOLD = 0.75


@dataclass(frozen=True)
class DashboardSettings:
    source: str
    device: int | None
    chunks: int
    chunk_seconds: int
    model_id: str
    local_files_only: bool
    min_rms: float
    min_peak: float
    alert_threshold: float
    real_time_replay: bool


class Dashboard:
    """Streamlit dashboard for the True-Tone live detection workflow."""

    def __init__(self):
        self.is_running = False

    def run(self):
        self.is_running = True
        self.render()

    def render(self):
        st.set_page_config(page_title="True-Tone", page_icon="TT", layout="wide")
        self._inject_styles()

        st.title("True-Tone")
        st.caption("Live audio deepfake detection")

        uploaded_file, settings = self._render_controls()
        score_box, status_box, stats_box, chart_box, log_box = self._render_live_shell()

        if st.button("Start detection", type="primary", use_container_width=True):
            self._run_detection(
                uploaded_file=uploaded_file,
                settings=settings,
                score_box=score_box,
                status_box=status_box,
                stats_box=stats_box,
                chart_box=chart_box,
                log_box=log_box,
            )

    def stop(self):
        self.is_running = False

    def _render_controls(self):
        with st.sidebar:
            st.header("Input")
            source = st.selectbox("Source", ("File replay", "Microphone", "System audio"))

            uploaded_file = None
            real_time_replay = True
            device = None
            if source == "File replay":
                uploaded_file = st.file_uploader("Audio file", type=("wav", "mp3", "flac", "ogg", "m4a"))
                real_time_replay = st.toggle("Pace replay", value=True)
            else:
                device_index = st.number_input("Device index", min_value=-1, value=-1, step=1)
                device = None if device_index < 0 else int(device_index)

            st.header("Run")
            chunks = st.number_input("Chunks", min_value=1, max_value=120, value=12, step=1)
            chunk_seconds = st.number_input("Seconds per chunk", min_value=1, max_value=10, value=CHUNK_DURATION, step=1)

            st.header("Detector")
            model_id = st.text_input("Model", value=DEFAULT_MODEL_ID)
            local_files_only = st.toggle("Use cached model files only", value=True)

            st.header("Gate")
            min_rms = st.number_input("Minimum RMS", min_value=0.0, max_value=0.1, value=0.002, step=0.001, format="%.4f")
            min_peak = st.number_input("Minimum peak", min_value=0.0, max_value=0.5, value=0.01, step=0.005, format="%.4f")
            alert_threshold = st.slider("Alert threshold", min_value=0.0, max_value=1.0, value=ALERT_THRESHOLD, step=0.05)

        settings = DashboardSettings(
            source=source,
            device=device,
            chunks=int(chunks),
            chunk_seconds=int(chunk_seconds),
            model_id=model_id.strip() or DEFAULT_MODEL_ID,
            local_files_only=local_files_only,
            min_rms=float(min_rms),
            min_peak=float(min_peak),
            alert_threshold=float(alert_threshold),
            real_time_replay=real_time_replay,
        )
        return uploaded_file, settings

    def _render_live_shell(self):
        left, right = st.columns((1, 2))
        with left:
            score_box = st.empty()
            status_box = st.empty()
            stats_box = st.empty()
        with right:
            chart_box = st.empty()
            log_box = st.empty()

        score_box.metric("AI probability", "0.000")
        status_box.info("Ready")
        stats_box.markdown("RMS `0.00000` | Peak `0.00000` | Latency `0.00s`")
        chart_box.pyplot(self._build_history_chart([], ALERT_THRESHOLD))
        log_box.caption("No chunks processed yet.")
        return score_box, status_box, stats_box, chart_box, log_box

    def _run_detection(self, uploaded_file, settings, score_box, status_box, stats_box, chart_box, log_box) -> None:
        temp_path = None
        history: list[PipelineScore] = []

        try:
            capture, temp_path = self._build_capture(uploaded_file, settings)
            detector = AudioDeepfakeDetector(
                model_id=settings.model_id,
                chunk_seconds=settings.chunk_seconds,
                local_files_only=settings.local_files_only,
                min_rms=settings.min_rms,
            )
            speech_gate = EnergySpeechGate(min_rms=settings.min_rms, min_peak=settings.min_peak)

            capture.start()
            try:
                for index in range(1, settings.chunks + 1):
                    captured_at = time.time()
                    samples = capture.read()
                    gate = speech_gate.process(samples)
                    if gate.is_speech:
                        ai_probability, _, _ = detector.predict_samples(gate.samples)
                    else:
                        ai_probability = 0.0

                    score = PipelineScore(
                        index=index,
                        ai_probability=ai_probability,
                        rms=gate.rms,
                        peak=gate.peak,
                        is_speech=gate.is_speech,
                        latency_seconds=time.time() - captured_at,
                    )
                    history.append(score)
                    self._update_live_view(score, history, settings.alert_threshold, score_box, status_box, stats_box, chart_box, log_box)

                    if settings.source == "File replay" and settings.real_time_replay:
                        time.sleep(min(settings.chunk_seconds, 1))
            finally:
                capture.stop()
        except Exception as exc:
            status_box.error(f"Detection stopped: {type(exc).__name__}: {exc}")
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)

    def _build_capture(self, uploaded_file, settings: DashboardSettings):
        if settings.source == "File replay":
            if uploaded_file is None:
                raise ValueError("Upload an audio file before starting file replay.")

            suffix = Path(uploaded_file.name).suffix or ".wav"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                temp_file.write(uploaded_file.getbuffer())
                temp_path = Path(temp_file.name)
            return FileReplayCapture(temp_path, chunk_duration=settings.chunk_seconds), temp_path

        if settings.source == "System audio":
            return SystemAudioCapture(device=settings.device, chunk_duration=settings.chunk_seconds), None

        return MicrophoneCapture(device=settings.device, chunk_duration=settings.chunk_seconds), None

    def _update_live_view(self, score, history, threshold, score_box, status_box, stats_box, chart_box, log_box) -> None:
        score_box.metric("AI probability", f"{score.ai_probability:.3f}")

        if not score.is_speech:
            status_box.info("Silence or low signal")
        elif score.ai_probability >= threshold:
            status_box.error("High likelihood of synthetic speech detected.")
        elif score.ai_probability >= threshold * 0.7:
            status_box.warning("Synthetic speech signal rising.")
        else:
            status_box.success("Human speech range")

        stats_box.markdown(
            f"RMS `{score.rms:.5f}` | Peak `{score.peak:.5f}` | "
            f"Latency `{score.latency_seconds:.2f}s` | Chunk `{score.index:03d}`"
        )
        chart_box.pyplot(self._build_history_chart(history, threshold))

        rows = [
            f"`{item.index:03d}` score `{item.ai_probability:.3f}` "
            f"{'speech' if item.is_speech else 'silence'} latency `{item.latency_seconds:.2f}s`"
            for item in history[-8:]
        ]
        log_box.markdown("  \n".join(rows))

    def _build_history_chart(self, history, threshold):
        fig, ax = plt.subplots(figsize=(8, 3))
        indexes = [item.index for item in history]
        scores = [item.ai_probability for item in history]

        ax.plot(indexes, scores, color="#1f7a6d", linewidth=2.5, marker="o")
        ax.axhline(threshold, color="#c43b3b", linestyle="--", linewidth=1.5)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Chunk")
        ax.set_ylabel("AI probability")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        return fig

    def _inject_styles(self) -> None:
        st.markdown(
            """
            <style>
            .stMetric {
                border: 1px solid rgba(49, 51, 63, 0.18);
                border-radius: 8px;
                padding: 16px;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
