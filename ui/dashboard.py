"""
Streamlit dashboard for True-Tone.

Displays a live AI-probability meter, waveform visualization,
historical score graph, and warning banner when synthetic speech
likelihood exceeds the configured threshold.

Run with:
    streamlit run ui/dashboard.py
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

# Ensure project root is on sys.path when running via streamlit
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

try:
    import streamlit as st
    import matplotlib.pyplot as plt
    import matplotlib

    matplotlib.use("Agg")
    _HAS_STREAMLIT = True
except ImportError:
    _HAS_STREAMLIT = False

# ── Constants ────────────────────────────────────────────────────────

DEFAULT_THRESHOLD = 0.20
POLL_INTERVAL = 0.8  # seconds between UI refreshes

# ── Role-based access credentials ────────────────────────────────────
# Simple session-based auth for demo/testing environments.
# Roles: admin (full access), tester (detection + logs), demo (detection only)

DEMO_CREDENTIALS = {
    "admin": {"password": "truetone2025", "role": "admin"},
    "tester": {"password": "testpass", "role": "tester"},
    "demo": {"password": "demo", "role": "demo"},
}

ROLE_PERMISSIONS = {
    "admin": {"can_configure_model": True, "can_view_logs": True, "can_change_threshold": True},
    "tester": {"can_configure_model": False, "can_view_logs": True, "can_change_threshold": True},
    "demo": {"can_configure_model": False, "can_view_logs": False, "can_change_threshold": False},
}


# ── Helper functions ─────────────────────────────────────────────────


def _render_waveform(samples: np.ndarray, sample_rate: int = 16000):
    """Render a waveform plot and return the matplotlib figure."""
    fig, ax = plt.subplots(figsize=(8, 2))
    t = np.linspace(0, len(samples) / sample_rate, len(samples))
    ax.plot(t, samples, color="#4a90d9", linewidth=0.5, alpha=0.9)
    ax.fill_between(t, samples, alpha=0.15, color="#4a90d9")
    ax.set_xlim(0, t[-1] if len(t) > 0 else 1)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("Time (s)", fontsize=8)
    ax.set_ylabel("Amplitude", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_facecolor("#0e1117")
    fig.patch.set_facecolor("#0e1117")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#333")
    ax.spines["left"].set_color("#333")
    ax.tick_params(colors="#888")
    ax.xaxis.label.set_color("#888")
    ax.yaxis.label.set_color("#888")
    plt.tight_layout()
    return fig


def _render_score_history(scores, threshold: float):
    """Render a score history line chart."""
    if not scores:
        return None
    probs = [s.ai_probability for s in scores]
    indices = list(range(1, len(probs) + 1))
    speech_mask = [s.is_speech for s in scores]

    fig, ax = plt.subplots(figsize=(10, 3))

    # Plot score line
    ax.plot(indices, probs, color="#d94370", linewidth=1.5, label="AI Probability", zorder=3)
    ax.fill_between(indices, probs, alpha=0.1, color="#d94370")

    # Mark speech vs silence
    for i, (idx, prob, is_sp) in enumerate(zip(indices, probs, speech_mask)):
        color = "#d94370" if is_sp else "#555"
        ax.scatter(idx, prob, color=color, s=15, zorder=4)

    # Threshold line
    ax.axhline(y=threshold, color="#ff6b6b", linestyle="--", linewidth=1, alpha=0.7, label=f"Threshold ({threshold:.0%})")

    ax.set_xlim(indices[0], indices[-1])
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Chunk #", fontsize=9)
    ax.set_ylabel("AI Probability", fontsize=9)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.5)
    ax.set_facecolor("#0e1117")
    fig.patch.set_facecolor("#0e1117")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#333")
    ax.spines["left"].set_color("#333")
    ax.tick_params(colors="#888", labelsize=8)
    ax.xaxis.label.set_color("#888")
    ax.yaxis.label.set_color("#888")
    plt.tight_layout()
    return fig


def _probability_color(prob: float, threshold: float) -> str:
    """Return a CSS color based on the probability level."""
    if prob >= threshold:
        return "#ff4444"
    if prob >= threshold * 0.6:
        return "#ffaa00"
    return "#44cc44"


# ── Dashboard class ──────────────────────────────────────────────────


class Dashboard:
    """Main dashboard interface for True-Tone application.

    When Streamlit is available, ``run()`` renders the full interactive
    dashboard. Otherwise it prints a message directing the user to
    launch via ``streamlit run``.
    """

    def __init__(self):
        """Initialize the dashboard."""
        self.is_running = False

    def run(self):
        """Start the dashboard application."""
        self.is_running = True
        self.render()

    def render(self):
        """Render the dashboard interface."""
        if not _HAS_STREAMLIT:
            print(
                "Streamlit is not installed or this file was not launched via "
                "'streamlit run'. Install with: pip install streamlit"
            )
            return
        _render_streamlit_app()

    def stop(self):
        """Stop the dashboard application."""
        self.is_running = False


# ── Streamlit app ────────────────────────────────────────────────────


def _render_login_page():
    """Render the login page for session-based authentication."""
    st.set_page_config(
        page_title="True-Tone · Login",
        page_icon="🔐",
        layout="centered",
    )
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("## 🔐 True-Tone Login")
    st.markdown("Enter your credentials to access the AI voice detection dashboard.")
    st.markdown("---")

    username = st.text_input("Username", key="login_user")
    password = st.text_input("Password", type="password", key="login_pass")

    if st.button("🔓 Login", use_container_width=True):
        if username in DEMO_CREDENTIALS and DEMO_CREDENTIALS[username]["password"] == password:
            st.session_state.authenticated = True
            st.session_state.username = username
            st.session_state.role = DEMO_CREDENTIALS[username]["role"]
            st.session_state.permissions = ROLE_PERMISSIONS[st.session_state.role]
            st.rerun()
        else:
            st.error("Invalid credentials. Try: demo / demo")

    st.markdown("---")
    st.caption("Demo accounts: `admin` / `truetone2025` · `tester` / `testpass` · `demo` / `demo`")


def _render_streamlit_app():
    """Full Streamlit application layout and logic."""

    # ── Authentication gate ──────────────────────────────────────
    if not st.session_state.get("authenticated", False):
        _render_login_page()
        return

    st.set_page_config(
        page_title="True-Tone · AI Voice Detector",
        page_icon="🎙️",
        layout="wide",
    )

    # ── Custom CSS ───────────────────────────────────────────────
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }
        .main-title {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 0;
        }
        .subtitle {
            color: #888;
            font-size: 1rem;
            margin-top: -10px;
        }
        .score-box {
            border-radius: 16px;
            padding: 24px;
            text-align: center;
            border: 1px solid #333;
            background: rgba(30, 30, 50, 0.6);
        }
        .score-value {
            font-size: 3.5rem;
            font-weight: 700;
            line-height: 1.1;
        }
        .score-label {
            font-size: 0.85rem;
            color: #888;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-top: 8px;
        }
        .status-pill {
            display: inline-block;
            padding: 4px 16px;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 600;
            letter-spacing: 0.5px;
        }
        .status-idle { background: #333; color: #888; }
        .status-running { background: #1a472a; color: #44cc44; }
        .status-error { background: #4a1a1a; color: #ff4444; }
        .metric-row {
            display: flex;
            gap: 12px;
            margin-top: 16px;
        }
        .mini-metric {
            flex: 1;
            background: rgba(30, 30, 50, 0.4);
            border: 1px solid #2a2a3a;
            border-radius: 10px;
            padding: 12px;
            text-align: center;
        }
        .mini-metric-value {
            font-size: 1.3rem;
            font-weight: 600;
        }
        .mini-metric-label {
            font-size: 0.7rem;
            color: #666;
            text-transform: uppercase;
        }
        div[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0e1117 0%, #1a1a2e 100%);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ── Header ───────────────────────────────────────────────────
    st.markdown('<div class="main-title">🎙️ True-Tone</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Real-Time AI Voice Detection — '
        "Estimates the likelihood of synthetic speech in live audio</div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # ── Initialize session state ─────────────────────────────────
    if "orchestrator" not in st.session_state:
        st.session_state.orchestrator = None
    if "pipeline_active" not in st.session_state:
        st.session_state.pipeline_active = False

    # ── Sidebar controls ─────────────────────────────────────────
    permissions = st.session_state.get("permissions", ROLE_PERMISSIONS["admin"])

    with st.sidebar:
        # Session info and logout
        role_label = st.session_state.get("role", "admin").upper()
        st.caption(f"👤 **{st.session_state.get('username', 'admin')}** · {role_label}")
        if st.button("🚪 Logout", use_container_width=True):
            for key in ["authenticated", "username", "role", "permissions"]:
                st.session_state.pop(key, None)
            _stop_pipeline()
            st.session_state.pipeline_active = False
            st.rerun()

        st.markdown("---")
        st.header("⚙️ Controls")

        source = st.radio(
            "Audio source",
            ["Microphone", "System Audio", "WAV File"],
            index=0,
        )

        if permissions.get("can_change_threshold", True):
            threshold = st.slider(
                "Alert threshold",
                min_value=0.0,
                max_value=1.0,
                value=DEFAULT_THRESHOLD,
                step=0.05,
                help="AI probability above this triggers a red warning.",
            )
        else:
            threshold = DEFAULT_THRESHOLD

        smoothing = st.slider(
            "Score smoothing (chunks)",
            min_value=1,
            max_value=10,
            value=5,
            help="Moving average window to reduce UI flicker.",
        )

        st.markdown("---")

        device_idx = st.number_input(
            "Device index (optional)",
            min_value=-1,
            max_value=20,
            value=-1,
            help="-1 = default device",
        )
        device = None if device_idx < 0 else int(device_idx)

        overlap = st.slider(
            "Overlap (seconds)",
            min_value=0.0,
            max_value=2.5,
            value=2.0,
            step=0.5,
            help="Overlap between audio chunks for smoother detection.",
        )

        # Model configuration — admin only
        model_id = ""
        model_weights = ""
        if permissions.get("can_configure_model", False):
            model_id = st.text_input(
                "Model IDs (optional)",
                value="",
                help="Hugging Face model ID, or comma-separated IDs for an ensemble. Leave blank for default.",
            )

            model_weights = st.text_input(
                "Model weights (optional)",
                value="",
                help="Comma-separated weights matching Model IDs, for example 0.7,0.3.",
            )

        uploaded_file = None
        if source == "WAV File":
            uploaded_file = st.file_uploader(
                "Upload audio file", type=["wav", "mp3", "flac", "ogg"]
            )

        st.markdown("---")

        col_start, col_stop = st.columns(2)
        with col_start:
            start_pressed = st.button("▶️ Start", width="stretch")
        with col_stop:
            stop_pressed = st.button("⏹️ Stop", width="stretch")

    # ── Handle start/stop ────────────────────────────────────────
    if start_pressed and not st.session_state.pipeline_active:
        try:
            with st.spinner("Loading AI model and starting pipeline... (this may take 10-30 seconds on first run)"):
                _start_pipeline(
                    source=source,
                    device=device,
                    overlap=overlap,
                    model_id=model_id if model_id.strip() else None,
                    model_weights=model_weights if model_weights.strip() else None,
                    smoothing=smoothing,
                    uploaded_file=uploaded_file,
                )
            st.session_state.pipeline_active = True
            st.rerun()
        except Exception as exc:
            import traceback
            st.error(f"Failed to start pipeline: {exc}")
            st.code(traceback.format_exc(), language="text")

    if stop_pressed and st.session_state.pipeline_active:
        _stop_pipeline()
        st.session_state.pipeline_active = False
        st.rerun()

    # ── Main content ─────────────────────────────────────────────
    orch = st.session_state.get("orchestrator")

    if orch is None or not st.session_state.pipeline_active:
        _render_idle_state()
        return

    # Get pipeline data
    snap_data = orch.to_streamlit_state()

    # ── Status pill ──────────────────────────────────────────────
    state_str = snap_data["pipeline_state"]
    if state_str == "running":
        st.markdown(
            '<span class="status-pill status-running">● LIVE</span>',
            unsafe_allow_html=True,
        )
    elif state_str == "error":
        st.markdown(
            '<span class="status-pill status-error">● ERROR</span>',
            unsafe_allow_html=True,
        )
        if snap_data["error_message"]:
            st.error(snap_data["error_message"])
    else:
        st.markdown(
            '<span class="status-pill status-idle">● IDLE</span>',
            unsafe_allow_html=True,
        )

    # ── Score + Waveform columns ─────────────────────────────────
    col_score, col_wave = st.columns([1, 2])

    with col_score:
        prob = snap_data["smoothed_probability"]
        color = _probability_color(prob, threshold)
        pct = f"{prob:.0%}"

        st.markdown(
            f"""
            <div class="score-box">
                <div class="score-value" style="color: {color}">{pct}</div>
                <div class="score-label">AI Probability (smoothed)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Warning banner
        if prob >= threshold:
            st.error("⚠️ **High Likelihood of Synthetic Speech Detected.**")
        elif prob >= threshold * 0.6:
            st.warning("⚡ Elevated AI-speech probability — monitoring…")
        else:
            st.success("✅ Speech appears human.")

        # Mini metrics
        latest = snap_data["latest_score"]
        if latest:
            st.markdown(
                f"""
                <div class="metric-row">
                    <div class="mini-metric">
                        <div class="mini-metric-value">{latest.rms:.4f}</div>
                        <div class="mini-metric-label">RMS</div>
                    </div>
                    <div class="mini-metric">
                        <div class="mini-metric-value">{latest.peak:.4f}</div>
                        <div class="mini-metric-label">Peak</div>
                    </div>
                    <div class="mini-metric">
                        <div class="mini-metric-value">{latest.latency_seconds:.2f}s</div>
                        <div class="mini-metric-label">Latency</div>
                    </div>
                </div>
                <div class="metric-row">
                    <div class="mini-metric">
                        <div class="mini-metric-value">{snap_data['total_chunks']}</div>
                        <div class="mini-metric-label">Chunks</div>
                    </div>
                    <div class="mini-metric">
                        <div class="mini-metric-value">{'🗣️' if latest.is_speech else '🔇'}</div>
                        <div class="mini-metric-label">{'Speech' if latest.is_speech else 'Silence'}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with col_wave:
        st.markdown("##### 🔊 Live Waveform")
        waveform = snap_data.get("latest_waveform")
        if waveform is not None and len(waveform) > 0:
            fig = _render_waveform(waveform)
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("Waiting for audio…")

    # ── Score history chart ──────────────────────────────────────
    st.markdown("---")
    st.markdown("##### 📈 Score History")

    scores = snap_data["recent_scores"]
    if scores:
        fig = _render_score_history(scores, threshold)
        if fig:
            st.pyplot(fig)
            plt.close(fig)
    else:
        st.info("No scores recorded yet.")

    # ── Detection Event Log (tester/admin) ───────────────────────
    if permissions.get("can_view_logs", False) and scores:
        st.markdown("---")
        st.markdown("##### 📋 Detection Event Log")

        import datetime

        log_data = []
        for s in reversed(scores[-20:]):
            log_data.append({
                "Chunk": s.index,
                "AI Prob": f"{s.ai_probability:.4f}",
                "Raw Prob": f"{s.raw_probability:.4f}",
                "State": s.decision_state,
                "Speech": "🗣️" if s.is_speech else "🔇",
                "RMS": f"{s.rms:.4f}",
                "Latency": f"{s.latency_seconds:.2f}s",
                "Alert": "⚠️" if s.ai_probability >= threshold else "✅",
            })
        if log_data:
            st.dataframe(log_data, use_container_width=True, height=300)

        # Analytics summary
        st.markdown("##### 📊 Session Analytics")
        speech_scores = [s for s in scores if s.is_speech]
        alert_count = sum(1 for s in scores if s.ai_probability >= threshold and s.is_speech)

        col_a1, col_a2, col_a3, col_a4, col_a5 = st.columns(5)
        col_a1.metric("Total Chunks", len(scores))
        col_a2.metric("Speech Chunks", len(speech_scores))
        col_a3.metric("Speech Ratio", f"{len(speech_scores)/max(len(scores),1):.0%}")
        if speech_scores:
            avg_prob = sum(s.ai_probability for s in speech_scores) / len(speech_scores)
            peak_prob = max(s.ai_probability for s in speech_scores)
            col_a4.metric("Avg AI Prob", f"{avg_prob:.4f}")
            col_a5.metric("Alerts Fired", alert_count)
        else:
            col_a4.metric("Avg AI Prob", "—")
            col_a5.metric("Alerts Fired", 0)

    # ── Auto-refresh ─────────────────────────────────────────────
    if st.session_state.pipeline_active:
        time.sleep(POLL_INTERVAL)
        st.rerun()


def _render_idle_state():
    """Render the idle/welcome state when no pipeline is running."""
    st.markdown(
        """
        <div style="text-align: center; padding: 60px 20px; color: #666;">
            <div style="font-size: 4rem; margin-bottom: 16px;">🎧</div>
            <div style="font-size: 1.2rem; margin-bottom: 8px;">
                Ready to detect AI-generated speech
            </div>
            <div style="font-size: 0.9rem; color: #555;">
                Select an audio source in the sidebar and press <b>▶️ Start</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def _get_detector(model_id: str | None = None, model_weights: str | None = None):
    """Cache the detector so it's only loaded once across reruns."""
    from inference.detector import AudioDeepfakeDetector, _parse_weights, _resolve_model_ids

    weights = (
        _parse_weights(model_weights, len(_resolve_model_ids(model_id)))
        if model_weights
        else None
    )
    return AudioDeepfakeDetector(model_id=model_id, model_weights=weights)


def _start_pipeline(
    source: str,
    device: int | None,
    overlap: float,
    model_id: str | None,
    model_weights: str | None,
    smoothing: int,
    uploaded_file=None,
):
    """Instantiate capture, detector, gate, and orchestrator."""
    from audio.mic_capture import MicrophoneCapture
    from audio.vad import EnergySpeechGate
    from pipeline.orchestrator import PipelineOrchestrator

    # Build capture source
    if source == "WAV File":
        if uploaded_file is None:
            raise ValueError("Please upload a WAV file first.")
        from audio.wav_loader import WAVFileReplay
        # Save uploaded file to temp location
        tmp_dir = Path(tempfile.mkdtemp())
        tmp_path = tmp_dir / uploaded_file.name
        tmp_path.write_bytes(uploaded_file.read())
        capture = WAVFileReplay(tmp_path, overlap_seconds=overlap, loop=True)
    elif source == "System Audio":
        try:
            from audio.system_capture import SystemAudioCapture
            capture = SystemAudioCapture(device=device, overlap_duration=overlap)
        except Exception as exc:
            raise RuntimeError(
                f"System audio capture failed: {exc}\n"
                "The 'soundcard' package may not support your audio setup. "
                "Try 'Microphone' source instead."
            ) from exc
    else:
        capture = MicrophoneCapture(
            device=device, overlap_duration=overlap
        )

    detector = _get_detector(model_id, model_weights)
    gate = EnergySpeechGate()

    orch = PipelineOrchestrator(
        capture=capture,
        detector=detector,
        speech_gate=gate,
        score_smoothing_window=smoothing,
    )
    orch.start()
    st.session_state.orchestrator = orch


def _stop_pipeline():
    """Stop the running orchestrator."""
    orch = st.session_state.get("orchestrator")
    if orch:
        orch.stop()
    st.session_state.orchestrator = None


# ── Streamlit entry point ────────────────────────────────────────────
if _HAS_STREAMLIT:
    dashboard = Dashboard()
    dashboard.run()
