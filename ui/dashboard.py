from __future__ import annotations

"""
Streamlit dashboard for True-Tone.

The rendering logic is split into a small view model plus a Streamlit adapter so
tests can verify dashboard behavior without launching a browser.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_ALERT_THRESHOLD = 0.70


@dataclass(frozen=True)
class DetectionEvent:
    """One user-visible detection event for the dashboard log."""

    timestamp: datetime
    probability: float
    source: str
    message: str


@dataclass(frozen=True)
class DashboardState:
    """Snapshot of detection state consumed by the dashboard."""

    ai_probability: float = 0.0
    waveform: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    sample_rate: int = 16000
    threshold: float = DEFAULT_ALERT_THRESHOLD
    events: tuple[DetectionEvent, ...] = ()
    source: str = "idle"
    is_running: bool = False


@dataclass(frozen=True)
class DashboardViewModel:
    """Presentation-ready values used by the Streamlit adapter."""

    probability_percent: int
    status_label: str
    alert_level: str
    waveform: np.ndarray
    event_rows: list[dict[str, Any]]


def build_view_model(state: DashboardState) -> DashboardViewModel:
    """Convert raw detection state into labels, meter values, and table rows."""
    probability = min(max(float(state.ai_probability), 0.0), 1.0)
    probability_percent = int(round(probability * 100))
    is_alert = probability >= state.threshold
    status_label = "High likelihood of synthetic speech" if is_alert else "No synthetic speech alert"
    alert_level = "error" if is_alert else "success"
    waveform = np.asarray(state.waveform, dtype=np.float32).reshape(-1)
    event_rows = [
        {
            "time": event.timestamp.strftime("%H:%M:%S"),
            "source": event.source,
            "probability": f"{event.probability:.2f}",
            "message": event.message,
        }
        for event in state.events
    ]
    return DashboardViewModel(
        probability_percent=probability_percent,
        status_label=status_label,
        alert_level=alert_level,
        waveform=waveform,
        event_rows=event_rows,
    )


class Dashboard:
    """Main dashboard interface for True-Tone application."""

    def __init__(self, threshold: float = DEFAULT_ALERT_THRESHOLD):
        """Initialize the dashboard."""
        self.is_running = False
        self.threshold = threshold
        self.state = DashboardState(threshold=threshold)

    def update_state(
        self,
        ai_probability: float,
        waveform: np.ndarray | None = None,
        sample_rate: int = 16000,
        source: str = "live",
    ) -> DashboardState:
        """Update dashboard state and append an alert event when threshold is crossed."""
        events = list(self.state.events)
        if ai_probability >= self.threshold:
            events.append(
                DetectionEvent(
                    timestamp=datetime.now(),
                    probability=float(ai_probability),
                    source=source,
                    message="High likelihood of synthetic speech detected.",
                )
            )

        self.state = DashboardState(
            ai_probability=float(ai_probability),
            waveform=np.asarray(waveform if waveform is not None else [], dtype=np.float32),
            sample_rate=sample_rate,
            threshold=self.threshold,
            events=tuple(events[-25:]),
            source=source,
            is_running=self.is_running,
        )
        return self.state

    def run(self) -> None:
        """Start the dashboard application."""
        self.is_running = True
        self.state = DashboardState(
            ai_probability=self.state.ai_probability,
            waveform=self.state.waveform,
            sample_rate=self.state.sample_rate,
            threshold=self.threshold,
            events=self.state.events,
            source=self.state.source,
            is_running=True,
        )
        self.render()

    def render(self, state: DashboardState | None = None) -> DashboardViewModel:
        """Render the dashboard interface and return the testable view model."""
        state = state or self.state
        view_model = build_view_model(state)

        try:
            import streamlit as st
        except ImportError:
            print(f"True-Tone: {view_model.status_label} ({view_model.probability_percent}%)")
            return view_model

        st.set_page_config(page_title="True-Tone", layout="wide")
        st.title("True-Tone")

        meter, waveform, log = st.columns([1, 2, 2])
        with meter:
            st.metric("AI voice probability", f"{view_model.probability_percent}%")
            st.progress(view_model.probability_percent / 100)
            if view_model.alert_level == "error":
                st.error(view_model.status_label)
            else:
                st.success(view_model.status_label)

        with waveform:
            st.subheader("Waveform")
            if view_model.waveform.size:
                st.line_chart(view_model.waveform)
            else:
                st.info("Waiting for audio...")

        with log:
            st.subheader("Event log")
            if view_model.event_rows:
                st.dataframe(view_model.event_rows, hide_index=True, use_container_width=True)
            else:
                st.info("No alerts recorded.")

        st.caption(f"Source: {state.source} | Threshold: {state.threshold:.2f}")
        return view_model

    def render_file_result(self, result: Any) -> DashboardViewModel:
        """Render a detector PredictionResult and keep source details in the event log."""
        source = str(Path(result.source).name)
        waveform = np.asarray([], dtype=np.float32)
        self.update_state(result.ai_probability, waveform=waveform, source=source)
        return self.render()

    def stop(self) -> None:
        """Stop the dashboard application."""
        self.is_running = False
