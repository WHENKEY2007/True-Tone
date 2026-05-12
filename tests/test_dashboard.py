from datetime import datetime
import unittest

import numpy as np

from ui.dashboard import Dashboard, DashboardState, DetectionEvent, build_view_model


class DashboardTests(unittest.TestCase):
    def test_view_model_clamps_probability_and_builds_alert(self):
        state = DashboardState(ai_probability=1.5, threshold=0.7)

        view_model = build_view_model(state)

        self.assertEqual(view_model.probability_percent, 100)
        self.assertEqual(view_model.alert_level, "error")
        self.assertEqual(view_model.status_label, "High likelihood of synthetic speech")

    def test_view_model_includes_waveform_and_event_rows(self):
        event = DetectionEvent(
            timestamp=datetime(2026, 5, 11, 9, 40, 0),
            probability=0.73,
            source="sample.wav",
            message="High likelihood of synthetic speech detected.",
        )
        state = DashboardState(
            ai_probability=0.20,
            waveform=np.array([[0.1], [0.2]], dtype=np.float32),
            events=(event,),
        )

        view_model = build_view_model(state)

        np.testing.assert_allclose(view_model.waveform, np.array([0.1, 0.2], dtype=np.float32))
        self.assertEqual(
            view_model.event_rows,
            [
                {
                    "time": "09:40:00",
                    "source": "sample.wav",
                    "probability": "0.73",
                    "message": "High likelihood of synthetic speech detected.",
                }
            ],
        )

    def test_dashboard_update_state_records_threshold_crossing(self):
        dashboard = Dashboard(threshold=0.7)

        state = dashboard.update_state(0.71, waveform=np.array([0.1, -0.1]), source="live")

        self.assertEqual(state.ai_probability, 0.71)
        self.assertEqual(state.source, "live")
        self.assertEqual(len(state.events), 1)


if __name__ == "__main__":
    unittest.main()
