import os
import unittest
from unittest.mock import Mock, patch

import numpy as np

from inference import detector


class DetectorScoringTests(unittest.TestCase):
    def test_fake_label_score_is_ai_probability(self):
        labels = [
            {"label": "real", "score": 0.20},
            {"label": "fake", "score": 0.80},
        ]

        self.assertEqual(detector._score_to_ai_probability(labels), 0.80)

    def test_real_label_is_inverted_when_no_fake_label_exists(self):
        labels = [{"label": "bonafide", "score": 0.90}]

        self.assertAlmostEqual(detector._score_to_ai_probability(labels), 0.10)

    def test_empty_labels_are_zero_probability(self):
        self.assertEqual(detector._score_to_ai_probability([]), 0.0)

    def test_model_ids_can_come_from_environment(self):
        with patch.dict(os.environ, {"TRUE_TONE_MODEL_IDS": "model/a, model/b"}, clear=False):
            self.assertEqual(detector.parse_model_ids(), ["model/a", "model/b"])

    def test_silent_chunks_skip_classifier(self):
        classifier = Mock()

        with patch("inference.detector.pipeline", return_value=classifier):
            model = detector.AudioDeepfakeDetector(model_id="local/test", chunk_seconds=1.0, min_rms=1.0)

        probability, chunk_scores, raw_outputs = model.predict_samples(np.zeros(16000, dtype=np.float32))

        classifier.assert_not_called()
        self.assertEqual(probability, 0.0)
        self.assertEqual(chunk_scores, [0.0])
        self.assertEqual(raw_outputs, [[{"label": "silence", "score": 1.0}]])

    def test_ensemble_averages_classifier_scores(self):
        classifier_a = Mock(return_value=[{"label": "fake", "score": 0.80}])
        classifier_b = Mock(return_value=[{"label": "real", "score": 0.80}])

        with patch("inference.detector.pipeline", side_effect=[classifier_a, classifier_b]):
            model = detector.AudioDeepfakeDetector(
                model_ids=["local/a", "local/b"],
                chunk_seconds=1.0,
                min_rms=0.0,
            )

        samples = np.ones(16000, dtype=np.float32) * 0.1
        probability, chunk_scores, raw_outputs = model.predict_samples(samples)

        self.assertAlmostEqual(probability, 0.50)
        self.assertEqual(chunk_scores, [0.50])
        self.assertEqual(len(raw_outputs[0]), 2)


if __name__ == "__main__":
    unittest.main()
