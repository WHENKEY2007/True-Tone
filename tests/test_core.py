"""
Unit tests for True-Tone core modules.

Run with:
    python -m pytest tests/ -v
    python tests/test_core.py        # standalone
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

# Ensure project root is importable
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class TestAudioPreprocessor(unittest.TestCase):
    """Tests for the audio preprocessing pipeline."""

    def test_to_mono_stereo(self):
        from processing.preprocessor import to_mono
        stereo = np.random.randn(16000, 2).astype(np.float32)
        mono = to_mono(stereo)
        self.assertEqual(mono.ndim, 1)
        self.assertEqual(mono.shape[0], 16000)

    def test_to_mono_already_mono(self):
        from processing.preprocessor import to_mono
        mono_in = np.random.randn(16000).astype(np.float32)
        mono_out = to_mono(mono_in)
        self.assertEqual(mono_out.shape, (16000,))

    def test_normalize_silent(self):
        from processing.preprocessor import normalize
        silent = np.zeros(16000, dtype=np.float32)
        result = normalize(silent)
        np.testing.assert_array_equal(result, silent)

    def test_normalize_scales(self):
        from processing.preprocessor import normalize
        audio = np.array([0.5, -0.5, 0.25], dtype=np.float32)
        result = normalize(audio)
        self.assertAlmostEqual(float(np.max(np.abs(result))), 0.5, places=4)

    def test_pad_or_trim_pad(self):
        from processing.preprocessor import pad_or_trim
        short = np.ones(100, dtype=np.float32)
        result, was_padded, was_trimmed = pad_or_trim(short, 200)
        self.assertEqual(result.shape[0], 200)
        self.assertTrue(was_padded)
        self.assertFalse(was_trimmed)

    def test_pad_or_trim_trim(self):
        from processing.preprocessor import pad_or_trim
        long = np.ones(300, dtype=np.float32)
        result, was_padded, was_trimmed = pad_or_trim(long, 200)
        self.assertEqual(result.shape[0], 200)
        self.assertFalse(was_padded)
        self.assertTrue(was_trimmed)

    def test_pad_or_trim_exact(self):
        from processing.preprocessor import pad_or_trim
        exact = np.ones(200, dtype=np.float32)
        result, was_padded, was_trimmed = pad_or_trim(exact, 200)
        self.assertEqual(result.shape[0], 200)
        self.assertFalse(was_padded)
        self.assertFalse(was_trimmed)

    def test_preprocess_chunk(self):
        from processing.preprocessor import preprocess_chunk
        raw = np.random.randn(48000).astype(np.float32)
        chunk = preprocess_chunk(raw, source_rate=48000, target_rate=16000, chunk_seconds=3.0)
        self.assertEqual(chunk.sample_rate, 16000)
        self.assertEqual(chunk.samples.shape[0], 48000)  # 16000 * 3
        self.assertGreaterEqual(chunk.rms, 0.0)
        self.assertGreaterEqual(chunk.peak, 0.0)

    def test_compute_stats(self):
        from processing.preprocessor import compute_stats
        audio = np.array([0.5, -0.5, 0.25, -0.25], dtype=np.float32)
        rms, peak = compute_stats(audio)
        self.assertAlmostEqual(peak, 0.5, places=4)
        self.assertGreater(rms, 0.0)


class TestEnergySpeechGate(unittest.TestCase):
    """Tests for the energy-based speech gate."""

    def test_silence_is_not_speech(self):
        from audio.vad import EnergySpeechGate
        gate = EnergySpeechGate(min_rms=0.002, min_peak=0.01)
        silence = np.zeros(16000, dtype=np.float32)
        result = gate.process(silence)
        self.assertFalse(result.is_speech)
        self.assertAlmostEqual(result.rms, 0.0)

    def test_loud_signal_is_speech(self):
        from audio.vad import EnergySpeechGate
        gate = EnergySpeechGate(min_rms=0.002, min_peak=0.01)
        loud = np.random.randn(16000).astype(np.float32) * 0.5
        result = gate.process(loud)
        self.assertTrue(result.is_speech)
        self.assertGreater(result.rms, 0.002)

    def test_result_preserves_samples(self):
        from audio.vad import EnergySpeechGate
        gate = EnergySpeechGate()
        audio = np.random.randn(48000).astype(np.float32) * 0.3
        result = gate.process(audio)
        self.assertEqual(result.samples.shape[0], 48000)


class TestWavUtils(unittest.TestCase):
    """Tests for WAV utility functions."""

    def test_chunk_audio_non_overlapping(self):
        from audio.wav_utils import chunk_audio
        audio = np.random.randn(48000).astype(np.float32)
        chunks = chunk_audio(audio, sample_rate=16000, chunk_seconds=1.0)
        self.assertEqual(len(chunks), 3)
        for chunk in chunks:
            self.assertEqual(chunk.shape[0], 16000)

    def test_chunk_audio_overlapping(self):
        from audio.wav_utils import chunk_audio
        audio = np.random.randn(48000).astype(np.float32)
        chunks = chunk_audio(audio, sample_rate=16000, chunk_seconds=1.0, hop_seconds=0.5)
        self.assertGreater(len(chunks), 3)

    def test_chunk_audio_short_input(self):
        from audio.wav_utils import chunk_audio
        audio = np.random.randn(8000).astype(np.float32)
        chunks = chunk_audio(audio, sample_rate=16000, chunk_seconds=1.0)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].shape[0], 16000)  # padded


class TestTemporalAggregator(unittest.TestCase):
    """Tests for the temporal confidence aggregation."""

    def test_initial_state(self):
        from pipeline.temporal import TemporalConfidenceAggregator
        agg = TemporalConfidenceAggregator()
        decision = agg.update(0.5, is_speech=True)
        self.assertEqual(decision.state, "insufficient_speech")

    def test_stabilizes_after_windows(self):
        from pipeline.temporal import TemporalConfidenceAggregator
        agg = TemporalConfidenceAggregator(min_speech_windows=2)
        agg.update(0.8, is_speech=True, timestamp=1.0)
        decision = agg.update(0.9, is_speech=True, timestamp=2.0)
        self.assertNotEqual(decision.state, "insufficient_speech")
        self.assertGreater(decision.stabilized_probability, 0.5)

    def test_silence_does_not_crash(self):
        from pipeline.temporal import TemporalConfidenceAggregator
        agg = TemporalConfidenceAggregator()
        for i in range(5):
            decision = agg.update(0.0, is_speech=False, timestamp=float(i))
        self.assertIsNotNone(decision)

    def test_reset_clears_state(self):
        from pipeline.temporal import TemporalConfidenceAggregator
        agg = TemporalConfidenceAggregator()
        agg.update(0.9, is_speech=True, timestamp=1.0)
        agg.update(0.9, is_speech=True, timestamp=2.0)
        agg.reset()
        decision = agg.update(0.1, is_speech=True, timestamp=3.0)
        self.assertEqual(decision.state, "insufficient_speech")


class TestAudioBuffer(unittest.TestCase):
    """Tests for the thread-safe audio buffer."""

    def test_put_and_get(self):
        from audio.buffer import AudioBuffer
        buf = AudioBuffer(max_size=10)
        buf.put("chunk_1")
        buf.put("chunk_2")
        self.assertEqual(buf.get(), "chunk_1")
        self.assertEqual(buf.get(), "chunk_2")
        self.assertIsNone(buf.get())

    def test_max_size(self):
        from audio.buffer import AudioBuffer
        buf = AudioBuffer(max_size=3)
        for i in range(5):
            buf.put(i)
        self.assertEqual(buf.size(), 3)

    def test_peek_latest(self):
        from audio.buffer import AudioBuffer
        buf = AudioBuffer()
        buf.put("a")
        buf.put("b")
        self.assertEqual(buf.peek_latest(), "b")
        self.assertEqual(buf.size(), 2)  # not consumed

    def test_get_all_drains(self):
        from audio.buffer import AudioBuffer
        buf = AudioBuffer()
        buf.put(1)
        buf.put(2)
        items = buf.get_all()
        self.assertEqual(items, [1, 2])
        self.assertTrue(buf.is_empty())


class TestFeatureExtraction(unittest.TestCase):
    """Tests for handcrafted audio feature extraction."""

    def test_extract_features_returns_vector(self):
        from processing.features import extract_audio_features
        audio = np.random.randn(48000).astype(np.float32) * 0.3
        features = extract_audio_features(audio, sample_rate=16000)
        self.assertGreaterEqual(features.rms, 0.0)
        self.assertGreaterEqual(features.spectral_entropy, 0.0)
        self.assertLessEqual(features.spectral_entropy, 1.0)
        self.assertGreaterEqual(features.synthetic_behavior_score, 0.0)
        self.assertLessEqual(features.synthetic_behavior_score, 1.0)

    def test_extract_features_silent_input(self):
        from processing.features import extract_audio_features
        silence = np.zeros(48000, dtype=np.float32)
        features = extract_audio_features(silence, sample_rate=16000)
        self.assertAlmostEqual(features.rms, 0.0, places=4)

    def test_as_dict_keys(self):
        from processing.features import extract_audio_features
        audio = np.random.randn(48000).astype(np.float32) * 0.3
        features = extract_audio_features(audio, sample_rate=16000)
        d = features.as_dict()
        self.assertIn("synthetic_behavior_score", d)
        self.assertIn("spectral_entropy", d)
        self.assertIn("cadence_consistency", d)
        self.assertIn("pitch_mean_hz", d)


if __name__ == "__main__":
    unittest.main()
