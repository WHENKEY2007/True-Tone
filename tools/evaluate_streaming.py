"""
Streaming-style evaluation for True-Tone.

This replays files as overlapping live windows, applies the same temporal
confidence aggregator used by the live pipeline, and reports both raw and
stabilized behavior. It is meant to catch instability that static file
averages hide.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from audio.vad import EnergySpeechGate
from audio.wav_utils import TARGET_SAMPLE_RATE, chunk_audio, load_wav
from inference.detector import AudioDeepfakeDetector
from pipeline.temporal import TemporalConfidenceAggregator


SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
AI_LABEL_HINTS = ("ai", "fake", "synthetic", "generated", "deepfake", "elevenlabs", "clone")
HUMAN_LABEL_HINTS = ("human", "real", "genuine", "natural", "bonafide")


def _classify_ground_truth(path: Path) -> str | None:
    parts = [part.lower() for part in (*path.parent.parts, path.name)]
    if any(part in {"fake", "ai", "synthetic", "generated", "deepfake"} for part in parts):
        return "ai"
    if any(part in {"real", "human", "bonafide", "genuine"} for part in parts):
        return "human"

    text = " ".join(parts)
    if any(hint in text for hint in AI_LABEL_HINTS):
        return "ai"
    if any(hint in text for hint in HUMAN_LABEL_HINTS):
        return "human"
    return None


def _verdict(score: float, threshold: float) -> str:
    return "AI" if score >= threshold else "HUMAN"


def _match(verdict: str, truth: str | None) -> str:
    if truth is None:
        return "-"
    expected = "AI" if truth == "ai" else "HUMAN"
    return "Y" if verdict == expected else "N"


def _evaluate_file(
    path: Path,
    detector: AudioDeepfakeDetector,
    gate: EnergySpeechGate,
    chunk_seconds: float,
    hop_seconds: float,
    threshold: float,
) -> dict[str, object]:
    audio = load_wav(path, target_sample_rate=TARGET_SAMPLE_RATE)
    chunks = chunk_audio(audio.samples, audio.sample_rate, chunk_seconds, hop_seconds)
    temporal = TemporalConfidenceAggregator()

    raw_scores: list[float] = []
    stable_scores: list[float] = []
    uncertainties: list[float] = []
    states: list[str] = []
    speech_chunks = 0
    first_alert_chunk: int | None = None

    for index, chunk in enumerate(chunks, 1):
        speech = gate.process(chunk)
        if speech.is_speech:
            raw_score, _, _ = detector.predict_samples(speech.samples)
            feature_summary = detector.last_feature_summaries[-1] if detector.last_feature_summaries else {}
            behavior_score = float(feature_summary.get("synthetic_behavior_score", 0.0))
            cadence_consistency = float(feature_summary.get("cadence_consistency", 0.0))
            speech_chunks += 1
        else:
            raw_score = 0.0
            behavior_score = 0.0
            cadence_consistency = 0.0

        decision = temporal.update(
            raw_score,
            is_speech=speech.is_speech,
            behavior_score=behavior_score,
            cadence_consistency=cadence_consistency,
        )
        raw_scores.append(raw_score)
        stable_scores.append(decision.stabilized_probability)
        uncertainties.append(decision.uncertainty)
        states.append(decision.state)
        if first_alert_chunk is None and decision.stabilized_probability >= threshold:
            first_alert_chunk = index

    raw_array = np.array(raw_scores, dtype=np.float32)
    stable_array = np.array(stable_scores, dtype=np.float32)
    final_score = float(stable_array[-1]) if stable_array.size else 0.0
    truth = _classify_ground_truth(path)
    verdict = _verdict(final_score, threshold)

    return {
        "file": path,
        "truth": truth,
        "verdict": verdict,
        "match": _match(verdict, truth),
        "chunks": len(chunks),
        "speech_chunks": speech_chunks,
        "raw_mean": float(np.mean(raw_array)) if raw_array.size else 0.0,
        "raw_std": float(np.std(raw_array)) if raw_array.size else 0.0,
        "stable_final": final_score,
        "stable_max": float(np.max(stable_array)) if stable_array.size else 0.0,
        "stable_std": float(np.std(stable_array)) if stable_array.size else 0.0,
        "mean_uncertainty": float(np.mean(uncertainties)) if uncertainties else 0.0,
        "final_state": states[-1] if states else "no_audio",
        "first_alert_seconds": None if first_alert_chunk is None else (first_alert_chunk - 1) * hop_seconds,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate True-Tone as a rolling-window streaming detector.")
    parser.add_argument("directory", help="Directory containing audio files.")
    parser.add_argument("--model-id", default=None, help="Hugging Face model ID.")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.65)
    parser.add_argument("--chunk-seconds", type=float, default=3.0)
    parser.add_argument("--hop-seconds", type=float, default=1.0)
    parser.add_argument("--min-rms", type=float, default=0.002)
    parser.add_argument("--min-peak", type=float, default=0.01)
    args = parser.parse_args()

    audio_dir = Path(args.directory)
    files = sorted(f for f in audio_dir.rglob("*") if f.suffix.lower() in SUPPORTED_EXTENSIONS)
    if not files:
        print(f"No audio files found in {audio_dir}")
        sys.exit(1)

    detector = AudioDeepfakeDetector(
        model_id=args.model_id,
        chunk_seconds=args.chunk_seconds,
        local_files_only=args.local_files_only,
        min_rms=args.min_rms,
    )
    gate = EnergySpeechGate(min_rms=args.min_rms, min_peak=args.min_peak)

    print(f"\nEvaluating {len(files)} file(s) with {args.chunk_seconds:.1f}s windows / {args.hop_seconds:.1f}s hop")
    print()
    print(
        f"{'File':<32} {'Truth':>6} {'Verdict':>7} {'OK':>3} "
        f"{'RawMean':>7} {'RawSD':>6} {'Final':>7} {'Max':>7} {'Unc':>6} {'State':>16} {'Alert':>7}"
    )
    print("-" * 118)

    results = []
    t0 = time.time()
    for path in files:
        result = _evaluate_file(
            path,
            detector,
            gate,
            args.chunk_seconds,
            args.hop_seconds,
            args.threshold,
        )
        results.append(result)
        alert = "-" if result["first_alert_seconds"] is None else f"{result['first_alert_seconds']:.1f}s"
        truth = (result["truth"] or "?").upper()
        print(
            f"{path.name:<32} {truth:>6} {result['verdict']:>7} {result['match']:>3} "
            f"{result['raw_mean']:>7.3f} {result['raw_std']:>6.3f} "
            f"{result['stable_final']:>7.3f} {result['stable_max']:>7.3f} "
            f"{result['mean_uncertainty']:>6.3f} {result['final_state']:>16} {alert:>7}"
        )

    labeled = [r for r in results if r["truth"] is not None]
    if labeled:
        correct = sum(1 for r in labeled if r["match"] == "Y")
        ai_total = sum(1 for r in labeled if r["truth"] == "ai")
        human_total = sum(1 for r in labeled if r["truth"] == "human")
        fp = sum(1 for r in labeled if r["truth"] == "human" and r["verdict"] == "AI")
        fn = sum(1 for r in labeled if r["truth"] == "ai" and r["verdict"] == "HUMAN")
        print()
        print("=" * 118)
        print(f"Labeled files: {len(labeled)} | Accuracy: {correct / len(labeled):.1%}")
        print(f"AI files: {ai_total} | Human files: {human_total} | False positives: {fp} | False negatives: {fn}")
    else:
        print("\nNo labeled files found. Use parent folders like real/ and fake/.")

    print(f"Elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
