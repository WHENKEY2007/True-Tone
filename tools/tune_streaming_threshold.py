"""
Tune the live detector threshold using rolling-window stabilized scores.

Unlike tools/tune_threshold.py, this script evaluates the same temporal
aggregator used by live detection, then sweeps thresholds over final session
scores. Use it to set dashboard/CLI alert thresholds from real streaming-like
behavior instead of static file averages.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from audio.vad import EnergySpeechGate
from inference.detector import AudioDeepfakeDetector, _parse_weights, _resolve_model_ids
from tools.evaluate_streaming import SUPPORTED_EXTENSIONS, _classify_ground_truth, _evaluate_file


def _metrics(scored: list[tuple[Path, float, str]], threshold: float) -> dict[str, float | int]:
    tp = sum(1 for _, score, truth in scored if truth == "ai" and score >= threshold)
    tn = sum(1 for _, score, truth in scored if truth == "human" and score < threshold)
    fp = sum(1 for _, score, truth in scored if truth == "human" and score >= threshold)
    fn = sum(1 for _, score, truth in scored if truth == "ai" and score < threshold)
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": threshold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune streaming stabilized-score threshold.")
    parser.add_argument("directory", help="Directory containing real/ and fake/ audio files.")
    parser.add_argument("--model-id", default=None, help="Hugging Face model ID.")
    parser.add_argument("--model-weights", default=None, help="Comma-separated model branch weights.")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--chunk-seconds", type=float, default=3.0)
    parser.add_argument("--hop-seconds", type=float, default=1.0)
    parser.add_argument("--min-rms", type=float, default=0.002)
    parser.add_argument("--min-peak", type=float, default=0.01)
    parser.add_argument(
        "--score-field",
        choices=("stable_final", "stable_max"),
        default="stable_final",
        help="Score to tune. final is conservative; max is faster but noisier.",
    )
    args = parser.parse_args()

    audio_dir = Path(args.directory)
    files = sorted(f for f in audio_dir.rglob("*") if f.suffix.lower() in SUPPORTED_EXTENSIONS)
    labeled = [f for f in files if _classify_ground_truth(f) is not None]
    if not labeled:
        print("No labeled files found. Use parent folders named real/ and fake/.")
        sys.exit(1)

    detector = AudioDeepfakeDetector(
        model_id=args.model_id,
        chunk_seconds=args.chunk_seconds,
        local_files_only=args.local_files_only,
        min_rms=args.min_rms,
        model_weights=_parse_weights(args.model_weights, len(_resolve_model_ids(args.model_id)))
        if args.model_weights
        else None,
    )
    gate = EnergySpeechGate(min_rms=args.min_rms, min_peak=args.min_peak)

    scored: list[tuple[Path, float, str]] = []
    print(f"Scoring {len(labeled)} labeled file(s)...")
    for path in labeled:
        result = _evaluate_file(
            path,
            detector,
            gate,
            args.chunk_seconds,
            args.hop_seconds,
            threshold=0.65,
        )
        truth = str(result["truth"])
        score = float(result[args.score_field])
        scored.append((path, score, truth))

    print()
    print(f"Tuning on `{args.score_field}`")
    print(f"{'Threshold':>10} {'Accuracy':>10} {'F1':>8} {'Precision':>10} {'Recall':>8} {'TP':>4} {'TN':>4} {'FP':>4} {'FN':>4}")
    print("-" * 78)

    rows = [_metrics(scored, i / 100.0) for i in range(5, 96, 5)]
    best = max(rows, key=lambda row: (float(row["f1"]), float(row["accuracy"]), -int(row["fp"])))
    for row in rows:
        marker = "*" if row is best else " "
        print(
            f"{marker}{row['threshold']:>9.2f} {row['accuracy']:>10.1%} {row['f1']:>8.3f} "
            f"{row['precision']:>10.3f} {row['recall']:>8.3f} "
            f"{row['tp']:>4} {row['tn']:>4} {row['fp']:>4} {row['fn']:>4}"
        )

    print()
    print(
        "Recommended threshold: "
        f"{best['threshold']:.2f} "
        f"(accuracy={best['accuracy']:.1%}, f1={best['f1']:.3f}, "
        f"fp={best['fp']}, fn={best['fn']})"
    )


if __name__ == "__main__":
    main()
