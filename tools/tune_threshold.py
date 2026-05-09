"""
Threshold tuning tool for True-Tone.

Evaluates multiple threshold values against a labeled test set
and recommends the optimal threshold that maximizes accuracy.

Usage:
    python tools/tune_threshold.py test_audio/
    python tools/tune_threshold.py test_audio/ --model-id <model>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference.detector import AudioDeepfakeDetector

SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
AI_LABEL_HINTS = ("ai", "fake", "synthetic", "generated", "deepfake", "elevenlabs", "clone")
HUMAN_LABEL_HINTS = ("human", "real", "genuine", "natural", "bonafide")


def _classify_ground_truth(path: str | Path) -> str | None:
    path = Path(path)
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Find the optimal AI detection threshold.")
    parser.add_argument("directory", help="Directory containing labeled audio files.")
    parser.add_argument("--model-id", default=None, help="Hugging Face model ID.")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    audio_dir = Path(args.directory)
    files = sorted(
        f for f in audio_dir.rglob("*")
        if f.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    # Score all files
    detector = AudioDeepfakeDetector(
        model_id=args.model_id,
        local_files_only=args.local_files_only,
    )

    scored = []
    for f in files:
        truth = _classify_ground_truth(f)
        if truth is None:
            continue
        try:
            result = detector.predict_file(f)
            scored.append((f.name, result.ai_probability, truth))
        except Exception as exc:
            print(f"Skipping {f.name}: {exc}")

    if not scored:
        print("No labeled files found. Name files with 'ai'/'human' keywords.")
        sys.exit(1)

    print(f"\nScored {len(scored)} labeled files.")
    print()

    # Evaluate thresholds from 0.05 to 0.95
    print(f"{'Threshold':>10} {'Accuracy':>10} {'TP':>5} {'TN':>5} {'FP':>5} {'FN':>5}")
    print("-" * 45)

    best_acc = 0.0
    best_thresh = 0.5

    for t_int in range(5, 100, 5):
        t = t_int / 100.0
        tp = sum(1 for _, s, g in scored if g == "ai" and s >= t)
        tn = sum(1 for _, s, g in scored if g == "human" and s < t)
        fp = sum(1 for _, s, g in scored if g == "human" and s >= t)
        fn = sum(1 for _, s, g in scored if g == "ai" and s < t)
        acc = (tp + tn) / len(scored) if scored else 0
        print(f"{t:>10.2f} {acc:>10.1%} {tp:>5} {tn:>5} {fp:>5} {fn:>5}")
        if acc > best_acc:
            best_acc = acc
            best_thresh = t

    print()
    print(f"✅ Recommended threshold: {best_thresh:.2f} (accuracy: {best_acc:.1%})")


if __name__ == "__main__":
    main()
