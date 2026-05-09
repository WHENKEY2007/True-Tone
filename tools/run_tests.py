"""
Batch test runner for True-Tone.

Classifies all audio files in a directory, prints per-file results,
and generates a summary table with accuracy metrics.

Usage:
    python tools/run_tests.py test_audio/
    python tools/run_tests.py test_audio/ --model-id <model>
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference.detector import AudioDeepfakeDetector

SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

# Files with these substrings in their name are assumed to be AI-generated
AI_LABEL_HINTS = ("ai", "fake", "synthetic", "generated", "deepfake", "elevenlabs", "clone")
HUMAN_LABEL_HINTS = ("human", "real", "genuine", "natural", "bonafide")


def _classify_ground_truth(filename: str) -> str | None:
    """Infer ground-truth label from the file name.

    Returns 'ai', 'human', or None if indeterminate.
    """
    name = filename.lower()
    if any(hint in name for hint in AI_LABEL_HINTS):
        return "ai"
    if any(hint in name for hint in HUMAN_LABEL_HINTS):
        return "human"
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch-test audio files for AI voice likelihood."
    )
    parser.add_argument(
        "directory",
        help="Directory containing audio files to test.",
    )
    parser.add_argument(
        "--model-id", default=None,
        help="Hugging Face model ID (default: project default).",
    )
    parser.add_argument(
        "--local-files-only", action="store_true",
        help="Use cached model files only.",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.65,
        help="AI detection threshold for pass/fail (default: 0.65).",
    )
    parser.add_argument(
        "--chunk-seconds", type=float, default=3.0,
        help="Seconds per inference chunk.",
    )
    args = parser.parse_args()

    audio_dir = Path(args.directory)
    if not audio_dir.is_dir():
        print(f"Error: {audio_dir} is not a directory.")
        sys.exit(1)

    files = sorted(
        f for f in audio_dir.rglob("*")
        if f.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not files:
        print(f"No audio files found in {audio_dir}")
        sys.exit(1)

    print(f"Found {len(files)} audio file(s) in {audio_dir}")
    print(f"Loading model...")

    detector = AudioDeepfakeDetector(
        model_id=args.model_id,
        chunk_seconds=args.chunk_seconds,
        local_files_only=args.local_files_only,
    )

    print()
    print(f"{'File':<40} {'Score':>8} {'Verdict':>10} {'Truth':>8} {'Match':>7} {'Time':>6}")
    print("-" * 85)

    results = []
    tp, tn, fp, fn = 0, 0, 0, 0

    for f in files:
        t0 = time.time()
        try:
            result = detector.predict_file(f)
            score = result.ai_probability
            elapsed = time.time() - t0
        except Exception as exc:
            print(f"{f.name:<40} {'ERROR':>8}   {type(exc).__name__}: {exc}")
            continue

        verdict = "AI" if score >= args.threshold else "HUMAN"
        truth = _classify_ground_truth(f.name)
        truth_label = truth.upper() if truth else "?"

        if truth == "ai":
            match = "✓" if verdict == "AI" else "✗"
            if verdict == "AI":
                tp += 1
            else:
                fn += 1
        elif truth == "human":
            match = "✓" if verdict == "HUMAN" else "✗"
            if verdict == "HUMAN":
                tn += 1
            else:
                fp += 1
        else:
            match = "—"

        print(
            f"{f.name:<40} {score:>8.4f} {verdict:>10} {truth_label:>8} {match:>7} {elapsed:>5.1f}s"
        )
        results.append((f.name, score, verdict, truth))

    # Summary
    total_labeled = tp + tn + fp + fn
    print()
    print("=" * 85)
    print(f"Total files: {len(results)}")

    if total_labeled > 0:
        accuracy = (tp + tn) / total_labeled
        print(f"Labeled files: {total_labeled}")
        print(f"  True Positives  (AI→AI):      {tp}")
        print(f"  True Negatives  (Human→Human): {tn}")
        print(f"  False Positives (Human→AI):    {fp}")
        print(f"  False Negatives (AI→Human):    {fn}")
        print(f"  Accuracy: {accuracy:.1%}")
    else:
        print("No labeled files found (name files with 'ai'/'human' for auto-labeling).")

    # Score distribution
    if results:
        scores = [r[1] for r in results]
        print(f"\nScore distribution:")
        print(f"  Min:  {min(scores):.4f}")
        print(f"  Max:  {max(scores):.4f}")
        print(f"  Mean: {sum(scores)/len(scores):.4f}")


if __name__ == "__main__":
    main()
