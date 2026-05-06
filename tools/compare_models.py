from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference.detector import AudioDeepfakeDetector


DEFAULT_MODELS = [
    "Vansh180/deepfake-audio-wav2vec2",
    "Hemgg/Deepfake-audio-detection",
    "DeepFake-Audio-Rangers/DeepfakeDetect_wav2vec2",
    "Gustking/wav2vec2-large-xlsr-deepfake-audio-classification",
    "garystafford/wav2vec2-deepfake-voice-detector",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare deepfake audio classifiers on one audio file.")
    parser.add_argument("audio_file", help="Audio file to classify.")
    parser.add_argument("--local-files-only", action="store_true", help="Use cached model files only.")
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS, help="Model IDs to test.")
    args = parser.parse_args()

    audio_file = Path(args.audio_file)
    print(f"Testing: {audio_file}")
    print()

    results = []
    for model_id in args.models:
        print(f"Model: {model_id}")
        try:
            detector = AudioDeepfakeDetector(
                model_id=model_id,
                local_files_only=args.local_files_only,
            )
            result = detector.predict_file(audio_file)
            labels = result.raw_outputs[0] if result.raw_outputs else []
            print(f"  AI probability: {result.ai_probability:.4f}")
            print("  Labels:", ", ".join(f"{item['label']}={float(item['score']):.4f}" for item in labels))
            results.append((result.ai_probability, model_id))
        except Exception as exc:
            print(f"  ERROR: {type(exc).__name__}: {exc}")
        print()

    if results:
        results.sort(reverse=True)
        print("Ranking:")
        for score, model_id in results:
            print(f"  {score:.4f}  {model_id}")


if __name__ == "__main__":
    main()
