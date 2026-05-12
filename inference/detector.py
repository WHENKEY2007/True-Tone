from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - exercised when optional dependency is absent.
    torch = None

try:
    from transformers import pipeline
except ImportError:  # pragma: no cover - exercised when optional dependency is absent.
    pipeline = None

from audio.wav_utils import TARGET_SAMPLE_RATE, chunk_audio, load_wav

DEFAULT_MODEL_ID = "DeepFake-Audio-Rangers/DeepfakeDetect_wav2vec2"
DEFAULT_MODEL_IDS = (DEFAULT_MODEL_ID,)
FAKE_LABEL_HINTS = ("fake", "spoof", "synthetic", "deepfake", "ai", "generated", "1")
REAL_LABEL_HINTS = ("real", "bonafide", "bona-fide", "human", "genuine", "authentic", "0")


@dataclass(frozen=True)
class PredictionResult:
    source: Path
    ai_probability: float
    chunk_scores: list[float]
    raw_outputs: list[list[dict[str, Any]]]
    model_id: str


def parse_model_ids(value: str | None = None) -> list[str]:
    """Return configured detector model ids from a comma-separated string or env var."""
    raw_value = value if value is not None else os.getenv("TRUE_TONE_MODEL_IDS") or os.getenv("TRUE_TONE_MODEL_ID")
    if raw_value:
        model_ids = [item.strip() for item in raw_value.split(",") if item.strip()]
        if model_ids:
            return model_ids
    return list(DEFAULT_MODEL_IDS)


def _score_to_ai_probability(labels: list[dict[str, Any]]) -> float:
    if not labels:
        return 0.0

    fake_score = None
    real_score = None

    for item in labels:
        label = str(item.get("label", "")).lower()
        score = float(item.get("score", 0.0))
        if any(hint in label for hint in FAKE_LABEL_HINTS):
            fake_score = max(score, fake_score or 0.0)
        if any(hint in label for hint in REAL_LABEL_HINTS):
            real_score = max(score, real_score or 0.0)

    if fake_score is not None:
        return fake_score
    if real_score is not None:
        return 1.0 - real_score

    best = max(labels, key=lambda item: float(item.get("score", 0.0)))
    return float(best.get("score", 0.0))


class AudioDeepfakeDetector:
    """CPU-friendly Hugging Face audio classification wrapper.

    The current production path uses Hugging Face audio-classification models
    such as Wav2Vec2 checkpoints. RawNetLite remains a documented candidate
    architecture, but it is not bundled here as a local PyTorch model.
    """

    def __init__(
        self,
        model_id: str | None = None,
        model_ids: list[str] | tuple[str, ...] | None = None,
        sample_rate: int = TARGET_SAMPLE_RATE,
        chunk_seconds: float = 3.0,
        local_files_only: bool = False,
        min_rms: float = 0.002,
    ):
        if model_id and model_ids:
            raise ValueError("Pass either model_id or model_ids, not both.")

        self.model_ids = list(model_ids) if model_ids else [model_id] if model_id else parse_model_ids()
        self.model_id = self.model_ids[0]
        self.sample_rate = sample_rate
        self.chunk_seconds = chunk_seconds
        self.min_rms = min_rms
        if pipeline is None:
            raise RuntimeError("Install transformers to run audio detection: pip install transformers")

        pipeline_kwargs = {
            "task": "audio-classification",
            "device": -1,
            "local_files_only": local_files_only,
        }
        if torch is not None:
            pipeline_kwargs["dtype"] = torch.float32

        self.classifiers = [
            pipeline(
                model=model,
                **pipeline_kwargs,
            )
            for model in self.model_ids
        ]
        self.classifier = self.classifiers[0]

    def predict_samples(self, samples: np.ndarray) -> tuple[float, list[float], list[list[dict[str, Any]]]]:
        chunks = chunk_audio(samples, self.sample_rate, self.chunk_seconds)
        chunk_scores = []
        raw_outputs = []

        for chunk in chunks:
            rms = float(np.sqrt(np.mean(np.square(chunk)))) if chunk.size else 0.0
            if rms < self.min_rms:
                chunk_scores.append(0.0)
                raw_outputs.append([{"label": "silence", "score": 1.0}])
                continue

            model_scores = []
            model_outputs = []
            for classifier in self.classifiers:
                output = classifier({"array": chunk, "sampling_rate": self.sample_rate}, top_k=None)
                labels = list(output)
                model_outputs.extend(labels)
                model_scores.append(_score_to_ai_probability(labels))

            raw_outputs.append(model_outputs)
            chunk_scores.append(float(np.mean(model_scores)) if model_scores else 0.0)

        return float(np.mean(chunk_scores)), chunk_scores, raw_outputs

    def predict_file(self, path: str | Path) -> PredictionResult:
        audio = load_wav(path, target_sample_rate=self.sample_rate)
        ai_probability, chunk_scores, raw_outputs = self.predict_samples(audio.samples)
        return PredictionResult(
            source=Path(path),
            ai_probability=ai_probability,
            chunk_scores=chunk_scores,
            raw_outputs=raw_outputs,
            model_id=",".join(self.model_ids),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify a static WAV/audio file for AI voice likelihood.")
    parser.add_argument("audio_file", help="Path to a WAV/audio file.")
    parser.add_argument("--model-id", default=None, help=f"Hugging Face model id. Default: {DEFAULT_MODEL_ID}")
    parser.add_argument(
        "--model-ids",
        default=None,
        help="Comma-separated Hugging Face model ids for simple averaged ensemble inference.",
    )
    parser.add_argument("--chunk-seconds", type=float, default=3.0, help="Seconds per inference chunk.")
    parser.add_argument("--local-files-only", action="store_true", help="Use only cached model files.")
    parser.add_argument("--min-rms", type=float, default=0.002, help="Treat quieter chunks as silence.")
    args = parser.parse_args()
    if args.model_id and args.model_ids:
        parser.error("Use either --model-id or --model-ids, not both.")

    detector = AudioDeepfakeDetector(
        model_id=args.model_id,
        model_ids=parse_model_ids(args.model_ids) if args.model_ids else None,
        chunk_seconds=args.chunk_seconds,
        local_files_only=args.local_files_only,
        min_rms=args.min_rms,
    )
    result = detector.predict_file(args.audio_file)

    print(f"Model: {result.model_id}")
    print(f"File: {result.source}")
    print(f"AI probability: {result.ai_probability:.4f}")
    print("Chunk scores:", ", ".join(f"{score:.4f}" for score in result.chunk_scores))
    if result.raw_outputs:
        print("First chunk labels:")
        for item in result.raw_outputs[0]:
            print(f"  {item['label']}: {float(item['score']):.4f}")


if __name__ == "__main__":
    main()
