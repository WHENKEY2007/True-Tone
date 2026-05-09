from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from transformers import pipeline

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from audio.wav_utils import TARGET_SAMPLE_RATE, chunk_audio, load_wav
from processing.features import extract_audio_features

DEFAULT_MODEL_ID = "DeepFake-Audio-Rangers/DeepfakeDetect_wav2vec2"
FAKE_LABEL_HINTS = ("fake", "spoof", "synthetic", "deepfake", "ai", "generated", "1")
REAL_LABEL_HINTS = ("real", "bonafide", "bona-fide", "human", "genuine", "authentic", "0")


@dataclass(frozen=True)
class ModelBranch:
    """One neural detector branch in the ensemble."""

    model_id: str
    weight: float
    classifier: Any


@dataclass(frozen=True)
class PredictionResult:
    source: Path
    ai_probability: float
    chunk_scores: list[float]
    raw_outputs: list[list[dict[str, Any]]]
    model_id: str
    feature_summaries: list[dict[str, float]] = field(default_factory=list)


def _parse_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_weights(value: str | None, count: int) -> list[float]:
    parsed = []
    if value:
        for item in value.split(","):
            item = item.strip()
            if item:
                parsed.append(float(item))
    if not parsed:
        return [1.0] * count
    if len(parsed) != count:
        raise ValueError(f"Expected {count} model weights, got {len(parsed)}.")
    return parsed


def _resolve_model_ids(model_id: str | None) -> list[str]:
    model_ids = _parse_csv(model_id)
    if not model_ids:
        model_ids = _parse_csv(os.getenv("TRUE_TONE_MODEL_IDS"))
    if not model_ids:
        single = os.getenv("TRUE_TONE_MODEL_ID", DEFAULT_MODEL_ID)
        model_ids = [single]
    return model_ids


def _score_to_ai_probability(labels: list[dict[str, Any]]) -> float:
    fake_score = None
    real_score = None

    for item in labels:
        label = str(item.get("label", "")).lower()
        score = float(item.get("score", 0.0))
        if any(hint in label for hint in FAKE_LABEL_HINTS):
            fake_score = max(score, fake_score or 0.0)
        if any(hint in label for hint in REAL_LABEL_HINTS):
            real_score = max(score, real_score or 0.0)

    if fake_score is not None and real_score is not None:
        total = fake_score + real_score
        if total > 0.0:
            return fake_score / total
        return fake_score

    if fake_score is not None:
        return fake_score
    if real_score is not None:
        return 1.0 - real_score

    # If the model returns unknown labels, avoid treating them as AI by default.
    return 0.0


class AudioDeepfakeDetector:
    """Hugging Face ensemble plus lightweight handcrafted feature fusion."""

    def __init__(
        self,
        model_id: str | None = None,
        sample_rate: int = TARGET_SAMPLE_RATE,
        chunk_seconds: float = 3.0,
        local_files_only: bool = False,
        min_rms: float = 0.002,
        enable_handcrafted_features: bool = True,
        model_weight: float | None = None,
        behavior_weight: float = 0.18,
        model_weights: list[float] | None = None,
    ):
        self.model_ids = _resolve_model_ids(model_id)
        if model_weights is None:
            model_weights = _parse_weights(os.getenv("TRUE_TONE_MODEL_WEIGHTS"), len(self.model_ids))
        if len(model_weights) != len(self.model_ids):
            raise ValueError(f"Expected {len(self.model_ids)} model weights, got {len(model_weights)}.")

        self.model_id = ",".join(self.model_ids)
        self.sample_rate = sample_rate
        self.chunk_seconds = chunk_seconds
        self.min_rms = min_rms
        self.enable_handcrafted_features = enable_handcrafted_features
        self.model_weight = 1.0 - behavior_weight if model_weight is None else model_weight
        self.behavior_weight = behavior_weight
        self.last_feature_summaries: list[dict[str, float]] = []

        # Auto-detect GPU: use CUDA if available, otherwise CPU
        if torch.cuda.is_available():
            device = 0  # first CUDA device
            gpu_name = torch.cuda.get_device_name(0)
            print(f"Loading model: {self.model_id} on GPU ({gpu_name}) ...")
        else:
            device = -1  # CPU
            print(f"Loading {len(self.model_ids)} detector model(s) on CPU ...")

        self.model_branches: list[ModelBranch] = []
        for branch_id, branch_weight in zip(self.model_ids, model_weights):
            location = f"GPU ({gpu_name})" if torch.cuda.is_available() else "CPU"
            print(f"  Loading model: {branch_id} on {location} weight={branch_weight:.3f} ...")
            classifier = pipeline(
                task="audio-classification",
                model=branch_id,
                device=device,
                dtype=torch.float32,
                local_files_only=local_files_only,
            )
            self.model_branches.append(
                ModelBranch(model_id=branch_id, weight=float(branch_weight), classifier=classifier)
            )

    def predict_samples(self, samples: np.ndarray) -> tuple[float, list[float], list[list[dict[str, Any]]]]:
        chunks = chunk_audio(samples, self.sample_rate, self.chunk_seconds)
        chunk_scores = []
        raw_outputs = []
        feature_summaries: list[dict[str, float]] = []

        for chunk in chunks:
            rms = float(np.sqrt(np.mean(np.square(chunk)))) if chunk.size else 0.0
            if rms < self.min_rms:
                chunk_scores.append(0.0)
                raw_outputs.append([{"label": "silence", "score": 1.0}])
                feature_summaries.append({})
                continue

            model_score, labels = self._predict_model_ensemble(chunk)
            feature_summary = self._extract_feature_summary(chunk)
            behavior_score = float(feature_summary.get("synthetic_behavior_score", 0.0))
            fused_score = self._fuse_model_and_behavior(model_score, behavior_score)

            if feature_summary:
                labels.append({"label": "model_ensemble", "score": model_score})
                labels.append({"label": "handcrafted_behavior", "score": behavior_score})
                labels.append({"label": "fused_ai_probability", "score": fused_score})

            raw_outputs.append(labels)
            chunk_scores.append(fused_score)
            feature_summaries.append(feature_summary)

        self.last_feature_summaries = feature_summaries
        return float(np.mean(chunk_scores)), chunk_scores, raw_outputs

    def _predict_model_ensemble(self, chunk: np.ndarray) -> tuple[float, list[dict[str, Any]]]:
        branch_scores = []
        labels: list[dict[str, Any]] = []
        for branch in self.model_branches:
            output = branch.classifier({"array": chunk, "sampling_rate": self.sample_rate}, top_k=None)
            branch_labels = list(output)
            branch_score = _score_to_ai_probability(branch_labels)
            branch_scores.append((branch_score, branch.weight))
            labels.append(
                {
                    "label": f"branch::{branch.model_id}",
                    "score": branch_score,
                    "weight": branch.weight,
                    "raw": branch_labels,
                }
            )

        total_weight = sum(weight for _, weight in branch_scores)
        if total_weight <= 0.0:
            return 0.0, labels
        score = sum(score * weight for score, weight in branch_scores) / total_weight
        return float(np.clip(score, 0.0, 1.0)), labels

    def _extract_feature_summary(self, chunk: np.ndarray) -> dict[str, float]:
        if not self.enable_handcrafted_features:
            return {}
        features = extract_audio_features(chunk, self.sample_rate)
        return features.as_dict()

    def _fuse_model_and_behavior(self, model_score: float, behavior_score: float) -> float:
        if not self.enable_handcrafted_features:
            return float(np.clip(model_score, 0.0, 1.0))
        total = self.model_weight + self.behavior_weight
        if total <= 0.0:
            return float(np.clip(model_score, 0.0, 1.0))
        fused = (self.model_weight * model_score + self.behavior_weight * behavior_score) / total
        return float(np.clip(fused, 0.0, 1.0))

    def predict_file(self, path: str | Path) -> PredictionResult:
        audio = load_wav(path, target_sample_rate=self.sample_rate)
        ai_probability, chunk_scores, raw_outputs = self.predict_samples(audio.samples)
        return PredictionResult(
            source=Path(path),
            ai_probability=ai_probability,
            chunk_scores=chunk_scores,
            raw_outputs=raw_outputs,
            model_id=self.model_id,
            feature_summaries=self.last_feature_summaries,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify a static WAV/audio file for AI voice likelihood.")
    parser.add_argument("audio_file", help="Path to a WAV/audio file.")
    parser.add_argument(
        "--model-id",
        default=None,
        help=f"HF model id, or comma-separated ids for an ensemble. Default: {DEFAULT_MODEL_ID}",
    )
    parser.add_argument("--model-weights", default=None, help="Comma-separated model branch weights.")
    parser.add_argument("--chunk-seconds", type=float, default=3.0, help="Seconds per inference chunk.")
    parser.add_argument("--local-files-only", action="store_true", help="Use only cached model files.")
    parser.add_argument("--min-rms", type=float, default=0.002, help="Treat quieter chunks as silence.")
    args = parser.parse_args()

    detector = AudioDeepfakeDetector(
        model_id=args.model_id,
        chunk_seconds=args.chunk_seconds,
        local_files_only=args.local_files_only,
        min_rms=args.min_rms,
        model_weights=_parse_weights(args.model_weights, len(_resolve_model_ids(args.model_id)))
        if args.model_weights
        else None,
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
