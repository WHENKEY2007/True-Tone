"""
Download a small balanced real/fake audio sample set from public datasets.

Default source:
  garystafford/deepfake-audio-detection on Hugging Face

The script intentionally downloads a bounded subset so local testing stays
fast and the repository does not balloon into a full dataset mirror.
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from huggingface_hub import HfApi, hf_hub_download


DEFAULT_REPO_ID = "garystafford/deepfake-audio-detection"
SUPPORTED_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}


def _select_files(files: list[str], prefix: str, count: int, seed: int) -> list[str]:
    candidates = [
        item for item in files
        if item.startswith(f"{prefix}/") and Path(item).suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    rng = random.Random(seed)
    rng.shuffle(candidates)
    return sorted(candidates[:count])


def _copy_hf_file(repo_id: str, repo_file: str, destination_root: Path) -> Path:
    cached_path = Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=repo_file,
            repo_type="dataset",
        )
    )
    destination = destination_root / repo_file
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cached_path, destination)
    return destination


def _write_source_note(destination_root: Path, repo_id: str, downloaded: list[str]) -> None:
    note = destination_root / "SOURCES.md"
    lines = [
        "# Online Audio Sample Sources",
        "",
        f"- Hugging Face dataset: `{repo_id}`",
        "- Dataset page: https://huggingface.co/datasets/garystafford/deepfake-audio-detection",
        "- Labels are preserved through `real/` and `fake/` folders.",
        "- Keep this directory out of git unless you intentionally want to version sample media.",
        "",
        "Downloaded files:",
        "",
    ]
    lines.extend(f"- `{item}`" for item in downloaded)
    note.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download balanced real/fake online audio samples.")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="Hugging Face dataset repo id.")
    parser.add_argument(
        "--output-dir",
        default="test_audio/online_samples",
        help="Destination directory for downloaded samples.",
    )
    parser.add_argument("--per-class", type=int, default=20, help="Number of real and fake files to download.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for sample selection.")
    args = parser.parse_args()

    if args.per_class <= 0:
        raise SystemExit("--per-class must be positive")

    destination_root = Path(args.output_dir)
    api = HfApi()
    files = api.list_repo_files(args.repo_id, repo_type="dataset")
    selected = (
        _select_files(files, "real", args.per_class, args.seed)
        + _select_files(files, "fake", args.per_class, args.seed + 1)
    )

    if not selected:
        raise SystemExit(f"No downloadable real/fake audio files found in {args.repo_id}")

    print(f"Downloading {len(selected)} files from {args.repo_id} to {destination_root}")
    downloaded: list[str] = []
    for repo_file in selected:
        destination = _copy_hf_file(args.repo_id, repo_file, destination_root)
        downloaded.append(repo_file)
        print(f"  {repo_file} -> {destination}")

    _write_source_note(destination_root, args.repo_id, downloaded)
    print()
    print(f"Done. Real files: {sum(item.startswith('real/') for item in downloaded)}")
    print(f"Done. Fake files: {sum(item.startswith('fake/') for item in downloaded)}")


if __name__ == "__main__":
    main()
