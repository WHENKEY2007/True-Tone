# Contributing to True-Tone

Thank you for your interest in contributing to True-Tone! This document provides guidelines and conventions for contributing to the project.

## Development Setup

1. Fork and clone the repository
2. Create a virtual environment and install dependencies:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
3. Run the dashboard to verify your setup:
   ```bash
   streamlit run ui/dashboard.py
   ```

## Project Conventions

### Code Style

- **Python 3.10+** type annotations throughout (use `from __future__ import annotations`)
- Docstrings on all public classes and functions (Google style)
- Module-level docstrings explaining purpose and usage
- Constants in `UPPER_SNAKE_CASE`
- Private helpers prefixed with `_`

### Module Contracts

All audio capture sources implement the same interface:

```python
class CaptureSource:
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def read(self) -> np.ndarray: ...  # float32 mono, 16 kHz
```

All speech gates implement:

```python
class SpeechGate:
    def process(self, samples: np.ndarray) -> SpeechGateResult: ...
```

### Architecture

- **`audio/`** — Capture and I/O only. No AI inference.
- **`processing/`** — Preprocessing and feature extraction. Stateless where possible.
- **`inference/`** — AI model loading and prediction. No audio I/O.
- **`pipeline/`** — Orchestration, threading, and temporal aggregation.
- **`ui/`** — Streamlit dashboard. Reads from the pipeline, never writes audio.
- **`tools/`** — CLI utilities for testing, evaluation, and tuning.

### Threading Rules

- Audio capture and AI inference run in separate daemon threads.
- Communication happens via bounded `queue.Queue` instances.
- UI reads pipeline state via `PipelineOrchestrator.snapshot()` (lock-minimal).
- Never block the capture thread on inference latency.

## Adding a New Detection Model

1. Ensure the model is hosted on Hugging Face as an `audio-classification` pipeline.
2. Test it standalone: `python -m inference.detector audio.wav --model-id your/model-id`
3. Add it to an ensemble: `--model-id "existing,your/model-id" --model-weights "0.6,0.4"`
4. Evaluate: `python tools/run_tests.py test_audio/ --model-id your/model-id`
5. Tune thresholds: `python tools/tune_streaming_threshold.py test_audio/online_samples`

## Adding a New Speech Gate

1. Create a class in `audio/vad.py` with a `process(samples) -> SpeechGateResult` method.
2. Export it in `audio/__init__.py`.
3. Add a selection option in `ui/dashboard.py` sidebar controls.

## Pull Request Guidelines

- One feature or fix per PR
- Include a clear description of what changed and why
- Ensure the pipeline runs without crashing in at least one mode (terminal, file, or dashboard)
- Update docstrings and README if adding new user-facing features

## Reporting Issues

When reporting bugs, please include:
- Operating system and Python version
- Audio device configuration (output of `--list-devices`)
- Full error traceback
- Steps to reproduce
