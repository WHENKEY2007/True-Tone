# 🎙️ True-Tone — Real-Time AI Voice Deepfake Detection

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Inference-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**True-Tone** is a real-time AI voice deepfake detection system that captures live audio from microphone or system speakers, processes it through a multi-stage detection pipeline with speech activity detection, and displays live probability scores via a Streamlit dashboard. It estimates the likelihood of synthetic speech in real time, running entirely on CPU.

> **Design Philosophy:** True-Tone deliberately avoids enterprise-grade cloud infrastructure and GPU dependencies in favor of a robust, standalone desktop application. The system estimates the *likelihood* of synthetic speech rather than claiming perfect detection.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🎤 **Real-Time Microphone Capture** | Continuous 16 kHz mono audio capture via `sounddevice` with configurable overlapping 3-second windows (default 1-second stride) |
| 🔊 **System Audio Loopback** | Capture speaker output through `soundcard` for monitoring playback or call audio |
| 📁 **WAV File Ingestion** | Drop-in file replay matching the live capture contract, with upload support in the dashboard |
| 🧠 **AI Detection Engine** | Pre-trained Wav2Vec2-based Hugging Face audio classification models with weighted ensemble support |
| 🎯 **Silero VAD Integration** | Neural voice activity detection via Silero VAD (`SileroSpeechGate`), plus a fast energy-based gate and a hybrid two-stage gate |
| 📊 **Streamlit Dashboard** | Live probability meter, waveform visualization, score history chart, and warning banners |
| 🔀 **Multi-Threaded Pipeline** | Separate capture and inference threads connected via `threading` + `queue` for non-blocking data flow |
| 📈 **Temporal Confidence Aggregation** | Session-aware EMA smoothing, trend analysis, uncertainty estimation, hysteresis states, and silence decay |
| 🔬 **Handcrafted Feature Fusion** | Spectral entropy, pitch drift, jitter, shimmer, HNR, cadence consistency, breathiness — fused with neural scores |
| ⚡ **CPU-Only Execution** | Runs on consumer hardware with no GPU requirement (auto-detects CUDA if available) |
| 🔧 **Overlapping Audio Chunks** | Configurable chunk overlap (default 2 seconds) for smoother, lower-latency detection |
| ⚠️ **Real-Time Warning Alerts** | Automatic red warning banner when AI probability exceeds configurable threshold |

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                        True-Tone Pipeline                           │
│                                                                     │
│  ┌─────────────┐    ┌──────────┐    ┌─────────────┐    ┌─────────┐ │
│  │   Capture    │───▶│  Queue   │───▶│  Inference   │───▶│  Scores │ │
│  │   Thread     │    │ (bounded)│    │   Thread     │    │  Deque  │ │
│  └─────────────┘    └──────────┘    └─────────────┘    └────┬────┘ │
│        │                                   │                 │      │
│  ┌─────┴──────┐              ┌─────────────┴──────┐   ┌─────┴────┐ │
│  │ Mic/System/ │              │  Speech Gate (VAD) │   │ Streamlit│ │
│  │ WAV Replay  │              │  + AI Detector     │   │ Dashboard│ │
│  │             │              │  + Feature Fusion   │   │   (UI)  │ │
│  └─────────────┘              │  + Temporal Agg.   │   └──────────┘ │
│                               └────────────────────┘                │
└──────────────────────────────────────────────────────────────────────┘
```

The pipeline runs two background threads orchestrated by `PipelineOrchestrator`:

1. **Capture Thread** — reads audio from the selected source (`MicrophoneCapture`, `SystemAudioCapture`, or `WAVFileReplay`) and enqueues 3-second chunks with configurable overlap.
2. **Inference Thread** — dequeues chunks, runs them through the speech gate (energy-based, Silero VAD, or hybrid), performs AI detection via the Hugging Face ensemble, extracts handcrafted behavioral features, fuses scores, and publishes `ScoreRecord` results through the temporal confidence aggregator.

The Streamlit dashboard polls `PipelineOrchestrator.snapshot()` to read results without blocking — a thread-safe, lock-minimal design.

---

## 📂 Project Structure

```
True-Tone/
├── app.py                          # Main entry point (dashboard / terminal / file modes)
├── live_pipeline.py                # Standalone threaded live detection pipeline
├── requirements.txt                # Python dependencies
│
├── audio/                          # Audio capture and I/O modules
│   ├── mic_capture.py              # Real-time microphone capture (sounddevice)
│   ├── system_capture.py           # System audio loopback capture (soundcard)
│   ├── wav_loader.py               # WAV file replay (drop-in capture replacement)
│   ├── wav_utils.py                # Audio loading, saving, and chunking utilities
│   ├── vad.py                      # VAD: EnergySpeechGate, SileroSpeechGate, HybridSpeechGate
│   └── buffer.py                   # Thread-safe audio buffer with peek/drain operations
│
├── processing/                     # Audio preprocessing and feature extraction
│   ├── preprocessor.py             # Mono conversion, resampling, normalization, pad/trim
│   └── features.py                 # Handcrafted DSP features (spectral, pitch, behavioral)
│
├── inference/                      # AI detection engine
│   └── detector.py                 # HuggingFace ensemble detector with feature fusion
│
├── pipeline/                       # Pipeline orchestration
│   ├── orchestrator.py             # Multi-threaded capture → gate → detector orchestrator
│   └── temporal.py                 # Temporal confidence aggregation with hysteresis
│
├── ui/                             # Frontend
│   └── dashboard.py                # Streamlit dashboard with live meter and charts
│
├── tools/                          # Testing and evaluation utilities
│   ├── run_tests.py                # Batch accuracy testing with confusion matrix
│   ├── compare_models.py           # Side-by-side model comparison on a single file
│   ├── evaluate_streaming.py       # Session-level streaming evaluation
│   ├── tune_threshold.py           # Threshold optimization on labeled datasets
│   ├── tune_streaming_threshold.py # Threshold tuning with temporal aggregation
│   ├── download_online_samples.py  # Download public real/fake test samples
│   └── generate_test_audio.py      # Generate synthetic test audio for validation
│
├── ARCHITECTURAL_BLUEPRINT.md      # Detailed architecture and team execution plan
└── ROBUSTNESS_UPGRADE_PLAN.md      # Roadmap for ensemble expansion and robustness
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or later
- A working microphone (for live capture) or audio files for file-based analysis
- ~500 MB disk space for model download (cached after first run)

### Installation

```bash
# Clone the repository
git clone https://github.com/WHENKEY2007/True-Tone.git
cd True-Tone

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

### Quick Start — Streamlit Dashboard

```bash
streamlit run ui/dashboard.py
```

This launches the full interactive dashboard where you can:
- Select audio source (Microphone / System Audio / WAV File upload)
- Start/stop live detection
- View the real-time AI probability meter and waveform
- Monitor score history with threshold visualization
- Receive warning alerts when synthetic speech is detected

### Quick Start — Terminal Mode

```bash
# Live microphone detection
python app.py --mode terminal --source mic

# System audio loopback detection
python app.py --mode terminal --source system --device 0

# File analysis
python app.py --mode file --source-file path/to/audio.wav
```

---

## 🎛️ Usage Guide

### 1. Audio Capture

**List available devices:**
```bash
python -m audio.mic_capture --list-devices        # Microphone devices
python -m audio.system_capture --list-devices      # System loopback devices
```

**Record test chunks:**
```bash
python -m audio.mic_capture --chunks 3 --device 1
python -m audio.system_capture --chunks 3 --device 0
```

### 2. Static File Detection

```bash
# Single file analysis
python -m inference.detector path/to/audio.wav

# Use cached model (offline)
python -m inference.detector path/to/audio.wav --local-files-only

# Compare multiple detector models
python tools/compare_models.py path/to/audio.wav
```

### 3. Live Terminal Pipeline

```bash
# Microphone with default settings (3s chunks, 2s overlap = 1s stride)
python live_pipeline.py --source mic --device 1

# System audio
python live_pipeline.py --source system --device 0

# Replay a file through the full live pipeline
python live_pipeline.py --source-file test_chunks/sample.wav --chunks 3 --local-files-only
```

Each chunk prints: `ai_probability`, `raw_probability`, `decision_state`, `uncertainty`, `rms`, `peak`, speech/silence label, and latency.

### 4. Ensemble Detection

True-Tone supports weighted multi-model ensembles:

```bash
# Two-model ensemble with custom weights
python -m inference.detector audio.wav \
  --model-id "DeepFake-Audio-Rangers/DeepfakeDetect_wav2vec2,Vansh180/deepfake-audio-wav2vec2" \
  --model-weights "0.6,0.4"

# Or via environment variables
export TRUE_TONE_MODEL_IDS="DeepFake-Audio-Rangers/DeepfakeDetect_wav2vec2,Vansh180/deepfake-audio-wav2vec2"
export TRUE_TONE_MODEL_WEIGHTS="0.6,0.4"
```

### 5. Batch Testing

```bash
# Test all files in a directory with accuracy metrics
python tools/run_tests.py test_audio/ --threshold 0.20

# Tune the optimal threshold on labeled data
python tools/tune_streaming_threshold.py test_audio/online_samples --local-files-only
```

---

## 🧩 Module Details

### Voice Activity Detection (`audio/vad.py`)

Three speech gate implementations:

| Gate | Method | Use Case |
|---|---|---|
| `EnergySpeechGate` | RMS + peak amplitude thresholds | Fast CPU-only pre-filter (default) |
| `SileroSpeechGate` | Silero VAD neural model (PyTorch Hub) | Higher accuracy in noisy environments |
| `HybridSpeechGate` | Energy pre-filter → Silero confirmation | Best of both: speed + accuracy |

### Temporal Confidence Aggregation (`pipeline/temporal.py`)

Raw per-chunk scores are noisy. The `TemporalConfidenceAggregator` stabilizes them using:

- **EMA smoothing** (α=0.35) with silence decay
- **Recency-weighted rolling average** across the speech memory window (30s)
- **Trend analysis** via linear regression on recent scores
- **Uncertainty estimation** from score variance
- **Anomaly scoring** from behavioral + cadence features
- **Hysteresis state machine**: `insufficient_speech` → `likely_real` / `suspicious` / `likely_synthetic`

### Handcrafted Feature Fusion (`processing/features.py`)

Complements neural detection with codec-surviving signals:

- Spectral: entropy, centroid, bandwidth, flatness, high-frequency ratio
- Prosody: pitch mean/std/drift (YIN), local jitter, local shimmer
- Temporal: energy variance/CV, pause ratio/count/duration CV, rhythm irregularity
- Voice quality: harmonic-to-noise ratio (dB), breathiness score, cadence consistency

These features are fused with the neural model score using configurable weights (default: 82% model, 18% behavioral).

---

## ⚙️ Configuration

| Parameter | Default | Description |
|---|---|---|
| `--chunk-seconds` | `3` | Duration of each audio chunk (seconds) |
| `--overlap` | `2.0` | Overlap between chunks (seconds); 2s overlap on 3s chunks = 1s stride |
| `--threshold` | `0.20` | AI probability alert threshold (tuned on streaming samples) |
| `--smoothing` | `5` | Moving average window for UI score smoothing |
| `--min-rms` | `0.002` | Minimum RMS energy for speech detection |
| `--min-peak` | `0.01` | Minimum peak amplitude for speech detection |
| `--gain` | `0.0` | Microphone gain boost in dB |
| `--model-id` | `DeepFake-Audio-Rangers/DeepfakeDetect_wav2vec2` | HuggingFace model ID(s) |
| `--model-weights` | Equal | Comma-separated ensemble branch weights |

---

## 🛡️ Technology Stack

| Layer | Technology |
|---|---|
| **Audio Capture** | `sounddevice` (microphone), `soundcard` (system loopback) |
| **Voice Activity Detection** | Silero VAD (neural), energy-based gate, hybrid two-stage |
| **AI Detection Engine** | Hugging Face `transformers` pipeline, Wav2Vec2-based classifiers |
| **Feature Extraction** | `scipy`, `librosa`, `numpy` — spectral, pitch, behavioral features |
| **Score Fusion** | Weighted neural + handcrafted behavioral fusion |
| **Temporal Aggregation** | EMA, rolling averages, trend, hysteresis state machine |
| **Pipeline Orchestration** | Python `threading` + `queue` (capture thread + inference thread) |
| **Frontend / Dashboard** | Streamlit with `matplotlib` visualizations |
| **Data Processing** | `numpy`, `scipy`, `torch`, `torchaudio` |

---

## 🗺️ Future Roadmap

See [ROBUSTNESS_UPGRADE_PLAN.md](ROBUSTNESS_UPGRADE_PLAN.md) for the detailed upgrade path.

**Planned enhancements:**

- 🔀 **Multi-Model Ensemble Expansion** — AASIST, RawNet2, WavLM, Whisper, and HuBERT embedding classifiers
- 📱 **Mobile Integration** — Lightweight ONNX-exported models for on-device inference
- 🌙 **Dark Mode** — Extended Streamlit theming with full dark/light mode toggle
- 📧 **Email/Webhook Alerts** — Configurable notification triggers when synthetic speech is sustained
- 🌐 **Browser Audio Monitoring** — WebRTC-based capture for in-browser detection
- 🔊 **Codec Robustness** — Augmentation and evaluation under Opus, MP3, G.711, G.722, AAC-LD conditions
- 📊 **Calibrated Stacking** — Meta-model (logistic regression → LightGBM) over all branch logits and features
- ⏱️ **Tiered Inference** — Cheap features every 1s, heavy models every 3–5s for latency/accuracy balance

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
