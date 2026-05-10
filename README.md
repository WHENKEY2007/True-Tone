# 🎙️ True Tone — AI-Powered Voice Authenticity Detection

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Inference-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**TrueTone** is an AI-powered voice authenticity detection system that identifies whether an audio clip is real or AI-generated. It analyzes speech patterns, tone variations, and audio characteristics using machine learning techniques to improve trust and security in digital communication.

> With the rapid growth of AI-generated voice technologies, detecting fake or manipulated audio has become a major challenge in digital communication. TrueTone addresses this by providing fast and accurate detection results, helping improve security, reduce misinformation, and build trust in digital audio communication.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🔐 **Authentication** | Secure session-based login with role-based access (admin / tester / demo) |
| 🎤 **Real-Time Microphone Capture** | Continuous 16 kHz mono audio capture via `sounddevice` with sliding-window chunk generation |
| 📁 **WAV Audio Upload** | File upload support through the Streamlit dashboard with full pipeline replay |
| 🔊 **System Audio Loopback** | Capture speaker output through `soundcard` for monitoring playback or call audio |
| 🧠 **AI Detection Engine** | Lightweight Wav2Vec2 / RawNetLite-compatible models via Hugging Face, optimized for CPU inference |
| 🎯 **Silero VAD Integration** | Speech detection using Silero VAD for filtering non-speech audio, plus energy-based and hybrid gates |
| 📊 **Streamlit Dashboard** | Real-time probability meter, waveform visualization, historical probability graph, and warning alerts |
| ⚠️ **Warning Alerts** | Automatic red warning banner when AI probability exceeds configurable threshold |
| 🔀 **Multi-Threaded Architecture** | Separate threads for audio capture, VAD, AI inference, and UI updates via Python `threading` and `queue` |
| 📈 **Temporal Aggregation** | False-positive reduction through EMA smoothing, trend analysis, and hysteresis state machine |
| 🔬 **Audio Feature Analysis** | Spectral entropy, pitch drift, jitter, shimmer, HNR, cadence consistency, breathiness scoring |
| 📋 **Detection Event Logging** | Event log table and session analytics visualization (admin/tester roles) |
| ⚡ **CPU-Only Execution** | Runs on standard consumer hardware without GPU (auto-detects CUDA if available) |
| 🧪 **Testing Suite** | Unit tests, batch testing tools, threshold tuning, and model comparison utilities |

---

## 🏗️ System Architecture

TRUE TONE uses a modular real-time architecture to capture microphone or uploaded audio and process it in small chunks. The system preprocesses the audio using normalization, silence filtering, and resampling before sending it to AI models like RawNetLite and Wav2Vec2 for synthetic voice detection. Detection results are displayed on a Streamlit dashboard with live alerts and waveform analysis, while Python threading and queues enable smooth real-time processing and responsive UI updates.

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         TRUE TONE — Pipeline Architecture                    │
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌─────────────────┐    ┌─────────┐ │
│  │  Audio Source │───▶│ Thread Queue │───▶│ Inference Thread │───▶│ Scores  │ │
│  │  (Capture     │    │  (bounded)   │    │                 │    │ (deque) │ │
│  │   Thread)     │    └──────────────┘    │ ┌─────────────┐ │    └────┬────┘ │
│  └──────┬───────┘                        │ │ Silero VAD  │ │         │      │
│         │                                │ │ Speech Gate │ │    ┌────┴────┐ │
│  ┌──────┴───────┐                        │ └──────┬──────┘ │    │Streamlit│ │
│  │ • Microphone  │                        │ ┌──────┴──────┐ │    │Dashboard│ │
│  │ • System Audio│                        │ │ Wav2Vec2 /  │ │    │  (UI)   │ │
│  │ • WAV Upload  │                        │ │ RawNetLite  │ │    │         │ │
│  │ • File Replay │                        │ │  Detector   │ │    │• Meter  │ │
│  └──────────────┘                        │ └──────┬──────┘ │    │• Wave   │ │
│                                          │ ┌──────┴──────┐ │    │• Graph  │ │
│                                          │ │  Feature    │ │    │• Alerts │ │
│                                          │ │  Fusion +   │ │    │• Logs   │ │
│                                          │ │  Temporal   │ │    └─────────┘ │
│                                          │ │  Aggregator │ │                 │
│                                          │ └─────────────┘ │                 │
│                                          └─────────────────┘                 │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Processing Pipeline

1. **Audio Capture** — 3-second overlapping chunks (48,000 samples at 16 kHz) via `sounddevice`
2. **Preprocessing** — Mono conversion, resampling to 16 kHz, volume normalization, noise filtering
3. **Speech Detection** — Silero VAD filters non-speech audio; energy-based gate as fast pre-filter
4. **AI Inference** — Lightweight Wav2Vec2 / RawNetLite models generate AI probability score (0.0–1.0)
5. **Feature Fusion** — Handcrafted DSP features fused with neural model scores
6. **Temporal Aggregation** — EMA smoothing, rolling averages, hysteresis for false-positive reduction
7. **Dashboard Display** — Real-time probability meter, waveform, score history, warning alerts

---

## 📂 Project Structure

```
True-Tone/
├── app.py                          # Main entry point (dashboard / terminal / file modes)
├── live_pipeline.py                # Standalone threaded live detection pipeline
├── requirements.txt                # Python dependencies
│
├── audio/                          # Audio capture and processing modules
│   ├── mic_capture.py              # Real-time microphone capture (sounddevice)
│   ├── system_capture.py           # System audio loopback capture (soundcard)
│   ├── wav_loader.py               # WAV file ingestion and replay
│   ├── wav_utils.py                # Audio loading, saving, chunking utilities
│   ├── vad.py                      # VAD: EnergySpeechGate, SileroSpeechGate, HybridSpeechGate
│   └── buffer.py                   # Thread-safe audio buffer with sliding window
│
├── processing/                     # Audio processing and feature extraction
│   ├── preprocessor.py             # Mono conversion, resampling, normalization, pad/trim
│   └── features.py                 # Handcrafted DSP/behavioral features
│
├── inference/                      # AI detection engine
│   └── detector.py                 # HuggingFace Wav2Vec2/RawNetLite ensemble detector
│
├── pipeline/                       # Multi-threaded pipeline orchestration
│   ├── orchestrator.py             # Capture → VAD → Detector orchestrator (threading + queue)
│   └── temporal.py                 # Temporal confidence aggregation with hysteresis
│
├── ui/                             # Frontend dashboard
│   └── dashboard.py                # Streamlit dashboard with auth, live meter, logs, analytics
│
├── tools/                          # Testing, evaluation, and tuning utilities
│   ├── run_tests.py                # Batch accuracy testing with confusion matrix
│   ├── compare_models.py           # Side-by-side model comparison
│   ├── evaluate_streaming.py       # Session-level streaming evaluation
│   ├── tune_threshold.py           # Threshold optimization on labeled datasets
│   ├── tune_streaming_threshold.py # Threshold tuning with temporal aggregation
│   ├── download_online_samples.py  # Download public real/fake test samples
│   └── generate_test_audio.py      # Generate synthetic test audio
│
├── tests/                          # Unit test suite
│   └── test_core.py                # Tests for preprocessing, VAD, chunking, features
│
├── ARCHITECTURAL_BLUEPRINT.md      # Detailed architecture and 6-day execution plan
├── ROBUSTNESS_UPGRADE_PLAN.md      # Roadmap for ensemble expansion and robustness
├── CONTRIBUTING.md                 # Development and contribution guidelines
└── LICENSE                         # MIT License
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or later
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

### Launch the Streamlit Dashboard

```bash
streamlit run ui/dashboard.py
```

**Login with demo credentials:**

| Role | Username | Password | Access |
|---|---|---|---|
| Admin | `admin` | `truetone2025` | Full access — model config, logs, threshold tuning |
| Tester | `tester` | `testpass` | Detection + event logs + analytics |
| Demo | `demo` | `demo` | Detection only — simplified interface |

After login, select an audio source (Microphone / System Audio / WAV File) and press **▶️ Start** to begin real-time detection.

### Terminal Mode

```bash
# Live microphone detection
python app.py --mode terminal --source mic

# System audio loopback
python app.py --mode terminal --source system --device 0

# Analyze a single audio file
python app.py --mode file --source-file path/to/audio.wav
```

---

## 📋 Requirements Fulfillment

### Functional Requirements

| Requirement | Status | Implementation |
|---|---|---|
| **Authentication** | ✅ | Session-based login with role credentials (`ui/dashboard.py`) |
| **Role-based access** | ✅ | Admin / Tester / Demo roles with permission controls |
| **Real-time microphone capture** | ✅ | `audio/mic_capture.py` — `sounddevice` with overlapping windows |
| **WAV audio upload** | ✅ | Streamlit file uploader in dashboard sidebar |
| **Sliding-window chunk generation** | ✅ | 3-second chunks with configurable overlap (default 2s = 1s stride) |
| **Speech detection (Silero VAD)** | ✅ | `audio/vad.py` — `SileroSpeechGate`, `HybridSpeechGate`, `EnergySpeechGate` |
| **Noise filtering & silence removal** | ✅ | Energy gate pre-filter + Silero VAD speech/silence classification |
| **Audio normalization & 16 kHz resampling** | ✅ | `processing/preprocessor.py` — mono, resample, peak-normalize |
| **AI probability score (0.0–1.0)** | ✅ | `inference/detector.py` — per-chunk and aggregated scores |
| **CPU-only execution** | ✅ | Default CPU inference, auto-detects CUDA if available |
| **Real-time probability meter** | ✅ | Large score display with color coding in dashboard |
| **Waveform visualization** | ✅ | Live matplotlib waveform plot in dashboard |
| **Historical probability graph** | ✅ | Score history chart with threshold line |
| **Warning alerts** | ✅ | Red/yellow/green alerts based on configurable threshold |
| **Detection event logging** | ✅ | Event log table + session analytics (tester/admin roles) |
| **False-positive reduction** | ✅ | `pipeline/temporal.py` — EMA, rolling average, hysteresis |

### Technical Requirements

| Requirement | Status | Implementation |
|---|---|---|
| **Python + Streamlit** | ✅ | Python 3.9+, Streamlit dashboard |
| **CPU hardware, no GPU** | ✅ | All inference on CPU by default |
| **Audio as NumPy arrays** | ✅ | float32 mono arrays throughout |
| **3-second chunks (48,000 samples at 16 kHz)** | ✅ | Configurable via `--chunk-seconds` |
| **Silero VAD integration** | ✅ | `SileroSpeechGate` in `audio/vad.py` |
| **Wav2Vec2 / RawNetLite models** | ✅ | HuggingFace `audio-classification` pipeline |
| **1–2 second inference latency** | ✅ | Measured latency per chunk displayed in dashboard |
| **Multi-threaded (threading + queue)** | ✅ | `pipeline/orchestrator.py` — capture thread + inference thread |
| **Separate threads for capture, VAD, inference, UI** | ✅ | Orchestrator manages thread lifecycle |

---

## 🛡️ Technologies Used

| Technology | Purpose |
|---|---|
| **Python 3.9+** | Core application language |
| **Streamlit** | Interactive dashboard frontend |
| **PyTorch** | Neural network inference runtime |
| **Torchaudio** | Audio processing and transforms |
| **NumPy** | Array operations and audio data handling |
| **SciPy** | Signal processing, resampling, spectral analysis |
| **SoundDevice** | Real-time microphone audio capture |
| **SoundCard** | System audio loopback capture |
| **PyAudio** | Optional fallback audio capture support |
| **Silero VAD** | Neural voice activity detection |
| **Hugging Face Transformers** | Pre-trained model loading and inference pipeline |
| **Wav2Vec2** | Primary speech representation model for detection |
| **RawNetLite** | Lightweight waveform-based countermeasure model |
| **Threading & Queue** | Multi-threaded pipeline orchestration |
| **Matplotlib** | Waveform and score visualization |
| **Librosa** | Pitch estimation (YIN) and audio feature extraction |

---

## 🎛️ Usage Guide

### Audio Capture

```bash
# List available input devices
python -m audio.mic_capture --list-devices
python -m audio.system_capture --list-devices

# Record test chunks
python -m audio.mic_capture --chunks 3 --device 1
python -m audio.system_capture --chunks 3 --device 0
```

### Static File Detection

```bash
# Single file analysis with AI probability score
python -m inference.detector path/to/audio.wav

# Use cached model (offline mode)
python -m inference.detector path/to/audio.wav --local-files-only

# Compare multiple detector models
python tools/compare_models.py path/to/audio.wav
```

### Live Terminal Pipeline

```bash
# Microphone detection (3s chunks, 2s overlap = 1s stride)
python live_pipeline.py --source mic --device 1

# System audio loopback
python live_pipeline.py --source system --device 0

# File replay through full pipeline
python live_pipeline.py --source-file test_chunks/sample.wav --chunks 3
```

Each chunk outputs: `ai_probability`, `raw_probability`, `decision_state`, `uncertainty`, `rms`, `peak`, speech/silence label, and inference latency.

### Ensemble Detection

```bash
# Weighted multi-model ensemble
python -m inference.detector audio.wav \
  --model-id "DeepFake-Audio-Rangers/DeepfakeDetect_wav2vec2,Vansh180/deepfake-audio-wav2vec2" \
  --model-weights "0.6,0.4"
```

### Batch Testing & Threshold Tuning

```bash
# Test all files with accuracy metrics
python tools/run_tests.py test_audio/ --threshold 0.20

# Tune optimal threshold on labeled data
python tools/tune_streaming_threshold.py test_audio/online_samples
```

### Unit Tests

```bash
python -m pytest tests/ -v
# or
python -m unittest discover tests/ -v
```

---

## ⚙️ Configuration

| Parameter | Default | Description |
|---|---|---|
| `--chunk-seconds` | `3` | Duration of each audio chunk (seconds) |
| `--overlap` | `2.0` | Overlap between chunks (seconds); 2s on 3s = 1s stride |
| `--threshold` | `0.20` | AI probability alert threshold |
| `--smoothing` | `5` | Moving average window for score smoothing |
| `--min-rms` | `0.002` | Minimum RMS energy for speech detection |
| `--min-peak` | `0.01` | Minimum peak amplitude for speech detection |
| `--gain` | `0.0` | Microphone gain boost in dB |
| `--model-id` | `DeepFake-Audio-Rangers/DeepfakeDetect_wav2vec2` | HuggingFace model ID(s) |
| `--model-weights` | Equal | Comma-separated ensemble branch weights |

---

## 🔍 In Scope

- ✅ Real-time microphone audio capture
- ✅ WAV audio ingestion
- ✅ Speech activity detection and filtering (Silero VAD)
- ✅ AI-based synthetic speech detection
- ✅ Live probability scoring (0.0–1.0)
- ✅ Streamlit visualization dashboard
- ✅ Real-time warning alerts
- ✅ CPU-only execution support
- ✅ Modular audio processing pipeline
- ✅ Demo-ready live detection workflow
- ✅ False-positive reduction through temporal aggregation

## 🚫 Out of Scope

- Training or fine-tuning AI models
- GPU acceleration and TensorRT optimization
- Kubernetes or cloud-native deployment
- Zoom SDK or Recall.ai integration
- Enterprise-scale distributed infrastructure
- Speaker diarization
- Custom dataset creation
- Mobile application support
- WebSocket-based streaming infrastructure
- Multi-cloud deployment environments

---

## 🗺️ Future Enhancements

See [ROBUSTNESS_UPGRADE_PLAN.md](ROBUSTNESS_UPGRADE_PLAN.md) for the detailed roadmap.

- 📱 Mobile platform integration
- 📊 Enhanced waveform visualization
- 🎯 Advanced probability meter UI
- 📧 Email-based alert notifications
- 🌍 Multi-language voice detection
- 🌐 Real-time browser audio monitoring
- 🔀 Advanced AI ensemble detection models (AASIST, RawNet2, WavLM, Whisper, HuBERT)
- 🛡️ Voice spoof attack classification
- 🌙 Dark mode dashboard support

---

## 📝 Conclusion

TRUE TONE is a lightweight and practical AI voice deepfake detection platform designed for real-time synthetic speech analysis. The project emphasizes modular architecture, CPU-efficient execution, and rapid deployment while delivering meaningful live AI detection capabilities. Its primary strengths lie in real-time responsiveness, simplified deployment, and effective integration of modern audio classification pipelines. Future growth opportunities include improving detection accuracy, expanding supported audio sources, and integrating advanced AI ensemble techniques for stronger real-world resilience.

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
