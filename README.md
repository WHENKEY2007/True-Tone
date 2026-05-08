# True-Tone

True-Tone is an advanced audio processing application designed for real-time audio capture, processing, and visualization from both microphone and system audio sources.

## Project Structure

```
true-tone/
├── app.py                 # Main application entry point
├── audio/                 # Audio processing modules
│   ├── mic_capture.py     # Microphone capture functionality
│   ├── system_capture.py  # System audio capture
│   ├── buffer.py          # Audio buffer management
│   └── wav_utils.py       # WAV file utilities
├── ui/                    # User interface modules
│   └── dashboard.py       # Main dashboard interface
├── test_chunks/           # Test audio files directory
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## Features

- Real-time microphone audio capture
- System audio capture and processing
- Audio buffering and queue management
- WAV file I/O utilities
- Interactive dashboard UI
- Audio visualization

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd true-tone
```

2. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit dashboard:
```bash
.\.venv\Scripts\streamlit.exe run app.py
```

## Day 1: Capture Checks

List microphone devices:
```bash
.\.venv\Scripts\python.exe audio\mic_capture.py --list-devices
```

Record three microphone chunks:
```bash
.\.venv\Scripts\python.exe audio\mic_capture.py --chunks 3 --device 1
```

List system speaker loopback devices:
```bash
.\.venv\Scripts\python.exe audio\system_capture.py --list-devices
```

Record three system audio chunks:
```bash
.\.venv\Scripts\python.exe audio\system_capture.py --chunks 3 --device 0
```

## Day 2: Static File Detection

Run AI-voice probability inference on a saved WAV file:
```bash
.\.venv\Scripts\python.exe -m inference.detector test_system_chunks\system_chunk_003_device_0_1778085454929809100.wav
```

After the model has downloaded once, run from the local Hugging Face cache:
```bash
.\.venv\Scripts\python.exe -m inference.detector test_system_chunks\system_chunk_003_device_0_1778085454929809100.wav --local-files-only
```

The detector prints an `AI probability` from `0.0` to `1.0` and the per-chunk scores used to calculate it.

Compare several candidate detector models on the same audio file:
```bash
.\.venv\Scripts\python.exe tools\compare_models.py test_chunks\your_sample.wav
```

## Day 3: Live Terminal Pipeline

Run live microphone capture through speech gating and the detector:
```bash
.\.venv\Scripts\python.exe live_pipeline.py --source mic --device 1
```

Run system-audio loopback through the same pipeline:
```bash
.\.venv\Scripts\python.exe live_pipeline.py --source system --device 0
```

Verify the wiring without live audio hardware by replaying a saved WAV file:
```bash
.\.venv\Scripts\python.exe live_pipeline.py --source-file test_chunks\your_sample.wav --chunks 3 --local-files-only
```

The Day 3 milestone is met when the terminal prints one line per chunk with `ai_probability`, `rms`, `peak`, speech/silence state, and latency.

## Day 4: Streamlit Dashboard

Run the product dashboard:
```bash
.\.venv\Scripts\streamlit.exe run app.py
```

The dashboard supports uploaded audio replay, microphone capture, and system-audio loopback. It shows the current AI probability, signal stats, per-chunk latency, a rolling history graph, and warning banners when the score crosses the configured threshold.

## Requirements

See `requirements.txt` for a complete list of dependencies.

## License

[Add license information here]

## Contributing

[Add contribution guidelines here]
