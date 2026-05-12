# True-Tone

True-Tone is a local, CPU-friendly audio application for capturing microphone or system audio, running AI-voice probability inference, and visualizing the result in a dashboard.

## Project Structure

```text
true-tone/
|-- app.py                 # Main application entry point
|-- audio/                 # Audio processing modules
|   |-- mic_capture.py     # Microphone capture functionality
|   |-- system_capture.py  # System audio capture
|   |-- buffer.py          # Audio buffer management
|   `-- wav_utils.py       # WAV file utilities
|-- inference/             # AI voice detector integration
|   `-- detector.py        # Hugging Face audio-classification wrapper
|-- tests/                 # Unit tests for core behavior
|-- ui/                    # User interface modules
|   `-- dashboard.py       # Streamlit dashboard interface
|-- requirements.txt       # Python dependencies
`-- README.md              # This file
```

## Features

- Real-time microphone audio capture
- System audio capture and processing
- Audio buffering and queue management
- WAV file I/O utilities
- Hugging Face Wav2Vec2-based AI voice probability inference
- Optional averaged detector ensembles via `TRUE_TONE_MODEL_IDS`
- Interactive Streamlit dashboard UI with probability meter, waveform view, warning banner, and event log
- Audio visualization

## Model Support

The implemented detector uses Hugging Face `audio-classification` checkpoints on CPU, with `DeepFake-Audio-Rangers/DeepfakeDetect_wav2vec2` as the default. You can test a single model with `--model-id` or run a simple averaged ensemble with `--model-ids` / `TRUE_TONE_MODEL_IDS`.

RawNetLite is a candidate lightweight architecture in the planning document, but this repository does not currently ship a RawNetLite implementation or weights.

## Installation

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

Run the application:

```bash
python app.py
```

Run the Streamlit dashboard directly:

```bash
streamlit run app.py
```

## Capture Checks

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

## Static File Detection

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

Run unit tests:

```bash
python -m unittest discover
```

## Requirements

See `requirements.txt` for a complete list of dependencies.

