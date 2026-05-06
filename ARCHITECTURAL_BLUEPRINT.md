# Architectural Blueprint for True Tone

## Real-Time Audio Deepfake Detection in Live Streaming Telemetry

## 1. Revised Project Scope and Objective

The primary objective of the True Tone project is to build a convincing, real-time AI voice detection demo that operates live on consumer hardware.

The project deliberately avoids enterprise-grade scalability, model training, and cloud infrastructure in favor of a robust, standalone desktop/web application executing entirely on a local CPU.

Instead of claiming to build a perfect deepfake detector, the system's technical posture is to estimate the likelihood of synthetic speech in real time.

The final deliverable will be a functional application that captures live microphone or system audio, processes it through a lightweight pre-trained AI classification pipeline, and displays a live confidence score and warning banner to the user.

## Out of Scope

The following items are strictly excluded:

- Training or fine-tuning neural network models.
- Integrating with the Zoom SDK, RTMS, or Recall.ai webhooks.
- Hardware acceleration such as TensorRT, GPUs, or Triton Inference Server.
- Cloud infrastructure, Kubernetes, or distributed deployment.
- Speaker diarization and custom dataset curation.

## 2. Final Technology Stack

The technology stack is aggressively simplified to ensure compatibility with a 6-day development timeline and CPU-only execution environments.

| Layer | Technology |
| --- | --- |
| Backend / Orchestration | Python standard library threading and queue |
| Frontend / UI | Streamlit |
| Audio Capture | sounddevice and pyaudio |
| Upstream Filtering | Silero VAD |
| AI Detection Engine | Pre-trained Hugging Face audio classification models, Wav2Vec2, or RawNetLite |
| Data Processing | numpy, scipy, torch, torchaudio |

RawNetLite is highly recommended for this stack because it is a convolutional-recurrent model explicitly tailored for lightweight, real-time execution directly on raw waveforms. This makes it highly competitive for CPU-only evaluation.

## 3. Team Structure and Responsibilities

To avoid merge conflicts and blocked dependencies within a 6-day sprint, the 6-person team must operate with strict modular separation.

## Person 1: Audio Capture Lead

### Tasks

- Implement continuous microphone and system audio capture using `sounddevice` or `pyaudio`.
- Construct a sliding window buffer to chunk the continuous stream.
- Write fallback logic for WAV file ingestion.

### Deliverables

- A stable Python module emitting raw, 3-second overlapping audio chunks.

## Person 2: VAD and Audio Processing

### Tasks

- Integrate Silero VAD to analyze incoming 3-second chunks.
- Filter out ambient room noise and silence.
- Normalize audio tensors to ensure consistent volume levels for the AI model.

### Deliverables

- A clean, speech-only stream of audio chunks.

## Person 3: AI Detection Integration

### Tasks

- Load the pre-trained PyTorch/Hugging Face model, such as RawNetLite or Wav2Vec2, onto the CPU.
- Build the inference pipeline that accepts a 3-second tensor and outputs a scalar probability score from `0.0` to `1.0`.

### Deliverables

- A deterministic prediction API/function ready for integration.

## Person 4: Backend and APIs, Pipeline Architect

### Tasks

- Connect the modules from Persons 1, 2, and 3 using Python `queue` and `threading`.
- Manage asynchronous data flow so the audio capture thread is not blocked by the AI inference thread.

### Deliverables

- The core stable live pipeline script that runs end to end.

## Person 5: Frontend / UI

### Tasks

- Build the Streamlit dashboard.
- Design a live probability meter.
- Add waveform visualization.
- Add an aggressive warning banner that triggers when the AI probability score exceeds a defined threshold.

### Deliverables

- A polished, working user interface.

## Person 6: Testing, Demo, and Presentation

### Tasks

- Curate a testing folder of clean human voices, ElevenLabs generations, and ChatGPT voices.
- Test the system for false positives.
- Tune the probabilistic aggregation logic, averaging scores over time to prevent UI flickering.
- Write the final presentation, architecture diagrams, and demo script.

### Deliverables

- A flawless, rehearsed live demo and pitch deck.

## 4. Six-Day Execution Plan

## Day 0: Setup Day

### Goal

Environment parity.

### Tasks

- Everyone installs Python.
- Set up a shared GitHub repository.
- Configure virtual environments with `venv`.
- Install the core dependencies:

```bash
pip install torch torchaudio transformers streamlit sounddevice numpy scipy silero-vad
```

- Establish folder structure.
- Establish team communication channels.

## Day 1: Audio Pipeline

### Goal

Can we continuously capture audio?

### Tasks

- Person 1 and Person 2 focus entirely on getting `sounddevice` to capture microphone and system audio.
- Segment live input into 3-second chunks.
- Save test WAV files to disk.

### Milestone

By the end of Day 1, live audio chunks are generated every 3 seconds without fail.

This is the critical path.

## Day 2: AI Detection Works

### Goal

Can the model classify audio?

### Tasks

- Person 3 loads the pre-trained model.
- Run inference exclusively on static uploaded WAV files.
- Extract a confidence score.

### Milestone

By the end of Day 2, the model successfully identifies uploaded AI voices versus human voices.

## Day 3: Connect Live Pipeline

### Goal

Live detection works.

### Tasks

- Person 4 wires Person 1's audio chunks through Person 2's VAD and into Person 3's model.

### Milestone

By the end of Day 3, speaking into the microphone prints live AI-probability scores directly to the terminal.

## Day 4: UI and Stability

### Goal

Looks like a real product.

### Tasks

- Person 5 connects the backend pipeline to the Streamlit UI.
- Add the live score.
- Add the historical graph.
- Add warning banners.
- Person 4 resolves threading crashes and latency spikes.

### Milestone

The Streamlit app runs without crashing and updates in real time.

## Day 5: Testing and Demo Optimization

### Goal

Hackathon-ready.

### Tasks

- Person 6 leads aggressive testing.
- Test human voices, noisy environments, and various AI generators.
- Implement a temporal averaging function, also called probabilistic aggregation, to smooth scores and reduce false positives.
- Tune the alert threshold.

### Milestone

The system reliably ignores human speech and flags synthetic speech without UI glitching.

## Day 6: Polish and Presentation

### Goal

Sell the idea.

### Tasks

- Finalize slides.
- Finalize the problem statement.
- Finalize architecture diagrams.
- Rehearse the live demo multiple times.
- Ensure a fallback pre-recorded video is ready.

## 5. Optimal Demo Flow and Risk Mitigation

Judges evaluate practical cybersecurity relevance and the smoothness of the real-time experience, not raw backend complexity.

## Three-Step Live Demo Sequence

## 1. Human Voice Baseline

The presenter speaks live into the microphone.

The Streamlit UI displays a stable, low AI-probability score, demonstrating that the system does not produce false positives on standard speech.

## 2. AI-Generated Voice: File Upload or Playback

The presenter plays an obvious AI-generated voice, such as a cloned celebrity or an ElevenLabs sample.

The confidence score immediately spikes on the UI.

## 3. Live Playback Attack

The presenter holds a phone playing a synthetic voice up to the microphone, simulating a live spoofing attack.

The Streamlit UI triggers a red warning banner:

```text
High Likelihood of Synthetic Speech Detected.
```

## Critical Risk Mitigation

## Risk 1: Live Audio Fails

If `sounddevice` loopbacks fail, rely on microphone capture.

If the microphone fails, seamlessly fall back to the Streamlit file upload utility to process static WAV files.

## Risk 2: Model Is Too Slow on CPU

Do not use heavy models.

Use lightweight pre-trained models like RawNetLite, which uses computationally inexpensive 1D convolutions and a GRU tailored specifically for embedded/CPU execution.

## Risk 3: False Positives or Jumpy UI

Do not display the raw inference score of a single 3-second chunk.

Implement a sliding window and calculate a moving average, or probabilistic aggregation, across the last 3 to 5 chunks so the UI meter moves smoothly and deliberately.

## Risk 4: UI Freezes

Keep the Streamlit application simple.

Run the audio and inference loops in a separate Python background thread that updates a shared state variable, which Streamlit simply reads and displays on a loop.

## 6. Day 1 Plan When Skipping Day 0

Since setup day is being skipped, Day 1 should compress Day 0 and Day 1 into a single practical milestone.

The only real goal for today is:

```text
By tonight, the project can continuously capture microphone audio and produce 3-second audio chunks reliably.
```

Do not touch AI detection yet unless audio capture is already stable. The live demo depends on the capture pipeline.

## Day 1 Priority Order

## 1. Set Up the Repository and Environment

Confirm or create the folder structure:

```text
True-Tone/
  audio/
    mic_capture.py
    wav_loader.py
  processing/
  inference/
  pipeline/
  ui/
  test_audio/
  requirements.txt
```

Create a virtual environment.

Install the minimum dependencies:

```bash
pip install sounddevice numpy scipy streamlit torch torchaudio
```

Other dependencies can be added later. Today does not need `transformers` or the detector model yet.

## 2. Get Microphone Capture Working

Person 1 should focus on `audio/mic_capture.py`.

Required behavior:

- Capture audio from the default microphone.
- Use a sample rate like `16000 Hz`.
- Use mono audio.
- Continuously collect raw PCM samples.
- Emit fixed-size chunks:
  - `3 seconds` per chunk.
  - Optional overlap later, such as `1.5 seconds`.

## 3. Save Captured Chunks to WAV Files

Before connecting more complex processing, prove the capture works.

Example target behavior:

```text
test_audio/chunk_001.wav
test_audio/chunk_002.wav
test_audio/chunk_003.wav
```

Then listen to the files manually. If they sound correct, Day 1 is on track.

## 4. Build a Fallback WAV Loader

This is important for demo safety.

Person 1 or Person 2 should add a simple utility that loads a `.wav` file and returns the same format as live microphone chunks.

That means the later pipeline can support both:

```python
get_audio_chunks_from_microphone()
get_audio_chunks_from_wav_file(path)
```

## 5. Start Basic Preprocessing

Person 2 can start basic preprocessing today.

Keep it simple:

- Convert stereo to mono.
- Resample to `16 kHz` if needed.
- Normalize volume.
- Ensure every chunk is shaped consistently.

Do not spend the whole day fighting Silero VAD yet. If there is time, start it, but the Day 1 milestone is capture first.

## 6. Define Module Contracts

By the end of today, everyone should agree that the audio module outputs something like:

```python
AudioChunk(
    samples: np.ndarray,
    sample_rate: int,
    timestamp: float,
)
```

Or, more simply:

```python
chunk: np.ndarray  # float32 mono waveform
sample_rate: int   # 16000
```

## Day 1 Success Criteria

By the end of Day 1, you should be able to run one command and see:

```text
Listening...
Generated chunk 1: shape=(48000,), sample_rate=16000
Generated chunk 2: shape=(48000,), sample_rate=16000
Generated chunk 3: shape=(48000,), sample_rate=16000
```

Since `16000 * 3 = 48000`, every 3-second chunk should contain `48000` samples.

You should also have several saved WAV files that contain your recorded voice clearly.

## Avoid on Day 1

Do not get distracted by:

- Hugging Face models.
- Streamlit UI.
- Warning banners.
- Architecture slides.
- Zoom/system audio loopback.
- GPU optimization.
- Model accuracy debates.

Today is plumbing day. Make the microphone-to-chunks path boringly reliable. That is the foundation everything else stands on.
