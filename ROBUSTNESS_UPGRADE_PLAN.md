# True-Tone Robustness Upgrade Plan

## Current System Diagnosis

The existing True-Tone pipeline is functional but thin:

- Capture emits 3-second chunks, with optional overlap only for microphone input.
- VAD is energy-based by default.
- Detection depends on one Hugging Face audio classifier.
- Live confidence is a short moving average, not a session-level decision.
- Evaluation tooling measures simple threshold accuracy, but not codec/noise/session stability.

The right upgrade path is evolutionary: keep capture, VAD, app wiring, and dashboard, then add stronger evidence streams, temporal confidence memory, calibrated fusion, and a real streaming evaluation suite.

## Implemented First-Pass Upgrade

This repo now includes:

- `processing/features.py`: low-latency DSP/behavioral features for spectral entropy, pitch drift, jitter, shimmer, HNR, cadence consistency, pauses, breathiness proxy, and temporal energy variance.
- `pipeline/temporal.py`: rolling session memory with EMA smoothing, weighted rolling averages, trend, uncertainty, anomaly score, confidence decay during silence, and hysteresis states.
- `inference/detector.py`: fuses the existing neural detector score with handcrafted behavioral evidence while preserving raw model outputs.
- `pipeline/orchestrator.py` and `live_pipeline.py`: publish stabilized session probability while retaining raw chunk probability for debugging.
- `tools/tune_streaming_threshold.py`: tunes alert thresholds on rolling-window stabilized scores.
- `tools/download_online_samples.py`: downloads a bounded balanced public real/fake test subset.
- `inference/detector.py`: supports comma-separated Hugging Face model IDs and branch weights for weighted ensemble inference.

## Target Architecture

Use a multi-branch detector:

```text
live audio
  -> capture with 3s windows and 1s stride
  -> preprocessing: mono, resample, loudness norm, light denoise, VAD
  -> parallel feature/model branches
       AASIST / AASIST-L
       RawNet2 or RawNetLite-style waveform CM
       WavLM embedding classifier
       Whisper encoder embedding classifier
       HuBERT embedding classifier
       handcrafted DSP + behavioral analyzer
  -> calibrated score fusion
  -> temporal confidence aggregator
  -> dashboard/API confidence band
```

The live display should show the stabilized score, not the raw chunk score. Raw scores stay available for diagnostics.

## Feature Engineering Upgrades

Prioritize features that survive compression:

- SSL embeddings: WavLM Base+ or Large, HuBERT Base/Large, Whisper encoder embeddings.
- Raw waveform anti-spoofing: AASIST/AASIST-L and RawNet2.
- Spectral features: entropy, flatness, high-frequency ratio, centroid/bandwidth drift.
- Voice naturalness: pitch drift, local jitter, local shimmer, approximate HNR.
- Behavior: pause ratio/count/CV, cadence consistency, temporal energy CV, speech rhythm irregularity.
- Call artifacts: clipping ratio, packet-drop style discontinuities, bandwidth estimate, noise-floor stability.

Treat handcrafted features as weak evidence. They are best for stabilizing and explaining model outputs, not as a standalone truth source.

## Ensemble Fusion Strategy

Start with calibrated weighted fusion:

```text
score = 0.30 * AASIST
      + 0.20 * RawNet2
      + 0.20 * WavLM classifier
      + 0.12 * Whisper classifier
      + 0.10 * HuBERT classifier
      + 0.08 * DSP/behavioral score
```

The current implementation supports this kind of weighted neural ensemble for Hugging Face audio-classification branches:

```bash
venv/bin/python tools/evaluate_streaming.py test_audio/online_samples \
  --model-id DeepFake-Audio-Rangers/DeepfakeDetect_wav2vec2,Vansh180/deepfake-audio-wav2vec2 \
  --model-weights 0.6,0.4
```

Or set it through environment variables:

```bash
export TRUE_TONE_MODEL_IDS=DeepFake-Audio-Rangers/DeepfakeDetect_wav2vec2,Vansh180/deepfake-audio-wav2vec2
export TRUE_TONE_MODEL_WEIGHTS=0.6,0.4
```

Do not make a second model the default until it has been downloaded, evaluated, and threshold-tuned locally. Good first candidates are `Vansh180/deepfake-audio-wav2vec2` and `Hemgg/Deepfake-audio-detection` because they are roughly the same size as the current branch; the large XLSR branches are around 1.26 GB each.

Initial measured ensemble result on `test_audio/online_samples`:

```text
Single current model, threshold 0.20: 75.0% accuracy, FP=5, FN=5
Current + Vansh180, weights 0.6/0.4, threshold 0.20: 77.5% accuracy, FP=5, FN=4
Current + Vansh180, weights 0.6/0.4, threshold 0.15: 77.5% accuracy, FP=6, FN=3
```

The ensemble improves recall slightly but roughly doubles CPU inference time on this machine. Use it for accuracy-focused testing, then decide whether the latency tradeoff is acceptable for live mode.

Then move to stacking:

- Inputs: each model logit, model uncertainty, SNR estimate, VAD ratio, codec/bandwidth features, behavioral features, recent temporal stats.
- Meta-model: logistic regression first, then LightGBM/XGBoost if enough validation data exists.
- Calibration: fit temperature scaling or isotonic regression per model and final stacker.
- Fallback: if a heavy model times out, renormalize weights over available branches and increase uncertainty.

## Temporal Aggregation

Use overlapping windows:

- 3-second window.
- 1-second stride for live capture.
- Keep 15 to 30 seconds of speech-window memory.
- Combine EMA, recency-weighted rolling mean, trend, and anomaly score.
- Use hysteresis:
  - `0.0-0.35`: likely real.
  - `0.35-0.65`: suspicious.
  - `0.65-1.0`: likely synthetic.
- Require repeated evidence before entering `likely_synthetic`.
- Decay confidence during silence instead of resetting instantly.

The current single-model baseline under-scores many fake examples, so the practical alert threshold is tuned separately from the ideal confidence bands. On the initial 40-file online sample set, a `0.20` alert threshold gave the best measured accuracy plateau. Re-tune this threshold whenever the model mix or dataset changes:

```bash
venv/bin/python tools/tune_streaming_threshold.py test_audio/online_samples --local-files-only
```

## Compression and Noise Robustness

Add augmentation and evaluation conditions:

- Opus at 6, 12, 16, 24, and 32 kbps.
- MP3 at 32, 64, and 96 kbps.
- G.711 A-law/u-law, G.722, AAC-LD-like bandwidth limits.
- Discord/Zoom/Meet style cascades: gain control, noise suppression, bandwidth limits, packet loss, resampling.
- Add room impulse responses, fan/keyboard/cafe noise, music bleed, mic distance variation, clipping, and overlap speech.

Preprocessing should use conservative normalization:

- RMS/LUFS-style loudness target, not peak-only normalization.
- Optional spectral gating only when SNR is low.
- Preserve high-frequency artifacts when possible; avoid aggressive denoise before anti-spoofing models.

## Model Improvement Strategy

Recommended model branches:

- AASIST/AASIST-L for efficient spectro-temporal anti-spoofing.
- RawNet2 for raw waveform cues.
- WavLM embedding classifier for robust speech representations.
- Whisper encoder embeddings for robustness to accents/noise and broad audio conditions.
- HuBERT embeddings as an independent SSL representation.

Training plan:

- Fine-tune only lightweight classifier heads first.
- Freeze SSL encoders for CPU deployment; unfreeze top layers only after validation improves.
- Use hard-negative mining from false positives: real speakers with clean studio mics, accents, singing, emotional speech, heavy compression, and background music.
- Use hard-positive mining from modern TTS: ElevenLabs, OpenAI Voice, Cartesia, Fish Audio, PlayHT, Resemble, Azure, Google, Amazon Polly, and voice conversion.
- Mix real/fake conversational turns, overlapped speech, and playback-through-speaker attacks.

## Confidence Calibration

Expose:

- `raw_probability`: current chunk/model fusion.
- `stabilized_probability`: temporal session confidence.
- `uncertainty`: disagreement or high volatility.
- `decision_state`: `likely_real`, `suspicious`, `likely_synthetic`, or `insufficient_speech`.

Calibration metrics:

- ECE/MCE for confidence reliability.
- ROC-AUC and PR-AUC.
- EER and min t-DCF for ASVspoof compatibility.
- False positive rate at fixed high recall.
- Detection latency: seconds until stable synthetic alert.

## Real-Time Optimization

Use tiered inference:

- Every 1s: cheap DSP + current lightweight model.
- Every 2s: AASIST/RawNet2 if CPU budget allows.
- Every 3-5s or async: WavLM/Whisper/HuBERT embedding branches.

Deployment optimizations:

- ONNX Runtime for AASIST/RawNet2/classifier heads.
- Quantize embedding classifiers and small anti-spoofing models.
- Keep SSL encoders optional or server-side unless local hardware is strong.
- Use a bounded async inference queue; never block audio capture.
- Cache overlapping-window embeddings when possible.

## Evaluation Methodology

Build a session-level test harness, not only per-file scoring:

- Replay files as 3-second windows with 1-second stride.
- Compute raw and stabilized metrics.
- Report per-domain metrics: clean, Discord, Zoom, Meet, low bitrate, noisy, overlapping, replayed.
- Track score volatility: standard deviation, number of false alert transitions, time-to-alert, time-to-clear.
- Include unknown generators and newly collected samples as a blind holdout.

Datasets and references:

- ASVspoof 2021 LA/DF for channel and compression effects.
- ASVspoof 5 for larger crowdsourced deepfake/adversarial attacks.
- In-house modern TTS/voice-conversion set for 2025-2026 systems.
- Real call recordings collected with consent across commodity microphones.

## Implementation Roadmap

1. Done: add handcrafted feature extraction and temporal confidence aggregation.
2. Done: add CLI/dashboard controls for `1s` stride overlap by default.
3. Add codec/noise augmentation tool using `ffmpeg` and room/noise mixing.
4. Add evaluation harness for session-level metrics and volatility.
5. Add calibrated weighted ensemble interface with optional model branches.
6. Integrate AASIST-L and RawNet2 first because they are anti-spoofing-specific.
7. Add WavLM/Whisper/HuBERT embedding heads after collecting enough modern/compressed validation data.
8. Train calibration stacker and freeze model versions.
9. Export fast branches to ONNX and add timeout-aware async inference.
10. Re-tune thresholds on real streaming sessions, not isolated clips.

## Primary References

- AASIST official implementation: https://github.com/clovaai/aasist
- WavLM paper page: https://www.microsoft.com/en-us/research/publication/wavlm-large-scale-self-supervised-pre-training-for-full-stack-speech-processing/
- Whisper release/model notes: https://openai.com/research/whisper/
- ASVspoof 2021 official challenge: https://www.asvspoof.org/index2021.html
- ASVspoof 5 overview: https://www.isca-archive.org/asvspoof_2024/wang24_asvspoof.html
