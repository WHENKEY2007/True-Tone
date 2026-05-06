from pathlib import Path
import time

import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write

SAMPLE_RATE = 16000
CHUNK_DURATION = 3  # seconds
OUTPUT_DIR = Path("test_chunks")


def _to_int16(audio):
    """Convert recorded audio to 16-bit PCM for reliable WAV playback."""
    if audio.dtype.kind == "f":
        audio = np.clip(audio, -1.0, 1.0)
        audio = (audio * np.iinfo(np.int16).max).astype(np.int16)
    return np.squeeze(audio)

def record_chunk():
    print("Recording...")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    audio = sd.rec(
        int(CHUNK_DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32"
    )

    sd.wait()

    audio = _to_int16(audio)

    if not np.any(audio):
        print("Warning: recorded chunk is silent. Check your input device and microphone permissions.")

    filename = OUTPUT_DIR / f"chunk_{int(time.time())}.wav"

    write(str(filename), SAMPLE_RATE, audio)

    print(f"Saved: {filename}")

while True:
    record_chunk()