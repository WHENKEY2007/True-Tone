"""Audio capture and processing package for True-Tone."""

from audio.mic_capture import MicrophoneCapture, SAMPLE_RATE, CHUNK_DURATION
from audio.wav_utils import load_wav, save_wav, chunk_audio, AudioData
from audio.wav_loader import WAVFileReplay, get_audio_chunks_from_wav_file
from audio.vad import EnergySpeechGate, SileroSpeechGate, HybridSpeechGate, SpeechGateResult
from audio.buffer import AudioBuffer

__all__ = [
    "MicrophoneCapture",
    "SAMPLE_RATE",
    "CHUNK_DURATION",
    "load_wav",
    "save_wav",
    "chunk_audio",
    "AudioData",
    "WAVFileReplay",
    "get_audio_chunks_from_wav_file",
    "EnergySpeechGate",
    "SileroSpeechGate",
    "HybridSpeechGate",
    "SpeechGateResult",
    "AudioBuffer",
]
