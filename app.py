"""
True-Tone: Real-Time AI Voice Detection
Main entry point for the True-Tone application.

Launch modes:
    1. Streamlit dashboard (default):
        streamlit run ui/dashboard.py

    2. Terminal live pipeline:
        python app.py --mode terminal

    3. File analysis:
        python app.py --mode file --source-file path/to/audio.wav

    4. Dashboard hint:
        python app.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def run_terminal(args) -> None:
    """Run the live detection pipeline in terminal mode."""
    from audio.mic_capture import MicrophoneCapture, CHUNK_DURATION
    from audio.system_capture import SystemAudioCapture
    from audio.wav_loader import WAVFileReplay
    from audio.vad import EnergySpeechGate
    from inference.detector import AudioDeepfakeDetector
    from pipeline.orchestrator import PipelineOrchestrator, PipelineState

    import time

    # Build capture source
    if args.source_file:
        capture = WAVFileReplay(
            args.source_file,
            chunk_seconds=args.chunk_seconds,
            loop=False,
        )
    elif args.source == "system":
        capture = SystemAudioCapture(device=args.device, chunk_duration=args.chunk_seconds)
    else:
        capture = MicrophoneCapture(
            device=args.device,
            chunk_duration=args.chunk_seconds,
            overlap_duration=args.overlap,
            gain_db=args.gain,
        )

    detector = AudioDeepfakeDetector(
        model_id=args.model_id if args.model_id else None,
        chunk_seconds=args.chunk_seconds,
        local_files_only=args.local_files_only,
    )
    gate = EnergySpeechGate(min_rms=args.min_rms, min_peak=args.min_peak)

    orch = PipelineOrchestrator(
        capture=capture,
        detector=detector,
        speech_gate=gate,
        score_smoothing_window=args.smoothing,
    )

    print("\n🎙️ True-Tone — Terminal Mode")
    print("Press Ctrl+C to stop.\n")

    orch.start()
    last_count = 0

    try:
        while orch.state == PipelineState.RUNNING:
            scores = orch.get_latest_scores()
            if len(scores) > last_count:
                for score in scores[last_count:]:
                    label = "🗣️ speech" if score.is_speech else "🔇 silence"
                    prob_bar = "█" * int(score.ai_probability * 20) + "░" * (20 - int(score.ai_probability * 20))
                    warning = " ⚠️  AI DETECTED" if score.ai_probability >= 0.65 else ""
                    print(
                        f"  [{score.index:03d}] {label}  "
                        f"|{prob_bar}| {score.ai_probability:.1%}  "
                        f"rms={score.rms:.4f} peak={score.peak:.4f} "
                        f"latency={score.latency_seconds:.2f}s{warning}",
                        flush=True,
                    )
                last_count = len(scores)
            time.sleep(0.3)

        # Check for errors
        if orch.state == PipelineState.ERROR:
            print(f"\n❌ Pipeline error: {orch.last_error}")

    except KeyboardInterrupt:
        print("\n\nStopping…")
    finally:
        orch.stop()
        print("Pipeline stopped.")

        # Print summary
        scores = orch.get_latest_scores()
        if scores:
            speech_scores = [s.ai_probability for s in scores if s.is_speech]
            print(f"\n{'─' * 50}")
            print(f"Total chunks: {len(scores)}")
            print(f"Speech chunks: {len(speech_scores)}")
            if speech_scores:
                import numpy as np
                print(f"Avg AI probability: {np.mean(speech_scores):.4f}")
                print(f"Max AI probability: {max(speech_scores):.4f}")


def run_file_analysis(args) -> None:
    """Analyze a single audio file and print results."""
    from inference.detector import AudioDeepfakeDetector

    if not args.source_file:
        print("Error: --source-file is required in file mode.")
        sys.exit(1)

    detector = AudioDeepfakeDetector(
        model_id=args.model_id if args.model_id else None,
        local_files_only=args.local_files_only,
    )

    result = detector.predict_file(args.source_file)

    print(f"\n🎙️ True-Tone — File Analysis")
    print(f"{'─' * 50}")
    print(f"File:            {result.source}")
    print(f"Model:           {result.model_id}")
    print(f"AI Probability:  {result.ai_probability:.4f} ({result.ai_probability:.1%})")
    print(f"Chunk scores:    {', '.join(f'{s:.4f}' for s in result.chunk_scores)}")

    if result.ai_probability >= 0.65:
        print(f"\n⚠️  HIGH likelihood of synthetic/AI-generated speech.")
    elif result.ai_probability >= 0.4:
        print(f"\n⚡ MODERATE likelihood — inspect further.")
    else:
        print(f"\n✅ Speech appears human.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="True-Tone: Real-Time AI Voice Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  streamlit run ui/dashboard.py          Launch the Streamlit dashboard
  python app.py --mode terminal          Live detection in the terminal
  python app.py --mode file -f audio.wav Analyze a single audio file
        """,
    )
    parser.add_argument(
        "--mode", choices=("dashboard", "terminal", "file"), default="dashboard",
        help="Run mode (default: dashboard).",
    )
    parser.add_argument("--source", choices=("mic", "system"), default="mic")
    parser.add_argument("-f", "--source-file", default=None, help="Audio file path.")
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--chunk-seconds", type=int, default=3)
    parser.add_argument("--overlap", type=float, default=0.0)
    parser.add_argument("--gain", type=float, default=0.0)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--min-rms", type=float, default=0.002)
    parser.add_argument("--min-peak", type=float, default=0.01)
    parser.add_argument("--smoothing", type=int, default=5)
    args = parser.parse_args()

    if args.mode == "dashboard":
        print("🎙️ True-Tone Dashboard")
        print()
        print("To launch the Streamlit dashboard, run:")
        print("  streamlit run ui/dashboard.py")
        print()
        print("For terminal mode, run:")
        print("  python app.py --mode terminal")
    elif args.mode == "terminal":
        run_terminal(args)
    elif args.mode == "file":
        run_file_analysis(args)


if __name__ == "__main__":
    main()
