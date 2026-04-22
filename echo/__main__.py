"""CLI entry for Echo."""

from __future__ import annotations

import argparse
from pathlib import Path

from echo.app import run_echo
from echo.asr import MODEL_ID_DEFAULT


def main() -> None:
    p = argparse.ArgumentParser(
        prog="echo-tui",
        description="Echo — transcript replay or local Kotoba-Whisper ASR + DuckDuckGo + optional LLM (Textual TUI).",
    )
    src = p.add_mutually_exclusive_group()
    src.add_argument(
        "--replay",
        type=Path,
        help="UTF-8 text file; non-empty lines are replayed as transcript (timestamps added if missing).",
    )
    src.add_argument(
        "--audio",
        type=Path,
        help=(
            "Local audio file (wav/mp3/flac/…). Uses Hugging Face weights "
            f"`{MODEL_ID_DEFAULT}` via Transformers (install: pip install -e '.[asr]')."
        ),
    )
    p.add_argument(
        "--line-delay",
        type=float,
        default=1.0,
        help="Seconds between replayed lines when using --replay (default: 1.0).",
    )
    p.add_argument(
        "--model-id",
        default=MODEL_ID_DEFAULT,
        help=f"Hugging Face model id (default: {MODEL_ID_DEFAULT}).",
    )
    p.add_argument(
        "--chunk-seconds",
        type=float,
        default=30.0,
        help="Fixed window length for file ASR in seconds (default: 30).",
    )
    p.add_argument(
        "--whisper-device",
        choices=["cpu", "cuda", "mps"],
        default=None,
        help="Force torch device for ASR (default: auto, or ECHO_WHISPER_DEVICE).",
    )
    args = p.parse_args()
    run_echo(
        replay_path=args.replay,
        line_delay=args.line_delay,
        audio_path=args.audio,
        model_id=args.model_id,
        chunk_seconds=args.chunk_seconds,
        whisper_device=args.whisper_device,
    )


if __name__ == "__main__":
    main()
