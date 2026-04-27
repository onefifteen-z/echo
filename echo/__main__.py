"""CLI entry for Echo."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from echo.app import run_echo
from echo.asr import MODEL_ID_DEFAULT


def main() -> None:
    p = argparse.ArgumentParser(
        prog="echo-tui",
        description="Echo — transcript replay or local Kotoba-Whisper ASR + DuckDuckGo + optional LLM (Textual TUI).",
    )
    p.add_argument(
        "--list-audio-devices",
        action="store_true",
        help="Print sounddevice input indices and exit (for --live).",
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
    src.add_argument(
        "--live",
        action="store_true",
        help=(
            "Real-time ASR from the default microphone plus optional second input "
            "(e.g. BlackHole for system audio). Install: pip install -e '.[asr]'."
        ),
    )
    p.add_argument(
        "--live-mic-device",
        type=str,
        default=None,
        metavar="ID_OR_SUBSTR",
        help="sounddevice input index or name substring for the microphone (default: host default).",
    )
    p.add_argument(
        "--live-system-device",
        type=str,
        default=None,
        metavar="ID_OR_SUBSTR",
        help="Second input (e.g. BlackHole) mixed 50/50 with the mic; omit for microphone-only.",
    )
    p.add_argument(
        "--live-system-authorized",
        action="store_true",
        help=(
            "Use the bundled macOS system-audio helper (System Audio Recording Only on macOS 14.2+, "
            "ScreenCaptureKit fallback otherwise) and mix with microphone. No BlackHole required."
        ),
    )
    p.add_argument(
        "--live-system-authorized-cmd",
        type=str,
        default=None,
        metavar="CMD",
        help=(
            "Command that streams system audio PCM to stdout. If omitted, Echo uses "
            "ECHO_SYSTEM_AUDIO_CMD or auto-builds a bundled ScreenCaptureKit helper. "
            "Expected format: s16le stereo 24kHz."
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
        default=None,
        help="ASR window in seconds (default: 5 with --live, 30 with --audio).",
    )
    p.add_argument(
        "--whisper-device",
        choices=["cpu", "cuda", "mps"],
        default=None,
        help="Force torch device for ASR (default: auto, or ECHO_WHISPER_DEVICE).",
    )
    p.add_argument(
        "--meeting-title",
        type=str,
        default=None,
        help="Meeting title. If omitted, Echo asks at startup.",
    )
    p.add_argument(
        "--meeting-context",
        type=str,
        default=None,
        help="Meeting background/context. If omitted, Echo asks at startup.",
    )
    args = p.parse_args()
    if args.list_audio_devices:
        from echo.live_audio import list_input_devices

        print(list_input_devices(), file=sys.stdout)
        raise SystemExit(0)

    modes = sum(1 for x in (args.replay, args.audio, args.live) if x)
    if modes > 1:
        p.error("Use only one of --replay, --audio, or --live.")
    if args.live_system_authorized and not args.live:
        p.error("--live-system-authorized requires --live.")
    if args.live_system_authorized and args.live_system_device:
        p.error("--live-system-authorized and --live-system-device cannot be used together.")
    if args.live_system_authorized_cmd and not args.live_system_authorized:
        p.error("--live-system-authorized-cmd requires --live-system-authorized.")

    chunk_seconds = args.chunk_seconds
    if chunk_seconds is None:
        chunk_seconds = 5.0 if args.live else 30.0

    live_mic: int | None = None
    live_sys: int | None = None
    if args.live:
        from echo.live_audio import resolve_input_device

        try:
            live_mic = resolve_input_device(args.live_mic_device, default=None)
            live_sys = resolve_input_device(args.live_system_device, default=None)
        except ValueError as e:
            p.error(str(e))
    run_echo(
        replay_path=args.replay,
        line_delay=args.line_delay,
        audio_path=args.audio,
        model_id=args.model_id,
        chunk_seconds=chunk_seconds,
        whisper_device=args.whisper_device,
        live_audio=args.live,
        live_mic_device=live_mic,
        live_system_device=live_sys,
        live_system_authorized=args.live_system_authorized,
        live_system_authorized_cmd=args.live_system_authorized_cmd,
        meeting_title=(args.meeting_title or "").strip() or "untitled-meeting",
        meeting_context=(args.meeting_context or "").strip(),
    )


if __name__ == "__main__":
    main()
