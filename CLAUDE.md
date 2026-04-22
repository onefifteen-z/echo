# Echo — notes for Claude / coding agents

## Language

- **Application UI strings** (TUI labels, errors surfaced to the user, CLI help where user-facing): **English**.
- **Repository documentation** (`README.md`, this file, inline product docs): **English**.

## Japanese fixtures (do not change)

- **Do not modify** Japanese-language **test and demo content**, including:
  - Built-in replay demo lines (`DEFAULT_DEMO_LINES` in `echo/app.py`).
  - Japanese sample transcripts under `samples/` (e.g. `ja_meeting_insight_chat.txt`).
  - Japanese ASR sample audio paths and expectations used in manual or automated checks.
- UI and docs stay English; **Japanese text in fixtures is intentional** for JP meeting / ASR scenarios.

## Live capture (`--live`)

- Uses **sounddevice** (`[asr]` extra): default **mic** + optional **second PortAudio input** (often a virtual loopback such as BlackHole) mixed **50/50** to mono **16 kHz** before Whisper.
- Device args accept a numeric **index** or a **case-insensitive substring** of the PortAudio device name (`--list-audio-devices`).
- **System authorization mode (no BlackHole):** `--live-system-authorized` auto-builds a bundled ScreenCaptureKit helper on macOS (or uses `ECHO_SYSTEM_AUDIO_CMD` / `--live-system-authorized-cmd`) for system output + mic in real time; keep UX/docs in English.
- Do not commit virtual-audio driver binaries; document OS-level setup in `README.md` only.
