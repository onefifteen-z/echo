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

- Uses **sounddevice** (`[asr]` extra): default **mic** + optional **second PortAudio input** (often a virtual loopback such as BlackHole) mixed **50/50** to mono **16 kHz** before ASR (**auto**: Apple on-device → Whisper API → local Kotoba only offline/unreachable API/`--asr-backend local`; see README).
- Device args accept a numeric **index** or a **case-insensitive substring** of the PortAudio device name (`--list-audio-devices`).
- **System authorization mode (no BlackHole):** `--live-system-authorized` auto-builds a bundled ScreenCaptureKit helper on macOS (or uses `ECHO_SYSTEM_AUDIO_CMD` / `--live-system-authorized-cmd`) for system output + mic in real time; keep UX/docs in English.
- Do not commit virtual-audio driver binaries; document OS-level setup in `README.md` only.
- Apple on-device ASR in `--live` is **streaming** (`AppleStreamingRecognizer` → long-running `macos/SpeechTranscribe.swift` helper); API/local are **windowed**. Backend choice, failure backoff and upgrades live in `AsrRouter` (`echo/asr_routing.py`) — do not load Kotoba weights from worker threads.

## Architecture

- `echo/engine/` holds the headless `EchoSession` (transcription, insights, summary, chat) and emits typed events (`engine/events.py`). The Textual TUI (`echo/app.py`) and the WebSocket sidecar (`echo/server.py`, `echo-server`) are both **subscribers**; put new behavior in the engine, not in a front end.
- When adding or renaming an event or command, update `gui/src/shared/types.ts` and the reducer in `gui/src/renderer/lib/session-store.ts` together with the Python side.
- `gui/` is Electron + React + Vite + Tailwind. The preload is sandboxed: **type-only imports** from `src/shared`, no runtime requires of local modules.

## Checks

- Python: use the repo **`.venv`** (`.venv/bin/python -m pytest`); the conda base interpreter can be SIGKILLed by macOS code-signing checks.
- GUI: `cd gui && npm run typecheck && npm test && npm run build`.
- GUI settings (`gui/src/main/services/settings-store.ts`) are injected as env into the sidecar and override `.env`; keep `gui/src/renderer/lib/settings-schema.ts` key validation in sync with the store.
