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
