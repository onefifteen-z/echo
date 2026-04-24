# Echo

<p align="center">
  <img src="echo-icon.svg" alt="Echo icon" width="108" />
</p>

<p align="center">
  <img alt="Python 3.10+" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" />
  <img alt="Built with Textual" src="https://img.shields.io/badge/TUI-Textual-5A48F5" />
  <img alt="ASR Kotoba Whisper v2.2" src="https://img.shields.io/badge/ASR-kotoba--whisper--v2.2-FFB020" />
  <img alt="Live audio mode" src="https://img.shields.io/badge/Live-mic%20%2B%20system-2EA043" />
</p>

Echo is a **terminal (TUI) demo** for meeting-style workflows: live-ish transcript, periodic **web search** via the **[`ddgs`](https://pypi.org/project/ddgs/)** metasearch library (DuckDuckGo / Bing / … backends), plus optional **LLM** summaries, and a bottom **chat** panel for questions over the transcript.

Built with [Textual](https://textual.textualize.io/).

## Features

- **Startup splash**: green **Braille-style ECHO** (Kiro-like dot logo) + short hints; any key / click or ~3.2s auto-dismiss (`ECHO_NO_SPLASH=1` to skip).
- **Transcript panel**: timed lines as they arrive.
- **Insights panel**: keywords, web hits, and a short summary (LLM optional).
- **Chat panel**: ask questions with full transcript as context (LLM optional).
- **Four input modes**:
  1. Built-in demo lines (no flags).
  2. `--replay` a UTF-8 text file (simulated streaming).
  3. `--audio` a local file transcribed with **[Kotoba-Whisper v2.2](https://huggingface.co/kotoba-tech/kotoba-whisper-v2.2)** weights via Transformers (see below).
  4. **`--live`** real-time capture from the **microphone** plus an optional **second PortAudio input** (often a virtual loopback such as [BlackHole](https://github.com/ExistentialAudio/BlackHole) on macOS—**not required**; see **Zoom without BlackHole** below). Both paths are resampled to **16 kHz mono**; when two devices are set, Echo **mixes them 50/50** before ASR. List devices with `echo-tui --list-audio-devices`.

## Requirements

- Python **3.10+**
- A virtual environment is recommended (macOS Homebrew Python is PEP 668–managed).

## Install

```bash
cd prj-echo
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
```

### Optional: local speech-to-text (`--audio`)

ASR uses PyTorch + Transformers + Torchaudio. Install the extra group:

```bash
pip install -e ".[asr]"
```

The `[asr]` extra includes **torchcodec**, which recent **torchaudio** builds use to decode **MP3** and many other formats. **ffmpeg** should be on your `PATH` (e.g. `brew install ffmpeg` on macOS). If TorchCodec is awkward to install, `pip install librosa` is enough for Echo’s **librosa fallback** when torchaudio fails. **Live capture** also needs **`sounddevice`** (included in `[asr]`).

First run downloads model weights from Hugging Face (large download; size depends on dtype/device).

With `--audio`, weights are loaded **before** the TUI starts (main thread) to avoid macOS `bad value(s) in fds_to_keep` errors when Hugging Face spawns helpers from a background thread.

> **Note:** The Hub “custom pipeline” for Kotoba-Whisper bundles **speaker diarization** (pyannote, etc.). Echo instead loads the same **`WhisperForConditionalGeneration`** checkpoint and runs **standard Japanese transcribe** in fixed time windows—no diarization, so you do not need pyannote tokens for this demo.

## Usage

The installed command is **`echo-tui`** (not `echo`). In zsh/bash, `echo` is almost always the **shell builtin**, so typing `echo --replay …` never starts this app—it only prints arguments.

```bash
# Built-in demo transcript
echo-tui

# Replay a UTF-8 file (one utterance per line; timestamps added if missing)
echo-tui --replay samples/sample_transcript.txt --line-delay 0.8

# Transcribe a local audio file (wav / mp3 / flac / …)
echo-tui --audio path/to/meeting.wav

# Live: mic + second input (e.g. loopback device index from --list-audio-devices)
echo-tui --list-audio-devices
echo-tui --live --live-mic-device 0 --live-system-device 2 --chunk-seconds 5

# Live without BlackHole: use bundled ScreenCaptureKit helper + mic
echo-tui --live --live-system-authorized

# Simple live shortcuts (no long argument list)
./bin/echo-live
./bin/echo-live-debug

# ASR options
echo-tui --audio meeting.wav --chunk-seconds 30 --model-id kotoba-tech/kotoba-whisper-v2.2
echo-tui --audio meeting.wav --whisper-device cuda   # or: cpu, mps
```

Same entry point without a script name clash (with the venv activated):

```bash
python -m echo --replay samples/sample_transcript.txt --line-delay 0.8
```

### Test LLM + web search / ddgs (no TUI)

```bash
echo-probe
# optional: echo-probe --llm-timeout 60 --ddg-timeout 20 --ddg-query "your query"
```

Loads the same `.env` rules as the app (repo root then cwd), prints which LLM route is active, runs one small **keywords/summary** call and one **ddgs** text search, then exits `0` / `1`.

### Environment variables

**LLM routing order** (first match wins; shell exports beat `.env`):

1. `ANTHROPIC_API_KEY` → Anthropic Messages API  
2. **`BASE_URL` + `API_KEY` + `MODEL` all set** → OpenAI-compatible Chat Completions (typical **[LiteLLM](https://github.com/BerriAI/litellm)** proxy: point `BASE_URL` at your proxy’s OpenAI API root, often ending in `/v1`)  
3. `OPENAI_API_KEY` → official OpenAI API (`ECHO_OPENAI_MODEL` optional, default `gpt-4o-mini`)

| Variable | Purpose |
|----------|---------|
| `ECHO_WHISPER_DEVICE` | `cpu`, `cuda`, or `mps` for ASR (if unset, auto). |
| `ANTHROPIC_API_KEY` | Claude for insights + chat. Optional: `ECHO_ANTHROPIC_MODEL`. |
| `BASE_URL`, `API_KEY`, `MODEL` | Together: OpenAI-compatible endpoint (LiteLLM, vLLM, etc.). Uses `openai` Python SDK with `base_url`. |
| `OPENAI_API_KEY` | Official OpenAI. Optional: `ECHO_OPENAI_MODEL`. |
| `ECHO_INSIGHT_LLM_TIMEOUT` | Seconds for each insights LLM call (default `120`). |
| `ECHO_INSIGHT_DDG_TIMEOUT` | Seconds for each ddgs web search in insights (default `45`). |
| `ECHO_DDGS_BACKEND` | Metasearch backend for the `ddgs` library: `auto` (default), `bing`, `duckduckgo`, `yahoo`, … (see [ddgs docs](https://pypi.org/project/ddgs/)). If unset and `auto` returns nothing, Echo tries `bing`, then `duckduckgo`, then `yahoo`. |
| `ECHO_NO_SPLASH` | Set to `1` / `true` / `yes` to skip the green ASCII **ECHO** startup splash (e.g. for automation). |
| `ECHO_WHISPER_LANGUAGE` | Whisper `language=` for forced decoder ids (default `japanese` for Kotoba). Set e.g. `english` for other models. |
| `ECHO_LIVE_EMIT_INTERVAL` | Legacy batch mode knob for live transcript flushing. `0` (default) keeps Echo in append-draft mode and uses semantic/pause/length segmentation rules. Positive values switch to interval batching. |
| `ECHO_LIVE_NOISE_TOKENS` | Comma-separated short tokens to suppress in live ASR when they look like low-energy noise hallucinations (default: `ごめん`). |
| `ECHO_LIVE_ALWAYS_DROP_NOISE_TOKENS` | Comma-separated tokens to always suppress when matched (default: `ごめん`), regardless of RMS or segment position. |
| `ECHO_LIVE_NOISE_MAX_CHARS` | Maximum normalized token length eligible for noise suppression (default `4`). |
| `ECHO_LIVE_NOISE_MAX_RMS_FACTOR` | Relative RMS ceiling for noise-token suppression, scaled by `ECHO_LIVE_MIN_RMS` (default `1.3`). |
| `ECHO_LIVE_NOISE_DROP_AT_SEGMENT_START` | If enabled (default `1`), drop configured noise tokens at the start of a new segment even when RMS is not low. |
| `ECHO_LIVE_ASR_LOG_INTERVAL_SECONDS` | Interval for live ASR latency diagnostics in debug/log output (default `2.0`; set `0` to disable). |
| `ECHO_LIVE_SEGMENT_PAUSE_SECONDS` | Main live UI segmentation trigger: finalize current draft when silence reaches this threshold (default `2.0`). |
| `ECHO_UI_SEGMENT_SEMANTIC_MIN_CHARS` | Minimum chars required before sentence-ending punctuation can finalize a segment (default `80`). |
| `ECHO_UI_SEGMENT_MAX_SECONDS` | Forced UI segment split when a draft grows too long in time (default `25`). |
| `ECHO_UI_SEGMENT_MAX_CHARS` | Forced UI segment split when a draft grows too long in text length (default `360`). |
| `ECHO_INSIGHT_LIVE_MIN_SECONDS` | Live insight auxiliary trigger on newly added transcript duration (default `10`). |
| `ECHO_INSIGHT_PAUSE_SECONDS` | Live idle-window trigger for insight evaluation (default `3`). |
| `ECHO_INSIGHT_PAUSE_MIN_CHARS` | Minimum new chars required when triggering insights from idle pause (default `80`). |
| `ECHO_INSIGHT_FINALIZED_SEGMENT_MIN_CHARS` | Minimum chars for a newly finalized segment to trigger insight directly (default `100`). |
| `ECHO_INSIGHT_SKIP_MIN_CHARS` | Skip insight calls when new content is below this size (default `50`). |
| `ECHO_INSIGHT_LIVE_MIN_INTERVAL` | Global live insight rate limiter in seconds (default `30`). |
| `ECHO_ENABLE_TRANSCRIPT_REWRITE` | Enable LLM rewrite-back to Transcript (`1`/`true`, default). Set `0` to keep Transcript raw ASR text and disable LLM rewrite-back. |
| `ECHO_DEBUG` | Enable debug panel/log events. Default is `0` (off). Set `1` to enable debug event output and log-file writing. |
| `ECHO_DEBUG_PANEL_LINES` | Debug panel history cap. Default `0` means unlimited history (no truncation). |
| `ECHO_OFFLINE` / `ECHO_ASSUME_ONLINE` / `ECHO_HF_LOCAL_ONLY` | See **Offline / LLM reachability** below. |
| `ECHO_SYSTEM_AUDIO_CMD` | Override command for `--live-system-authorized` (if `--live-system-authorized-cmd` is omitted). Built-in default auto-compiles a bundled ScreenCaptureKit helper on macOS. Custom command must output `s16le`, 24kHz, stereo PCM to stdout. |

`API_KEY` / `MODEL` are generic names—keep Echo’s `.env` dedicated to this app or use a separate file / shell exports so they do not clash with other tools.

If the **Insights** panel stays on “Updating…”, it was usually a **hung or failing LLM/search request** with no UI recovery (now fixed: timeouts + error text). Check `BASE_URL` / keys, corporate proxy, or search rate limits; try `ECHO_DDGS_BACKEND=bing`; increase timeouts via the env vars above if needed.

If no LLM configuration is detected, Echo still runs: keywords use a small heuristic, summaries note that LLM is off, and chat replies tell you to set a key.

### Offline / LLM reachability

- On startup Echo **TCP-probes the configured LLM host** (Anthropic, OpenAI, or your `BASE_URL`). If that fails—or no API keys are set—**Insights and chat are disabled**, but **Transcript** (replay, file ASR, or `--live`) still runs.
- **`ECHO_ASSUME_ONLINE=1`**: skip the LLM probe (useful on split-tunnel / internal-only API routes).
- **`ECHO_OFFLINE=1`**: force local Hugging Face cache for weights and treat the LLM as unavailable.
- **`ECHO_HF_LOCAL_ONLY=1`**: load ASR weights from disk only (no Hub download) without forcing the LLM offline probe to fail.

### Live audio (macOS)

#### Option A — second *input* device (e.g. BlackHole / Loopback)

macOS does not expose “what Zoom plays to your speakers” as a microphone. A **second input** is usually a **virtual loopback** (BlackHole, or paid tools such as **Loopback**—same idea, different vendor).

1. Route Zoom’s **speaker** output through that loopback (often via a **Multi-Output Device** in Audio MIDI Setup: built-in speakers + loopback).
2. `echo-tui --list-audio-devices` → pick **mic** + **loopback** indices or name substrings.
3. `echo-tui --live --live-mic-device … --live-system-device …`  
   Or build an **Aggregate Device** (mic + loopback) and pass **only** `--live-mic-device` to that aggregate.

Echo does **not** ship or install audio drivers.

In `--live` default mode (`ECHO_LIVE_EMIT_INTERVAL=0`), Transcript keeps one mutable draft line and finalizes it when one of these happens:

- pause reaches `ECHO_LIVE_SEGMENT_PAUSE_SECONDS` (primary trigger),
- sentence-ending punctuation is seen and draft length reaches `ECHO_UI_SEGMENT_SEMANTIC_MIN_CHARS`,
- draft exceeds `ECHO_UI_SEGMENT_MAX_SECONDS` or `ECHO_UI_SEGMENT_MAX_CHARS` (forced split).

Insights consume only new live content since the previous call and trigger on new chars / newly finalized segment / new duration / idle pause, with `ECHO_INSIGHT_LIVE_MIN_INTERVAL` as a hard rate limit.
For debugging, Echo writes live debug events to `./log/echo_live_<timestamp>.log` (created from your launch directory).

#### Option B — Zoom **without** BlackHole (system authorization mode)

Echo supports a real-time path that mixes:

- microphone from `sounddevice`, and
- system output captured by a bundled macOS helper.

By default, Echo auto-builds and runs `echo/macos/SystemAudioCapture.swift` on first use. The helper writes `s16le` stereo PCM @ 24kHz to stdout and uses **ScreenCaptureKit** by default (requires **Screen Recording** permission).

Steps:

1. Run Echo live mode:
   `echo-tui --live --live-system-authorized`
2. Approve the macOS prompt (microphone for your terminal app if not granted yet). For system output capture, grant **Screen Recording** permission.
3. Optional: provide your own command instead of the bundled helper:
   `echo-tui --live --live-system-authorized --live-system-authorized-cmd "/path/to/cmd"`
4. Optional persistent override:
   `export ECHO_SYSTEM_AUDIO_CMD="/path/to/cmd"`

Notes:

- `--live-system-authorized` cannot be combined with `--live-system-device`.
- If helper build or system capture startup fails, Echo shows a startup error in Transcript.
- Optional advanced path: on macOS 14.2+, set `ECHO_ENABLE_AUDIO_TAP=1` to opt into the Core Audio process-tap mode (System Audio Recording Only).

Fallback (always works, no real-time requirements): Zoom/QuickTime recording → `echo-tui --audio file`.

### `.env` (python-dotenv)

On startup, Echo loads **python-dotenv** in this order (variables already set in your shell are **not** overwritten):

1. `.env` next to `pyproject.toml` (repository root)
2. `.env` in the **current working directory** (handy if you run from another folder)

Example `.env` at the repo root:

```bash
# --- Option A: Anthropic (highest priority if set) ---
# ANTHROPIC_API_KEY=sk-ant-...
# ECHO_ANTHROPIC_MODEL=claude-3-5-haiku-20241022

# --- Option B: LiteLLM (or any OpenAI-compatible proxy) ---
# All three required; do not set ANTHROPIC above if you want this path.
# BASE_URL=http://127.0.0.1:4000/v1
# API_KEY=sk-...          # whatever your proxy expects (often master key)
# MODEL=gpt-4o-mini       # LiteLLM model name / route

# --- Option C: OpenAI official ---
# OPENAI_API_KEY=sk-...
# ECHO_OPENAI_MODEL=gpt-4o-mini

# ECHO_WHISPER_DEVICE=mps
```

Add `.env` to `.gitignore` (already in this repo) so keys are not committed.

### Keyboard shortcuts

| Key | Action |
|-----|--------|
| `p` | Pause / resume (text replay or between ASR chunks). |
| `s` | Save current transcript to `echo_transcript_<timestamp>.txt` in the working directory. |
| `c` | Clear chat log. |
| `q` | Quit. |

## Project layout

```
echo/
  __main__.py   # CLI
  splash.py     # ASCII ECHO startup modal
  app.py        # Textual UI + workers
  live_audio.py # sounddevice capture (mic + optional loopback)
  macos/SystemAudioCapture.swift  # bundled ScreenCaptureKit helper for --live-system-authorized
  asr.py        # Kotoba-Whisper weights, chunked inference
  llm.py        # Optional Anthropic / OpenAI + fallbacks
  search.py     # DuckDuckGo (threaded)
samples/
  sample_transcript.txt
pyproject.toml
```

## License

See model and dependency licenses (e.g. Kotoba-Whisper **Apache-2.0** on Hugging Face). This demo code is provided as-is for experimentation.
