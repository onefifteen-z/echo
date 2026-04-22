# Echo

Echo is a **terminal (TUI) demo** for meeting-style workflows: live-ish transcript, periodic **web search** via the **[`ddgs`](https://pypi.org/project/ddgs/)** metasearch library (DuckDuckGo / Bing / … backends), plus optional **LLM** summaries, and a bottom **chat** panel for questions over the transcript.

Built with [Textual](https://textual.textualize.io/).

## Features

- **Startup splash**: green **Braille-style ECHO** (Kiro-like dot logo) + short hints; any key / click or ~3.2s auto-dismiss (`ECHO_NO_SPLASH=1` to skip).
- **Transcript panel**: timed lines as they arrive.
- **Insights panel**: keywords, web hits, and a short summary (LLM optional).
- **Chat panel**: ask questions with full transcript as context (LLM optional).
- **Three input modes**:
  1. Built-in demo lines (no flags).
  2. `--replay` a UTF-8 text file (simulated streaming).
  3. `--audio` a local file transcribed with **[Kotoba-Whisper v2.2](https://huggingface.co/kotoba-tech/kotoba-whisper-v2.2)** weights via Transformers (see below).

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

The `[asr]` extra includes **torchcodec**, which recent **torchaudio** builds use to decode **MP3** and many other formats. **ffmpeg** should be on your `PATH` (e.g. `brew install ffmpeg` on macOS). If TorchCodec is awkward to install, `pip install librosa` is enough for Echo’s **librosa fallback** when torchaudio fails.

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

`API_KEY` / `MODEL` are generic names—keep Echo’s `.env` dedicated to this app or use a separate file / shell exports so they do not clash with other tools.

If the **Insights** panel stays on “Updating…”, it was usually a **hung or failing LLM/search request** with no UI recovery (now fixed: timeouts + error text). Check `BASE_URL` / keys, corporate proxy, or search rate limits; try `ECHO_DDGS_BACKEND=bing`; increase timeouts via the env vars above if needed.

If no LLM configuration is detected, Echo still runs: keywords use a small heuristic, summaries note that LLM is off, and chat replies tell you to set a key.

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
  asr.py        # Kotoba-Whisper weights, chunked inference
  llm.py        # Optional Anthropic / OpenAI + fallbacks
  search.py     # DuckDuckGo (threaded)
samples/
  sample_transcript.txt
pyproject.toml
```

## License

See model and dependency licenses (e.g. Kotoba-Whisper **Apache-2.0** on Hugging Face). This demo code is provided as-is for experimentation.
