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

Echo is a meeting assistant: a live transcript (Apple on-device speech, a Whisper API or local Kotoba-Whisper), periodic **insights** from an **LLM** with **web search** and an optional **knowledge vault**, a **chat** over the transcript, your own notes, and a post-meeting **summary**.

Most people use the **[desktop app](#desktop-app-gui)** (macOS, Electron). The same engine also runs as a **[terminal app](#terminal-app-tui)** built with [Textual](https://textual.textualize.io/).

## Features

**Desktop app (macOS):**

- **Setup dialog**: title, time, context, audio source (mic + system audio without extra drivers), language, speech engine, knowledge vault.
- **Live view**: transcript with draft/final lines, insights with web results, meeting chat and a rich-text notebook (with screenshots).
- **Meetings archive**: past meetings with time, duration, summary, notebook, transcript and insights; regenerate summaries later.
- **Menu bar item** and **Zoom / Teams meeting detection** (start transcribing from a floating card).
- **Settings view** for the LLM, speech recognition, web search and knowledge vault.

**Terminal app (TUI):**

- **Startup splash**: green **Braille-style ECHO** (Kiro-like dot logo) + short hints; any key / click or ~3.2s auto-dismiss (`ECHO_NO_SPLASH=1` to skip).
- **Transcript panel**: timed lines as they arrive.
- **Insights panel**: keywords, web hits, and a short summary (LLM optional).
- **Chat panel**: ask questions with full transcript as context (LLM optional).
- **Four input modes**:
  1. Built-in demo lines (no flags).
  2. `--replay` a UTF-8 text file (simulated streaming).
  3. `--audio` a local file transcribed with **[Kotoba-Whisper v2.2](https://huggingface.co/kotoba-tech/kotoba-whisper-v2.2)** weights via Transformers (see below).
  4. **`--live`** real-time capture from the **microphone** plus an optional **second PortAudio input** (often a virtual loopback such as [BlackHole](https://github.com/ExistentialAudio/BlackHole) on macOS—**not required**; see **Zoom without BlackHole** below). Both paths are resampled to **16 kHz mono**; when two devices are set, Echo **mixes them 50/50** before ASR. List devices with `echo-tui --list-audio-devices`.

## Desktop app (`gui/`)

Requirements: macOS, Python 3.10+ (Homebrew recommended) and, to build from source, Node.js with npm. Meeting detection needs macOS 14+; Apple's SpeechAnalyzer needs macOS 26 (older versions fall back to `SFSpeechRecognizer`).

An Electron + React + Tailwind front end (same stack as Lumina) that drives the Python engine through a local WebSocket sidecar (`echo-server`). It has a setup dialog (title, time, context, audio source, language, speech engine, microphone, knowledge vault, Apple model download), live transcript with draft/final lines, level meter, status chips (ASR backend, mic-only warning, LLM availability), insights with collapsible web results, and meeting chat. Outputs go to `~/Documents/Echo/<YYYY-MM-DD HHMM> <title>/` by default (the folder can be changed in Settings; dated folders, so meetings with the same title never overwrite each other; the folder name is the meeting id). The files in each folder are listed under [Save output](#save-output-s).

- **Notebook:** during a meeting, the *My notebook* tab (next to chat) is a rich-text editor (headings, lists, checklists, tables; Markdown shortcuts such as `# ` and `- [ ] ` work) that autosaves into the session; it is written to `notebook.md` and included in the summary.
  - **Images:** paste a screenshot, drag an image file in, or use *Image* in the toolbar (PNG, JPEG, GIF, WebP, up to 20 MB). Files are saved in the meeting folder under `notes-assets/` and linked from `notebook.md` as `![name](notes-assets/…)`. When the summary is generated (at the end of the meeting or with *Regenerate summary*), up to 8 notebook screenshots (the last ones, each under 3.75 MB) are sent to the LLM as images alongside the notes, so slides, whiteboards and shared documents count toward takeaways and next actions. If the model doesn't accept images, the summary is generated from text only and says so.
- **Meeting time:** every meeting has a time (minute precision). It defaults to when recording started; set it in the setup dialog's *Time* field, or later under **Details**. Meetings started from the Zoom/Teams prompt use the moment the meeting was detected. The time is passed to the summary prompt and orders the meetings list.
- **Meetings view** (left rail): lists past meetings (newest first by meeting time, searchable by title, context and summary) with time, a duration badge, line count and summary preview. A meeting shows:
  - **Summary** with a *Regenerate summary* button (re-runs the LLM with transcript, insights, context and notebook).
  - **Notebook** (editable, autosaved).
  - **Transcript** and **Insights** (read-only).
  - **Details**: editable time and context plus metadata (*Use start time* restores the default time).
  - The header shows the meeting time and a duration badge. The title is editable inline; *Open folder* opens the meeting in Finder.
  - *Delete* moves the folder to the macOS Trash, so it can be recovered.
  - The meeting in progress is marked **Live**: edits go to the running session, while regenerate and delete wait until the meeting ends.
- Folders written by older versions (no `meeting.json`) are still listed, using the folder name as title and its modification time as date.
- **Menu bar icon (macOS):**
  - While a meeting runs, it shows the elapsed time and a recording dot; it shows *Paused* when paused.
  - Its menu has the meeting status, *Open Echo*, *New meeting…*, *Pause* / *Resume*, *Stop meeting* and *Quit Echo*.
  - Closing the window keeps Echo running in the Dock and the menu bar.
  - On MacBooks with a notch, macOS hides menu bar icons that don't fit. If Echo's is missing, ⌘-drag it further right, or remove other menu bar items.
- **Meeting detection (macOS 14+):** when the Zoom or Microsoft Teams desktop app starts using the microphone, Echo shows a floating *meeting detected* card in the top-right corner.
  - *Start transcribing* starts right away with the source, engine, language and mic of your last meeting. The title comes from the meeting window, or falls back to "Zoom meeting 14:05". The meeting time is when the meeting was detected (Zoom and Teams don't expose the scheduled time).
  - The chevron menu has *Set up first…* (opens the setup dialog with the title and time filled in), *Don't ask for Zoom/Teams* and *Turn off meeting detection*.
  - The card closes by itself after 45 seconds, or when the meeting ends. It never takes focus from the meeting app, and it isn't shown while an Echo meeting is already running.
  - Turn detection on or off, and re-enable muted apps, under **Settings → Meeting detection**.
  - Requirements: the Xcode Command Line Tools (`xcode-select --install`). Echo compiles a small watcher helper the first time. Meeting titles need the **Screen Recording** permission; without it, the fallback title is used. Browser meetings (Zoom or Teams on the web) are not detected.
  - Development: `ECHO_DEV_FAKE_ZOOM_BUNDLE=org.python.python npm run dev` treats any Python process using the mic as Zoom, so `python -c "import sounddevice"`-style captures can trigger the card.
- **Knowledge vault:** optionally attach a Lumina vault in the setup dialog or from the top bar; insights and chat then consult its wiki on demand (see [Knowledge vault (Lumina)](#knowledge-vault-lumina)).
- **Editing a live meeting:** click the title in the top bar to rename it, and use *Context…* to edit the meeting context; changes are saved to the running session and its `meeting.json`.
- **Settings view** (bottom of the left rail): edits the engine configuration for the desktop app.
  - **Groups:** Workspace (meetings folder, glossary file), Language model, Speech recognition (default engine, Whisper API, local device, Apple on-device model status with *Download*), Web search, Knowledge vault (Lumina app path and the vaults Echo can see), Insights & transcript, Network & debugging.
  - **Advanced:** any other variable from [Environment variables](#environment-variables).
  - **Source badges:** each field shows where its current value comes from, *Echo settings* or *.env* (shell environment / `.env`). An empty field uses the built-in default, shown as the placeholder.
  - **Storage:** settings are stored in `~/Library/Application Support/Echo/settings.json`. Values whose name contains `KEY`, `TOKEN`, `SECRET` or `PASSWORD` are encrypted with the macOS Keychain (Electron `safeStorage`) and never sent back to the window.
  - **Precedence:** values from settings are passed to the engine as environment variables, so they override your shell and `.env`. *Remove override* falls back to those again. The terminal app (`echo`) does not read these settings.
  - **Saving:** saving restarts the engine. If a meeting is running, you can end it (it is saved first) and restart, or save now and apply later.
  - **Meetings folder:** changing it affects new meetings and the Meetings list; existing folders are not moved.

```bash
# Development (uses the repo .venv: python3 -m venv .venv && .venv/bin/pip install -e '.[asr]')
cd gui && npm install && npm run dev

# Type-check / unit tests / production build / macOS package
npm run typecheck && npm test && npm run build
npm run package

# Build and install to /Applications with a stable local signature (keeps macOS permissions
# such as Screen Recording across reinstalls). Create the identity once first.
./scripts/create-signing-cert.sh
npm run install:local

# Regenerate app icons (gui/assets/icons) after editing echo-icon.svg
npm run icons
```

- **Python runtime:** in development the app runs `../.venv/bin/python -m echo.server`. A packaged app bundles the `echo` sources and on first launch creates **`~/.echo/venv`** (needs Python ≥ 3.10, Homebrew preferred) and installs Echo into it; it is re-installed when the app version changes.
- **Configuration:** the packaged app reads **`~/.echo/.env`** (see [Configuration](#configuration)). In development the repo `.env` is used. Values from the Settings view take precedence over both.
- **Permissions:** the app requests **Microphone** access; **Speech Recognition** and **Screen & System Audio Recording** prompts appear the first time Apple ASR or system-audio capture is used. macOS ties these grants to the app's signature: an ad-hoc signed build gets a new signature on every rebuild, so system-audio capture fails with `permissionDenied` until you grant access again. Install with `npm run install:local` (stable local signature) to keep the grants across reinstalls; if a stale grant is stuck, run `tccutil reset ScreenCapture com.echo.app` and relaunch.
- **Crashes:** if the sidecar dies it is restarted automatically (up to 3 times per minute); after that the window shows the error tail and a *Restart engine* button. Sidecar logs: `~/Library/Application Support/Echo/echo-sidecar.log`. Set `ECHO_PYTHON=/path/to/python` to use a specific interpreter.

### Sidecar protocol (`echo-server`)

`echo-server [--host 127.0.0.1] [--port 0] [--output-root DIR]` prints one JSON line `{"port", "token", "protocol"}` to stdout, then serves `ws://127.0.0.1:<port>/?token=<token>`. It exits on SIGTERM or when stdin closes. Messages are JSON:

- Request `{"id", "cmd", "args"}` → response `{"id", "ok", "result" | "error"}`.
- Commands: `ping`, `probe`, `devices.list`, `apple.status`, `apple.install_assets`, `session.start` (`title`, `context`, `source`: `mic`/`both`/`system`/`demo`, `backend`, `language`, `mic_device`, `chunk_seconds`, `line_delay`, optional `vault`, optional `meeting_time` as ISO 8601, floored to the minute), `session.stop`, `session.pause`, `session.resume`, `session.reset`, `session.set_language`, `session.save`, `session.snapshot`, `session.notebook.set` (`text`), `chat.ask`, `meetings.list`, `meetings.get` (`id`), `meetings.update` (`id`, optional `title` / `context` / `notebook` / `meeting_time`; `""` resets the time to the start time), `meetings.regenerate_summary` (`id`; emits `summary.written` with `id`), `config.effective` (`keys`; returns `{value, source: app|env|default, secret}` per key, with secret values masked), `wiki.vaults` (returns `{available, reason?, vaults: [{name, path, default, initialized}]}`), `session.set_vault` (`vault`: name or `null`). `probe` also reports the active web search provider. Deleting is done by the Electron main process (`shell.trashItem`), not the sidecar.
- Server-pushed events `{"event": {...}}` mirror `echo/engine/events.py` (`transcript.draft`, `transcript.final`, `transcript.corrected`, `insight.pending`, `insight.pass`, `meter`, `status`, `notice`, `chat.reply`, `summary.written`, `session.ended`, …).

## Configuration

Shared by the desktop app and the terminal app. In the desktop app most of these can be set in **Settings**, which override the shell and `.env`.

### Environment variables

**LLM routing order** (first match wins; shell exports beat `.env`):

1. `ANTHROPIC_API_KEY` → Anthropic Messages API  
2. **`BASE_URL` + `API_KEY` + `MODEL` all set** → OpenAI-compatible Chat Completions (typical **[LiteLLM](https://github.com/BerriAI/litellm)** proxy: point `BASE_URL` at your proxy’s OpenAI API root, often ending in `/v1`)  
3. `OPENAI_API_KEY` → official OpenAI API (`ECHO_OPENAI_MODEL` optional, default `gpt-4o-mini`)

**ASR routing order** (with `--audio` or `--live`; override via `--asr-backend` or `ECHO_ASR_BACKEND`):

1. **macOS on-device** — Apple **SpeechAnalyzer / SpeechTranscriber** (macOS 26+, falls back to on-device `SFSpeechRecognizer`). In `--live` this is a **streaming** backend: one long-running helper process receives audio continuously and returns partial + final results (no fixed chunks). Requires **Speech Recognition** permission and the per-language model; install it with `echo-tui --install-apple-speech ja` (or accept the prompt at startup / the GUI's *Download model* button).  
2. **Whisper API** — OpenAI-compatible **`/v1/audio/transcriptions`**. Prefer **`ECHO_WHISPER_API_BASE_URL` + `API_KEY` + `ECHO_WHISPER_API_MODEL`** when the transcription endpoint differs from chat; otherwise `BASE_URL` + `API_KEY` (chat `MODEL` is ignored for ASR), or `OPENAI_API_KEY` for the official API.  
3. **Local Kotoba-Whisper** — Hugging Face weights via Transformers (`pip install -e '.[asr]'`). Used in `auto` when offline/API unreachable, or when forced with `--asr-backend local`.

The router caches probe results, switches away from a backend after **3 consecutive failures** (exponential cool-down 30–300 s, re-checked every 60 s so it upgrades back to Apple/API when they recover), never loads Kotoba weights mid-session from a worker thread, and in a **forced** mode (`apple`/`api`/`local`) fails loudly instead of silently falling back. The Whisper API path uses a request timeout, passes the recent transcript as `prompt` for continuity, and drops typical silence hallucinations (“Thanks for watching”, “ご視聴ありがとうございました”…) on near-silent chunks.

Default is **`auto`**: **Apple (on-device)** for the **current live language** → **Whisper API** when Apple is unavailable for that language → **local Kotoba** only when offline/API unreachable, or **`--asr-backend local`**. Apple support differs by locale (e.g. ja-JP may download later while en-US/zh-CN are ready); Echo **re-probes on each language switch** and does **not** download Kotoba weights at startup unless the resolved backend is local.

| Variable | Purpose |
|----------|---------|
| `ECHO_ASR_BACKEND` | `auto` (default), `apple`, `api`, or `local`. Same as `--asr-backend`. |
| `ECHO_WHISPER_API_MODEL` | Whisper model id for API ASR (default `whisper-1`). Used with `BASE_URL`+`API_KEY` or `OPENAI_API_KEY`. |
| `ECHO_WHISPER_API_BASE_URL` | Optional **separate** OpenAI-compatible root for `/audio/transcriptions` (e.g. `https://whisper-proxy.example/v1`). Uses the same `API_KEY` as LLM when set; does not use the chat `MODEL`. |
| `ECHO_WHISPER_API_TIMEOUT` | Seconds per Whisper API request (default `20`; one retry). |
| `ECHO_WHISPER_API_CHUNK_SECONDS` | Live window length when the Whisper API is the active backend (default `8`). `--chunk-seconds` overrides. |
| `ECHO_NO_APPLE_INSTALL_PROMPT` | Set to `1` to skip the startup prompt offering to download a missing Apple on-device speech model. |
| `ECHO_WHISPER_DEVICE` | `cpu`, `cuda`, or `mps` for **local** Kotoba ASR (if unset, auto). |
| `ANTHROPIC_API_KEY` | Claude for insights + chat. Optional: `ECHO_ANTHROPIC_MODEL`. |
| `BASE_URL`, `API_KEY`, `MODEL` | Together: OpenAI-compatible endpoint (LiteLLM, vLLM, etc.). Uses `openai` Python SDK with `base_url`. |
| `OPENAI_API_KEY` | Official OpenAI. Optional: `ECHO_OPENAI_MODEL`. |
| `ECHO_INSIGHT_LLM_TIMEOUT` | Seconds for each insights LLM call (default `120`). |
| `ECHO_INSIGHT_DDG_TIMEOUT` | Seconds for each ddgs web search in insights (default `45`). |
| `ECHO_PERPLEXITY_BASE_URL` | OpenAI-compatible root for **Perplexity Sonar** web search (e.g. an AI-gateway Perplexity route, `https://…/v1`). When set together with a key, insights search uses Perplexity first. Alias: `AI_GATEWAY_PERPLEXITY_BASE_URL`. Must be `https://`. |
| `ECHO_PERPLEXITY_API_KEY` | Key for Perplexity search. Falls back to `AI_GATEWAY_API_KEY`, then the shared `API_KEY`. |
| `ECHO_PERPLEXITY_MODEL` | Sonar model (default `sonar`). Alias: `AI_GATEWAY_PERPLEXITY_MODEL`. |
| `ECHO_PERPLEXITY_TIMEOUT` | Seconds per Perplexity request (default `30`; one retry). Keep `ECHO_INSIGHT_DDG_TIMEOUT` larger, since it bounds Perplexity + ddgs fallback together. |
| `ECHO_DDGS_BACKEND` | Metasearch backend for the `ddgs` library: `auto` (default), `bing`, `duckduckgo`, `yahoo`, … (see [ddgs docs](https://pypi.org/project/ddgs/)). If unset and `auto` returns nothing, Echo tries `bing`, then `duckduckgo`, then `yahoo`. |
| `ECHO_LUMINA_APP` | Path to `Lumina.app` used for [knowledge vaults](#knowledge-vault-lumina) (default `/Applications/Lumina.app`). |
| `ECHO_LUMINA_MCP_CMD` | Full Lumina MCP CLI command (shell-style), e.g. a dev build; `--vault <path>` is appended. Overrides `ECHO_LUMINA_APP`. |
| `ECHO_LUMINA_USER_DATA` | Lumina's userData folder, where the vault list is read (default `~/Library/Application Support/Lumina`). |
| `ECHO_NO_SPLASH` | Set to `1` / `true` / `yes` to skip the green ASCII **ECHO** startup splash (e.g. for automation). |
| `ECHO_WHISPER_LANGUAGE` | Whisper `language=` for forced decoder ids (default `japanese` for Kotoba). Set e.g. `english` or `chinese` for other models. |
| `ECHO_LIVE_EMIT_INTERVAL` | Legacy batch mode knob for live transcript flushing. `0` (default) keeps Echo in append-draft mode and uses semantic/pause/length segmentation rules. Positive values switch to interval batching. |
| `ECHO_LIVE_NOISE_TOKENS` | Comma-separated short tokens to suppress in live ASR when they look like low-energy noise hallucinations (default: `ごめん`). |
| `ECHO_LIVE_ALWAYS_DROP_NOISE_TOKENS` | Comma-separated tokens to always suppress when matched (default: `ごめん`), regardless of RMS or segment position. |
| `ECHO_LIVE_NOISE_MAX_CHARS` | Maximum normalized token length eligible for noise suppression (default `4`). |
| `ECHO_LIVE_NOISE_MAX_RMS_FACTOR` | Relative RMS ceiling for noise-token suppression, scaled by `ECHO_LIVE_MIN_RMS` (default `1.3`). |
| `ECHO_LIVE_NOISE_DROP_AT_SEGMENT_START` | If enabled (default `1`), drop configured noise tokens at the start of a new segment even when RMS is not low. |
| `ECHO_LIVE_ASR_LOG_INTERVAL_SECONDS` | Interval for live ASR latency diagnostics in debug/log output (default `2.0`; set `0` to disable). |
| `ECHO_LIVE_CAPTURE_OUT_QUEUE_MAX` | Max items in the live mixed-audio queue read by ASR (default `512`; was `256`). |
| `ECHO_LIVE_CAPTURE_MIX_QUEUE_MAX` | Max items in mic/system per-side queues before mixing (default `256`; was `128`). |
| `ECHO_LIVE_CAPTURE_OUT_Q_DROP_THRESHOLD` | **Live only.** If capture `out_q` depth is **greater than** this, discard the oldest `ECHO_LIVE_CAPTURE_OUT_Q_DROP_BATCH` blocks before draining (default `400`; `0` disables). |
| `ECHO_LIVE_CAPTURE_OUT_Q_DROP_BATCH` | **Live only.** Oldest blocks removed when over threshold (default `30`; ~4s of audio at default blocksize). |
| `ECHO_LIVE_SEGMENT_PAUSE_SECONDS` | Main live UI segmentation trigger: finalize current draft when silence reaches this threshold (default `2.0`). |
| `ECHO_UI_SEGMENT_SEMANTIC_MIN_CHARS` | Minimum chars required before sentence-ending punctuation can finalize a segment (default `80`). |
| `ECHO_UI_SEGMENT_MAX_SECONDS` | Forced UI segment split when a draft grows too long in time (default `25`). |
| `ECHO_UI_SEGMENT_MAX_CHARS` | Forced UI segment split when a draft grows too long in text length (default `360`). |
| `ECHO_INSIGHT_LIVE_MIN_SECONDS` | **`--live` only.** Auxiliary trigger on newly added transcript duration (default `10`). |
| `ECHO_INSIGHT_PAUSE_SECONDS` | **`--live` only.** Idle-window trigger for insight evaluation (default `3`). |
| `ECHO_INSIGHT_PAUSE_MIN_CHARS` | **`--live` only.** Minimum new chars when triggering insights from idle pause (default `80`). |
| `ECHO_INSIGHT_FINALIZED_SEGMENT_MIN_CHARS` | **`--live` only.** Minimum chars for a newly finalized segment to trigger insight directly (default `100`). |
| `ECHO_INSIGHT_SKIP_MIN_CHARS` | **`--live` only.** Skip insight calls when new content is below this size (default `50`). |
| `ECHO_INSIGHT_LIVE_MIN_INTERVAL` | **`--live` only.** Insight rate limiter in seconds (default `45`). |
| `ECHO_INSIGHT_PAUSE_CAPTURE_Q` | **`--live` only.** Do not start a new insight pass while capture `out_q` depth is at least this (default `400`; `0` disables). |
| `ECHO_ENABLE_TRANSCRIPT_REWRITE` | Enable LLM rewrite-back to Transcript (`1`/`true`, default). Set `0` to keep Transcript raw ASR text and disable LLM rewrite-back. |
| `ECHO_DEBUG` | Enable debug panel/log events. Default is `0` (off). Set `1` to enable debug event output and log-file writing. |
| `ECHO_DEBUG_PANEL_LINES` | Debug panel history cap. Default `0` means unlimited history (no truncation). |
| `ECHO_OFFLINE` / `ECHO_ASSUME_ONLINE` / `ECHO_HF_LOCAL_ONLY` | See **Offline / LLM reachability** below. |
| `ECHO_SYSTEM_AUDIO_CMD` | Override command for `--live-system-authorized` (if `--live-system-authorized-cmd` is omitted). Built-in default auto-compiles a bundled ScreenCaptureKit helper on macOS. Custom command must output `s16le`, 24kHz, stereo PCM to stdout. |

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

### Web search

Insight passes routed to **web search** use **Perplexity Sonar** when `ECHO_PERPLEXITY_BASE_URL` (or `AI_GATEWAY_PERPLEXITY_BASE_URL`) and a key are set; only **cited URLs** become results. If Perplexity is not configured, errors (auth, quota, rate limit, timeout) or returns no citations, Echo falls back to the keyless **ddgs** metasearch (`auto` → `bing` → `duckduckgo` → `yahoo`, or just `ECHO_DDGS_BACKEND`). The Insights web section shows which provider answered and any fallback reason; `echo-probe` reports the active provider.

### Knowledge vault (Lumina)

A meeting can optionally use one of your [Lumina](../prj-lumina) vaults (an LLM-maintained personal wiki) as extra context. Pick it in the setup dialog under *Knowledge vault (optional)*, or switch or clear it during the meeting from the vault pill in the top bar.

- **Insights:** when a vault is attached, the insight router gets a third action, `wiki_lookup`, next to `comment` and `web_search`. It sees the vault name and some page titles, and picks `wiki_lookup` when the conversation touches topics, people, projects, clients or past decisions the wiki likely covers. Echo then searches the vault, reads the top pages and writes a short *From <vault>* insight that cites the `wiki/...` pages; the matched pages are listed under *Vault pages*.
- **Chat:** every question is also searched in the vault; matching pages are added to the prompt, and the answer uses them only when relevant (with the filename cited).
- **Failures never block the meeting:** if Lumina is slow, missing or errors, the insight says so in one line and the next pass carries on; chat simply answers without the vault.
- **How it connects:** Echo starts Lumina's bundled MCP server over stdio (`ELECTRON_RUN_AS_NODE=1 Lumina.app/Contents/MacOS/Lumina …/app.asar/dist/main/mcp/cli.js --vault <path>`), one process per vault in use. Lumina doesn't need to be running. Echo only calls read tools (`search_wiki`, `get_article`, `list_articles`). The vault list is Lumina's own vault history (`~/Library/Application Support/Lumina/llm-wiki-config.json`). The first start of a vault can take up to a minute (index load, OneDrive); Echo warms it up in the background when the vault is selected.
- **Overrides:** `ECHO_LUMINA_APP` (path to `Lumina.app`; also under **Settings → Knowledge vault**), `ECHO_LUMINA_MCP_CMD` (full CLI command, e.g. a dev build; `--vault` is appended), `ECHO_LUMINA_USER_DATA` (Lumina's userData folder).
- The chosen vault is stored as `vault` in `meeting.json`, and remembered for the next meeting (including meetings started from the Zoom/Teams prompt).

### Offline / LLM reachability

- On startup Echo **TCP-probes the configured LLM host** (Anthropic, OpenAI, or your `BASE_URL`). If that fails—or no API keys are set—**Insights and chat are disabled**, but **Transcript** (replay, file ASR, or `--live`) still runs.
- **`ECHO_ASSUME_ONLINE=1`**: skip the LLM probe (useful on split-tunnel / internal-only API routes).
- **`ECHO_OFFLINE=1`**: force local Hugging Face cache for weights and treat the LLM as unavailable.
- **`ECHO_HF_LOCAL_ONLY=1`**: load ASR weights from disk only (no Hub download) without forcing the LLM offline probe to fail.

### VAD and denoising

The live pipeline ships with an energy-only voice gate by default. For noisier rooms (HVAC,
keyboard, music bleed) you can opt into a real **VAD** model and a spectral **denoiser** in
front of Whisper. Both stages live in `echo/audio_filters.py`, are off by default, and
fail safely (missing dep -> single warning, fall back to passthrough) so disabling them is
just a matter of unsetting the env var.

Install the optional extra:

```bash
pip install -e '.[asr,asr-vad]'
```

Recommended starting config (Silero VAD + spectral gating):

```bash
ECHO_LIVE_VAD=silero \
ECHO_LIVE_VAD_MIN_SPEECH_RATIO=0.10 \
ECHO_LIVE_DENOISE=noisereduce \
echo-tui --live
```

| Env var | Default | Notes |
|---|---|---|
| `ECHO_LIVE_VAD` | `off` | `silero`, `webrtc`, or `off`. `silero` automatically falls back to `webrtc` if the model can't load, then to `off`. |
| `ECHO_LIVE_VAD_MIN_SPEECH_RATIO` | `0.10` | Drop a chunk when fewer than this fraction of frames are classified as speech (0..1). |
| `ECHO_LIVE_VAD_SPEECH_PROB_THRESHOLD` | `0.5` | Per-frame Silero probability cutoff (0..1). |
| `ECHO_LIVE_VAD_AGGRESSIVENESS` | `2` | WebRTC mode 0..3 (higher = stricter). |
| `ECHO_LIVE_DENOISE` | `off` | `noisereduce` or `off` (stationary spectral gating). |
| `ECHO_LIVE_DENOISE_STRENGTH` | `0.85` | `prop_decrease` for `noisereduce` (0..1). |

The energy gate (`ECHO_LIVE_MIN_RMS` / `ECHO_LIVE_ACTIVE_AMP`) still runs first as a cheap
pre-filter; the model VAD only sees chunks that already passed it. Rejections show up as
`VAD gate` in the input meter note and as `vad drop ratio=...` lines in the debug log.

**Heads up:** aggressive denoising can *hurt* Whisper accuracy on already-clean audio
(it can over-suppress voiced fricatives). Leave `ECHO_LIVE_DENOISE=off` unless background
noise is a concrete problem, and start with a low strength (`0.4`-`0.6`) before pushing it
higher.

### Glossary / dictionary

Drop a `dictionary.yml` at the repo root (or set `ECHO_DICTIONARY_PATH=/abs/path.yml`) to bias
**LLM prompts** (transcript correction, web synthesis, chat) toward your preferred spellings of
proper nouns, jargon, names, and acronyms. The LLM substitutes the canonical spelling when the
transcript clearly refers to a glossary entry.

The file is mtime-cached, so live edits are picked up without restarting Echo. A starter file
ships at `dictionary.yml.example` — copy it to `dictionary.yml` (gitignored). All three shapes
below are equivalent and flatten to the same deduped list (capped at ~200 terms / 600 chars
when serialized into prompts):

```yaml
# Form 1 - bare list
- EchoApp
- Kotoba-Whisper
- 石井
- FOMC
```

```yaml
# Form 2 - explicit `terms` key
terms:
  - EchoApp
  - 石井
```

```yaml
# Form 3 - cosmetic groupings
people: [石井, 田中]
products: [EchoApp, EchoHub]
acronyms: [FOMC, ARPU]
```

Missing file = feature off (no warnings). Missing PyYAML = one-time hint in the debug log;
LLM calls continue normally without the glossary.

### Test LLM + web search (no TUI)

```bash
echo-probe
# optional: echo-probe --llm-timeout 60 --ddg-timeout 20 --ddg-query "your query"
```

Loads the same `.env` rules as the app (repo root then cwd), prints which LLM route is active, runs one small **keywords/summary** call and one web search (Perplexity when configured, else ddgs; prints the provider), then exits `0` / `1`.

## Terminal app (TUI)

The terminal app (`echo-tui`) runs the same engine in a Textual UI: transcript, insights and chat panels, with demo, replay, audio-file and live input modes.

### Requirements

- Python **3.10+**
- A virtual environment is recommended (macOS Homebrew Python is PEP 668–managed).

### Install

```bash
cd prj-echo
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
```

**Use a dedicated virtualenv, not a conda base env.** On Apple Silicon, macOS kills a process outright (`SIGKILL (Code Signature Invalid)`) when it loads a native extension whose arm64 signature is missing or modified. Long-lived conda base envs often contain such files (for example a modified `lib-dynload/_sqlite3`), and Echo imports several native modules lazily mid-session, so the symptom used to be a silent kill during a meeting plus mouse escape codes left in the terminal.

- On startup, `echo-tui` and the desktop engine import the risky native modules in a child process first and exit with a message naming the broken module. A passing result is remembered per Python environment (`<venv>/.echo-preflight-ok`), so the check only runs again after packages or the interpreter change (`ECHO_SKIP_PREFLIGHT=1` disables it). The desktop engine pre-imports these modules in the background after it is ready.
- Fatal-signal tracebacks are written to `./log/echo_crash_<timestamp>.log`.
- `bin/echo-live` / `bin/echo-live-debug` prefer `.venv/bin/echo-tui` and restore the terminal on exit.
- `echo-probe` reports the interpreter in use, arm64 signature problems, Apple speech model status, and a real Whisper API call.

#### Optional: local speech-to-text (`--audio`)

ASR uses PyTorch + Transformers + Torchaudio. Install the extra group:

```bash
pip install -e ".[asr]"
```

The `[asr]` extra includes **torchcodec**, which recent **torchaudio** builds use to decode **MP3** and many other formats. **ffmpeg** should be on your `PATH` (e.g. `brew install ffmpeg` on macOS). If TorchCodec is awkward to install, `pip install librosa` is enough for Echo’s **librosa fallback** when torchaudio fails. **Live capture** also needs **`sounddevice`** (included in `[asr]`).

First run downloads model weights from Hugging Face (large download; size depends on dtype/device).

With `--audio`, weights are loaded **before** the TUI starts (main thread) to avoid macOS `bad value(s) in fds_to_keep` errors when Hugging Face spawns helpers from a background thread.

> **Note:** The Hub “custom pipeline” for Kotoba-Whisper bundles **speaker diarization** (pyannote, etc.). Echo instead loads the same **`WhisperForConditionalGeneration`** checkpoint and runs **standard Japanese transcribe** in fixed time windows—no diarization, so you do not need pyannote tokens for this demo.

### Usage

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
echo-tui --audio meeting.wav --whisper-device cuda   # local Kotoba only: cpu, cuda, mps
echo-tui --live --asr-backend apple                  # force Apple on-device (macOS)
echo-tui --live --asr-backend api                    # force Whisper API
echo-tui --live --asr-backend local                  # force local Kotoba-Whisper
echo-tui --install-apple-speech ja                   # one-time download of Apple's on-device ja model
```

Same entry point without a script name clash (with the venv activated):

```bash
python -m echo --replay samples/sample_transcript.txt --line-delay 0.8
```

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

### Keyboard shortcuts

| Key | Action |
|-----|--------|
| `p` | Pause / resume (text replay or between ASR chunks). |
| `s` | Save the meeting bundle to the per-meeting output directory (see below). |
| `c` | Clear chat log. |
| `q` | Quit. |

### Save output (`s`)

Pressing `s` writes the current meeting to its output directory:

| File | Contents |
|------|----------|
| `notes.txt` | Plain-text transcript (one line per utterance). |
| `insights.md` | All live insight passes (comments + research summaries + web hits). |
| `meeting_context.md` | Meeting title and the background context entered at start. |
| `summary.md` | **LLM-generated post-meeting summary** with two sections — `## Key takeaways` and `## Next actions`. |
| `notebook.md` | Your own free-form notes (desktop app notebook; empty in the TUI). |
| `notes-assets/` | Images pasted or dropped into the notebook (desktop app only). |
| `insights.json` | Structured insight passes, used to regenerate the summary later. |
| `meeting.json` | Metadata: id, title, context, meeting time, start/end time, duration, mode, language, ASR backend, line count. |

`summary.md` is rendered by the configured LLM using the transcript, the live insight comments,
the meeting title / context, your notebook (treated as authoritative for decisions and action items), and the optional glossary. While the LLM call is in flight the file
contains a `_Generating meeting summary…_` placeholder, then is overwritten when the response
arrives. If no LLM provider is configured (or the call fails), `summary.md` falls back to a static
markdown stub explaining how to enable it.

## Project layout

```
echo/
  __main__.py   # CLI
  splash.py     # ASCII ECHO startup modal
  app.py        # Textual UI (subscriber of the engine)
  engine/       # Headless EchoSession, event types, text helpers, ASR setup
  server.py     # echo-server WebSocket sidecar for the desktop app
  live_session.py   # Live capture → ASR (streaming Apple / windowed API & local) → drafts
  asr_routing.py    # AsrRouter: Apple → Whisper API → local Kotoba, backoff + upgrade
  asr_apple.py      # Apple Speech helper build, model install, streaming recognizer
  asr_api.py        # OpenAI-compatible Whisper API client
  meetings.py       # Saved-meeting archive (list / read / edit / regenerate summary)
  live_audio.py # sounddevice capture (mic + optional loopback)
  macos/SystemAudioCapture.swift  # bundled ScreenCaptureKit helper for --live-system-authorized
  macos/SpeechTranscribe.swift    # bundled on-device Speech helper for ASR (macOS)
  asr.py        # Kotoba-Whisper weights, chunked inference
  audio_filters.py  # Optional VAD (Silero/WebRTC) + denoise (noisereduce)
  llm.py        # Optional Anthropic / OpenAI + fallbacks
  search.py     # DuckDuckGo (threaded)
  wiki.py       # Read-only Lumina vault client (MCP over stdio)
  dictionary.py # Optional glossary loader (dictionary.yml -> LLM prompts)
samples/
  sample_transcript.txt
gui/            # Electron + React desktop app (see "Desktop app")
dictionary.yml.example
pyproject.toml
```

## License

Echo is licensed under the [Apache License 2.0](LICENSE).

Models and dependencies have their own licenses (e.g. Kotoba-Whisper is Apache-2.0 on Hugging Face).
