"""Textual TUI for Echo — offline transcript replay + insights + chat."""

from __future__ import annotations

import asyncio
import math
import os
import re
import time
from dataclasses import dataclass
from datetime import timedelta
from functools import partial
from pathlib import Path
from typing import Iterable, Literal

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Collapsible, Footer, Header, Input, Markdown, RichLog, Rule, Static

from echo.asr import MODEL_ID_DEFAULT
from echo.connectivity import echo_env_hub_local_only, echo_llm_online, hub_local_only_from_env
from echo.splash import SplashScreen
from echo.llm import (
    InsightRouteResult,
    llm_chat_answer,
    llm_route_insight,
    llm_synthesize_web_findings,
)
from echo.search import SearchHit, duckduckgo_search


@dataclass
class InsightPassRecord:
    n: int
    stamp: str
    kind: Literal["comment", "web_search"]
    body_md: str
    routing_md: str | None = None
    corrected_md: str | None = None
    web_md: str | None = None


@dataclass
class TranscriptSegment:
    line_id: int
    line: str
    spoken_text: str
    stable_after: float
    allow_correction: bool


_LEADING_BRACKET_TS = re.compile(r"^(\[[^\]]+\]\s*)")
_LIVE_SENTENCE_END_RE = re.compile(r"[。！？!?]+$")
_LIVE_SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？!?])\s*")
_LIVE_CLAUSE_SPLIT_RE = re.compile(r"(?<=[、，,;；])\s*")
_LIVE_STOP_WORDS_DEFAULT: tuple[str, ...] = (
    "that's all",
    "thank you",
    "over",
    "next topic",
    "以上",
    "以上です",
    "谢谢",
)
_METER_BAR_WIDTH = 26
_METER_FLOOR_DB = -60.0
_METER_HISTORY_WIDTH = 52
_METER_TIMELINE_LEVELS = " .:-=+*#%@"
_METER_CHART_ROWS = 8


def _leading_ts_prefix(line: str) -> str:
    m = _LEADING_BRACKET_TS.match(line)
    return m.group(1) if m else ""


def _strip_leading_ts(line: str) -> str:
    return _LEADING_BRACKET_TS.sub("", line, count=1).strip()


def _normalize_spaces(text: str) -> str:
    return " ".join(text.split())


def _trim_chunk_overlap(prev: str, cur: str, *, min_chars: int = 8, max_chars: int = 80) -> str:
    p = _normalize_spaces(prev)
    c = _normalize_spaces(cur)
    if not c:
        return ""
    if not p:
        return c
    lim = min(len(p), len(c), max_chars)
    for n in range(lim, min_chars - 1, -1):
        if p.endswith(c[:n]):
            return c[n:].lstrip()
    return c


def _split_live_groups(text: str, *, max_chars: int = 36) -> list[str]:
    normalized = _normalize_spaces(text)
    if not normalized:
        return []

    units = [u.strip() for u in _LIVE_SENTENCE_SPLIT_RE.split(normalized) if u.strip()]
    groups: list[str] = []
    for unit in units:
        if len(unit) <= max_chars:
            groups.append(unit)
            continue

        clauses = [c.strip() for c in _LIVE_CLAUSE_SPLIT_RE.split(unit) if c.strip()]
        if len(clauses) <= 1:
            groups.extend(
                unit[i : i + max_chars].strip()
                for i in range(0, len(unit), max_chars)
                if unit[i : i + max_chars].strip()
            )
            continue

        cur = ""
        for clause in clauses:
            if len(clause) > max_chars:
                if cur:
                    groups.append(cur)
                    cur = ""
                groups.extend(
                    clause[i : i + max_chars].strip()
                    for i in range(0, len(clause), max_chars)
                    if clause[i : i + max_chars].strip()
                )
                continue
            if not cur:
                cur = clause
                continue
            if len(cur) + 1 + len(clause) <= max_chars:
                cur = f"{cur} {clause}"
                continue
            groups.append(cur)
            cur = clause
        if cur:
            groups.append(cur)
    return groups


def _strip_live_tail_punct(text: str) -> str:
    return text.rstrip("。！？!?.,、，,;；:： ")


def _live_hits_stop_word(text: str, stop_words: tuple[str, ...]) -> bool:
    if not stop_words:
        return False
    tail = _strip_live_tail_punct(_normalize_spaces(text)).lower()
    if not tail:
        return False
    return any(tail.endswith(w.lower()) for w in stop_words)


def _amp_to_dbfs(amplitude: float) -> float:
    a = max(0.0, float(amplitude))
    if a <= 1e-8:
        return -120.0
    return 20.0 * math.log10(a)


def _db_to_unit(db: float, floor_db: float = _METER_FLOOR_DB) -> float:
    if db <= floor_db:
        return 0.0
    if db >= 0.0:
        return 1.0
    return (db - floor_db) / abs(floor_db)


def _meter_bar(unit: float, width: int = _METER_BAR_WIDTH) -> str:
    v = max(0.0, min(1.0, float(unit)))
    filled = int(round(v * width))
    return "#" * filled + "-" * (width - filled)


def _fmt_db(db: float) -> str:
    if db <= -119.0:
        return "-inf"
    return f"{db:5.1f}"


def _meter_timeline(units: list[float], width: int = _METER_HISTORY_WIDTH) -> str:
    if width <= 0:
        return ""
    levels = _METER_TIMELINE_LEVELS
    top = len(levels) - 1
    tail = units[-width:]
    if len(tail) < width:
        tail = ([0.0] * (width - len(tail))) + tail
    out: list[str] = []
    for u in tail:
        v = max(0.0, min(1.0, float(u)))
        idx = int(round(v * top))
        out.append(levels[idx])
    return "".join(out)


def _meter_vertical_chart(
    units: list[float],
    *,
    width: int = _METER_HISTORY_WIDTH,
    rows: int = _METER_CHART_ROWS,
) -> list[str]:
    """Render a vertical equalizer-like chart (bottom darker, top lighter)."""
    if width <= 0 or rows <= 0:
        return []
    tail = units[-width:]
    if len(tail) < width:
        tail = ([0.0] * (width - len(tail))) + tail
    heights = [max(0, min(rows, int(round(max(0.0, min(1.0, u)) * rows)))) for u in tail]

    out: list[str] = []
    for r in range(rows, 0, -1):
        if r <= rows // 3:
            blk = "██"
        elif r <= (rows * 2) // 3:
            blk = "▓▓"
        else:
            blk = "▒▒"
        line = "".join(blk if h >= r else "  " for h in heights)
        out.append(line)
    return out


def _live_chunk_metrics(
    samples: object,
    *,
    active_amp: float,
) -> tuple[float, float, float]:
    import numpy as np

    x = np.asarray(samples, dtype=np.float32).reshape(-1)
    if x.size == 0:
        return 0.0, 0.0, 0.0
    absx = np.abs(x)
    rms = float(np.sqrt(np.mean(np.square(x), dtype=np.float64)))
    peak = float(absx.max())
    active_ratio = float(np.mean(absx >= max(active_amp, 1e-8)))
    return rms, peak, active_ratio


def _passes_live_voice_gate(
    *,
    rms: float,
    peak: float,
    active_ratio: float,
    min_rms: float,
    active_amp: float,
    min_active_ratio: float,
) -> bool:
    # Gate low-energy chunks before Whisper to reduce "silence hallucinations".
    if rms >= min_rms:
        return True
    if active_ratio >= min_active_ratio and peak >= active_amp * 1.2:
        return True
    return False


def _should_trigger_insight_live(
    *,
    eligible_segments: int,
    idle_tick: bool,
    idle_for: float,
    min_segments: int,
    pause_seconds: float,
) -> tuple[bool, str]:
    if eligible_segments <= 0:
        return False, ""
    if eligible_segments >= max(1, min_segments):
        return True, "segment_count"
    if idle_tick and idle_for >= max(0.0, pause_seconds):
        return True, "pause"
    return False, ""


def _insight_corrected_segment_md(raw: str, corrected: str) -> str | None:
    """Markdown for collapsible when routing returned a materially different cleaned segment."""
    c = corrected.strip()
    if not c:
        return None
    if " ".join(raw.split()) == " ".join(c.split()):
        return None
    safe = c.replace("```", "′′′")
    return (
        "_Routing-time segment (ASR / wording cleanup; same facts as raw chunk):_\n\n"
        f"```text\n{safe}\n```"
    )


def _transcript_lines_after_correction(source_lines: list[str], corrected: str) -> list[str] | None:
    """If ``corrected`` materially changes the chunk, return new Transcript lines; else ``None``."""
    raw = "\n".join(source_lines)
    if _insight_corrected_segment_md(raw, corrected) is None:
        return None
    c = corrected.strip()
    if not c:
        return None
    parts = [ln.strip() for ln in c.splitlines() if ln.strip()]
    if not parts:
        return None
    if len(parts) == len(source_lines):
        out: list[str] = []
        for old, p in zip(source_lines, parts, strict=True):
            if p.lstrip().startswith("["):
                out.append(p)
            else:
                out.append(_leading_ts_prefix(old) + p)
        return out
    if len(parts) == 1 and len(source_lines) > 1:
        return [_leading_ts_prefix(source_lines[0]) + parts[0]]
    out2: list[str] = []
    for i, p in enumerate(parts):
        old = source_lines[min(i, len(source_lines) - 1)]
        if p.startswith("["):
            out2.append(p)
        else:
            pref = _leading_ts_prefix(old)
            out2.append(pref + p if pref else p)
    return out2


def _spoken_lines_after_correction(source_spoken: list[str], corrected: str) -> list[str] | None:
    raw = "\n".join(source_spoken)
    c = corrected.strip()
    if not c:
        return None
    if " ".join(raw.split()) == " ".join(c.split()):
        return None
    parts = [ln.strip() for ln in c.splitlines() if ln.strip()]
    if len(parts) != len(source_spoken):
        return None
    return parts


def _insight_routing_collapsible(route: InsightRouteResult, *, web_path: bool) -> str | None:
    if web_path:
        return f"**Choice:** web search\n\n**Reason:** {route.reason or '—'}"
    if route.reason in ("", "no_llm"):
        return None
    return f"**Choice:** comment / draft\n\n**Reason:** {route.reason}"


def _format_web_search_md(query: str, hits: list[SearchHit], ddg_err: str) -> str:
    lines: list[str] = []
    if ddg_err:
        lines.append(f"> {ddg_err}")
        lines.append("")
    if query.strip():
        lines.append(f"_Query:_ `{query}`")
        lines.append("")
    if hits:
        for h in hits:
            title = h.title or "(no title)"
            body = (h.body or "").replace("\n", " ")
            if h.href:
                lines.append(f"- [{title}]({h.href}) — {body}")
            else:
                lines.append(f"- **{title}** — {body}")
    else:
        lines.append("- (no results, offline, or error above)")
    return "\n".join(lines).strip()


DEFAULT_DEMO_LINES: list[str] = [
    "[00:01] 今日の議題は四半期レビューと予算の見直しです。",
    "[00:12] まず予算について、マーケティングは前四半期比で 8% 増です。",
    "[00:28] Q4 の売上は計画を上回りましたが、在庫コストが圧迫しています。",
    "[00:45] 次に、来期のキャップエクスをどう配分するか決めたいです。",
    "[01:02] 予算の再配分案は金曜までにドラフトを共有しましょう。",
    "[01:18] それでは次のトピックに移ります。",
]


class EchoApp(App[None]):
    """Offline-first meeting assistant demo."""

    TITLE = "Echo"
    SUB_TITLE = "🎙 Echo"

    CSS = """
    Screen {
        layout: vertical;
    }
    #main_row {
        height: 1fr;
        min-height: 12;
    }
    #left_column {
        width: 1fr;
        height: 1fr;
        layout: vertical;
    }
    #input_meter {
        height: 11;
        min-height: 11;
        border: tall $warning;
        margin: 0 1 1 0;
    }
    #live_preview {
        height: 3;
        min-height: 3;
        border: tall $secondary;
        margin: 0 1 1 0;
    }
    #debug_panel {
        height: 7;
        min-height: 7;
        border: tall $panel;
        margin: 0 1 1 0;
    }
    #transcript {
        height: 1fr;
        min-height: 6;
        border: tall $accent;
        margin: 0 1 0 0;
        overflow-x: hidden;
    }
    #insights {
        width: 1fr;
        height: 1fr;
        border: tall $primary;
        margin: 0 0 0 1;
    }
    #insights:focus {
        border: tall $success;
    }
    #insights Markdown {
        width: 100%;
        overflow-x: hidden;
    }
    #bottom {
        height: auto;
        max-height: 16;
        min-height: 6;
        border: tall $success;
        margin-top: 1;
    }
    #chat_log {
        height: 1fr;
        min-height: 3;
        max-height: 11;
        margin-bottom: 1;
        overflow-x: hidden;
    }
    #chat_input {
        dock: bottom;
        height: auto;
    }
    """

    BINDINGS = [
        ("p", "toggle_pause", "Pause"),
        ("s", "save_transcript", "Save"),
        ("c", "clear_chat", "Clear chat"),
        ("i", "focus_insights", "Insights"),
        ("t", "focus_transcript", "Transcript"),
        ("q", "quit", "Quit"),
    ]

    def __init__(
        self,
        replay_path: Path | None = None,
        line_delay: float = 1.0,
        insight_min_chars: int = 120,
        insight_min_interval: float = 25.0,
        audio_path: Path | None = None,
        model_id: str = MODEL_ID_DEFAULT,
        chunk_seconds: float = 30.0,
        whisper_device: str | None = None,
        llm_offline: bool = False,
        live_audio: bool = False,
        live_mic_device: int | None = None,
        live_system_device: int | None = None,
        live_system_authorized: bool = False,
        live_system_authorized_cmd: str | None = None,
        live_emit_interval: float = 15.0,
        live_group_delay: float = 0.16,
        live_group_chars: int = 36,
        live_stop_words: tuple[str, ...] | None = None,
    ) -> None:
        super().__init__()
        self.replay_path = replay_path
        self.line_delay = line_delay
        self.insight_min_chars = insight_min_chars
        self.insight_min_interval = insight_min_interval
        self.audio_path = audio_path
        self.live_audio = live_audio
        self.live_mic_device = live_mic_device
        self.live_system_device = live_system_device
        self.live_system_authorized = live_system_authorized
        self.live_system_authorized_cmd = live_system_authorized_cmd
        self.live_emit_interval = max(2.0, live_emit_interval)
        self.live_group_delay = max(0.0, live_group_delay)
        self.live_group_chars = max(12, live_group_chars)
        if live_stop_words is None:
            env_words = os.environ.get("ECHO_LIVE_STOP_WORDS", "")
            words = [w.strip() for w in env_words.split(",") if w.strip()]
            self.live_stop_words = tuple(words) if words else _LIVE_STOP_WORDS_DEFAULT
        else:
            self.live_stop_words = tuple(w.strip() for w in live_stop_words if w.strip())
        self.model_id = model_id
        self.chunk_seconds = chunk_seconds
        self.whisper_device = whisper_device
        self._llm_offline = llm_offline
        self._live_wall_t0: float | None = None

        self._paused = False
        self._t0 = time.monotonic()
        self._lines: list[str] = []
        self._line_id_to_index: dict[int, int] = {}
        self._line_id_to_spoken: dict[int, str] = {}
        self._next_line_id = 1
        self._transcript_log_prefix: list[str] = []
        self._transcript_queue: asyncio.Queue[object] = asyncio.Queue()
        self._insight_buffer: list[TranscriptSegment] = []
        self._last_insight_at = time.monotonic()
        self._last_transcript_activity_at = time.monotonic()
        self._insight_lock = asyncio.Lock()
        self._eof_sentinel = "__ECHO_EOF__"
        self._insight_passes: list[InsightPassRecord] = []
        self._live_meter_rms = 0.0
        self._live_meter_peak = 0.0
        self._live_meter_last_signal_at = 0.0
        self._live_meter_note = "Listening..." if self.live_audio else "Available in --live mode."
        self.live_meter_history_width = max(
            16,
            int(os.environ.get("ECHO_LIVE_METER_HISTORY_WIDTH", str(_METER_HISTORY_WIDTH))),
        )
        self._live_rms_hist: list[float] = [0.0] * self.live_meter_history_width
        self._live_peak_hist: list[float] = [0.0] * self.live_meter_history_width
        self._live_preview_text = ""
        self.live_char_delay = max(
            0.0,
            float(os.environ.get("ECHO_LIVE_CHAR_DELAY", "0.025")),
        )
        self.live_min_rms = max(
            0.0,
            float(os.environ.get("ECHO_LIVE_MIN_RMS", "0.0035")),
        )
        self.live_active_amp = max(
            1e-4,
            float(os.environ.get("ECHO_LIVE_ACTIVE_AMP", "0.012")),
        )
        self.live_min_active_ratio = min(
            1.0,
            max(
                0.0,
                float(os.environ.get("ECHO_LIVE_MIN_ACTIVE_RATIO", "0.015")),
            ),
        )
        self.live_chunk_overlap_seconds = max(
            0.0,
            float(os.environ.get("ECHO_LIVE_CHUNK_OVERLAP_SECONDS", "0.5")),
        )
        self.live_rewrite_stable_seconds = max(
            0.0,
            float(os.environ.get("ECHO_LIVE_REWRITE_STABLE_SECONDS", "10.0")),
        )
        self.insight_min_segments = max(
            1,
            int(os.environ.get("ECHO_INSIGHT_MIN_SEGMENTS", "4")),
        )
        self.insight_pause_seconds = max(
            0.2,
            float(os.environ.get("ECHO_INSIGHT_PAUSE_SECONDS", "2.2")),
        )
        self.debug_event_limit = max(
            6,
            int(os.environ.get("ECHO_DEBUG_PANEL_LINES", "14")),
        )
        self._debug_events: list[str] = []
        self._last_silence_debug_at = 0.0

    def _elapsed_hms(self) -> str:
        dt = timedelta(seconds=int(time.monotonic() - self._t0))
        h, r = divmod(dt.seconds, 3600)
        m, s = divmod(r, 60)
        if dt.days:
            h += dt.days * 24
        return f"{h:02d}:{m:02d}:{s:02d}"

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="main_row"):
            with Vertical(id="left_column"):
                yield Static(self._input_meter_text(), id="input_meter", markup=False)
                yield Static(self._live_preview_view(), id="live_preview", markup=False)
                yield Static(self._debug_panel_text(), id="debug_panel", markup=False)
                yield RichLog(
                    id="transcript",
                    highlight=True,
                    markup=True,
                    wrap=True,
                    min_width=0,
                )
                with Vertical(id="bottom"):
                    yield RichLog(
                        id="chat_log",
                        highlight=True,
                        markup=True,
                        wrap=True,
                        min_width=0,
                    )
                    yield Input(
                        placeholder=(
                            "LLM unavailable — chat disabled"
                            if self._llm_offline
                            else "Ask about this meeting…"
                        ),
                        id="chat_input",
                        disabled=self._llm_offline,
                    )
            with VerticalScroll(id="insights"):
                yield Markdown(
                    _llm_unavailable_insights_markdown()
                    if self._llm_offline
                    else _initial_insights_markdown()
                )
        yield Footer()

    def on_mount(self) -> None:
        if self._llm_offline:
            self.query_one("#chat_log", RichLog).write(
                "[dim]LLM unavailable (no API keys or TCP probe to the configured host failed): "
                "Insights and chat are disabled. The Transcript panel still works (replay or ASR).[/]"
            )
        if os.environ.get("ECHO_NO_SPLASH", "").lower() not in ("1", "true", "yes"):
            self.push_screen(SplashScreen())
        if self.live_audio:
            mode = "Live ASR"
        elif self.audio_path is not None:
            mode = "ASR file"
        else:
            mode = "Text replay"
        off = " · no LLM" if self._llm_offline else ""
        self.sub_title = f"🎙 {mode}{off}  [{self._elapsed_hms()}]  p/s/c/q"
        self.set_interval(1.0, self._tick_header)
        self.set_interval(0.15, self._tick_input_meter)
        self.set_interval(0.12, self._tick_live_preview)
        self.set_interval(0.2, self._tick_debug_panel)
        self._debug_note(
            "startup "
            f"mode={mode} live={self.live_audio} stable={self.live_rewrite_stable_seconds:.1f}s "
            f"overlap={self.live_chunk_overlap_seconds:.2f}s "
            f"insight[min_segments={self.insight_min_segments},pause={self.insight_pause_seconds:.1f}s]",
        )
        # Keep exclusive=False so the insight loop does not cancel transcription.
        self.run_worker(self._transcription_worker(), exclusive=False)
        self.run_worker(self._insight_worker(), exclusive=False)

    def _tick_header(self) -> None:
        if self.live_audio:
            state = "Paused" if self._paused else "Listening"
        elif self.audio_path is not None:
            state = "Paused" if self._paused else "Transcribing"
        else:
            state = "Paused" if self._paused else "Replaying"
        if self.live_audio:
            mode = "Live"
        elif self.audio_path is not None:
            mode = "ASR"
        else:
            mode = "Text"
        off = " · no LLM" if self._llm_offline else ""
        self.sub_title = f"🎙 {mode} · {state}{off}  [{self._elapsed_hms()}]  p/s/c/q"

    def _input_meter_text(self) -> str:
        if not self.live_audio:
            return "Input level\navailable in --live mode"

        rms_db = _amp_to_dbfs(self._live_meter_rms)
        peak_db = _amp_to_dbfs(self._live_meter_peak)
        rms_unit = _db_to_unit(rms_db)
        peak_unit = _db_to_unit(peak_db)

        age = (
            time.monotonic() - self._live_meter_last_signal_at
            if self._live_meter_last_signal_at > 0
            else float("inf")
        )
        if self._paused:
            state = "Paused"
        elif age > 2.0:
            state = "No recent signal"
        else:
            state = self._live_meter_note or "Listening"

        rms_bar = _meter_bar(rms_unit)
        peak_bar = _meter_bar(peak_unit)
        # Use mostly peak dynamics; blend in RMS for smoother columns.
        blend = [
            max(0.0, min(1.0, (p * 0.75) + (r * 0.25)))
            for r, p in zip(self._live_rms_hist, self._live_peak_hist, strict=False)
        ]
        chart = _meter_vertical_chart(
            blend,
            width=self.live_meter_history_width,
            rows=_METER_CHART_ROWS,
        )
        chart_block = "\n".join(chart)
        return (
            f"Input level · {state}\n"
            f"RMS  [{rms_bar}] {_fmt_db(rms_db)} dBFS\n"
            f"Peak [{peak_bar}] {_fmt_db(peak_db)} dBFS\n"
            f"{chart_block}"
        )

    def _tick_input_meter(self) -> None:
        if self.live_audio:
            rms_unit = _db_to_unit(_amp_to_dbfs(self._live_meter_rms))
            peak_unit = _db_to_unit(_amp_to_dbfs(self._live_meter_peak))
            self._live_rms_hist.append(rms_unit)
            self._live_peak_hist.append(peak_unit)
            if len(self._live_rms_hist) > self.live_meter_history_width:
                self._live_rms_hist = self._live_rms_hist[-self.live_meter_history_width :]
            if len(self._live_peak_hist) > self.live_meter_history_width:
                self._live_peak_hist = self._live_peak_hist[-self.live_meter_history_width :]
        meter = self.query_one("#input_meter", Static)
        meter.update(self._input_meter_text())

    def _live_preview_view(self) -> str:
        if not self.live_audio:
            return "Local ASR preview\navailable in --live mode"
        if self._paused:
            state = "Local ASR preview · paused"
        elif self._live_preview_text:
            state = "Local ASR preview · typing"
        else:
            state = "Local ASR preview · waiting"
        preview = self._live_preview_text or "..."
        return f"{state}\n{preview}"

    def _tick_live_preview(self) -> None:
        box = self.query_one("#live_preview", Static)
        box.update(self._live_preview_view())

    def _debug_note(self, message: str) -> None:
        stamp = self._elapsed_hms()
        line = f"[{stamp}] {message}"
        self._debug_events.append(line)
        if len(self._debug_events) > self.debug_event_limit:
            self._debug_events = self._debug_events[-self.debug_event_limit :]

    def _debug_panel_text(self) -> str:
        if not self._debug_events:
            return "Debug panel\n(no events yet)"
        return "Debug panel\n" + "\n".join(self._debug_events)

    def _tick_debug_panel(self) -> None:
        panel = self.query_one("#debug_panel", Static)
        panel.update(self._debug_panel_text())

    def action_toggle_pause(self) -> None:
        self._paused = not self._paused
        if self.live_audio:
            self._live_meter_note = "Paused" if self._paused else "Listening"
            if self._paused:
                self._live_preview_text = ""
            self._tick_live_preview()
        self._debug_note(f"pause toggled -> {self._paused}")

    def action_save_transcript(self) -> None:
        path = Path.cwd() / f"echo_transcript_{int(time.time())}.txt"
        path.write_text("\n".join(self._lines), encoding="utf-8")
        self.notify(f"Saved {path}", timeout=4)

    def action_clear_chat(self) -> None:
        chat = self.query_one("#chat_log", RichLog)
        chat.clear()
        if self._llm_offline:
            chat.write(
                "[dim]LLM unavailable: Insights and chat are disabled. Transcript still works.[/]"
            )

    def action_focus_insights(self) -> None:
        self.query_one("#insights", VerticalScroll).focus()

    def action_focus_transcript(self) -> None:
        self.query_one("#transcript", RichLog).focus()

    def _log_transcript_meta(self, log: RichLog, text: str) -> None:
        """Status / error lines above replay or ASR lines (preserved when transcript body is rewritten)."""
        self._transcript_log_prefix.append(text)
        log.write(text)

    async def _refresh_transcript_log(self) -> None:
        log = self.query_one("#transcript", RichLog)
        log.clear()
        for m in self._transcript_log_prefix:
            log.write(m)
        for line in self._lines:
            log.write(line)

    async def _emit_transcript_line(
        self,
        log: RichLog,
        line: str,
        *,
        spoken_text: str | None = None,
        allow_correction: bool = True,
        stable_delay: float = 0.0,
    ) -> TranscriptSegment:
        spoken = _normalize_spaces(spoken_text or _strip_leading_ts(line))
        seg = TranscriptSegment(
            line_id=self._next_line_id,
            line=line,
            spoken_text=spoken,
            stable_after=time.monotonic() + max(0.0, stable_delay),
            allow_correction=allow_correction,
        )
        self._next_line_id += 1
        self._line_id_to_index[seg.line_id] = len(self._lines)
        self._line_id_to_spoken[seg.line_id] = seg.spoken_text
        self._lines.append(line)
        log.write(line)
        await self._transcript_queue.put(seg)
        if self.live_audio:
            self._debug_note(
                f"emit segment id={seg.line_id} stable={max(seg.stable_after - time.monotonic(), 0.0):.1f}s allow_fix={seg.allow_correction}",
            )
        return seg

    async def _flush_live_group(
        self,
        log: RichLog,
        *,
        t0w: float,
        t1w: float,
        text: str,
        allow_correction: bool,
        pretyped: bool = False,
    ) -> None:
        from echo.asr import format_timestamp_hms

        payload = _normalize_spaces(text)
        if not payload:
            return
        groups = _split_live_groups(payload, max_chars=self.live_group_chars)
        if not groups:
            return

        ts = f"[{format_timestamp_hms(t0w)}–{format_timestamp_hms(t1w)}]"
        for i, g in enumerate(groups):
            await self._typewriter_emit_line(
                log,
                ts,
                g,
                allow_correction=allow_correction,
                pretyped=pretyped,
            )
            if i < len(groups) - 1 and self.live_group_delay > 0:
                await asyncio.sleep(self.live_group_delay)

    async def _typewriter_preview_append(self, ts: str, text: str) -> None:
        if not self.live_audio:
            return
        incoming = _normalize_spaces(text)
        if not incoming:
            return
        current = self._live_preview_text
        if not current.startswith(f"{ts} "):
            current = f"{ts} "
        if self.live_char_delay <= 0:
            self._live_preview_text = current + incoming
            self._tick_live_preview()
            return
        for ch in incoming:
            current += ch
            self._live_preview_text = current
            self._tick_live_preview()
            await asyncio.sleep(self.live_char_delay)

    async def _typewriter_emit_line(
        self,
        log: RichLog,
        ts: str,
        text: str,
        *,
        allow_correction: bool,
        pretyped: bool = False,
    ) -> None:
        line = f"{ts} {text}"
        if pretyped or not self.live_audio or self.live_char_delay <= 0:
            self._live_preview_text = ""
            self._tick_live_preview()
            await self._emit_transcript_line(
                log,
                line,
                spoken_text=text,
                allow_correction=allow_correction,
                stable_delay=self.live_rewrite_stable_seconds if self.live_audio else 0.0,
            )
            return

        typed = ""
        for ch in text:
            typed += ch
            self._live_preview_text = f"{ts} {typed}"
            self._tick_live_preview()
            await asyncio.sleep(self.live_char_delay)

        self._live_preview_text = ""
        self._tick_live_preview()
        # First flush (local ASR) completes here; only then enqueue for LLM insight passes.
        await self._emit_transcript_line(
            log,
            line,
            spoken_text=text,
            allow_correction=allow_correction,
            stable_delay=self.live_rewrite_stable_seconds if self.live_audio else 0.0,
        )

    async def _transcription_worker(self) -> None:
        log = self.query_one("#transcript", RichLog)
        if self.live_audio:
            await self._transcription_from_live(log)
            await self._transcript_queue.put(self._eof_sentinel)
            return
        if self.audio_path is not None:
            await self._transcription_from_audio(log)
            await self._transcript_queue.put(self._eof_sentinel)
            return

        lines = _load_replay_lines(self.replay_path)
        for raw in lines:
            while self._paused:
                await asyncio.sleep(0.15)
            line = raw.strip()
            if not line:
                continue
            if not line.startswith("["):
                line = f"[{self._elapsed_hms()}] {line}"
            await self._emit_transcript_line(log, line)
            await asyncio.sleep(self.line_delay)
        await self._transcript_queue.put(self._eof_sentinel)

    async def _transcription_from_audio(self, log: RichLog) -> None:
        from echo.asr import TranscribeSession, format_timestamp_hms

        path = self.audio_path
        if path is None or not path.is_file():
            self._log_transcript_meta(log, f"[red]Audio file not found:[/] {path}")
            return

        load_msg = (
            f"[dim]Loading `{self.model_id}` from local cache (no Hugging Face Hub download)…[/]"
            if hub_local_only_from_env()
            else f"[dim]Loading `{self.model_id}` (first run may download ~3GB of weights)…[/]"
        )
        self._log_transcript_meta(log, load_msg)
        try:
            session = await asyncio.to_thread(
                partial(
                    TranscribeSession,
                    path,
                    model_id=self.model_id,
                    chunk_seconds=self.chunk_seconds,
                    device=self.whisper_device,
                ),
            )
        except RuntimeError as e:
            self._log_transcript_meta(log, f"[red]{e}[/]")
            return
        except Exception as e:  # noqa: BLE001
            self._log_transcript_meta(log, f"[red]ASR init failed:[/] {e!s}")
            return

        self._log_transcript_meta(log, "[dim]Transcription started (p = pause between chunks).[/]")
        while True:
            while self._paused:
                await asyncio.sleep(0.15)
            try:
                seg = await asyncio.to_thread(session.next_segment)
            except Exception as e:  # noqa: BLE001
                self._log_transcript_meta(log, f"[red]Chunk failed:[/] {e!s}")
                break
            if seg is None:
                break
            t0, t1, text = seg
            if not text.strip():
                continue
            line = f"[{format_timestamp_hms(t0)}–{format_timestamp_hms(t1)}] {text.strip()}"
            await self._emit_transcript_line(
                log,
                line,
                spoken_text=text.strip(),
            )

    async def _transcription_from_live(self, log: RichLog) -> None:
        import numpy as np
        from echo.asr import format_timestamp_hms, transcribe_mono16k_chunk
        from echo.live_audio import (
            LiveMixerConfig,
            LivePairedCapture,
            LiveSystemAuthorizedCapture,
            LiveSystemAuthorizedConfig,
        )

        self._log_transcript_meta(
            log,
            (
                f"[dim]Live ASR (`{self.model_id}`) — local Hub cache only, no download.[/]"
                if hub_local_only_from_env()
                else f"[dim]Live ASR (`{self.model_id}`) — model already loaded on the main thread.[/]"
            ),
        )

        if self.live_system_authorized:
            from echo.live_audio import ensure_macos_system_audio_helper

            cmd = self.live_system_authorized_cmd or os.environ.get("ECHO_SYSTEM_AUDIO_CMD")
            if not cmd:
                try:
                    cmd = await asyncio.to_thread(ensure_macos_system_audio_helper)
                except Exception as e:  # noqa: BLE001
                    self._log_transcript_meta(
                        log,
                        "[red]Failed to build bundled macOS system-audio helper:[/] "
                        f"{e!s}\n"
                        "[dim]You can still provide an explicit command via "
                        "--live-system-authorized-cmd '/path/to/cmd'.[/]",
                    )
                    return
            self._log_transcript_meta(
                log,
                "[dim]Live: microphone + system output via authorized macOS capture command "
                "(Screen Recording + Microphone permissions may be requested). "
                f"Transcript flushes on stop words/punctuation or every ~{self.live_emit_interval:.0f}s.[/]",
            )
        elif self.live_system_device is None:
            self._log_transcript_meta(
                log,
                "[dim]Live: microphone only. Add --live-system-device (e.g. BlackHole 2ch index) "
                "to mix system audio, or create an aggregate input in Audio MIDI Setup. "
                f"Transcript flushes on stop words/punctuation or every ~{self.live_emit_interval:.0f}s.[/]",
            )
        else:
            self._log_transcript_meta(
                log,
                "[dim]Live: microphone + second input (50/50 mono mix). "
                "p = pause (discards audio while paused). "
                f"Transcript flushes on stop words/punctuation or every ~{self.live_emit_interval:.0f}s.[/]",
            )

        if self.live_system_authorized:
            cap = LiveSystemAuthorizedCapture(
                LiveSystemAuthorizedConfig(
                    mic_device=self.live_mic_device,
                    command=cmd,
                    samplerate=16000,
                    blocksize=2048,
                )
            )
            try:
                cap.start()
            except Exception as e:  # noqa: BLE001
                self._log_transcript_meta(
                    log,
                    "[red]Failed to start system-authorized capture:[/] "
                    f"{e!s}\n"
                    "[dim]Tip: install/build SystemAudioDump and pass "
                    "--live-system-authorized-cmd '/path/to/SystemAudioDump'.[/]",
                )
                return
        else:
            cfg = LiveMixerConfig(
                mic_device=self.live_mic_device,
                system_device=self.live_system_device,
                samplerate=16000,
                blocksize=2048,
            )
            cap = LivePairedCapture(cfg)
            cap.start()
        sr = 16000
        chunk_samples = max(int(self.chunk_seconds * sr), int(0.5 * sr))
        overlap_samples = max(0, int(self.live_chunk_overlap_seconds * sr))
        if overlap_samples >= chunk_samples:
            overlap_samples = max(0, chunk_samples // 2)
        hop_samples = max(1, chunk_samples - overlap_samples)
        buf = np.array([], dtype=np.float32)
        self._live_wall_t0 = time.monotonic()
        lang = (os.environ.get("ECHO_WHISPER_LANGUAGE") or "japanese").strip().lower() or "japanese"
        self._live_meter_note = "Listening"
        last_chunk_text = ""
        pending_texts: list[str] = []
        pending_t0w: float | None = None
        pending_t1w: float | None = None
        pending_allow_correction = True
        pending_pretyped = False

        async def _flush_pending() -> None:
            nonlocal pending_texts, pending_t0w, pending_t1w, pending_allow_correction, pending_pretyped
            if not pending_texts or pending_t0w is None or pending_t1w is None:
                return
            await self._flush_live_group(
                log,
                t0w=pending_t0w,
                t1w=pending_t1w,
                text=" ".join(pending_texts),
                allow_correction=pending_allow_correction,
                pretyped=pending_pretyped,
            )
            self._debug_note(
                f"flush local lines={len(pending_texts)} allow_fix={pending_allow_correction}",
            )
            pending_texts = []
            pending_t0w = None
            pending_t1w = None
            pending_allow_correction = True
            pending_pretyped = False

        try:
            while True:
                while self._paused:
                    await _flush_pending()
                    await asyncio.to_thread(cap.read_block, 0.2)
                    await asyncio.sleep(0.05)

                block = await asyncio.to_thread(cap.read_block, 0.35)
                if block is None:
                    if self.live_system_authorized and hasattr(cap, "process_exited"):
                        if cap.process_exited():
                            msg = ""
                            if hasattr(cap, "process_error_text"):
                                msg = cap.process_error_text()
                            await _flush_pending()
                            self._log_transcript_meta(
                                log,
                                "[red]System-authorized capture stopped:[/] "
                                f"{msg or 'unknown helper failure'}",
                            )
                            self._live_meter_note = "Capture stopped"
                            self._debug_note(f"capture stopped: {msg or 'unknown helper failure'}")
                            self._live_preview_text = ""
                            self._tick_live_preview()
                            return
                    await asyncio.sleep(0.02)
                    continue
                arr = np.asarray(block, dtype=np.float32).reshape(-1)
                if arr.size == 0:
                    await asyncio.sleep(0.02)
                    continue
                block_rms = float(np.sqrt(np.mean(np.square(arr), dtype=np.float64)))
                block_peak = float(np.max(np.abs(arr)))
                self._live_meter_rms = (self._live_meter_rms * 0.75) + (block_rms * 0.25)
                self._live_meter_peak = max(block_peak, self._live_meter_peak * 0.88)
                if block_peak > 1e-4:
                    self._live_meter_last_signal_at = time.monotonic()
                    if not self._paused:
                        self._live_meter_note = "Signal detected"
                buf = np.concatenate([buf, arr])

                while buf.size >= chunk_samples:
                    piece = buf[:chunk_samples]
                    buf = buf[hop_samples:]
                    t0w = time.monotonic() - (self._live_wall_t0 or time.monotonic())
                    t1w = t0w + (chunk_samples / sr)
                    piece_rms, piece_peak, piece_active_ratio = _live_chunk_metrics(
                        piece,
                        active_amp=self.live_active_amp,
                    )
                    if not _passes_live_voice_gate(
                        rms=piece_rms,
                        peak=piece_peak,
                        active_ratio=piece_active_ratio,
                        min_rms=self.live_min_rms,
                        active_amp=self.live_active_amp,
                        min_active_ratio=self.live_min_active_ratio,
                    ):
                        if not self._paused:
                            self._live_meter_note = "Silence gate"
                            now_dbg = time.monotonic()
                            if now_dbg - self._last_silence_debug_at >= 1.8:
                                self._debug_note(
                                    f"gate silence rms={piece_rms:.4f} peak={piece_peak:.4f} active={piece_active_ratio:.3f}",
                                )
                                self._last_silence_debug_at = now_dbg
                        continue
                    try:
                        text = await asyncio.to_thread(
                            transcribe_mono16k_chunk,
                            piece,
                            model_id=self.model_id,
                            device=self.whisper_device,
                            language=lang,
                        )
                    except Exception as e:  # noqa: BLE001
                        self._log_transcript_meta(log, f"[red]Live chunk failed:[/] {e!s}")
                        self._live_meter_note = "ASR error"
                        self._debug_note(f"asr chunk error: {e!s}")
                        continue
                    if not (text or "").strip():
                        continue
                    cleaned = _normalize_spaces(text)
                    if not cleaned:
                        continue
                    deduped = _trim_chunk_overlap(last_chunk_text, cleaned)
                    last_chunk_text = cleaned
                    if not deduped:
                        self._debug_note("dedup dropped fully-overlapped chunk")
                        continue

                    if pending_t0w is None:
                        pending_t0w = t0w
                    pending_t1w = t1w
                    preview_ts = f"[{format_timestamp_hms(pending_t0w)}–{format_timestamp_hms(pending_t1w)}]"
                    await self._typewriter_preview_append(preview_ts, deduped)
                    pending_pretyped = True
                    pending_texts.append(deduped)
                    allow_piece_correction = piece_rms >= (self.live_min_rms * 1.25) or (
                        piece_active_ratio >= (self.live_min_active_ratio * 2.0)
                    )
                    pending_allow_correction = pending_allow_correction and allow_piece_correction

                    window = (pending_t1w - pending_t0w) if pending_t0w is not None else 0.0
                    hit_boundary = bool(_LIVE_SENTENCE_END_RE.search(deduped)) or _live_hits_stop_word(
                        deduped,
                        self.live_stop_words,
                    )
                    if hit_boundary or window >= self.live_emit_interval:
                        await _flush_pending()
        finally:
            await _flush_pending()
            self._live_meter_note = "Stopped"
            self._debug_note("live transcription stopped")
            self._live_preview_text = ""
            self._tick_live_preview()
            cap.stop()

    async def _insight_worker(self) -> None:
        while True:
            try:
                chunk = await asyncio.wait_for(self._transcript_queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                chunk = None
            if chunk == self._eof_sentinel:
                source_segments = list(self._insight_buffer)
                buf_text = "\n".join(seg.spoken_text for seg in source_segments).strip()
                self._insight_buffer.clear()
                if buf_text and not self._llm_offline:
                    asyncio.create_task(self._run_insight_pass(buf_text, source_segments))
                break
            if isinstance(chunk, TranscriptSegment) and not self._llm_offline:
                self._insight_buffer.append(chunk)
                self._last_transcript_activity_at = time.monotonic()
            if self._paused:
                # Pause means "freeze insight generation": keep buffering but do not trigger new passes.
                continue
            now = time.monotonic()
            eligible = [seg for seg in self._insight_buffer if seg.stable_after <= now]
            buf_text = "\n".join(seg.spoken_text for seg in eligible).strip()
            trigger_reason = ""
            if self.live_audio:
                should_run, trigger_reason = _should_trigger_insight_live(
                    eligible_segments=len(eligible),
                    idle_tick=(chunk is None),
                    idle_for=now - self._last_transcript_activity_at,
                    min_segments=self.insight_min_segments,
                    pause_seconds=self.insight_pause_seconds,
                )
            else:
                should_run = bool(buf_text) and (
                    len(buf_text) >= self.insight_min_chars
                    or (now - self._last_insight_at) >= self.insight_min_interval
                )
                if should_run:
                    trigger_reason = "chars_or_interval"
            if chunk is None and not should_run:
                continue
            if not should_run:
                continue

            snapshot = buf_text
            source_segments = list(eligible)
            consumed = {seg.line_id for seg in source_segments}
            self._insight_buffer = [seg for seg in self._insight_buffer if seg.line_id not in consumed]
            self._debug_note(
                "insight trigger "
                f"reason={trigger_reason or '-'} chars={len(snapshot)} "
                f"segments={len(source_segments)} buffered_left={len(self._insight_buffer)}",
            )
            self._last_insight_at = now
            asyncio.create_task(self._run_insight_pass(snapshot, source_segments))

    async def _refresh_insights_panel(self, *, tail: str | None = None) -> None:
        scroll = self.query_one("#insights", VerticalScroll)

        def _scroll_bottom() -> None:
            try:
                scroll.scroll_end(animate=False)
            except Exception:
                # Keep rendering robust across Textual version differences.
                return

        for child in list(scroll.children):
            await child.remove()
        if not self._insight_passes:
            if tail is None:
                md = (
                    _llm_unavailable_insights_markdown()
                    if self._llm_offline
                    else _initial_insights_markdown()
                )
                await scroll.mount(Markdown(md))
            else:
                await scroll.mount(Markdown(f"## Insights\n\n{tail}\n"))
            _scroll_bottom()
            return
        await scroll.mount(Markdown("## Insights"))
        await scroll.mount(Rule())
        for i, rec in enumerate(self._insight_passes):
            if i:
                await scroll.mount(Rule())
            heading = "Comment" if rec.kind == "comment" else "Research summary"
            await scroll.mount(
                Markdown(
                    f"### Pass {rec.n} @ `{rec.stamp}`\n\n"
                    f"## {heading}\n\n{rec.body_md}\n",
                ),
            )
            if rec.routing_md:
                await scroll.mount(
                    Collapsible(
                        Markdown(rec.routing_md),
                        title="Routing",
                        collapsed=True,
                    ),
                )
            if rec.corrected_md:
                await scroll.mount(
                    Collapsible(
                        Markdown(rec.corrected_md),
                        title="Cleaned segment",
                        collapsed=True,
                    ),
                )
            if rec.web_md:
                await scroll.mount(
                    Collapsible(
                        Markdown(rec.web_md),
                        title="Web search",
                        collapsed=True,
                    ),
                )
        if tail:
            await scroll.mount(Rule())
            await scroll.mount(Markdown(tail))
        _scroll_bottom()

    def _apply_segment_text(self, seg: TranscriptSegment, new_spoken: str) -> bool:
        new_text = _normalize_spaces(new_spoken)
        if not new_text:
            return False
        old_text = self._line_id_to_spoken.get(seg.line_id, seg.spoken_text)
        if " ".join(old_text.split()) == " ".join(new_text.split()):
            return False
        idx = self._line_id_to_index.get(seg.line_id)
        if idx is None or idx < 0 or idx >= len(self._lines):
            return False
        old_line = self._lines[idx]
        pref = _leading_ts_prefix(old_line)
        updated = new_text if new_text.startswith("[") else (pref + new_text if pref else new_text)
        self._lines[idx] = updated
        self._line_id_to_spoken[seg.line_id] = new_text
        return True

    def _try_apply_corrected_transcript(
        self,
        source_segments: list[TranscriptSegment],
        corrected: str,
        corrected_segments: list[tuple[int, str]] | None,
    ) -> bool:
        """Apply LLM transcript correction using segment ids first, then strict line-count fallback."""
        if not source_segments:
            return False

        applied = False
        id_map = {seg.line_id: seg for seg in source_segments if seg.allow_correction}
        for sid, txt in corrected_segments or []:
            seg = id_map.get(int(sid))
            if seg is None:
                continue
            applied = self._apply_segment_text(seg, txt) or applied
        if applied:
            return True

        fallback_segments = [seg for seg in source_segments if seg.allow_correction]
        if not fallback_segments:
            return False
        src_spoken = [seg.spoken_text for seg in fallback_segments]
        repl = _spoken_lines_after_correction(src_spoken, corrected)
        if not repl:
            return False
        out = False
        for seg, new_txt in zip(fallback_segments, repl, strict=True):
            out = self._apply_segment_text(seg, new_txt) or out
        return out

    async def _run_insight_pass(
        self,
        snapshot: str,
        source_segments: list[TranscriptSegment] | None = None,
    ) -> None:
        if self._llm_offline:
            return
        llm_timeout = float(os.environ.get("ECHO_INSIGHT_LLM_TIMEOUT", "120"))
        ddg_timeout = float(os.environ.get("ECHO_INSIGHT_DDG_TIMEOUT", "45"))

        async with self._insight_lock:
            next_n = len(self._insight_passes) + 1
            stamp = self._elapsed_hms()
            tail = (
                "_Updating…_"
                if not self._insight_passes
                else f"_[Generating pass {next_n} @ `{stamp}`…]_\n"
            )
            await self._refresh_insights_panel(tail=tail)

            try:
                try:
                    seg_payload = (
                        [(seg.line_id, seg.spoken_text) for seg in source_segments]
                        if source_segments
                        else None
                    )
                    if seg_payload:
                        first_id = seg_payload[0][0]
                        last_id = seg_payload[-1][0]
                        self._debug_note(
                            f"insight pass input segments={len(seg_payload)} ids={first_id}-{last_id}",
                        )
                    route = await asyncio.wait_for(
                        llm_route_insight(snapshot, segments=seg_payload),
                        timeout=llm_timeout,
                    )
                except asyncio.TimeoutError:
                    route = InsightRouteResult(
                        "comment",
                        f"_Insight routing LLM timed out after {llm_timeout:.0f}s "
                        "(check network, `BASE_URL`, API key, or proxy)._",
                        "",
                        "timeout",
                        "",
                    )
                except Exception as e:  # noqa: BLE001
                    route = InsightRouteResult(
                        "comment",
                        f"_Insight routing failed:_ `{e!s}`",
                        "",
                        "error",
                        "",
                    )

                ctx = route.corrected_transcript.strip() or snapshot
                raw_for_md = (
                    "\n".join(seg.spoken_text for seg in source_segments)
                    if source_segments
                    else snapshot
                )
                self._debug_note(
                    f"route action={route.action} structured_edits={len(route.corrected_segments or [])} reason={route.reason or '-'}",
                )
                corrected_md = _insight_corrected_segment_md(raw_for_md, route.corrected_transcript)
                applied = bool(source_segments) and self._try_apply_corrected_transcript(
                    source_segments,
                    route.corrected_transcript,
                    route.corrected_segments,
                )
                if applied:
                    await self._refresh_transcript_log()
                    corrected_md = None
                    self._debug_note("transcript rewrite applied")
                elif source_segments and (route.corrected_segments or route.corrected_transcript.strip()):
                    self._debug_note("transcript rewrite skipped (no eligible mapping)")

                if route.action == "comment":
                    body = route.comment.strip() or "_(empty)_"
                    routing = _insight_routing_collapsible(route, web_path=False)
                    self._insight_passes.append(
                        InsightPassRecord(
                            n=next_n,
                            stamp=stamp,
                            kind="comment",
                            body_md=body,
                            routing_md=routing,
                            corrected_md=corrected_md,
                            web_md=None,
                        ),
                    )
                else:
                    query = route.search_query.strip() or ctx[:100].strip()
                    hits: list[SearchHit] = []
                    ddg_err = ""
                    if query:
                        try:
                            hits = await asyncio.wait_for(
                                duckduckgo_search(query, max_results=5),
                                timeout=ddg_timeout,
                            )
                        except asyncio.TimeoutError:
                            ddg_err = (
                                f"ddgs search timed out after {ddg_timeout:.0f}s "
                                "(network, rate limit, or proxy). Try ECHO_DDGS_BACKEND=bing."
                            )
                        except Exception as e:  # noqa: BLE001
                            ddg_err = f"ddgs: `{e!s}`"
                    try:
                        summary = await asyncio.wait_for(
                            llm_synthesize_web_findings(ctx, query, hits),
                            timeout=llm_timeout,
                        )
                    except asyncio.TimeoutError:
                        summary = (
                            f"_Web synthesis LLM timed out after {llm_timeout:.0f}s; "
                            f"showing search excerpts._\n\n**Query:** `{query}`"
                        )
                    except Exception as e:  # noqa: BLE001
                        summary = f"_Web synthesis failed:_ `{e!s}`\n\n**Query:** `{query}`"

                    web_md = _format_web_search_md(query, hits, ddg_err)
                    routing = _insight_routing_collapsible(route, web_path=True)
                    self._insight_passes.append(
                        InsightPassRecord(
                            n=next_n,
                            stamp=stamp,
                            kind="web_search",
                            body_md=summary.strip() or "_(empty)_",
                            routing_md=routing,
                            corrected_md=corrected_md,
                            web_md=web_md,
                        ),
                    )
                await self._refresh_insights_panel(tail=None)
            except Exception as e:  # noqa: BLE001
                self.log.exception("insights panel failed")
                self._insight_passes.append(
                    InsightPassRecord(
                        n=next_n,
                        stamp=stamp,
                        kind="comment",
                        body_md=f"```\n{e!s}\n```",
                        routing_md=None,
                        corrected_md=None,
                        web_md=None,
                    ),
                )
                await self._refresh_insights_panel(tail=None)

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "chat_input":
            return
        q = event.value.strip()
        if not q:
            return
        event.input.clear()
        chat = self.query_one("#chat_log", RichLog)
        if self._llm_offline:
            chat.write(f"[bold cyan]You:[/] {q}")
            chat.write("[yellow]LLM unavailable; chat is disabled.[/]")
            return
        chat.write(f"[bold cyan]You:[/] {q}")
        full = "\n".join(self._lines)
        try:
            ans = await llm_chat_answer(full, q)
        except Exception as e:  # noqa: BLE001 — demo surface errors in UI
            ans = f"(error) {e!s}"
        chat.write(f"[bold green]AI:[/] {ans}")


def _initial_insights_markdown() -> str:
    return (
        "## Insights\n\n"
        "Each pass is routed by the **LLM** using **`AGENT.md`** (digest; editable). "
        "Outputs are kept **short for live meetings**: brief **Comment** or compact **Research summary**, "
        "plus optional **Cleaned segment** / **Web search** (collapsible).\n\n"
        "_Waiting for enough text…_\n"
    )


def _llm_unavailable_insights_markdown() -> str:
    return (
        "## Insights\n\n"
        "**LLM unavailable:** no `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / "
        "(`BASE_URL` + `API_KEY` + `MODEL`), or the **TCP probe** to the configured API host failed.\n\n"
        "Insights and chat require an LLM; those requests are disabled here.\n\n"
        "**Transcript** is unaffected: text replay works as usual; whether file/live ASR hits "
        "Hugging Face Hub is controlled separately via `ECHO_OFFLINE`, `ECHO_HF_LOCAL_ONLY`, "
        "`HF_HUB_OFFLINE`, etc.\n\n"
        "_If the API is reachable but the probe is blocked (e.g. corporate split DNS), set "
        "`ECHO_ASSUME_ONLINE=1` to skip LLM reachability checks._\n"
    )


def _load_replay_lines(replay_path: Path | None) -> Iterable[str]:
    if replay_path is None:
        return list(DEFAULT_DEMO_LINES)
    text = replay_path.read_text(encoding="utf-8")
    parts = [ln.rstrip("\n") for ln in text.splitlines()]
    return [ln for ln in parts if ln.strip()]


def _load_dotenv_files() -> None:
    """Load `.env` from repo root, then from cwd (defaults: do not override existing os.environ)."""
    from dotenv import load_dotenv

    repo_root = Path(__file__).resolve().parent.parent
    load_dotenv(repo_root / ".env")
    load_dotenv()


def run_echo(
    replay_path: Path | None = None,
    line_delay: float = 1.0,
    audio_path: Path | None = None,
    model_id: str = MODEL_ID_DEFAULT,
    chunk_seconds: float = 30.0,
    whisper_device: str | None = None,
    *,
    live_audio: bool = False,
    live_mic_device: int | None = None,
    live_system_device: int | None = None,
    live_system_authorized: bool = False,
    live_system_authorized_cmd: str | None = None,
) -> None:
    _load_dotenv_files()
    if live_system_authorized and not live_audio:
        raise ValueError("live_system_authorized requires live_audio=True")
    if echo_env_hub_local_only():
        os.environ.setdefault("ECHO_HF_LOAD_LOCAL_ONLY", "1")
    llm_ok = echo_llm_online()
    if audio_path is not None or live_audio:
        import sys

        if audio_path is not None:
            ap = Path(audio_path)
            if not ap.is_file():
                raise FileNotFoundError(f"Audio file not found: {ap.resolve()}")

        from echo.asr import ensure_model_loaded

        if hub_local_only_from_env():
            print(
                "Echo: ASR loading from local Hugging Face cache only (Hub download disabled)…",
                file=sys.stderr,
                flush=True,
            )
        elif live_audio and live_system_authorized:
            print(
                "Echo: loading ASR for live capture (mic + system output via authorized command)…",
                file=sys.stderr,
                flush=True,
            )
        elif live_audio:
            print(
                "Echo: loading ASR weights on the main thread for live capture…",
                file=sys.stderr,
                flush=True,
            )
        else:
            print(
                "Echo: loading ASR weights on the main thread (first run may download ~3GB)…",
                file=sys.stderr,
                flush=True,
            )
        # Main-thread load avoids HF subprocess + asyncio.to_thread + Textual (macOS fds_to_keep).
        ensure_model_loaded(model_id, whisper_device)
    insight_min_chars = 220 if live_audio else 120
    insight_min_interval = 45.0 if live_audio else 25.0
    EchoApp(
        replay_path=replay_path,
        line_delay=line_delay,
        insight_min_chars=insight_min_chars,
        insight_min_interval=insight_min_interval,
        audio_path=audio_path,
        model_id=model_id,
        chunk_seconds=chunk_seconds,
        whisper_device=whisper_device,
        llm_offline=not llm_ok,
        live_audio=live_audio,
        live_mic_device=live_mic_device,
        live_system_device=live_system_device,
        live_system_authorized=live_system_authorized,
        live_system_authorized_cmd=live_system_authorized_cmd,
        live_emit_interval=15.0 if live_audio else 5.0,
    ).run()
