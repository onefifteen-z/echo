"""Textual TUI for Echo — renders an ``EchoSession`` (transcript, insights, chat)."""

from __future__ import annotations

import asyncio
import math
import os
import time
from pathlib import Path
from typing import Awaitable

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen, Screen
from textual.widgets import Collapsible, Footer, Header, Input, Markdown, RichLog, Rule, Static

from echo.asr import MODEL_ID_DEFAULT
from echo.connectivity import echo_env_hub_local_only, echo_llm_online
from echo.engine.events import (
    EngineEvent,
    InsightPass,
    InsightPending,
    Notice,
    Status,
    SummaryWritten,
    TranscriptCorrected,
    TranscriptDraft,
    TranscriptFinal,
)
from echo.engine.session import EchoSession
from echo.engine.text import (  # noqa: F401 — re-exported for callers/tests importing from echo.app
    InsightPassRecord,
    TranscriptSegment,
    _LEADING_BRACKET_TS,
    _LIVE_LINE_SPAN_RE,
    _LIVE_SENTENCE_END_RE,
    _LIVE_SENTENCE_END_ANYWHERE_RE,
    _LIVE_SENTENCE_SPLIT_RE,
    _LIVE_CLAUSE_SPLIT_RE,
    _UNTITLED_MEETING,
    _default_meeting_title,
    _leading_ts_prefix,
    _strip_leading_ts,
    _line_span_seconds,
    _normalize_spaces,
    meeting_title_to_folder_name,
    _render_asr_load_source_label,
    _short_asr_label,
    _trim_chunk_overlap,
    _split_live_groups,
    _strip_live_tail_punct,
    _live_hits_stop_word,
    _is_live_noise_token,
    _should_drop_live_noise_token,
    _live_chunk_metrics,
    _passes_live_voice_gate,
    _should_trigger_insight_live,
    _should_flush_live_pending,
    _insight_corrected_segment_md,
    _insight_segments_signature,
    _has_meaningful_new_content_since_comment,
    _transcript_lines_after_correction,
    _spoken_lines_after_correction,
    _insight_routing_collapsible,
    _format_web_search_md,
    _initial_insights_markdown,
    _llm_unavailable_insights_markdown,
    _load_replay_lines,
)
from echo.splash import SplashScreen


class MeetingSetupScreen(Screen[tuple[str, str]]):
    """Full screen prompting for meeting title and context."""

    DEFAULT_CSS = """
    MeetingSetupScreen {
        align: center middle;
        background: $background;
    }
    #meeting-setup {
        width: 72;
        max-width: 90%;
        height: auto;
        border: tall $accent;
        background: $panel;
        padding: 2 3;
    }
    #meeting-setup-heading {
        text-align: center;
        text-style: bold;
        margin-bottom: 1;
    }
    #meeting-title-input {
        margin-bottom: 1;
    }
    #meeting-context-input {
        margin-bottom: 1;
    }
    #meeting-setup-hint {
        text-align: center;
    }
    """

    def __init__(self, *, title: str = "", context: str = "") -> None:
        super().__init__()
        self._init_title = _default_meeting_title(title)
        self._init_context = context

    def compose(self) -> ComposeResult:
        with Vertical(id="meeting-setup"):
            yield Static("Meeting Setup", id="meeting-setup-heading")
            yield Static("Title:")
            yield Input(
                value=self._init_title,
                placeholder="e.g. Weekly Sync",
                id="meeting-title-input",
            )
            yield Static("Background / context:")
            yield Input(
                value=self._init_context,
                placeholder="e.g. Review blockers and next milestones",
                id="meeting-context-input",
            )
            yield Static(
                "[dim]Enter in title -> moves to context.  Enter in context -> start.[/]",
                id="meeting-setup-hint",
                markup=True,
            )

    def on_mount(self) -> None:
        self.query_one("#meeting-title-input", Input).focus()

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "meeting-title-input":
            self.query_one("#meeting-context-input", Input).focus()
            return
        title = self.query_one("#meeting-title-input", Input).value.strip()
        context = self.query_one("#meeting-context-input", Input).value.strip()
        self.dismiss((title or _UNTITLED_MEETING, context))


_METER_BAR_WIDTH = 26
_METER_FLOOR_DB = -60.0
_METER_HISTORY_WIDTH = 52
_METER_TIMELINE_LEVELS = " .:-=+*#%@"
_METER_CHART_ROWS = 8


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


def _meter_bar_markup(unit: float, width: int = _METER_BAR_WIDTH) -> str:
    """Color meter bars: green -> yellow -> orange -> red."""
    v = max(0.0, min(1.0, float(unit)))
    filled = int(round(v * width))
    out: list[str] = []
    for i in range(width):
        if i >= filled:
            out.append("[#5f5f5f]-[/]")
            continue
        pos = (i + 1) / max(1, width)
        if pos <= 0.6:
            color = "green"
        elif pos <= 0.8:
            color = "yellow"
        elif pos <= 0.9:
            color = "orange3"
        else:
            color = "red"
        out.append(f"[{color}]#[/]")
    return "".join(out)


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
    """Render a vertical equalizer-like chart with level colors."""
    if width <= 0 or rows <= 0:
        return []
    tail = units[-width:]
    if len(tail) < width:
        tail = ([0.0] * (width - len(tail))) + tail
    heights = [max(0, min(rows, int(round(max(0.0, min(1.0, u)) * rows)))) for u in tail]

    out: list[str] = []
    for r in range(rows, 0, -1):
        level = r / max(1, rows)
        if level >= 0.85:
            color = "red"
        elif level >= 0.65:
            color = "orange3"
        elif level >= 0.45:
            color = "yellow"
        else:
            color = "green"
        filled = f"[{color}]█[/] "
        empty = "  "
        line = "".join(filled if h >= r else empty for h in heights).rstrip()
        out.append(line)
    return out


DEFAULT_DEMO_LINES: list[str] = [
    "[00:01] 今日の議題は四半期レビューと予算の見直しです。",
    "[00:12] まず予算について、マーケティングは前四半期比で 8% 増です。",
    "[00:28] Q4 の売上は計画を上回りましたが、在庫コストが圧迫しています。",
    "[00:45] 次に、来期のキャップエクスをどう配分するか決めたいです。",
    "[01:02] 予算の再配分案は金曜までにドラフトを共有しましょう。",
    "[01:18] それでは次のトピックに移ります。",
]


class QuitConfirmScreen(ModalScreen[bool]):
    """Confirmation dialog shown when quitting while summary is generating."""

    DEFAULT_CSS = """
    QuitConfirmScreen {
        align: center middle;
    }
    #quit-confirm-box {
        width: auto;
        height: auto;
        padding: 2 4;
        background: $surface;
        border: solid $warning;
    }
    """
    BINDINGS = [
        Binding("q", "force_quit", "Quit", priority=True),
        Binding("w", "wait", "Wait", priority=True),
        Binding("escape", "wait", "Cancel", show=False, priority=True),
    ]

    def compose(self) -> ComposeResult:
        with Vertical(id="quit-confirm-box"):
            yield Static("Summary is still generating. Quit anyway?\n\n[q] Quit  [w] Wait")

    def action_force_quit(self) -> None:
        self.dismiss(True)

    def action_wait(self) -> None:
        self.dismiss(False)


class EchoApp(App[None]):
    """Textual front-end: renders ``EchoSession`` events and forwards key bindings to it."""

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
        height: auto;
        min-height: 11;
        border: tall $warning;
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
        Binding("p", "toggle_pause", "Pause", priority=True),
        Binding("s", "save_transcript", "Save", priority=True),
        Binding("c", "clear_chat", "Clear chat", priority=True),
        Binding("r", "reset", "Reset", priority=True),
        Binding("l", "cycle_live_language", "Language", priority=True),
        Binding("d", "toggle_live_diagnostics", "Delay", priority=True),
        Binding("i", "focus_insights", "Insights", priority=True),
        Binding("t", "focus_transcript", "Transcript", priority=True),
        Binding("q", "quit", "Quit", priority=True),
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
        asr_load_source: str | None = None,
        meeting_title: str = _UNTITLED_MEETING,
        meeting_context: str = "",
        output_dir: Path | None = None,
        session: EchoSession | None = None,
    ) -> None:
        super().__init__()
        self.engine = session or EchoSession(
            replay_path=replay_path,
            replay_lines=DEFAULT_DEMO_LINES,
            line_delay=line_delay,
            insight_min_chars=insight_min_chars,
            insight_min_interval=insight_min_interval,
            audio_path=audio_path,
            model_id=model_id,
            chunk_seconds=chunk_seconds,
            whisper_device=whisper_device,
            llm_offline=llm_offline,
            live_audio=live_audio,
            live_mic_device=live_mic_device,
            live_system_device=live_system_device,
            live_system_authorized=live_system_authorized,
            live_system_authorized_cmd=live_system_authorized_cmd,
            live_emit_interval=live_emit_interval,
            live_group_delay=live_group_delay,
            live_group_chars=live_group_chars,
            live_stop_words=live_stop_words,
            asr_load_source=asr_load_source,
            meeting_title=meeting_title,
            meeting_context=meeting_context,
            output_dir=output_dir,
        )
        self.engine.subscribe(self._on_engine_event)
        self._workers_started = False
        self._resume_after_setup = False
        # RichLog bookkeeping: which transcript line ids are on screen, and how many strip rows
        # the trailing line occupies (so a growing draft line can be replaced in place).
        self._rendered_line_ids: list[int] = []
        self._draft_log_strip_count = 0
        self._transcript_log_prefix: list[str] = []
        # Incremental insight-panel rendering state (avoid full re-mount on every refresh).
        self._rendered_pass_count = 0
        self._insight_header_mounted = False
        self._insight_tail_widgets: list = []
        self.live_meter_history_width = self.engine.config.live.meter_history_width
        self._live_rms_hist: list[float] = [0.0] * self.live_meter_history_width
        self._live_peak_hist: list[float] = [0.0] * self.live_meter_history_width
        self._show_live_diagnostics = False

    # ------------------------------------------------------------------ engine passthroughs

    @property
    def live_audio(self) -> bool:
        return self.engine.live_audio

    @property
    def audio_path(self) -> Path | None:
        return self.engine.audio_path

    @property
    def debug_enabled(self) -> bool:
        return self.engine.debug_enabled

    @property
    def _llm_offline(self) -> bool:
        return self.engine.llm_offline

    @property
    def _lines(self) -> list[str]:
        return self.engine._lines

    @_lines.setter
    def _lines(self, value: list[str]) -> None:
        self.engine._lines = value

    @property
    def _insight_passes(self) -> list[InsightPassRecord]:
        return self.engine._insight_passes

    @_insight_passes.setter
    def _insight_passes(self, value: list[InsightPassRecord]) -> None:
        self.engine._insight_passes = value

    # ------------------------------------------------------------------ layout

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="main_row"):
            with Vertical(id="left_column"):
                yield Static(self._input_meter_text(), id="input_meter", markup=True)
                if self.debug_enabled:
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
        eng = self.engine
        eng.set_meeting_metadata(eng.meeting_title, eng.meeting_context)
        if self._llm_offline:
            self.query_one("#chat_log", RichLog).write(
                "[dim]LLM unavailable (no API keys or TCP probe to the configured host failed): "
                "Insights and chat are disabled. The Transcript panel still works (replay or ASR).[/]"
            )
        self._tick_header()
        self.set_interval(1.0, self._tick_header)
        self.set_interval(0.15, self._tick_input_meter)
        if self.debug_enabled:
            self.set_interval(0.2, self._tick_debug_panel)
        if eng.live_audio:
            insight_dbg = (
                f"insight_live[min_chars={eng.insight_min_chars},min_sec={eng.insight_live_min_seconds:.1f}s,"
                f"pause={eng.insight_pause_seconds:.1f}s,pause_min={eng.insight_pause_min_chars},"
                f"seg_min={eng.insight_finalized_segment_min_chars},skip_min={eng.insight_skip_min_chars},"
                f"min_interval={eng.insight_live_min_interval:.1f}s,"
                f"pause_cap_q={eng.insight_pause_capture_q}]"
            )
        else:
            insight_dbg = (
                f"insight_replay[min_chars={eng.insight_min_chars},"
                f"min_interval={eng.insight_min_interval:.1f}s]"
            )
        if eng.live_vad_backend != "off":
            vad_dbg = (
                f"vad[{eng.live_vad_backend},min_ratio={eng.live_vad_min_speech_ratio:.2f},"
                f"prob_thr={eng.live_vad_speech_prob_threshold:.2f},"
                f"agg={eng.live_vad_aggressiveness}]"
            )
        else:
            vad_dbg = "vad[off]"
        if eng.live_denoise_backend != "off":
            denoise_dbg = f"denoise[{eng.live_denoise_backend},strength={eng.live_denoise_strength:.2f}]"
        else:
            denoise_dbg = "denoise[off]"
        eng._debug_note(
            "startup "
            f"mode={eng.mode} stable={eng.live_rewrite_stable_seconds:.1f}s "
            f"overlap={eng.live_chunk_overlap_seconds:.2f}s seg_pause={eng.live_segment_pause_seconds:.1f}s "
            f"ui[max_sec={eng.ui_segment_max_seconds:.1f},max_chars={eng.ui_segment_max_chars},"
            f"semantic_min={eng.ui_segment_semantic_min_chars}] "
            f"asr_log_interval={eng.live_asr_log_interval_seconds:.1f}s "
            f"rewrite={'on' if eng.enable_transcript_rewrite else 'off'} "
            f"lang={eng.live_asr_language} "
            f"{vad_dbg} {denoise_dbg} "
            f"{insight_dbg}",
        )
        if eng._debug_log_path is not None:
            eng._debug_note(f"debug log file: {eng._debug_log_path}")
        show_splash = os.environ.get("ECHO_NO_SPLASH", "").lower() not in ("1", "true", "yes")
        if show_splash:
            self.push_screen(SplashScreen(), callback=self._on_splash_done)
        else:
            self._open_meeting_setup()

    def _on_splash_done(self, _result: object) -> None:
        self._open_meeting_setup()

    def _open_meeting_setup(self) -> None:
        self.push_screen(
            MeetingSetupScreen(title=self.engine.meeting_title, context=self.engine.meeting_context),
            callback=self._on_meeting_setup_done,
        )

    def _on_meeting_setup_done(self, result: tuple[str, str]) -> None:
        title, context = result
        self.engine.set_meeting_metadata(title, context)
        if not self._workers_started:
            self._workers_started = True
            self.engine.start()
        if self._resume_after_setup:
            self.engine.set_paused(False)
            self._resume_after_setup = False
            self.notify("Session reset", timeout=2)

    # ------------------------------------------------------------------ engine events

    def _on_engine_event(self, ev: EngineEvent) -> Awaitable[None] | None:
        if not self.is_running:
            return None
        if isinstance(ev, (TranscriptDraft, TranscriptFinal)):
            self._render_line_update(ev.line_id, ev.line)
        elif isinstance(ev, TranscriptCorrected):
            self._refresh_transcript_log()
        elif isinstance(ev, Notice):
            self._write_notice(ev.text)
        elif isinstance(ev, InsightPending):
            tail = (
                "_Updating…_"
                if not self._insight_passes
                else f"_[Generating pass {ev.n} @ `{ev.stamp}`…]_\n"
            )
            return self._refresh_insights_panel(tail=tail)
        elif isinstance(ev, InsightPass):
            return self._refresh_insights_panel(tail=None)
        elif isinstance(ev, SummaryWritten):
            if ev.announce and ev.ok:
                self.notify("Summary generated ✓", timeout=4)
        elif isinstance(ev, Status):
            self._tick_header()
        return None

    def _transcript_log(self) -> RichLog | None:
        try:
            return self.query_one("#transcript", RichLog)
        except Exception:  # noqa: BLE001 — not mounted yet / screen switched
            return None

    def _write_notice(self, text: str) -> None:
        self._transcript_log_prefix.append(text)
        log = self._transcript_log()
        if log is not None:
            log.write(text)
            # A notice now sits below the trailing transcript line; it can no longer be replaced in place.
            self._draft_log_strip_count = 0

    def _render_transcript_line(self, line_id: int, line: str) -> str:
        if self.engine._line_id_to_flushed.get(line_id, False):
            return f"[white]{line}[/]"
        return f"[dim]{line}[/]"

    def _write_transcript_line(self, log: RichLog, line_id: int, line: str) -> int:
        """Write one transcript line and return how many strip rows it occupied."""
        rendered = self._render_transcript_line(line_id, line)
        before = len(log.lines)
        log.write(rendered)
        return max(0, len(log.lines) - before)

    def _render_line_update(self, line_id: int, line: str) -> None:
        log = self._transcript_log()
        if log is None:
            return
        if self._rendered_line_ids and self._rendered_line_ids[-1] == line_id:
            count = self._draft_log_strip_count
            if count <= 0 or count > len(log.lines):
                self._refresh_transcript_log()
                return
            try:
                for _ in range(count):
                    log.lines.pop()
                log.refresh()
            except Exception:  # noqa: BLE001 — Textual internals not contractually stable
                self._refresh_transcript_log()
                return
            self._draft_log_strip_count = self._write_transcript_line(log, line_id, line)
        elif line_id not in self._rendered_line_ids:
            self._rendered_line_ids.append(line_id)
            self._draft_log_strip_count = self._write_transcript_line(log, line_id, line)
        else:
            self._refresh_transcript_log()

    def _refresh_transcript_log(self) -> None:
        log = self._transcript_log()
        if log is None:
            return
        log.clear()
        for m in self._transcript_log_prefix:
            log.write(m)
        self._draft_log_strip_count = 0
        self._rendered_line_ids = []
        for st in self.engine.transcript_snapshot():
            self._rendered_line_ids.append(st.line_id)
            self._draft_log_strip_count = self._write_transcript_line(log, st.line_id, st.line)

    # ------------------------------------------------------------------ header / meter / debug

    def _elapsed_hms(self) -> str:
        return self.engine._elapsed_hms()

    def _tick_header(self) -> None:
        eng = self.engine
        paused = eng.paused
        if eng.live_audio:
            state = "Paused" if paused else "Recording"
            mode = "Live"
        elif eng.audio_path is not None:
            state = "Paused" if paused else "Transcribing"
            mode = "ASR"
        else:
            state = "Paused" if paused else "Replaying"
            mode = "Text"
        off = " · no LLM" if self._llm_offline else ""
        lang = f" · lang={eng.live_asr_language}" if eng.live_audio else ""
        asr = f" · {eng.short_asr_label()}" if eng.asr_load_source else ""
        mic_only = " · MIC ONLY (system audio not captured)" if eng.mic_only else ""
        self.sub_title = f"🎙 {mode} · {state}{off}{lang}{asr}{mic_only}  [{self._elapsed_hms()}]  p/s/c/r/l/d/q"

    def _input_meter_text(self, *, chart_cols: int | None = None) -> str:
        eng = self.engine
        if not eng.live_audio:
            return "Input level\navailable in --live mode"

        age = (
            time.monotonic() - eng._live_meter_last_signal_at
            if eng._live_meter_last_signal_at > 0
            else float("inf")
        )
        if eng.paused:
            state = "Paused"
        elif age > 2.0:
            state = "Recording · no recent signal"
        else:
            state = f"Recording · {eng._live_meter_note or 'Listening'}"

        blend = [
            max(0.0, min(1.0, (p * 0.75) + (r * 0.25)))
            for r, p in zip(self._live_rms_hist, self._live_peak_hist, strict=False)
        ]
        cols = max(8, chart_cols if chart_cols is not None else self.live_meter_history_width)
        chart_block = "\n".join(_meter_vertical_chart(blend, width=cols, rows=_METER_CHART_ROWS))
        diag = ""
        if self._show_live_diagnostics:
            if eng._live_diag_updated_at <= 0.0:
                diag = "\nDelay: waiting for first ASR chunk..."
            else:
                age_s = max(0.0, time.monotonic() - eng._live_diag_updated_at)
                cap_q_txt = str(eng._live_diag_cap_q) if eng._live_diag_cap_q >= 0 else "?"
                diag = (
                    "\n"
                    "Delay: "
                    f"decode={eng._live_diag_decode_ms:.0f}ms "
                    f"rtf={eng._live_diag_rtf:.2f} "
                    f"lag={eng._live_diag_lag_secs:.2f}s "
                    f"e2e={eng._live_diag_e2e_delay_secs:.2f}s\n"
                    "Buffer: "
                    f"cap_q={cap_q_txt} "
                    f"asr_buf={eng._live_diag_buf_s:.2f}s "
                    f"chunks={eng._live_diag_buf_chunks} "
                    f"updated={age_s:.1f}s ago"
                )
        return f"Input level · {state}\n{chart_block}{diag}"

    def _tick_input_meter(self) -> None:
        eng = self.engine
        if eng.live_audio:
            self._live_rms_hist.append(_db_to_unit(_amp_to_dbfs(eng._live_meter_rms)))
            self._live_peak_hist.append(_db_to_unit(_amp_to_dbfs(eng._live_meter_peak)))
            w = self.live_meter_history_width
            self._live_rms_hist = self._live_rms_hist[-w:]
            self._live_peak_hist = self._live_peak_hist[-w:]
        meter = self.query_one("#input_meter", Static)
        chart_cols = self.live_meter_history_width
        if meter.size.width > 0:
            # 2 chars per column (bar + space), reserve a bit for panel borders.
            chart_cols = max(8, min(self.live_meter_history_width, (meter.size.width - 4) // 2))
        meter.update(self._input_meter_text(chart_cols=chart_cols))

    def _debug_panel_text(self) -> str:
        events = self.engine._debug_events
        if not events:
            return "Debug panel\n(no events yet)"
        return "Debug panel\n" + "\n".join(events)

    def _tick_debug_panel(self) -> None:
        if not self.debug_enabled:
            return
        self.query_one("#debug_panel", Static).update(self._debug_panel_text())

    # ------------------------------------------------------------------ actions

    def action_toggle_pause(self) -> None:
        self.engine.toggle_pause()

    def action_quit(self) -> None:
        if self.engine.summary_in_progress:
            self.push_screen(QuitConfirmScreen(), self._handle_quit_confirm)
        else:
            self.exit()

    async def _handle_quit_confirm(self, quit_now: bool) -> None:
        if not quit_now:
            await self.engine.wait_for_summary()
        self.exit()

    def action_save_transcript(self) -> None:
        self._save_meeting_outputs(show_notification=True)

    def _save_meeting_outputs(self, *, show_notification: bool) -> None:
        out = self.engine.save_outputs(announce=show_notification)
        if show_notification:
            if self._llm_offline:
                self.notify(f"Saved notes and insights to {out}", timeout=4)
            else:
                self.notify("Saved notes and insights. Summary generating…", timeout=4)

    def action_clear_chat(self) -> None:
        chat = self.query_one("#chat_log", RichLog)
        chat.clear()
        if self._llm_offline:
            chat.write("[dim]LLM unavailable: Insights and chat are disabled. Transcript still works.[/]")

    def action_reset(self) -> None:
        self.engine.reset()
        self._rendered_line_ids = []
        self._draft_log_strip_count = 0
        self._transcript_log_prefix.clear()
        self._rendered_pass_count = 0
        self._insight_header_mounted = False
        self._insight_tail_widgets = []
        self._live_rms_hist = [0.0] * self.live_meter_history_width
        self._live_peak_hist = [0.0] * self.live_meter_history_width
        self.query_one("#transcript", RichLog).clear()
        self.action_clear_chat()
        self._tick_input_meter()
        if self.debug_enabled:
            self._tick_debug_panel()
        asyncio.create_task(self._refresh_insights_panel(tail=None))
        self._resume_after_setup = True
        self._open_meeting_setup()

    def action_focus_insights(self) -> None:
        self.query_one("#insights", VerticalScroll).focus()

    def action_focus_transcript(self) -> None:
        self.query_one("#transcript", RichLog).focus()

    def action_cycle_live_language(self) -> None:
        if not self.live_audio:
            self.notify("Language switch is available in --live mode.", timeout=2)
            return
        ok, message = self.engine.cycle_language()
        self.notify(message, severity="information" if ok else "error", timeout=3 if ok else 6)
        self._tick_header()

    def action_toggle_live_diagnostics(self) -> None:
        if not self.live_audio:
            self.notify("Delay diagnostics are available in --live mode.", timeout=2)
            return
        self._show_live_diagnostics = not self._show_live_diagnostics
        self._tick_input_meter()
        self.notify(
            "Delay diagnostics: ON" if self._show_live_diagnostics else "Delay diagnostics: OFF",
            timeout=2,
        )

    def on_unmount(self) -> None:
        # Failures here must not block app shutdown (slow disk, permission errors, etc.).
        try:
            self._save_meeting_outputs(show_notification=False)
        except Exception:  # noqa: BLE001
            self.log.exception("on_unmount: failed to save meeting outputs")
        self.engine.cancel_workers()

    # ------------------------------------------------------------------ insights panel

    async def _refresh_insights_panel(self, *, tail: str | None = None) -> None:
        """Render the insights panel incrementally (mount only new passes plus a fresh tail).

        Falls back to a full rebuild when the rendered prefix is out of sync (e.g. after reset).
        """
        scroll = self.query_one("#insights", VerticalScroll)

        def _scroll_bottom() -> None:
            try:
                scroll.scroll_end(animate=False)
            except Exception:  # noqa: BLE001
                return

        rendered = self._rendered_pass_count
        passes = self._insight_passes
        can_incremental = rendered > 0 and rendered <= len(passes) and self._insight_header_mounted
        if not can_incremental:
            await self._rebuild_insights_panel_full(scroll, tail)
            _scroll_bottom()
            return

        await self._remove_insight_tail_widgets()
        for i in range(rendered, len(passes)):
            await self._mount_insight_pass(scroll, passes[i], with_leading_rule=(i > 0))
        self._rendered_pass_count = len(passes)
        await self._mount_insight_tail(scroll, tail)
        _scroll_bottom()

    async def _rebuild_insights_panel_full(self, scroll: VerticalScroll, tail: str | None) -> None:
        for child in list(scroll.children):
            await child.remove()
        self._rendered_pass_count = 0
        self._insight_header_mounted = False
        self._insight_tail_widgets = []

        passes = self._insight_passes
        if not passes:
            if tail is None:
                md = _llm_unavailable_insights_markdown() if self._llm_offline else _initial_insights_markdown()
                placeholder = Markdown(md)
            else:
                placeholder = Markdown(f"## Insights\n\n{tail}\n")
            await scroll.mount(placeholder)
            self._insight_tail_widgets = [placeholder]
            return

        await scroll.mount(Markdown("## Insights"))
        await scroll.mount(Rule())
        self._insight_header_mounted = True
        for i, rec in enumerate(passes):
            await self._mount_insight_pass(scroll, rec, with_leading_rule=(i > 0))
        self._rendered_pass_count = len(passes)
        await self._mount_insight_tail(scroll, tail)

    async def _mount_insight_pass(
        self, scroll: VerticalScroll, rec: InsightPassRecord, *, with_leading_rule: bool,
    ) -> None:
        if with_leading_rule:
            await scroll.mount(Rule())
        heading = "Comment" if rec.kind == "comment" else "Research summary"
        await scroll.mount(Markdown(f"### Pass {rec.n} @ `{rec.stamp}`\n\n## {heading}\n\n{rec.body_md}\n"))
        if self.debug_enabled and rec.routing_md:
            await scroll.mount(Collapsible(Markdown(rec.routing_md), title="Routing", collapsed=True))
        if self.debug_enabled and rec.corrected_md:
            await scroll.mount(Collapsible(Markdown(rec.corrected_md), title="Cleaned segment", collapsed=True))
        if rec.web_md:
            await scroll.mount(Collapsible(Markdown(rec.web_md), title="Web search", collapsed=True))

    async def _remove_insight_tail_widgets(self) -> None:
        for w in self._insight_tail_widgets:
            try:
                await w.remove()
            except Exception:  # noqa: BLE001 — widget may already be detached
                continue
        self._insight_tail_widgets = []

    async def _mount_insight_tail(self, scroll: VerticalScroll, tail: str | None) -> None:
        if not tail:
            self._insight_tail_widgets = []
            return
        rule = Rule()
        body = Markdown(tail)
        await scroll.mount(rule)
        await scroll.mount(body)
        self._insight_tail_widgets = [rule, body]

    # ------------------------------------------------------------------ chat

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "chat_input":
            return
        q = event.value.strip()
        if not q:
            return
        event.input.clear()
        chat = self.query_one("#chat_log", RichLog)
        chat.write(f"[bold cyan]You:[/] {q}")
        if self._llm_offline:
            chat.write("[yellow]LLM unavailable; chat is disabled.[/]")
            return
        ans = await self.engine.ask(q)
        chat.write(f"[bold green]AI:[/] {ans}")


def _load_dotenv_files() -> None:
    """Load `.env` from repo root, then from cwd (defaults: do not override existing os.environ)."""
    from dotenv import load_dotenv

    repo_root = Path(__file__).resolve().parent.parent
    load_dotenv(repo_root / ".env")
    load_dotenv()


def install_apple_speech_cli(language: str) -> None:
    """Download Apple's on-device speech model for ``language`` with a terminal progress line."""
    import sys

    from echo.asr_apple import install_apple_speech

    def _progress(fraction: float) -> None:
        print(f"\rEcho: downloading Apple speech model… {fraction * 100:5.1f}%", end="", file=sys.stderr, flush=True)

    install_apple_speech(language, on_progress=_progress)
    print(f"\nEcho: Apple on-device speech model for {language} is installed.", file=sys.stderr, flush=True)


def _maybe_offer_apple_speech_install(router: object, language: str) -> None:
    """When the Apple model for ``language`` can be downloaded, ask once on an interactive TTY."""
    import sys

    if getattr(router, "forced", None) not in (None, "apple_on_device") or sys.platform != "darwin":
        return
    if os.environ.get("ECHO_NO_APPLE_INSTALL_PROMPT") or not (sys.stdin.isatty() and sys.stderr.isatty()):
        return
    try:
        status = router.apple_status(language)  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        return
    if status != "supported":
        return
    try:
        answer = input(
            f"Echo: Apple's on-device {language} speech model is not installed "
            "(system download, runs offline afterwards). Download now? [y/N] "
        )
    except EOFError:
        return
    if answer.strip().lower() not in ("y", "yes"):
        return
    try:
        install_apple_speech_cli(language)
    except Exception as e:  # noqa: BLE001
        print(f"\nEcho: Apple speech model download failed: {e}", file=sys.stderr, flush=True)
    router.invalidate_apple()  # type: ignore[attr-defined]


def run_echo(
    replay_path: Path | None = None,
    line_delay: float = 1.0,
    audio_path: Path | None = None,
    model_id: str = MODEL_ID_DEFAULT,
    chunk_seconds: float | None = None,
    whisper_device: str | None = None,
    asr_backend: str | None = None,
    *,
    live_audio: bool = False,
    live_mic_device: int | None = None,
    live_system_device: int | None = None,
    live_system_authorized: bool = False,
    live_system_authorized_cmd: str | None = None,
    meeting_title: str = _UNTITLED_MEETING,
    meeting_context: str = "",
    output_dir: Path | None = None,
) -> None:
    _load_dotenv_files()
    if live_system_authorized and not live_audio:
        raise ValueError("live_system_authorized requires live_audio=True")
    if echo_env_hub_local_only():
        os.environ.setdefault("ECHO_HF_LOAD_LOCAL_ONLY", "1")
    llm_ok = echo_llm_online()
    asr_load_source: str | None = None
    if audio_path is not None or live_audio:
        import sys

        from echo.asr import _default_asr_language_from_env
        from echo.asr_routing import AsrUnavailableError
        from echo.engine.asr_setup import prepare_asr

        if audio_path is not None:
            ap = Path(audio_path)
            if not ap.is_file():
                raise FileNotFoundError(f"Audio file not found: {ap.resolve()}")

        try:
            # Main-thread local load avoids HF subprocess + asyncio.to_thread + Textual (macOS fds_to_keep).
            _router, asr_load_source, chunk_seconds = prepare_asr(
                asr_backend,
                language=_default_asr_language_from_env(),
                live_audio=live_audio,
                chunk_seconds=chunk_seconds,
                model_id=model_id,
                whisper_device=whisper_device,
                live_system_authorized=live_system_authorized,
                before_select=_maybe_offer_apple_speech_install,
                log=lambda m: print(m, file=sys.stderr, flush=True),
            )
        except AsrUnavailableError as e:
            print(f"Echo: {e}", file=sys.stderr, flush=True)
            raise SystemExit(2) from e
        print(
            f"Echo: ASR backend: {_render_asr_load_source_label(asr_load_source)}.",
            file=sys.stderr,
            flush=True,
        )
    if chunk_seconds is None:
        chunk_seconds = 30.0
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
        live_emit_interval=0.0 if live_audio else 5.0,
        asr_load_source=asr_load_source,
        meeting_title=meeting_title,
        meeting_context=meeting_context,
        output_dir=output_dir,
    ).run()
