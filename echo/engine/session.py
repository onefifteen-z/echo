"""Headless Echo meeting session: transcript state, ASR workers, insights, summary, and chat.

``EchoSession`` owns everything that is not rendering. UIs (the Textual TUI, the WebSocket
sidecar used by the Electron app) subscribe to its events and call its control methods.
``LiveTranscriptionSession`` drives it as its host (``app``), so the live-capture loop never
touches a UI toolkit.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import time
from datetime import datetime, timedelta
from functools import partial
from pathlib import Path
from typing import Any, Callable, Iterable

from echo.asr import MODEL_ID_DEFAULT
from echo.config import EchoConfig
from echo.connectivity import hub_local_only_from_env
from echo.engine.events import (
    ChatReply,
    EngineEvent,
    InsightPass,
    InsightPending,
    Meter,
    Notice,
    SessionEnded,
    SessionReset,
    Status,
    SummaryWritten,
    TranscriptCorrected,
    TranscriptDraft,
    TranscriptFinal,
    TranscriptLineState,
)
from echo.engine.text import (
    _UNTITLED_MEETING,
    InsightPassRecord,
    TranscriptSegment,
    _format_web_search_md,
    _format_wiki_md,
    _has_meaningful_new_content_since_comment,
    _insight_corrected_segment_md,
    _insight_routing_collapsible,
    _insight_segments_signature,
    _leading_ts_prefix,
    _load_replay_lines,
    _normalize_spaces,
    _render_asr_load_source_label,
    _short_asr_label,
    patch_meeting_json,
    summary_preview,
    _should_trigger_insight_live,
    _split_live_groups,
    _spoken_lines_after_correction,
    _strip_leading_ts,
    meeting_title_to_folder_name,
)
from echo.llm import (
    MEETING_SUMMARY_OFFLINE_MARKDOWN,
    InsightRouteResult,
    llm_chat_answer,
    llm_meeting_summary,
    llm_route_insight,
    llm_synthesize_web_findings,
    llm_synthesize_wiki_findings,
    wiki_sources_block,
)
from echo.search import SearchHit, web_search
from echo.wiki import LuminaWiki, WikiError, WikiHit, WikiPage
from echo.text_utils import format_timestamp_hms

logger = logging.getLogger("echo.engine")

Listener = Callable[[EngineEvent], Any]

LIVE_ASR_LANGUAGES: tuple[str, ...] = ("japanese", "english", "chinese")


class EchoSession:
    """One meeting: replay / audio file / live capture → transcript → insights → outputs."""

    def __init__(
        self,
        *,
        replay_path: Path | None = None,
        replay_lines: Iterable[str] = (),
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
        language: str | None = None,
        vault: str | None = None,
        wiki: LuminaWiki | None = None,
    ) -> None:
        self.replay_path = replay_path
        self.replay_lines = list(replay_lines)
        self.line_delay = line_delay
        self.insight_min_chars = insight_min_chars
        self.insight_min_interval = insight_min_interval
        self.audio_path = audio_path
        self.live_audio = live_audio
        self.live_mic_device = live_mic_device
        self.live_system_device = live_system_device
        self.live_system_authorized = live_system_authorized
        self.live_system_authorized_cmd = live_system_authorized_cmd
        self.live_emit_interval = max(0.0, live_emit_interval)
        self.live_group_delay = max(0.0, live_group_delay)
        self.live_group_chars = max(12, live_group_chars)

        cfg = EchoConfig.from_env()
        self.config = cfg
        live_cfg = cfg.live
        ui_cfg = cfg.ui_segment
        ins_cfg = cfg.insight

        if live_stop_words is None:
            self.live_stop_words = live_cfg.stop_words
        else:
            self.live_stop_words = tuple(w.strip() for w in live_stop_words if w.strip())
        self.live_noise_tokens = live_cfg.noise_tokens
        self.live_always_drop_noise_tokens = live_cfg.always_drop_noise_tokens
        self.live_noise_max_chars = live_cfg.noise_max_chars
        self.live_noise_max_rms_factor = live_cfg.noise_max_rms_factor
        self.live_noise_drop_at_segment_start = live_cfg.noise_drop_at_segment_start
        self.model_id = model_id
        self.chunk_seconds = chunk_seconds
        self.whisper_device = whisper_device
        self.asr_load_source = asr_load_source
        self.meeting_title = (meeting_title or "").strip() or _UNTITLED_MEETING
        self.meeting_context = (meeting_context or "").strip()
        self._explicit_output_dir = Path(output_dir) if output_dir is not None else None
        self.output_dir = (
            self._explicit_output_dir
            if self._explicit_output_dir is not None
            else Path.cwd() / "output" / meeting_title_to_folder_name(self.meeting_title)
        )
        self.live_asr_languages = LIVE_ASR_LANGUAGES
        env_lang = (
            language or os.environ.get("ECHO_WHISPER_LANGUAGE") or "japanese"
        ).strip().lower() or "japanese"
        self.live_asr_language = env_lang if env_lang in self.live_asr_languages else "japanese"
        self._llm_offline = llm_offline
        self._wiki = wiki
        self.vault: str | None = (vault or "").strip() or None
        self._vault_titles: dict[str, list[str]] = {}
        self._vault_warm_task: asyncio.Task[None] | None = None
        self._summary_task: asyncio.Task[None] | None = None
        self._live_wall_t0: float | None = None
        self._live_reset_seq = 0

        self._listeners: list[Listener] = []
        self._tasks: list[asyncio.Task[None]] = []
        self._started = False
        self._finished = False
        self._stopped = False

        self._paused = False
        self._t0 = time.monotonic()
        self.started_at = datetime.now().astimezone()
        self.notebook_md = ""
        self._lines: list[str] = []
        self._line_ids: list[int] = []
        self._line_id_to_index: dict[int, int] = {}
        self._line_id_to_spoken: dict[int, str] = {}
        self._line_id_to_flushed: dict[int, bool] = {}
        self._next_line_id = 1
        self._transcript_queue: asyncio.Queue[object] = asyncio.Queue()
        self._insight_seg_by_id: dict[int, TranscriptSegment] = {}
        self._insight_seg_order: list[int] = []
        self._insight_line_state: dict[int, tuple[str, bool]] = {}
        self._last_insight_at = time.monotonic()
        self._last_transcript_activity_at = time.monotonic()
        self._insight_lock = asyncio.Lock()
        self._last_insight_submission_sig = ""
        self._last_comment_input_snapshot = ""
        self._eof_sentinel = "__ECHO_EOF__"
        self._insight_passes: list[InsightPassRecord] = []
        self._force_live_insight_flush = False
        self._insight_tasks: set[asyncio.Task[None]] = set()
        self._insight_session_id = 0
        self._chat_history: list[tuple[str, str]] = []
        self._live_meter_rms = 0.0
        self._live_meter_peak = 0.0
        self._live_meter_last_signal_at = 0.0
        self._live_meter_note = "Listening..." if self.live_audio else "Available in --live mode."
        self.live_min_rms = live_cfg.min_rms
        self.live_active_amp = live_cfg.active_amp
        self.live_min_active_ratio = live_cfg.min_active_ratio
        self.live_vad_backend = live_cfg.vad_backend
        self.live_vad_min_speech_ratio = live_cfg.vad_min_speech_ratio
        self.live_vad_speech_prob_threshold = live_cfg.vad_speech_prob_threshold
        self.live_vad_aggressiveness = live_cfg.vad_aggressiveness
        self.live_denoise_backend = live_cfg.denoise_backend
        self.live_denoise_strength = live_cfg.denoise_strength
        self.live_chunk_overlap_seconds = live_cfg.chunk_overlap_seconds
        self.live_segment_pause_seconds = live_cfg.segment_pause_seconds
        self.live_overlap_min_chars = live_cfg.overlap_min_chars
        self.live_rewrite_stable_seconds = live_cfg.rewrite_stable_seconds
        self.live_asr_log_interval_seconds = live_cfg.asr_log_interval_seconds
        self.ui_segment_semantic_min_chars = ui_cfg.semantic_min_chars
        self.ui_segment_max_seconds = ui_cfg.max_seconds
        self.ui_segment_max_chars = ui_cfg.max_chars
        self.insight_live_min_seconds = ins_cfg.live_min_seconds
        self.insight_pause_min_chars = ins_cfg.pause_min_chars
        self.insight_finalized_segment_min_chars = ins_cfg.finalized_segment_min_chars
        self.insight_skip_min_chars = ins_cfg.skip_min_chars
        self.insight_pause_seconds = ins_cfg.pause_seconds
        self.insight_live_min_interval = ins_cfg.live_min_interval
        self.insight_pause_capture_q = ins_cfg.pause_capture_q
        self.enable_transcript_rewrite = ins_cfg.enable_transcript_rewrite
        self._live_capture_queue_depth: int = -1
        self._last_capture_backlog_debug_at: float = 0.0
        self.chat_context_max_lines = cfg.chat_context_max_lines
        self.debug_enabled = os.environ.get("ECHO_DEBUG", "0").strip().lower() in ("1", "true", "yes", "on")
        self.debug_event_limit = max(0, int(os.environ.get("ECHO_DEBUG_PANEL_LINES", "0")))
        self._debug_events: list[str] = []
        self._debug_log_path: Path | None = None
        self._last_min_interval_debug_at = 0.0
        self._last_asr_latency_debug_at = 0.0
        self._live_diag_decode_ms = 0.0
        self._live_diag_rtf = 0.0
        self._live_diag_lag_secs = 0.0
        self._live_diag_e2e_delay_secs = 0.0
        self._live_diag_cap_q = -1
        self._live_diag_buf_s = 0.0
        self._live_diag_buf_chunks = 0
        self._live_diag_updated_at = 0.0
        if self.debug_enabled:
            try:
                log_dir = Path.cwd() / "log"
                log_dir.mkdir(parents=True, exist_ok=True)
                self._debug_log_path = log_dir / f"echo_live_{int(time.time())}.log"
            except Exception:  # noqa: BLE001
                self._debug_log_path = None
        self._last_silence_debug_at = 0.0
        self._last_live_emitted_spoken = ""

    # ------------------------------------------------------------------ events

    def subscribe(self, listener: Listener) -> Callable[[], None]:
        """Register ``listener(event)``; it may return an awaitable. Returns an unsubscribe fn."""
        self._listeners.append(listener)

        def _unsubscribe() -> None:
            if listener in self._listeners:
                self._listeners.remove(listener)

        return _unsubscribe

    async def _emit(self, event: EngineEvent) -> None:
        for fn in list(self._listeners):
            try:
                r = fn(event)
                if inspect.isawaitable(r):
                    await r
            except Exception:  # noqa: BLE001 — a broken subscriber must not stop the engine
                logger.exception("event listener failed for %s", event.type)

    def _emit_nowait(self, event: EngineEvent) -> None:
        for fn in list(self._listeners):
            try:
                r = fn(event)
            except Exception:  # noqa: BLE001
                logger.exception("event listener failed for %s", event.type)
                continue
            if inspect.isawaitable(r):
                try:
                    asyncio.get_running_loop().create_task(r)  # type: ignore[arg-type]
                except RuntimeError:
                    if inspect.iscoroutine(r):
                        r.close()

    # ------------------------------------------------------------------ status

    def _elapsed_hms(self) -> str:
        dt = timedelta(seconds=int(time.monotonic() - self._t0))
        h, r = divmod(dt.seconds, 3600)
        m, s = divmod(r, 60)
        if dt.days:
            h += dt.days * 24
        return f"{h:02d}:{m:02d}:{s:02d}"

    @property
    def mode(self) -> str:
        if self.live_audio:
            return "live"
        if self.audio_path is not None:
            return "audio"
        return "replay"

    @property
    def mic_only(self) -> bool:
        return self.live_audio and not self.live_system_authorized and self.live_system_device is None

    @property
    def llm_offline(self) -> bool:
        return self._llm_offline

    @property
    def paused(self) -> bool:
        return self._paused

    @property
    def lines(self) -> list[str]:
        return self._lines

    @property
    def insight_passes(self) -> list[InsightPassRecord]:
        return self._insight_passes

    def status(self) -> Status:
        if self._stopped:
            state = "stopped"
        elif self._finished:
            state = "finished"
        elif not self._started:
            state = "idle"
        else:
            state = "paused" if self._paused else "running"
        return Status(
            mode=self.mode,  # type: ignore[arg-type]
            state=state,  # type: ignore[arg-type]
            language=self.live_asr_language,
            languages=list(self.live_asr_languages),
            asr_source=self.asr_load_source,
            asr_label=_render_asr_load_source_label(self.asr_load_source) if self.asr_load_source else "",
            mic_only=self.mic_only,
            llm_offline=self._llm_offline,
            meeting_title=self.meeting_title,
            elapsed=self._elapsed_hms(),
            vault=self.vault,
        )

    def short_asr_label(self) -> str:
        return _short_asr_label(self.asr_load_source)

    def _tick_header(self) -> None:
        self._emit_nowait(self.status())

    def _debug_note(self, message: str) -> None:
        if not self.debug_enabled:
            return
        line = f"[{self._elapsed_hms()}] {message}"
        self._debug_events.append(line)
        if self.debug_event_limit > 0 and len(self._debug_events) > self.debug_event_limit:
            self._debug_events = self._debug_events[-self.debug_event_limit :]
        if self._debug_log_path is not None:
            try:
                with self._debug_log_path.open("a", encoding="utf-8") as f:
                    f.write(line + "\n")
            except Exception:  # noqa: BLE001
                pass

    def _update_live_diagnostics(
        self,
        *,
        decode_secs: float,
        rtf: float,
        lag_secs: float,
        e2e_delay_secs: float,
        cap_q: int,
        buf_s: float,
        buf_chunks: int,
    ) -> None:
        self._live_diag_decode_ms = max(0.0, float(decode_secs)) * 1000.0
        self._live_diag_rtf = max(0.0, float(rtf))
        self._live_diag_lag_secs = max(0.0, float(lag_secs))
        self._live_diag_e2e_delay_secs = max(0.0, float(e2e_delay_secs))
        self._live_diag_cap_q = int(cap_q)
        self._live_diag_buf_s = max(0.0, float(buf_s))
        self._live_diag_buf_chunks = max(0, int(buf_chunks))
        self._live_diag_updated_at = time.monotonic()

    def _log_transcript_meta(self, _log: object, text: str, *, debug_only: bool = False) -> None:
        """Status / error line shown above the transcript (``text`` may contain Rich markup)."""
        if debug_only and not self.debug_enabled:
            return
        if "[red]" in text:
            level = "error"
        elif "[yellow]" in text:
            level = "warning"
        else:
            level = "debug" if debug_only else "info"
        self._emit_nowait(Notice(level=level, text=text))  # type: ignore[arg-type]

    # ------------------------------------------------------------------ control

    def set_notebook(self, text: str) -> None:
        self.notebook_md = text or ""

    def set_meeting_metadata(self, title: str, context: str) -> None:
        self.meeting_title = (title or "").strip() or _UNTITLED_MEETING
        self.meeting_context = (context or "").strip()
        if self._explicit_output_dir is None:
            self.output_dir = Path.cwd() / "output" / meeting_title_to_folder_name(self.meeting_title)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._tick_header()

    @property
    def wiki(self) -> LuminaWiki:
        if self._wiki is None:
            self._wiki = LuminaWiki()
        return self._wiki

    def set_vault(self, vault: str | None) -> None:
        """Attach (or clear with ``None``) the Lumina knowledge vault used by insights and chat."""
        self.vault = (vault or "").strip() or None
        self._debug_note(f"knowledge vault -> {self.vault or '-'}")
        self._warm_vault()
        self._tick_header()

    def _warm_vault(self) -> None:
        """Start the vault's MCP process and cache page titles in the background (first start can be slow)."""
        vault = self.vault
        if not vault or vault in self._vault_titles:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return

        async def _run() -> None:
            try:
                self._vault_titles[vault] = await self.wiki.titles(vault)
            except WikiError as e:
                self._debug_note(f"knowledge vault warm-up failed: {e}")
            except Exception:  # noqa: BLE001
                logger.exception("knowledge vault warm-up failed")

        self._vault_warm_task = loop.create_task(_run())

    def _dictionary_terms(self) -> tuple[str, ...]:
        from echo.dictionary import load_dictionary

        return load_dictionary(notify=self._debug_note)

    def start(self) -> None:
        """Spawn the transcription + insight workers on the running loop (idempotent)."""
        if self._started:
            return
        self._started = True
        self.started_at = datetime.now().astimezone()
        self._t0 = time.monotonic()
        loop = asyncio.get_running_loop()
        self._tasks = [
            loop.create_task(self._transcription_worker()),
            loop.create_task(self._insight_worker()),
            loop.create_task(self._status_ticker()),
        ]
        if self.live_audio:
            self._tasks.append(loop.create_task(self._meter_ticker()))
        self._warm_vault()
        self._tick_header()

    async def stop(self) -> None:
        for t in self._tasks:
            if not t.done():
                t.cancel()
        for t in self._tasks:
            try:
                await t
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        self._tasks = []
        self._finished = True
        self._stopped = True
        self._tick_header()

    def cancel_workers(self) -> None:
        for t in self._tasks:
            if not t.done():
                t.cancel()

    def set_paused(self, paused: bool) -> None:
        self._paused = bool(paused)
        if self.live_audio:
            self._live_meter_note = "Paused" if self._paused else "Listening"
        self._debug_note(f"pause toggled -> {self._paused}")
        self._tick_header()

    def toggle_pause(self) -> None:
        self.set_paused(not self._paused)

    def set_language(self, language: str) -> tuple[bool, str]:
        """Switch the live ASR language and re-route the backend. Returns ``(ok, message)``."""
        from echo.asr_routing import AsrUnavailableError, get_router

        lang = (language or "").strip().lower()
        if lang not in self.live_asr_languages:
            return False, f"Unsupported language: {language!r}"
        self.live_asr_language = lang
        self._debug_note(f"live ASR language -> {lang}")
        router = get_router()
        try:
            router.select(lang)
        except AsrUnavailableError as e:
            self._tick_header()
            return False, f"Live ASR language: {lang} · {e}"
        self.asr_load_source = router.display_source()
        self._tick_header()
        return True, f"Live ASR language: {lang} · {_render_asr_load_source_label(self.asr_load_source)}"

    def cycle_language(self) -> tuple[bool, str]:
        cur = self.live_asr_languages.index(self.live_asr_language)
        return self.set_language(self.live_asr_languages[(cur + 1) % len(self.live_asr_languages)])

    def reset(self) -> None:
        """Clear transcript + insights and pause; the live worker re-arms on the next resume."""
        self._paused = True
        self._insight_session_id += 1
        for task in list(self._insight_tasks):
            if not task.done():
                task.cancel()
        self._insight_tasks.clear()
        self._lines.clear()
        self._line_ids.clear()
        self._line_id_to_index.clear()
        self._line_id_to_spoken.clear()
        self._line_id_to_flushed.clear()
        self._next_line_id = 1
        self._insight_seg_by_id.clear()
        self._insight_seg_order.clear()
        self._insight_line_state.clear()
        self._insight_passes.clear()
        self._chat_history.clear()
        self._last_insight_at = time.monotonic()
        self._last_transcript_activity_at = self._last_insight_at
        self._last_insight_submission_sig = ""
        self._last_comment_input_snapshot = ""
        self._force_live_insight_flush = False
        while not self._transcript_queue.empty():
            try:
                self._transcript_queue.get_nowait()
            except Exception:  # noqa: BLE001
                break
        self._live_meter_rms = 0.0
        self._live_meter_peak = 0.0
        self._live_meter_last_signal_at = 0.0
        self._last_live_emitted_spoken = ""
        self._live_meter_note = "Listening..." if self.live_audio else "Available in --live mode."
        self._live_diag_decode_ms = 0.0
        self._live_diag_rtf = 0.0
        self._live_diag_lag_secs = 0.0
        self._live_diag_e2e_delay_secs = 0.0
        self._live_diag_cap_q = -1
        self._live_diag_buf_s = 0.0
        self._live_diag_buf_chunks = 0
        self._live_diag_updated_at = 0.0
        self._live_reset_seq += 1
        self._debug_events.clear()
        self._debug_note("session reset")
        self._emit_nowait(SessionReset())
        self._tick_header()

    async def _status_ticker(self) -> None:
        while not self._stopped:
            await asyncio.sleep(1.0)
            if self._listeners:
                self._tick_header()

    async def _meter_ticker(self) -> None:
        while True:
            await asyncio.sleep(0.1)
            if self._listeners:
                await self._emit(
                    Meter(rms=self._live_meter_rms, peak=self._live_meter_peak, note=self._live_meter_note)
                )

    # ------------------------------------------------------------------ outputs

    @property
    def summary_in_progress(self) -> bool:
        return self._summary_task is not None and not self._summary_task.done()

    async def wait_for_summary(self) -> None:
        if self._summary_task is not None:
            try:
                await self._summary_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass

    def save_outputs(self, *, announce: bool = True) -> Path:
        """Write notes / insights / context now and (re)generate ``summary.md`` in the background."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "notes.txt").write_text("\n".join(self._lines), encoding="utf-8")
        (self.output_dir / "insights.md").write_text(self._insights_export_markdown(), encoding="utf-8")
        (self.output_dir / "meeting_context.md").write_text(self._meeting_context_markdown(), encoding="utf-8")
        (self.output_dir / "notebook.md").write_text(self.notebook_md, encoding="utf-8")
        (self.output_dir / "insights.json").write_text(
            json.dumps(
                [
                    {"n": r.n, "stamp": r.stamp, "kind": r.kind, "body_md": r.body_md, **({"vault": r.vault} if r.vault else {})}
                    for r in self._insight_passes
                ],
                ensure_ascii=False,
                indent=1,
            ),
            encoding="utf-8",
        )
        (self.output_dir / "meeting.json").write_text(
            json.dumps(self.meeting_meta(), ensure_ascii=False, indent=1), encoding="utf-8"
        )
        self._schedule_meeting_summary_write(announce=announce)
        return self.output_dir

    def _schedule_meeting_summary_write(self, *, announce: bool) -> None:
        """Generate ``summary.md`` via LLM in the background; write a static fallback when offline."""
        summary_path = self.output_dir / "summary.md"
        if self._llm_offline:
            try:
                summary_path.write_text(MEETING_SUMMARY_OFFLINE_MARKDOWN, encoding="utf-8")
            except OSError:
                logger.exception("save: failed to write offline summary.md")
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is None or not loop.is_running():
            try:
                summary_path.write_text(MEETING_SUMMARY_OFFLINE_MARKDOWN, encoding="utf-8")
            except OSError:
                logger.exception("save: failed to write fallback summary.md (no loop)")
            return
        try:
            summary_path.write_text("_Generating meeting summary…_\n", encoding="utf-8")
        except OSError:
            logger.exception("save: failed to write summary.md placeholder")
        if self.summary_in_progress:
            assert self._summary_task is not None
            self._summary_task.cancel()
        self._summary_task = asyncio.create_task(
            self._async_write_meeting_summary(summary_path, announce=announce)
        )

    async def _async_write_meeting_summary(self, summary_path: Path, *, announce: bool) -> None:
        try:
            transcript = "\n".join(self._lines)
            insight_passes = [(rec.kind, rec.body_md) for rec in self._insight_passes]
            md = await llm_meeting_summary(
                transcript,
                insight_passes=insight_passes,
                meeting_title=self.meeting_title,
                meeting_context=self.meeting_context,
                dictionary_terms=self._dictionary_terms(),
                notebook=self.notebook_md,
                assets_dir=self.output_dir / "notes-assets",
            )
            summary_path.write_text(md, encoding="utf-8")
            patch_meeting_json(self.output_dir, summary_preview=summary_preview(md))
            await self._emit(SummaryWritten(path=str(summary_path), ok=True, announce=announce))
        except Exception:  # noqa: BLE001 - never let summary failure crash the app
            logger.exception("save: failed to generate meeting summary")
            try:
                summary_path.write_text(
                    MEETING_SUMMARY_OFFLINE_MARKDOWN
                    + "\n_Summary generation raised an unexpected error; see app logs._\n",
                    encoding="utf-8",
                )
            except OSError:
                logger.exception("save: failed to write fallback summary.md after error")
            await self._emit(SummaryWritten(path=str(summary_path), ok=False, announce=announce))

    def meeting_meta(self) -> dict[str, object]:
        return {
            "version": 1,
            "id": self.output_dir.name,
            "title": self.meeting_title,
            "context": self.meeting_context,
            "started_at": self.started_at.isoformat(timespec="seconds"),
            "ended_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "duration_s": int(time.monotonic() - self._t0),
            "mode": self.mode,
            "language": self.live_asr_language if self.live_audio else "",
            "asr_source": self.asr_load_source or "",
            "line_count": len(self._lines),
            "has_notebook": bool(self.notebook_md.strip()),
            "vault": self.vault,
        }

    def _meeting_context_markdown(self) -> str:
        ctx = self.meeting_context or "_(none)_"
        return f"# Meeting\n\n- Title: {self.meeting_title}\n\n## Context\n\n{ctx}\n"

    def _insights_export_markdown(self) -> str:
        lines: list[str] = [f"# Insights — {self.meeting_title}", ""]
        if not self._insight_passes:
            lines.append("_No insights generated yet._")
            lines.append("")
            return "\n".join(lines)
        for rec in self._insight_passes:
            heading = {"comment": "Comment", "wiki": f"From {rec.vault or 'knowledge vault'}"}.get(rec.kind, "Research summary")
            lines.append(f"## Pass {rec.n} @ {rec.stamp}")
            lines.append("")
            lines.append(f"### {heading}")
            lines.append("")
            lines.append(rec.body_md.strip() or "_(empty)_")
            lines.append("")
            if rec.web_md:
                lines.append("### Vault pages" if rec.kind == "wiki" else "### Web search")
                lines.append("")
                lines.append(rec.web_md.strip())
                lines.append("")
        return "\n".join(lines)

    # ------------------------------------------------------------------ transcript

    def transcript_snapshot(self) -> list[TranscriptLineState]:
        return [
            TranscriptLineState(
                line_id=lid,
                line=line,
                text=self._line_id_to_spoken.get(lid, ""),
                flushed=self._line_id_to_flushed.get(lid, False),
            )
            for lid, line in zip(self._line_ids, self._lines, strict=True)
        ]

    async def _refresh_transcript_log(self) -> None:
        await self._emit(TranscriptCorrected(lines=self.transcript_snapshot()))

    async def _refresh_last_transcript_line(self) -> None:
        if not self._line_ids:
            return
        lid = self._line_ids[-1]
        await self._emit(
            TranscriptDraft(line_id=lid, line=self._lines[-1], text=self._line_id_to_spoken.get(lid, ""))
        )

    async def _on_segment_finalized(self, seg: TranscriptSegment) -> None:
        idx = self._line_id_to_index.get(seg.line_id)
        line = self._lines[idx] if idx is not None and 0 <= idx < len(self._lines) else seg.line
        await self._emit(TranscriptFinal(line_id=seg.line_id, line=line, text=seg.spoken_text))

    async def _emit_transcript_line(
        self,
        _log: object,
        line: str,
        *,
        spoken_text: str | None = None,
        allow_correction: bool = True,
        stable_delay: float = 0.0,
        enqueue: bool = True,
        finalized: bool = True,
    ) -> TranscriptSegment:
        spoken = _normalize_spaces(spoken_text or _strip_leading_ts(line))
        seg = TranscriptSegment(
            line_id=self._next_line_id,
            line=line,
            spoken_text=spoken,
            stable_after=time.monotonic() + max(0.0, stable_delay),
            allow_correction=allow_correction,
            finalized=finalized,
        )
        self._next_line_id += 1
        self._line_id_to_index[seg.line_id] = len(self._lines)
        self._line_id_to_spoken[seg.line_id] = seg.spoken_text
        self._line_id_to_flushed[seg.line_id] = False
        self._line_ids.append(seg.line_id)
        self._lines.append(line)
        cls = TranscriptFinal if finalized else TranscriptDraft
        await self._emit(cls(line_id=seg.line_id, line=line, text=spoken))
        if enqueue:
            await self._transcript_queue.put(seg)
        if self.live_audio:
            self._debug_note(
                f"emit segment id={seg.line_id} "
                f"stable={max(seg.stable_after - time.monotonic(), 0.0):.1f}s allow_fix={seg.allow_correction}",
            )
        return seg

    async def _flush_live_group(
        self,
        log: object,
        *,
        t0w: float,
        t1w: float,
        text: str,
        allow_correction: bool,
        split_groups: bool = True,
    ) -> None:
        payload = _normalize_spaces(text)
        if not payload:
            return
        groups = _split_live_groups(payload, max_chars=self.live_group_chars) if split_groups else [payload]
        if not groups:
            return
        ts = f"[{format_timestamp_hms(t0w)}–{format_timestamp_hms(t1w)}]"
        for i, g in enumerate(groups):
            await self._emit_transcript_line(
                log,
                f"{ts} {g}",
                spoken_text=g,
                allow_correction=allow_correction,
                stable_delay=self.live_rewrite_stable_seconds if self.live_audio else 0.0,
            )
            if i < len(groups) - 1 and self.live_group_delay > 0:
                await asyncio.sleep(self.live_group_delay)

    # ------------------------------------------------------------------ workers

    async def _transcription_worker(self) -> None:
        try:
            if self.live_audio:
                await self._transcription_from_live()
            elif self.audio_path is not None:
                await self._transcription_from_audio()
            else:
                await self._transcription_from_replay()
        finally:
            await self._transcript_queue.put(self._eof_sentinel)

    async def _transcription_from_replay(self) -> None:
        for raw in _load_replay_lines(self.replay_path, self.replay_lines):
            while self._paused:
                await asyncio.sleep(0.15)
            line = raw.strip()
            if not line:
                continue
            if not line.startswith("["):
                line = f"[{self._elapsed_hms()}] {line}"
            await self._emit_transcript_line(None, line)
            await asyncio.sleep(self.line_delay)

    async def _transcription_from_audio(self) -> None:
        from echo.asr import TranscribeSession

        path = self.audio_path
        if path is None or not path.is_file():
            self._log_transcript_meta(None, f"[red]Audio file not found:[/] {path}")
            return
        load_msg = (
            f"[dim]Loading `{self.model_id}` from local cache (no Hugging Face Hub download)…[/]"
            if hub_local_only_from_env()
            else f"[dim]Loading `{self.model_id}` (first run may download ~3GB of weights)…[/]"
        )
        self._log_transcript_meta(None, load_msg, debug_only=True)
        self._log_transcript_meta(
            None,
            f"[dim]ASR model load source: {_render_asr_load_source_label(self.asr_load_source)}.[/]",
            debug_only=True,
        )
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
            self._log_transcript_meta(None, f"[red]{e}[/]")
            return
        except Exception as e:  # noqa: BLE001
            self._log_transcript_meta(None, f"[red]ASR init failed:[/] {e!s}")
            return

        self._log_transcript_meta(None, "[dim]Transcription started (p = pause between chunks).[/]", debug_only=True)
        while True:
            while self._paused:
                await asyncio.sleep(0.15)
            try:
                seg = await asyncio.to_thread(session.next_segment)
            except Exception as e:  # noqa: BLE001
                self._log_transcript_meta(None, f"[red]Chunk failed:[/] {e!s}")
                break
            if seg is None:
                break
            t0, t1, text = seg
            if not text.strip():
                continue
            line = f"[{format_timestamp_hms(t0)}–{format_timestamp_hms(t1)}] {text.strip()}"
            await self._emit_transcript_line(None, line, spoken_text=text.strip())

    async def _transcription_from_live(self) -> None:
        from echo.live_session import LiveTranscriptionSession

        await LiveTranscriptionSession(self, None).run()

    async def _insight_worker(self) -> None:
        try:
            await self._insight_loop()
        finally:
            self._finished = True
            self._emit_nowait(SessionEnded())
            self._tick_header()

    async def _insight_loop(self) -> None:
        while True:
            try:
                chunk = await asyncio.wait_for(self._transcript_queue.get(), timeout=1.5)
            except asyncio.TimeoutError:
                chunk = None
            if chunk == self._eof_sentinel:
                source_segments = [self._insight_seg_by_id[i] for i in self._insight_seg_order]
                buf_text = "\n".join(seg.spoken_text for seg in source_segments).strip()
                self._insight_seg_by_id.clear()
                self._insight_seg_order.clear()
                self._insight_line_state.clear()
                if buf_text and not self._llm_offline:
                    await self._run_insight_pass(buf_text, source_segments, session_id=self._insight_session_id)
                break
            if isinstance(chunk, TranscriptSegment) and not self._llm_offline:
                norm_spoken = _normalize_spaces(chunk.spoken_text)
                state = (norm_spoken, bool(chunk.finalized))
                if self._insight_line_state.get(chunk.line_id) == state:
                    continue
                self._insight_line_state[chunk.line_id] = state
                lid = chunk.line_id
                if lid not in self._insight_seg_by_id:
                    self._insight_seg_order.append(lid)
                self._insight_seg_by_id[lid] = chunk
                self._last_transcript_activity_at = time.monotonic()
            if self.live_audio and self._paused and not self._force_live_insight_flush:
                continue
            now = time.monotonic()
            if self.live_audio:
                if self._force_live_insight_flush:
                    eligible = [self._insight_seg_by_id[i] for i in self._insight_seg_order]
                else:
                    eligible = [
                        self._insight_seg_by_id[i]
                        for i in self._insight_seg_order
                        if self._insight_seg_by_id[i].finalized
                    ]
            else:
                eligible = [
                    self._insight_seg_by_id[i]
                    for i in self._insight_seg_order
                    if self._insight_seg_by_id[i].stable_after <= now
                ]
            buf_text = "\n".join(seg.spoken_text for seg in eligible).strip()
            new_finalized_segments = sum(1 for seg in eligible if seg.finalized)
            max_finalized_segment_chars = max(
                (len(seg.spoken_text) for seg in eligible if seg.finalized),
                default=0,
            )
            trigger_reason = ""
            if self.live_audio:
                force_final = self._force_live_insight_flush
                should_run, trigger_reason = _should_trigger_insight_live(
                    new_chars=len(buf_text),
                    new_finalized_segments=new_finalized_segments,
                    max_finalized_segment_chars=max_finalized_segment_chars,
                    finalized_segment_min_chars=self.insight_finalized_segment_min_chars,
                    skip_min_chars=self.insight_skip_min_chars,
                )
                if force_final and buf_text:
                    should_run = True
                    trigger_reason = "forced_final"
                elif force_final and not buf_text:
                    self._force_live_insight_flush = False
                elif should_run and (now - self._last_insight_at) < self.insight_live_min_interval:
                    should_run = False
                    trigger_reason = "min_interval"
                    if now - self._last_min_interval_debug_at >= 1.0:
                        wait_for = max(0.0, self.insight_live_min_interval - (now - self._last_insight_at))
                        self._debug_note(
                            "insight skipped "
                            f"reason=min_interval wait={wait_for:.1f}s chars={len(buf_text)} "
                            f"finalized={new_finalized_segments}",
                        )
                        self._last_min_interval_debug_at = now
            else:
                should_run = bool(buf_text) and (
                    len(buf_text) >= self.insight_min_chars
                    or (now - self._last_insight_at) >= self.insight_min_interval
                )
                if should_run:
                    trigger_reason = "chars_or_interval"
            if not should_run:
                continue
            if self._insight_lock.locked():
                if trigger_reason == "forced_final":
                    self._force_live_insight_flush = True
                self._debug_note("insight skipped reason=busy")
                continue

            if (
                self.live_audio
                and not self._force_live_insight_flush
                and self.insight_pause_capture_q > 0
                and self._live_capture_queue_depth >= self.insight_pause_capture_q
            ):
                if now - self._last_capture_backlog_debug_at >= 1.0:
                    self._debug_note(
                        "insight skipped "
                        f"reason=capture_backlog cap_q={self._live_capture_queue_depth}"
                        f">={self.insight_pause_capture_q}",
                    )
                    self._last_capture_backlog_debug_at = now
                continue

            snapshot = buf_text
            source_segments = list(eligible)
            if self.live_audio and not self._force_live_insight_flush:
                if not _has_meaningful_new_content_since_comment(
                    current_text=snapshot,
                    previous_comment_input=self._last_comment_input_snapshot,
                    min_delta_chars=self.insight_skip_min_chars,
                ):
                    self._debug_note("insight skipped reason=no_new_content_since_last_comment")
                    continue
            sig = _insight_segments_signature(source_segments)
            if self.live_audio and sig == self._last_insight_submission_sig:
                self._debug_note("insight skipped reason=duplicate_input")
                continue
            consumed = {seg.line_id for seg in source_segments}
            for sid in consumed:
                self._insight_seg_by_id.pop(sid, None)
                self._insight_line_state.pop(sid, None)
            self._insight_seg_order = [i for i in self._insight_seg_order if i not in consumed]
            self._debug_note(
                "insight trigger "
                f"reason={trigger_reason or '-'} chars={len(snapshot)} "
                f"segments={len(source_segments)} buffered_left={len(self._insight_seg_order)}",
            )
            self._last_insight_at = now
            self._last_insight_submission_sig = sig
            if trigger_reason == "forced_final":
                self._force_live_insight_flush = False
            task = asyncio.create_task(
                self._run_insight_pass(snapshot, source_segments, session_id=self._insight_session_id)
            )
            self._insight_tasks.add(task)
            task.add_done_callback(self._insight_tasks.discard)

    # ------------------------------------------------------------------ insights

    def _apply_segment_text(self, seg: TranscriptSegment, new_spoken: str) -> bool:
        new_text = _normalize_spaces(new_spoken)
        if not new_text:
            return False
        old_text = self._line_id_to_spoken.get(seg.line_id, seg.spoken_text)
        if old_text.split() == new_text.split():
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

    async def _append_insight_pass(self, rec: InsightPassRecord) -> None:
        self._insight_passes.append(rec)
        await self._emit(
            InsightPass(
                n=rec.n,
                stamp=rec.stamp,
                kind=rec.kind,
                body_md=rec.body_md,
                routing_md=rec.routing_md,
                corrected_md=rec.corrected_md,
                web_md=rec.web_md,
                vault=rec.vault,
            )
        )

    async def _run_insight_pass(
        self,
        snapshot: str,
        source_segments: list[TranscriptSegment] | None = None,
        *,
        session_id: int | None = None,
    ) -> None:
        if self._llm_offline:
            return
        llm_timeout = float(os.environ.get("ECHO_INSIGHT_LLM_TIMEOUT", "120"))
        ddg_timeout = float(os.environ.get("ECHO_INSIGHT_DDG_TIMEOUT", "45"))

        def _is_stale() -> bool:
            return session_id is not None and session_id != self._insight_session_id

        async with self._insight_lock:
            if _is_stale():
                self._debug_note("insight pass aborted reason=session_reset")
                return
            next_n = len(self._insight_passes) + 1
            stamp = self._elapsed_hms()
            await self._emit(InsightPending(n=next_n, stamp=stamp))

            try:
                try:
                    seg_payload = (
                        [(seg.line_id, seg.spoken_text) for seg in source_segments] if source_segments else None
                    )
                    if seg_payload:
                        self._debug_note(
                            f"insight pass input segments={len(seg_payload)} "
                            f"ids={seg_payload[0][0]}-{seg_payload[-1][0]}",
                        )
                    vault = self.vault
                    vault_hits = await self._vault_hints(vault, snapshot) if vault else []
                    route = await asyncio.wait_for(
                        llm_route_insight(
                            snapshot,
                            segments=seg_payload,
                            meeting_context=self.meeting_context,
                            meeting_title=self.meeting_title,
                            dictionary_terms=self._dictionary_terms(),
                            vault=vault,
                            vault_titles=self._vault_titles.get(vault or "", []),
                            vault_hits=vault_hits,
                        ),
                        timeout=llm_timeout,
                    )
                    if _is_stale():
                        self._debug_note("insight pass aborted reason=session_reset post-route")
                        return
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
                    route = InsightRouteResult("comment", f"_Insight routing failed:_ `{e!s}`", "", "error", "")

                ctx = route.corrected_transcript.strip() or snapshot
                raw_for_md = "\n".join(seg.spoken_text for seg in source_segments) if source_segments else snapshot
                self._debug_note(
                    f"route action={route.action} structured_edits={len(route.corrected_segments or [])} "
                    f"reason={route.reason or '-'}",
                )
                corrected_md = _insight_corrected_segment_md(raw_for_md, route.corrected_transcript)
                applied = False
                if self.enable_transcript_rewrite and source_segments:
                    applied = self._try_apply_corrected_transcript(
                        source_segments,
                        route.corrected_transcript,
                        route.corrected_segments,
                    )
                if applied:
                    for seg in source_segments or []:
                        self._line_id_to_flushed[seg.line_id] = True
                    await self._refresh_transcript_log()
                    corrected_md = None
                    self._debug_note("transcript rewrite applied")
                elif source_segments and (route.corrected_segments or route.corrected_transcript.strip()):
                    if self.enable_transcript_rewrite:
                        self._debug_note("transcript rewrite skipped (no eligible mapping)")
                    else:
                        self._debug_note("transcript rewrite disabled (raw mode)")

                if route.action == "comment":
                    await self._append_insight_pass(
                        InsightPassRecord(
                            n=next_n,
                            stamp=stamp,
                            kind="comment",
                            body_md=route.comment.strip() or "_(empty)_",
                            routing_md=_insight_routing_collapsible(route, web_path=False),
                            corrected_md=corrected_md,
                            web_md=None,
                        ),
                    )
                    self._last_comment_input_snapshot = _normalize_spaces(snapshot)
                elif route.action == "wiki_lookup" and self.vault:
                    if not await self._run_wiki_pass(route, ctx, corrected_md, next_n, stamp, llm_timeout, _is_stale):
                        return
                else:
                    query = route.search_query.strip() or ctx[:100].strip()
                    hits: list[SearchHit] = []
                    search_err = ""
                    search_provider = ""
                    if query:
                        try:
                            outcome = await asyncio.wait_for(web_search(query, max_results=5), timeout=ddg_timeout)
                            hits, search_provider, search_err = outcome.hits, outcome.provider, outcome.note
                        except asyncio.TimeoutError:
                            search_err = (
                                f"Web search timed out after {ddg_timeout:.0f}s "
                                "(network, rate limit, or proxy). Raise ECHO_INSIGHT_DDG_TIMEOUT or try ECHO_DDGS_BACKEND=bing."
                            )
                        except Exception as e:  # noqa: BLE001
                            search_err = f"Web search failed: `{e!s}`"
                        if search_provider:
                            self._debug_note(f"web search provider={search_provider} hits={len(hits)}")
                    try:
                        summary = await asyncio.wait_for(
                            llm_synthesize_web_findings(
                                ctx,
                                query,
                                hits,
                                self.meeting_context,
                                meeting_title=self.meeting_title,
                                dictionary_terms=self._dictionary_terms(),
                            ),
                            timeout=llm_timeout,
                        )
                    except asyncio.TimeoutError:
                        summary = (
                            f"_Web synthesis LLM timed out after {llm_timeout:.0f}s; "
                            f"showing search excerpts._\n\n**Query:** `{query}`"
                        )
                    except Exception as e:  # noqa: BLE001
                        summary = f"_Web synthesis failed:_ `{e!s}`\n\n**Query:** `{query}`"

                    if _is_stale():
                        self._debug_note("insight pass aborted reason=session_reset post-web")
                        return
                    await self._append_insight_pass(
                        InsightPassRecord(
                            n=next_n,
                            stamp=stamp,
                            kind="web_search",
                            body_md=summary.strip() or "_(empty)_",
                            routing_md=_insight_routing_collapsible(route, web_path=True),
                            corrected_md=corrected_md,
                            web_md=_format_web_search_md(query, hits, search_err, provider=search_provider),
                        ),
                    )
                if source_segments:
                    for seg in source_segments:
                        self._line_id_to_flushed[seg.line_id] = True
                    if not applied:
                        await self._refresh_transcript_log()
            except Exception as e:  # noqa: BLE001
                logger.exception("insight pass failed")
                await self._append_insight_pass(
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

    async def _run_wiki_pass(
        self,
        route: InsightRouteResult,
        ctx: str,
        corrected_md: str | None,
        n: int,
        stamp: str,
        llm_timeout: float,
        is_stale: Callable[[], bool],
    ) -> bool:
        """Search the vault and append a ``wiki`` insight. Returns False when the pass went stale."""
        vault = self.vault or ""
        query = route.wiki_query.strip() or route.search_query.strip() or ctx[:100].strip()
        hits: list[WikiHit] = []
        pages: list[WikiPage] = []
        err = ""
        try:
            hits, pages = await asyncio.wait_for(self.wiki.lookup(vault, query, limit=5, read=3), timeout=30)
        except asyncio.TimeoutError:
            err = f"Lumina is still loading **{vault}** (the first start can take up to a minute); the next pass will retry."
        except WikiError as e:
            err = f"Knowledge vault lookup failed: {e}"
        except Exception as e:  # noqa: BLE001 — Lumina must never break an insight pass
            err = f"Knowledge vault lookup failed: `{e!s}`"
        self._debug_note(f"wiki lookup vault={vault} hits={len(hits)} pages={len(pages)} err={err or '-'}")
        if err or not hits:
            body = f"_{err}_" if err else f"_Nothing in **{vault}** matched `{query}`._"
            if route.comment.strip():
                body = f"{route.comment.strip()}\n\n{body}"
        else:
            try:
                body = await asyncio.wait_for(
                    llm_synthesize_wiki_findings(
                        ctx,
                        query,
                        hits,
                        pages,
                        vault=vault,
                        meeting_context=self.meeting_context,
                        meeting_title=self.meeting_title,
                        dictionary_terms=self._dictionary_terms(),
                    ),
                    timeout=llm_timeout,
                )
            except asyncio.TimeoutError:
                body = f"_Vault synthesis LLM timed out after {llm_timeout:.0f}s; showing vault matches._"
            except Exception as e:  # noqa: BLE001
                body = f"_Vault synthesis failed:_ `{e!s}`"
        if is_stale():
            self._debug_note("insight pass aborted reason=session_reset post-wiki")
            return False
        await self._append_insight_pass(
            InsightPassRecord(
                n=n,
                stamp=stamp,
                kind="wiki",
                body_md=body.strip() or "_(empty)_",
                routing_md=_insight_routing_collapsible(route, web_path=False, wiki_path=True),
                corrected_md=corrected_md,
                web_md=_format_wiki_md(query, hits, err, vault=vault),
                vault=vault,
            ),
        )
        return True

    async def _vault_hints(self, vault: str, snapshot: str) -> list[WikiHit]:
        """Quick keyword search on the chunk so the router can judge whether the vault is relevant."""
        self._warm_vault()
        if vault not in self._vault_titles:
            return []
        try:
            return await asyncio.wait_for(self.wiki.search(vault, snapshot[-600:], limit=6), timeout=4)
        except Exception as e:  # noqa: BLE001 — hints are optional
            self._debug_note(f"vault hint search skipped: {e or 'timeout'}")
            return []

    async def _chat_wiki_excerpts(self, question: str) -> str:
        vault = self.vault
        if not vault:
            return ""
        try:
            hits, pages = await asyncio.wait_for(self.wiki.lookup(vault, question, limit=5, read=2), timeout=20)
        except (WikiError, asyncio.TimeoutError) as e:
            self._debug_note(f"chat wiki lookup skipped: {e or 'timeout'}")
            return ""
        except Exception:  # noqa: BLE001
            logger.exception("chat wiki lookup failed")
            return ""
        if not hits:
            return ""
        return f"Knowledge vault \"{vault}\" (search results for the question):\n<<<\n{wiki_sources_block(hits, pages)}\n>>>"

    # ------------------------------------------------------------------ chat

    def _chat_transcript_context(self) -> str:
        lines = self._lines
        cap = self.chat_context_max_lines
        if len(lines) > cap:
            omitted = len(lines) - cap
            return (
                f"(…{omitted} earlier transcript line(s) omitted; "
                f"raise ECHO_CHAT_CONTEXT_MAX_LINES to include more.)\n" + "\n".join(lines[-cap:])
            )
        return "\n".join(lines)

    async def ask(self, question: str) -> str:
        """Answer a question about the meeting so far; also emits ``ChatReply``."""
        q = (question or "").strip()
        if not q:
            return ""
        if self._llm_offline:
            ans = "LLM unavailable; chat is disabled."
        else:
            try:
                ans = await llm_chat_answer(
                    self._chat_transcript_context(),
                    q,
                    self.meeting_context,
                    meeting_title=self.meeting_title,
                    dictionary_terms=self._dictionary_terms(),
                    wiki_excerpts=await self._chat_wiki_excerpts(q),
                )
            except Exception as e:  # noqa: BLE001 — surface errors in the chat pane
                ans = f"(error) {e!s}"
        self._chat_history.append((q, ans))
        await self._emit(ChatReply(question=q, answer=ans))
        return ans
