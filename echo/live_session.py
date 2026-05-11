"""Live ASR session worker.

Encapsulates the per-session state of ``EchoApp._transcription_from_live`` (capture handle,
chunk buffer, pending pieces, draft line, snapshot bookkeeping) so the app class can stay focused
on UI orchestration. The class is intentionally tightly coupled to ``EchoApp`` — it borrows the
app's transcript queue, debug helpers, and per-line bookkeeping rather than duplicating them.
"""

from __future__ import annotations

import asyncio
import os
import queue
import time
from collections import deque
from typing import TYPE_CHECKING

import numpy as np

from echo.text_utils import format_timestamp_hms

if TYPE_CHECKING:
    from textual.widgets import RichLog

    from echo.app import EchoApp, TranscriptSegment


class LiveTranscriptionSession:
    """Owns the lifecycle of one ``--live`` capture run.

    All previously-``nonlocal`` state of ``_transcription_from_live`` lives here as instance
    attributes, which makes the segment-finalization rules far easier to read and to test.
    """

    def __init__(self, app: "EchoApp", log: "RichLog") -> None:
        self.app = app
        self.log = log

        # Capture state.
        self._cap = None  # type: ignore[assignment]
        self._system_cap = None  # type: ignore[assignment]
        self._sr = 16000
        self._chunk_samples = 0
        self._hop_samples = 1
        # Live audio backlog: queue of float32 blocks + byte offset into the first block.
        # Avoids O(n) ``np.concatenate`` / giant reallocations when ASR falls behind realtime.
        self._audio_blocks: deque[np.ndarray] = deque()
        self._audio_block0_off = 0
        self._piece_workspace: np.ndarray | None = None

        # Sliding chunk dedup.
        self._last_chunk_text = ""

        # Pending ASR pieces that haven't been pushed to the live draft yet.
        self._pending_texts: list[str] = []
        self._pending_t0w: float | None = None
        self._pending_t1w: float | None = None
        self._pending_allow_correction = True
        self._last_voiced_wall = 0.0

        # Live draft (one mutable transcript line that grows until a finalize trigger fires).
        self._draft_segment: "TranscriptSegment | None" = None
        self._draft_t0w: float | None = None
        self._draft_allow_correction = True
        self._draft_stable_after = 0.0
        self._draft_last_snapshot_text = ""
        self._draft_last_snapshot_at = 0.0

        # Cached language for the duration of this session (cycled via the 'l' key).
        self._lang = self.app.live_asr_language
        # After pause, arm at most one post-drain forced insight per "idle" episode (see main loop).
        self._post_pause_drain_insight_done = False

    # ------------------------------------------------------------------ helpers

    def _pause_drain_quiescent(self) -> bool:
        """True when capture out-queue is empty and no full ASR window remains in the stream."""
        cap_q = self._capture_out_queue_depth()
        if cap_q > 0:
            return False
        if cap_q < 0:
            # Unknown depth (no capture handle); do not treat as idle.
            return False
        return self._pending_asr_chunk_iterations() == 0

    def _clear_pending(self) -> None:
        self._pending_texts = []
        self._pending_t0w = None
        self._pending_t1w = None
        self._pending_allow_correction = True

    def _reset_draft_state(self) -> None:
        self._draft_segment = None
        self._draft_t0w = None
        self._draft_allow_correction = True
        self._draft_stable_after = 0.0
        self._draft_last_snapshot_text = ""
        self._draft_last_snapshot_at = 0.0

    async def _start_draft(
        self,
        *,
        t0w: float,
        t1w: float,
        payload: str,
        allow_correction: bool,
    ) -> None:
        """Begin a new draft transcript line and queue an initial snapshot for insights."""
        line = f"[{format_timestamp_hms(t0w)}–{format_timestamp_hms(t1w)}] {payload}"
        seg = await self.app._emit_transcript_line(
            self.log,
            line,
            spoken_text=payload,
            allow_correction=allow_correction,
            stable_delay=self.app.live_rewrite_stable_seconds if self.app.live_audio else 0.0,
            enqueue=False,
            finalized=False,
        )
        self._draft_segment = seg
        self._draft_t0w = t0w
        self._draft_allow_correction = allow_correction
        self._draft_stable_after = seg.stable_after
        self._draft_last_snapshot_text = payload
        self._draft_last_snapshot_at = time.monotonic()
        from echo.app import TranscriptSegment as _Seg

        await self.app._transcript_queue.put(
            _Seg(
                line_id=seg.line_id,
                line=line,
                spoken_text=payload,
                stable_after=self._draft_stable_after,
                allow_correction=allow_correction,
                finalized=False,
            )
        )

    async def _append_live_draft(
        self,
        *,
        t0w: float,
        t1w: float,
        text: str,
        allow_correction: bool,
    ) -> None:
        from echo.app import _normalize_spaces

        payload = _normalize_spaces(text)
        if not payload:
            return

        if self._draft_segment is None or self._draft_t0w is None:
            await self._start_draft(t0w=t0w, t1w=t1w, payload=payload, allow_correction=allow_correction)
            return

        seg_id = self._draft_segment.line_id
        idx = self.app._line_id_to_index.get(seg_id)
        if idx is None or idx < 0 or idx >= len(self.app._lines):
            # The draft line was wiped externally (e.g. reset between chunks); start fresh.
            await self._start_draft(t0w=t0w, t1w=t1w, payload=payload, allow_correction=allow_correction)
            return

        prev_spoken = self.app._line_id_to_spoken.get(seg_id, self._draft_segment.spoken_text)
        merged = _normalize_spaces(f"{prev_spoken} {payload}")
        line = f"[{format_timestamp_hms(self._draft_t0w)}–{format_timestamp_hms(t1w)}] {merged}"
        self.app._lines[idx] = line
        self.app._line_id_to_spoken[seg_id] = merged
        self._draft_allow_correction = self._draft_allow_correction and allow_correction
        self._draft_segment.line = line
        self._draft_segment.spoken_text = merged
        self._draft_segment.allow_correction = self._draft_allow_correction
        self._draft_segment.stable_after = self._draft_stable_after
        self._draft_segment.finalized = False
        # Hot path: only the trailing draft line changed, do a surgical replace instead of
        # clearing and rewriting the entire transcript every chunk.
        await self.app._refresh_last_transcript_line()
        now = time.monotonic()
        should_snapshot = (
            merged != self._draft_last_snapshot_text
            and (now - self._draft_last_snapshot_at) >= self.app.insight_pause_seconds
        )
        if should_snapshot:
            self._draft_last_snapshot_text = merged
            self._draft_last_snapshot_at = now
            from echo.app import TranscriptSegment as _Seg

            await self.app._transcript_queue.put(
                _Seg(
                    line_id=self._draft_segment.line_id,
                    line=line,
                    spoken_text=merged,
                    stable_after=self._draft_stable_after,
                    allow_correction=self._draft_allow_correction,
                    finalized=False,
                )
            )

    async def _finalize_live_draft(self, reason: str) -> bool:
        """Mark the current draft line as finalized; return True if a finalized segment was emitted."""
        if self._draft_segment is None:
            return False
        spoken = self.app._line_id_to_spoken.get(
            self._draft_segment.line_id, self._draft_segment.spoken_text,
        ).strip()
        if not spoken:
            self._reset_draft_state()
            return False
        if spoken == self.app._last_live_emitted_spoken:
            self.app._debug_note("pause flush dropped identical segment")
            self._reset_draft_state()
            return False
        seg = self._draft_segment
        seg.spoken_text = spoken
        seg.allow_correction = self._draft_allow_correction
        seg.stable_after = self._draft_stable_after
        seg.finalized = True
        await self.app._transcript_queue.put(seg)
        self.app._last_live_emitted_spoken = spoken
        self.app._debug_note(
            f"flush local reason={reason} chars={len(spoken)} allow_fix={self._draft_allow_correction}",
        )
        self._reset_draft_state()
        return True

    def _draft_finalize_reason(self, *, reason: str, idle_for: float) -> str:
        """Decide whether the in-flight draft should be finalized now, and why."""
        from echo.app import (
            _LIVE_SENTENCE_END_ANYWHERE_RE,
            _LIVE_SENTENCE_END_RE,
            _line_span_seconds,
        )

        seg = self._draft_segment
        if seg is None:
            return ""
        spoken = self.app._line_id_to_spoken.get(seg.line_id, seg.spoken_text).strip()
        if not spoken:
            return ""
        if reason in ("pause", "paused", "capture_stopped", "stop"):
            return reason
        if idle_for >= self.app.live_segment_pause_seconds:
            return "pause"

        over_limit = False
        if len(spoken) >= self.app.ui_segment_max_chars:
            over_limit = True
        else:
            span = _line_span_seconds(seg.line)
            if span is not None:
                t0, t1 = span
                if max(0.0, t1 - t0) >= self.app.ui_segment_max_seconds:
                    over_limit = True

        ends_at_sentence = _LIVE_SENTENCE_END_RE.search(spoken)

        # Over length/time limit: prefer cutting at a sentence boundary, but if ASR output has no
        # punctuation, cut immediately — the segment is already long enough.
        if over_limit:
            if ends_at_sentence:
                return "max_sentence"
            if _LIVE_SENTENCE_END_ANYWHERE_RE.search(spoken):
                return "max_mid_sentence"
            return "max_no_punct"

        # Normal sentence-end: need a brief pause to avoid cutting between back-to-back sentences
        # in continuous speech.
        if (
            ends_at_sentence
            and len(spoken) >= self.app.ui_segment_semantic_min_chars
            and idle_for >= min(self.app.live_segment_pause_seconds, 0.6)
        ):
            return "sentence_end"
        return ""

    async def _flush_pending(self, reason: str) -> None:
        from echo.app import _normalize_spaces

        if not self._pending_texts or self._pending_t0w is None or self._pending_t1w is None:
            idle_for = (
                time.monotonic() - self._last_voiced_wall if self._last_voiced_wall > 0 else 0.0
            )
            fin_reason = self._draft_finalize_reason(reason=reason, idle_for=idle_for)
            if fin_reason:
                emitted = await self._finalize_live_draft(fin_reason)
                if emitted and fin_reason in ("stop", "capture_stopped", "paused"):
                    self.app._force_live_insight_flush = True
            return

        joined = _normalize_spaces(" ".join(self._pending_texts))
        if not joined:
            self._clear_pending()
            return

        if self.app.live_emit_interval <= 0.0:
            await self._append_live_draft(
                t0w=self._pending_t0w,
                t1w=self._pending_t1w,
                text=joined,
                allow_correction=self._pending_allow_correction,
            )
            self._clear_pending()
            idle_for = (
                time.monotonic() - self._last_voiced_wall if self._last_voiced_wall > 0 else 0.0
            )
            fin_reason = self._draft_finalize_reason(reason=reason, idle_for=idle_for)
            if fin_reason:
                emitted = await self._finalize_live_draft(fin_reason)
                if emitted and fin_reason in ("stop", "capture_stopped", "paused"):
                    self.app._force_live_insight_flush = True
            return

        if reason in ("pause", "paused", "capture_stopped", "stop"):
            await self._append_live_draft(
                t0w=self._pending_t0w,
                t1w=self._pending_t1w,
                text=joined,
                allow_correction=self._pending_allow_correction,
            )
            self._clear_pending()
            idle_for = (
                time.monotonic() - self._last_voiced_wall if self._last_voiced_wall > 0 else 0.0
            )
            fin_reason = self._draft_finalize_reason(reason=reason, idle_for=idle_for)
            if fin_reason:
                emitted = await self._finalize_live_draft(fin_reason)
                if emitted:
                    self.app._force_live_insight_flush = True
            return

        # Legacy interval batching path.
        if joined == self.app._last_live_emitted_spoken:
            self.app._debug_note("pause flush dropped identical segment")
        else:
            await self.app._flush_live_group(
                self.log,
                t0w=self._pending_t0w,
                t1w=self._pending_t1w,
                text=joined,
                allow_correction=self._pending_allow_correction,
                split_groups=False,
            )
            self.app._last_live_emitted_spoken = joined
            self.app._debug_note(
                f"flush local reason={reason} chars={len(joined)} "
                f"allow_fix={self._pending_allow_correction}",
            )
        self._clear_pending()

    async def _flush_if_needed(self, *, idle_tick: bool) -> None:
        from echo.app import _should_flush_live_pending

        idle_for = (
            time.monotonic() - self._last_voiced_wall if self._last_voiced_wall > 0 else 0.0
        )
        if not self._pending_texts:
            if (
                idle_tick
                and self.app.live_emit_interval <= 0.0
                and self._draft_segment is not None
            ):
                fin_reason = self._draft_finalize_reason(reason="", idle_for=idle_for)
                if fin_reason:
                    await self._flush_pending(fin_reason)
            return

        pending_duration = 0.0
        if self._pending_t0w is not None and self._pending_t1w is not None:
            pending_duration = max(0.0, self._pending_t1w - self._pending_t0w)
        should_flush, reason = _should_flush_live_pending(
            has_pending=bool(self._pending_texts),
            pending_duration=pending_duration,
            idle_tick=idle_tick,
            idle_for=idle_for,
            pause_seconds=self.app.live_segment_pause_seconds,
            emit_interval=self.app.live_emit_interval,
        )
        if should_flush:
            await self._flush_pending(reason)

    # ------------------------------------------------------------------ capture lifecycle

    async def _start_capture(self) -> bool:
        """Build the platform-specific capture and start it. Return False on a fatal init error."""
        from echo.live_audio import (
            LiveMixerConfig,
            LivePairedCapture,
            LiveSystemAuthorizedCapture,
            LiveSystemAuthorizedConfig,
        )

        app = self.app
        lc = app.config.live
        if app.live_system_authorized:
            from echo.live_audio import ensure_macos_system_audio_helper

            cmd = app.live_system_authorized_cmd or os.environ.get("ECHO_SYSTEM_AUDIO_CMD")
            if not cmd:
                try:
                    cmd = await asyncio.to_thread(ensure_macos_system_audio_helper)
                except Exception as e:  # noqa: BLE001
                    app._log_transcript_meta(
                        self.log,
                        "[red]Failed to build bundled macOS system-audio helper:[/] "
                        f"{e!s}\n"
                        "[dim]You can still provide an explicit command via "
                        "--live-system-authorized-cmd '/path/to/cmd'.[/]",
                    )
                    return False
            self._system_cap = LiveSystemAuthorizedCapture(
                LiveSystemAuthorizedConfig(
                    mic_device=app.live_mic_device,
                    command=cmd,
                    samplerate=self._sr,
                    blocksize=2048,
                    capture_out_queue_max=lc.capture_out_queue_max,
                    capture_mix_queue_max=lc.capture_mix_queue_max,
                )
            )
            self._cap = self._system_cap
            try:
                self._cap.start()
            except Exception as e:  # noqa: BLE001
                app._log_transcript_meta(
                    self.log,
                    "[red]Failed to start system-authorized capture:[/] "
                    f"{e!s}\n"
                    "[dim]Tip: install/build SystemAudioDump and pass "
                    "--live-system-authorized-cmd '/path/to/SystemAudioDump'.[/]",
                )
                return False
        else:
            cfg = LiveMixerConfig(
                mic_device=app.live_mic_device,
                system_device=app.live_system_device,
                samplerate=self._sr,
                blocksize=2048,
                capture_out_queue_max=lc.capture_out_queue_max,
                capture_mix_queue_max=lc.capture_mix_queue_max,
            )
            self._cap = LivePairedCapture(cfg)
            self._cap.start()
        return True

    def _emit_capture_banners(self) -> None:
        app = self.app
        if app.live_system_authorized:
            app._log_transcript_meta(
                self.log,
                "[dim]Live: microphone + system output via authorized macOS capture command "
                "(System Audio Recording Only on macOS 14.2+, otherwise Screen Recording; "
                "Microphone permission may also be requested). "
                "p = pause (stops live capture feeding the queue; drains backlog through ASR). "
                "Transcript appends continuously and creates a new segment only after a long pause.[/]",
                debug_only=True,
            )
        elif app.live_system_device is None:
            app._log_transcript_meta(
                self.log,
                "[dim]Live: microphone only. Add --live-system-device (e.g. BlackHole 2ch index) "
                "to mix system audio, or create an aggregate input in Audio MIDI Setup. "
                "p = pause (stops live capture feeding the queue; drains backlog through ASR). "
                "Transcript appends continuously and creates a new segment only after a long pause.[/]",
                debug_only=True,
            )
        else:
            app._log_transcript_meta(
                self.log,
                "[dim]Live: microphone + second input (50/50 mono mix). "
                "p = pause (stops live capture feeding the queue; drains backlog through ASR). "
                "Transcript appends continuously and creates a new segment only after a long pause.[/]",
                debug_only=True,
            )

    def _set_capture_paused(self, paused: bool) -> None:
        cap = self._cap
        if cap is None:
            return
        fn = getattr(cap, "set_capture_paused", None)
        if callable(fn):
            fn(paused)

    def _drain_capture_queue(self) -> None:
        if self._cap is None:
            return
        while not self._cap.out_q.empty():
            try:
                self._cap.out_q.get_nowait()
            except Exception:  # noqa: BLE001
                break

    def _reset_for_new_session(self) -> None:
        self._clear_pending()
        self._reset_draft_state()
        self._last_chunk_text = ""
        self._last_voiced_wall = 0.0
        self._audio_blocks.clear()
        self._audio_block0_off = 0
        self.app._live_wall_t0 = time.monotonic()
        self.app._last_live_emitted_spoken = ""
        self._drain_capture_queue()
        self.app._debug_note("live worker state reset")

    def _audio_stream_sample_count(self) -> int:
        """Total float32 samples queued (logical stream length)."""
        if not self._audio_blocks:
            return 0
        total = 0
        for i, b in enumerate(self._audio_blocks):
            if i == 0:
                total += b.size - self._audio_block0_off
            else:
                total += b.size
        return total

    def _append_audio_block(self, arr: np.ndarray) -> None:
        if arr.size == 0:
            return
        self._audio_blocks.append(arr)

    def _peek_copy_first_n_samples(self, dest: np.ndarray, n: int) -> bool:
        """Copy the first ``n`` samples of the logical stream into ``dest[:n]`` without consuming."""
        if self._audio_stream_sample_count() < n:
            return False
        filled = 0
        for block_i, b in enumerate(self._audio_blocks):
            start = self._audio_block0_off if block_i == 0 else 0
            avail = b.size - start
            take = min(avail, n - filled)
            dest[filled : filled + take] = b[start : start + take]
            filled += take
            if filled >= n:
                break
        return filled == n

    def _consume_stream_samples(self, n: int) -> None:
        """Drop the first ``n`` samples from the logical stream (sliding-window hop)."""
        remaining = n
        while remaining > 0 and self._audio_blocks:
            b0 = self._audio_blocks[0]
            avail = b0.size - self._audio_block0_off
            take = min(avail, remaining)
            self._audio_block0_off += take
            remaining -= take
            if self._audio_block0_off >= b0.size:
                self._audio_blocks.popleft()
                self._audio_block0_off = 0

    def _capture_out_queue_depth(self) -> int:
        """Approximate number of mixed-audio blocks waiting in the capture thread queue (-1 if unknown)."""
        cap = self._cap
        if cap is None:
            return -1
        q = getattr(cap, "out_q", None)
        if q is None:
            return -1
        try:
            return int(q.qsize())
        except Exception:
            return -1

    def _pending_asr_chunk_iterations(self) -> int:
        """How many more ASR windows could be taken from the internal stream without new capture (0+)."""
        s = self._audio_stream_sample_count()
        cs = self._chunk_samples
        hop = self._hop_samples
        n = 0
        while s >= cs:
            n += 1
            s -= hop
        return n

    def _drain_out_q_into_stream_nonblocking(self) -> int:
        """Move all blocks currently in capture ``out_q`` into ``_audio_blocks`` (non-blocking)."""
        cap = self._cap
        if cap is None:
            return 0
        q = getattr(cap, "out_q", None)
        if q is None:
            return 0
        app = self.app
        if app.live_audio:
            lc = app.config.live
            thr = lc.capture_out_q_drop_threshold
            if thr > 0:
                try:
                    depth0 = int(q.qsize())
                except Exception:
                    depth0 = -1
                if depth0 > thr:
                    batch = lc.capture_out_q_drop_batch
                    dropped = 0
                    while dropped < batch:
                        try:
                            q.get_nowait()
                        except queue.Empty:
                            break
                        dropped += 1
                    if dropped:
                        app._live_capture_queue_depth = self._capture_out_queue_depth()
                        app._debug_note(
                            "capture out_q trim "
                            f"count={dropped} depth_before={depth0} thr={thr}",
                        )
        n_moved = 0
        while True:
            try:
                block = q.get_nowait()
            except queue.Empty:
                break
            arr = np.asarray(block, dtype=np.float32).reshape(-1)
            if arr.size == 0:
                continue
            block_rms = float(np.sqrt(np.mean(np.square(arr), dtype=np.float64)))
            block_peak = float(np.max(np.abs(arr)))
            app._live_meter_rms = (app._live_meter_rms * 0.75) + (block_rms * 0.25)
            app._live_meter_peak = max(block_peak, app._live_meter_peak * 0.88)
            if block_peak > 1e-4:
                app._live_meter_last_signal_at = time.monotonic()
            self._append_audio_block(arr)
            n_moved += 1
        if n_moved:
            app._live_capture_queue_depth = self._capture_out_queue_depth()
        return n_moved

    async def _async_run_buffered_asr_pass(self, log: "RichLog") -> bool:
        """Run ASR on every full window in ``_audio_blocks`` (same rules as the main live loop)."""
        from echo.app import (
            _live_chunk_metrics,
            _normalize_spaces,
            _passes_live_voice_gate,
            _should_drop_live_noise_token,
            _trim_chunk_overlap,
        )
        from echo.asr import transcribe_mono16k_chunk
        from echo.audio_filters import denoise, vad_speech_ratio

        app = self.app
        ws = self._piece_workspace
        assert ws is not None
        progressed = False
        while self._audio_stream_sample_count() >= self._chunk_samples:
            if not self._peek_copy_first_n_samples(ws, self._chunk_samples):
                break
            self._consume_stream_samples(self._hop_samples)
            piece = ws
            t0w = time.monotonic() - (app._live_wall_t0 or time.monotonic())
            t1w = t0w + (self._chunk_samples / self._sr)
            piece_rms, piece_peak, piece_active_ratio = _live_chunk_metrics(
                piece, active_amp=app.live_active_amp,
            )
            if not _passes_live_voice_gate(
                rms=piece_rms,
                peak=piece_peak,
                active_ratio=piece_active_ratio,
                min_rms=app.live_min_rms,
                active_amp=app.live_active_amp,
                min_active_ratio=app.live_min_active_ratio,
            ):
                if not app._paused:
                    app._live_meter_note = "Silence gate"
                    now_dbg = time.monotonic()
                    if now_dbg - app._last_silence_debug_at >= 1.8:
                        app._debug_note(
                            f"gate silence rms={piece_rms:.4f} peak={piece_peak:.4f} "
                            f"active={piece_active_ratio:.3f}",
                        )
                        app._last_silence_debug_at = now_dbg
                await self._flush_if_needed(idle_tick=True)
                progressed = True
                continue

            # Optional denoise stage (spectral gating). Runs in a worker so a slow noisereduce
            # call does not stall the event loop.
            if app.live_denoise_backend != "off":
                try:
                    piece = await asyncio.to_thread(
                        denoise,
                        piece,
                        backend=app.live_denoise_backend,
                        sr=self._sr,
                        prop_decrease=app.live_denoise_strength,
                        notify=app._debug_note,
                    )
                except Exception as e:  # noqa: BLE001 -- never let filters break the loop
                    app._debug_note(f"denoise error: {e!s}")

            # Optional model-based VAD layered on top of the cheap energy gate.
            if app.live_vad_backend != "off":
                try:
                    speech_ratio = await asyncio.to_thread(
                        vad_speech_ratio,
                        piece,
                        backend=app.live_vad_backend,
                        sr=self._sr,
                        speech_prob_threshold=app.live_vad_speech_prob_threshold,
                        aggressiveness=app.live_vad_aggressiveness,
                        notify=app._debug_note,
                    )
                except Exception as e:  # noqa: BLE001
                    app._debug_note(f"vad error: {e!s}")
                    speech_ratio = 1.0
                if speech_ratio < app.live_vad_min_speech_ratio:
                    if not app._paused:
                        app._live_meter_note = "VAD gate"
                        app._debug_note(
                            f"vad drop ratio={speech_ratio:.2f} "
                            f"thr={app.live_vad_min_speech_ratio:.2f} "
                            f"backend={app.live_vad_backend}",
                        )
                    await self._flush_if_needed(idle_tick=True)
                    progressed = True
                    continue

            app._live_capture_queue_depth = self._capture_out_queue_depth()
            try:
                asr_t0 = time.monotonic()
                text = await asyncio.to_thread(
                    transcribe_mono16k_chunk,
                    piece,
                    model_id=app.model_id,
                    device=app.whisper_device,
                    language=app.live_asr_language or self._lang,
                )
                asr_t1 = time.monotonic()
            except Exception as e:  # noqa: BLE001
                app._log_transcript_meta(log, f"[red]Live chunk failed:[/] {e!s}")
                app._live_meter_note = "ASR error"
                app._debug_note(f"asr chunk error: {e!s}")
                progressed = True
                continue
            chunk_secs = max(1e-6, (self._chunk_samples / self._sr))
            decode_secs = max(0.0, asr_t1 - asr_t0)
            rtf = decode_secs / chunk_secs
            live_now = time.monotonic() - (app._live_wall_t0 or time.monotonic())
            lag_secs = max(0.0, live_now - t1w)
            e2e_delay_secs = max(0.0, live_now - t0w)
            cap_q = self._capture_out_queue_depth()
            app._live_capture_queue_depth = cap_q
            buf_s = self._audio_stream_sample_count() / float(self._sr)
            buf_chunks = self._pending_asr_chunk_iterations()
            app._update_live_diagnostics(
                decode_secs=decode_secs,
                rtf=rtf,
                lag_secs=lag_secs,
                e2e_delay_secs=e2e_delay_secs,
                cap_q=cap_q,
                buf_s=buf_s,
                buf_chunks=buf_chunks,
            )
            if app.live_asr_log_interval_seconds > 0.0:
                if (
                    time.monotonic() - app._last_asr_latency_debug_at
                ) >= app.live_asr_log_interval_seconds:
                    app._debug_note(
                        "asr latency "
                        f"decode={decode_secs*1000:.0f}ms chunk={chunk_secs:.2f}s "
                        f"rtf={rtf:.2f} lag={lag_secs:.2f}s e2e_delay={e2e_delay_secs:.2f}s "
                        f"cap_q={cap_q} buf_s={buf_s:.2f} buf_chunks={buf_chunks}",
                    )
                    app._last_asr_latency_debug_at = time.monotonic()
            progressed = True
            if not (text or "").strip():
                continue
            cleaned = _normalize_spaces(text)
            if not cleaned:
                continue
            deduped = _trim_chunk_overlap(
                self._last_chunk_text,
                cleaned,
                min_chars=app.live_overlap_min_chars,
            )
            self._last_chunk_text = cleaned
            if not deduped:
                app._debug_note("dedup dropped fully-overlapped chunk")
                continue
            is_segment_start = self._draft_segment is None and not self._pending_texts
            if _should_drop_live_noise_token(
                text=deduped,
                noise_tokens=app.live_noise_tokens,
                always_drop_tokens=app.live_always_drop_noise_tokens,
                max_chars=app.live_noise_max_chars,
                piece_rms=piece_rms,
                min_rms=app.live_min_rms,
                max_rms_factor=app.live_noise_max_rms_factor,
                is_segment_start=is_segment_start,
                drop_at_segment_start=app.live_noise_drop_at_segment_start,
            ):
                app._debug_note(
                    f"noise token dropped text={deduped!r} rms={piece_rms:.4f} "
                    f"segment_start={is_segment_start}",
                )
                continue

            allow_piece_correction = piece_rms >= (app.live_min_rms * 1.25) or (
                piece_active_ratio >= (app.live_min_active_ratio * 2.0)
            )
            if self._pending_t0w is None:
                self._pending_t0w = t0w
            self._pending_t1w = t1w
            self._pending_texts.append(deduped)
            self._pending_allow_correction = (
                self._pending_allow_correction and allow_piece_correction
            )
            self._last_voiced_wall = time.monotonic()
            await self._flush_if_needed(idle_tick=False)
        return progressed

    async def _drain_capture_and_asr_while_paused(self, log: "RichLog") -> bool:
        """While paused: do not call ``read_block``; capture is paused at the device/reader layer.

        Drains ``out_q`` into ``_audio_blocks`` and runs ASR on full windows already buffered.
        """
        n = self._drain_out_q_into_stream_nonblocking()
        progressed_asr = await self._async_run_buffered_asr_pass(log)
        return bool(n or progressed_asr)

    # ------------------------------------------------------------------ main loop

    async def run(self) -> None:
        from echo.app import (
            _render_asr_load_source_label,
        )
        from echo.connectivity import hub_local_only_from_env

        app = self.app
        log = self.log

        app._log_transcript_meta(
            log,
            (
                f"[dim]Live ASR (`{app.model_id}`) — local Hub cache only, no download.[/]"
                if hub_local_only_from_env()
                else f"[dim]Live ASR (`{app.model_id}`) — model already loaded on the main thread.[/]"
            ),
            debug_only=True,
        )
        app._log_transcript_meta(
            log,
            f"[dim]ASR model load source: {_render_asr_load_source_label(app.asr_load_source)}.[/]",
            debug_only=True,
        )
        self._emit_capture_banners()
        if not await self._start_capture():
            return

        self._chunk_samples = max(int(app.chunk_seconds * self._sr), int(0.5 * self._sr))
        overlap_samples = max(0, int(app.live_chunk_overlap_seconds * self._sr))
        if overlap_samples >= self._chunk_samples:
            overlap_samples = max(0, self._chunk_samples // 2)
        self._hop_samples = max(1, self._chunk_samples - overlap_samples)
        self._piece_workspace = np.empty(self._chunk_samples, dtype=np.float32)
        app._live_wall_t0 = time.monotonic()
        app._live_meter_note = "Listening"

        try:
            pause_drain_done = False
            reset_seq = app._live_reset_seq
            while True:
                while app._paused:
                    if not pause_drain_done:
                        self._set_capture_paused(True)
                        await self._flush_pending("paused")
                        pause_drain_done = True
                        self._post_pause_drain_insight_done = False
                    worked = await self._drain_capture_and_asr_while_paused(log)
                    if worked:
                        # More audio arrived or ASR produced text; allow another post-drain arm when idle again.
                        self._post_pause_drain_insight_done = False
                    elif (
                        not self._post_pause_drain_insight_done
                        and self._pause_drain_quiescent()
                    ):
                        # Pause-time forced_final may have run before backlog ASR finished; once the
                        # capture queue and internal stream have no full windows left, nudge insight
                        # so the worker can flush on the next tick (it wakes via queue timeout).
                        app._force_live_insight_flush = True
                        self._post_pause_drain_insight_done = True
                        app._debug_note("insight arm reason=pause_drain_quiescent")
                    if not worked:
                        await asyncio.sleep(0.05)
                self._set_capture_paused(False)
                if app._live_reset_seq != reset_seq:
                    reset_seq = app._live_reset_seq
                    self._reset_for_new_session()
                pause_drain_done = False

                block = await asyncio.to_thread(self._cap.read_block, 0.35)
                if block is None:
                    if self._system_cap is not None and self._system_cap.process_exited():
                        msg = self._system_cap.process_error_text()
                        await self._flush_pending("capture_stopped")
                        app._log_transcript_meta(
                            log,
                            "[red]System-authorized capture stopped:[/] "
                            f"{msg or 'unknown helper failure'}",
                        )
                        app._live_meter_note = "Capture stopped"
                        app._debug_note(
                            f"capture stopped: {msg or 'unknown helper failure'}"
                        )
                        return
                    await self._flush_if_needed(idle_tick=True)
                    await asyncio.sleep(0.02)
                    continue
                arr = np.asarray(block, dtype=np.float32).reshape(-1)
                if arr.size == 0:
                    await self._flush_if_needed(idle_tick=True)
                    await asyncio.sleep(0.02)
                    continue
                block_rms = float(np.sqrt(np.mean(np.square(arr), dtype=np.float64)))
                block_peak = float(np.max(np.abs(arr)))
                app._live_meter_rms = (app._live_meter_rms * 0.75) + (block_rms * 0.25)
                app._live_meter_peak = max(block_peak, app._live_meter_peak * 0.88)
                if block_peak > 1e-4:
                    app._live_meter_last_signal_at = time.monotonic()
                    if not app._paused:
                        app._live_meter_note = "Signal detected"
                self._append_audio_block(arr)
                app._live_capture_queue_depth = self._capture_out_queue_depth()

                await self._async_run_buffered_asr_pass(log)
        finally:
            await self._flush_pending("stop")
            app._live_meter_note = "Stopped"
            app._debug_note("live transcription stopped")
            self._set_capture_paused(False)
            if self._cap is not None:
                self._cap.stop()
