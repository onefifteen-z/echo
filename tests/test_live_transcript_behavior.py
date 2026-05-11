from __future__ import annotations


def test_split_live_groups_breaks_long_chunks() -> None:
    from echo.app import _split_live_groups

    text = "Quarterly update is done, action items are confirmed, next topic starts now."
    groups = _split_live_groups(text, max_chars=24)

    assert len(groups) >= 2
    assert all(g.strip() for g in groups)
    assert all(len(g) <= 24 for g in groups)


def test_live_stop_word_detection_handles_trailing_punctuation() -> None:
    from echo.app import _live_hits_stop_word

    assert _live_hits_stop_word("Thanks everyone, done.", ("done",))
    assert _live_hits_stop_word("以上です。", ("以上です",))
    assert not _live_hits_stop_word("Still discussing open tasks", ("done",))


def test_live_noise_token_detection_handles_punctuation_and_case() -> None:
    from echo.app import _is_live_noise_token

    assert _is_live_noise_token("ごめん。", noise_tokens=("ごめん",), max_chars=4)
    assert _is_live_noise_token("GOMEN?", noise_tokens=("gomen",), max_chars=5)
    assert not _is_live_noise_token("ごめん です", noise_tokens=("ごめん",), max_chars=4)


def test_should_drop_live_noise_token_on_segment_start_even_with_louder_audio() -> None:
    from echo.app import _should_drop_live_noise_token

    assert _should_drop_live_noise_token(
        text="ごめん",
        noise_tokens=("ごめん",),
        always_drop_tokens=(),
        max_chars=4,
        piece_rms=0.02,
        min_rms=0.0035,
        max_rms_factor=1.3,
        is_segment_start=True,
        drop_at_segment_start=True,
    )


def test_should_drop_live_noise_token_by_rms_when_not_segment_start() -> None:
    from echo.app import _should_drop_live_noise_token

    assert _should_drop_live_noise_token(
        text="ごめん",
        noise_tokens=("ごめん",),
        always_drop_tokens=(),
        max_chars=4,
        piece_rms=0.003,
        min_rms=0.0035,
        max_rms_factor=1.3,
        is_segment_start=False,
        drop_at_segment_start=True,
    )
    assert not _should_drop_live_noise_token(
        text="ごめん",
        noise_tokens=("ごめん",),
        always_drop_tokens=(),
        max_chars=4,
        piece_rms=0.02,
        min_rms=0.0035,
        max_rms_factor=1.3,
        is_segment_start=False,
        drop_at_segment_start=True,
    )


def test_should_drop_live_noise_token_always_drop_list_ignores_rms_and_position() -> None:
    from echo.app import _should_drop_live_noise_token

    assert _should_drop_live_noise_token(
        text="ごめん",
        noise_tokens=("ごめん",),
        always_drop_tokens=("ごめん",),
        max_chars=4,
        piece_rms=0.2,
        min_rms=0.0035,
        max_rms_factor=1.3,
        is_segment_start=False,
        drop_at_segment_start=False,
    )


def test_trim_chunk_overlap_removes_prefix_duplication() -> None:
    from echo.app import _trim_chunk_overlap

    prev = "today we review budget and timeline"
    cur = "budget and timeline then decide owners"
    out = _trim_chunk_overlap(prev, cur)
    assert out == "then decide owners"


def test_meter_helpers_map_amplitude_to_bar() -> None:
    from echo.app import _amp_to_dbfs, _db_to_unit, _meter_bar, _meter_timeline, _meter_vertical_chart

    silent_db = _amp_to_dbfs(0.0)
    assert silent_db <= -119.0
    assert _db_to_unit(silent_db) == 0.0
    assert _meter_bar(0.0, width=8) == "--------"

    loud_db = _amp_to_dbfs(1.0)
    assert loud_db == 0.0
    assert _db_to_unit(loud_db) == 1.0
    assert _meter_bar(1.0, width=8) == "########"
    assert _meter_timeline([0.0, 0.5, 1.0], width=6).endswith(" =@")
    chart = _meter_vertical_chart([0.0, 0.5, 1.0], width=6, rows=4)
    assert len(chart) == 4
    assert any("█" in row or "▓" in row or "▒" in row for row in chart)


def test_live_voice_gate_rejects_silence_chunk() -> None:
    from echo.app import _live_chunk_metrics, _passes_live_voice_gate

    samples = [0.0002] * 2000
    rms, peak, active_ratio = _live_chunk_metrics(samples, active_amp=0.012)
    assert not _passes_live_voice_gate(
        rms=rms,
        peak=peak,
        active_ratio=active_ratio,
        min_rms=0.0035,
        active_amp=0.012,
        min_active_ratio=0.015,
    )


def test_live_voice_gate_accepts_speech_like_chunk() -> None:
    from echo.app import _live_chunk_metrics, _passes_live_voice_gate

    samples = ([0.0] * 1700) + ([0.03] * 300)
    rms, peak, active_ratio = _live_chunk_metrics(samples, active_amp=0.012)
    assert _passes_live_voice_gate(
        rms=rms,
        peak=peak,
        active_ratio=active_ratio,
        min_rms=0.0035,
        active_amp=0.012,
        min_active_ratio=0.015,
    )


def test_live_vad_drops_chunk_below_threshold(monkeypatch) -> None:
    """When VAD ratio < min_speech_ratio, the chunk is dropped before Whisper."""
    import asyncio
    from types import SimpleNamespace

    import numpy as np

    import echo.app as app_mod
    import echo.audio_filters as af
    import echo.live_session as live_session_mod
    from echo.live_session import LiveTranscriptionSession

    # Force the energy gate to always pass so we exercise the new VAD branch.
    monkeypatch.setattr(app_mod, "_passes_live_voice_gate", lambda **_kw: True)

    # Stub VAD: report only 5% speech.
    monkeypatch.setattr(af, "vad_speech_ratio", lambda *_a, **_k: 0.05)

    # Whisper must NOT be invoked when the gate trips.
    transcribe_calls: list[object] = []

    def _fake_transcribe(*_args, **_kwargs):
        transcribe_calls.append(_args)
        return "should-not-run"

    import echo.asr as asr_mod

    monkeypatch.setattr(asr_mod, "transcribe_mono16k_chunk", _fake_transcribe)

    debug_notes: list[str] = []
    fake_app = SimpleNamespace(
        # VAD knobs
        live_vad_backend="silero",
        live_vad_min_speech_ratio=0.10,
        live_vad_speech_prob_threshold=0.5,
        live_vad_aggressiveness=2,
        # Denoise knobs (off so it does not interfere)
        live_denoise_backend="off",
        live_denoise_strength=0.85,
        # Energy gate knobs (irrelevant - we forced the gate above)
        live_min_rms=0.0035,
        live_active_amp=0.012,
        live_min_active_ratio=0.015,
        # ASR knobs
        live_asr_language="japanese",
        model_id="whisper-stub",
        whisper_device=None,
        live_asr_log_interval_seconds=0.0,
        live_emit_interval=15.0,
        # Pause / wall-clock state
        _paused=False,
        _live_wall_t0=None,
        _live_meter_rms=0.0,
        _live_meter_peak=0.0,
        _live_meter_last_signal_at=0.0,
        _live_meter_note="Listening...",
        _live_capture_queue_depth=-1,
        _last_silence_debug_at=0.0,
        _last_asr_latency_debug_at=0.0,
        _debug_note=debug_notes.append,
        _update_live_diagnostics=lambda **_kw: None,
    )

    # Build the session and seed its audio buffer with one full chunk of "loud" audio.
    log = SimpleNamespace()
    session = LiveTranscriptionSession.__new__(LiveTranscriptionSession)
    session.app = fake_app
    session.log = log
    session._cap = None
    session._system_cap = None
    session._sr = 16000
    session._chunk_samples = 16000  # 1s window
    session._hop_samples = 16000
    session._audio_blocks = __import__("collections").deque()
    session._audio_block0_off = 0
    session._piece_workspace = np.zeros(session._chunk_samples, dtype=np.float32)
    session._last_chunk_text = ""
    session._pending_texts = []
    session._pending_t0w = None
    session._pending_t1w = None
    session._pending_allow_correction = True
    session._last_voiced_wall = 0.0
    session._draft_segment = None
    session._draft_t0w = None
    session._draft_allow_correction = True
    session._draft_stable_after = 0.0
    session._draft_last_snapshot_text = ""
    session._draft_last_snapshot_at = 0.0
    session._lang = "japanese"
    session._post_pause_drain_insight_done = False

    # 1 second of saturated audio: easily passes the (forced) energy gate.
    session._append_audio_block(np.full(16000, 0.5, dtype=np.float32))

    asyncio.run(session._async_run_buffered_asr_pass(log))

    assert transcribe_calls == [], "Whisper was called even though VAD gate should have dropped the chunk"
    assert fake_app._live_meter_note == "VAD gate"
    assert any("vad drop" in n for n in debug_notes), debug_notes
    # Buffer should be drained after the hop.
    assert session._audio_stream_sample_count() == 0


def test_live_denoise_runs_before_vad(monkeypatch) -> None:
    """When denoise backend is enabled, the pipeline calls denoise() *before* vad_speech_ratio()."""
    import asyncio
    from types import SimpleNamespace

    import numpy as np

    import echo.app as app_mod
    import echo.audio_filters as af
    from echo.live_session import LiveTranscriptionSession

    monkeypatch.setattr(app_mod, "_passes_live_voice_gate", lambda **_kw: True)

    call_order: list[str] = []

    def _fake_denoise(samples, **_kwargs):
        call_order.append("denoise")
        # Mark the array so we can prove the VAD step received the post-denoise audio.
        return np.full_like(samples, 0.123, dtype=np.float32)

    def _fake_vad(samples, **_kwargs):
        call_order.append("vad")
        # Confirm we are inspecting the denoised tensor.
        assert abs(float(samples[0]) - 0.123) < 1e-6
        # Force a drop so we do not have to stub the full Whisper post-processing chain.
        return 0.0

    monkeypatch.setattr(af, "denoise", _fake_denoise)
    monkeypatch.setattr(af, "vad_speech_ratio", _fake_vad)

    fake_app = SimpleNamespace(
        live_vad_backend="silero",
        live_vad_min_speech_ratio=0.10,
        live_vad_speech_prob_threshold=0.5,
        live_vad_aggressiveness=2,
        live_denoise_backend="noisereduce",
        live_denoise_strength=0.85,
        live_min_rms=0.0035,
        live_active_amp=0.012,
        live_min_active_ratio=0.015,
        live_asr_language="japanese",
        model_id="whisper-stub",
        whisper_device=None,
        live_asr_log_interval_seconds=0.0,
        live_emit_interval=15.0,
        _paused=False,
        _live_wall_t0=None,
        _live_meter_rms=0.0,
        _live_meter_peak=0.0,
        _live_meter_last_signal_at=0.0,
        _live_meter_note="Listening...",
        _live_capture_queue_depth=-1,
        _last_silence_debug_at=0.0,
        _last_asr_latency_debug_at=0.0,
        _debug_note=lambda _msg: None,
        _update_live_diagnostics=lambda **_kw: None,
    )

    log = SimpleNamespace()
    session = LiveTranscriptionSession.__new__(LiveTranscriptionSession)
    session.app = fake_app
    session.log = log
    session._cap = None
    session._system_cap = None
    session._sr = 16000
    session._chunk_samples = 16000
    session._hop_samples = 16000
    session._audio_blocks = __import__("collections").deque()
    session._audio_block0_off = 0
    session._piece_workspace = np.zeros(session._chunk_samples, dtype=np.float32)
    session._last_chunk_text = ""
    session._pending_texts = []
    session._pending_t0w = None
    session._pending_t1w = None
    session._pending_allow_correction = True
    session._last_voiced_wall = 0.0
    session._draft_segment = None
    session._draft_t0w = None
    session._draft_allow_correction = True
    session._draft_stable_after = 0.0
    session._draft_last_snapshot_text = ""
    session._draft_last_snapshot_at = 0.0
    session._lang = "japanese"
    session._post_pause_drain_insight_done = False

    session._append_audio_block(np.full(16000, 0.5, dtype=np.float32))

    asyncio.run(session._async_run_buffered_asr_pass(log))

    assert call_order == ["denoise", "vad"]


def test_should_trigger_insight_live_skips_without_finalized_segment() -> None:
    from echo.app import _should_trigger_insight_live

    ok, reason = _should_trigger_insight_live(
        new_chars=80,
        new_finalized_segments=0,
        max_finalized_segment_chars=0,
        finalized_segment_min_chars=100,
        skip_min_chars=50,
    )
    assert ok is False
    assert reason == ""


def test_insight_segments_signature_distinguishes_finalized_state() -> None:
    from echo.app import TranscriptSegment, _insight_segments_signature

    seg_a = TranscriptSegment(
        line_id=7,
        line="[00:00:01–00:00:02] hello",
        spoken_text="hello",
        stable_after=0.0,
        allow_correction=True,
        finalized=False,
    )
    seg_b = TranscriptSegment(
        line_id=7,
        line="[00:00:01–00:00:02] hello",
        spoken_text="hello",
        stable_after=0.0,
        allow_correction=True,
        finalized=True,
    )
    assert _insight_segments_signature([seg_a]) != _insight_segments_signature([seg_b])


def test_meaningful_new_content_since_comment_requires_delta_threshold() -> None:
    from echo.app import _has_meaningful_new_content_since_comment

    assert not _has_meaningful_new_content_since_comment(
        current_text="status update",
        previous_comment_input="status update",
        min_delta_chars=50,
    )
    assert not _has_meaningful_new_content_since_comment(
        current_text="status update plus tiny",
        previous_comment_input="status update",
        min_delta_chars=50,
    )
    assert _has_meaningful_new_content_since_comment(
        current_text="status update " + ("x" * 80),
        previous_comment_input="status update",
        min_delta_chars=50,
    )


def test_should_trigger_insight_live_skips_on_pause_without_finalized() -> None:
    from echo.app import _should_trigger_insight_live

    ok, reason = _should_trigger_insight_live(
        new_chars=90,
        new_finalized_segments=0,
        max_finalized_segment_chars=0,
        finalized_segment_min_chars=100,
        skip_min_chars=50,
    )
    assert ok is False
    assert reason == ""


def test_render_asr_load_source_label() -> None:
    from echo.app import _render_asr_load_source_label

    assert _render_asr_load_source_label("local-cache-only") == "local cache only (offline mode)"
    assert _render_asr_load_source_label("local-cache") == "local cache (no network fetch needed)"
    assert _render_asr_load_source_label("hub-fetch") == "network (Hugging Face Hub fallback)"
    assert _render_asr_load_source_label("other") == "unknown"


def test_should_trigger_insight_live_skips_on_char_count_without_finalized() -> None:
    from echo.app import _should_trigger_insight_live

    ok, reason = _should_trigger_insight_live(
        new_chars=240,
        new_finalized_segments=0,
        max_finalized_segment_chars=0,
        finalized_segment_min_chars=100,
        skip_min_chars=50,
    )
    assert ok is False
    assert reason == ""


def test_should_trigger_insight_live_stays_off_when_short_and_active() -> None:
    from echo.app import _should_trigger_insight_live

    ok, reason = _should_trigger_insight_live(
        new_chars=40,
        new_finalized_segments=1,
        max_finalized_segment_chars=200,
        finalized_segment_min_chars=100,
        skip_min_chars=50,
    )
    assert ok is False
    assert reason == ""


def test_should_trigger_insight_live_on_new_finalized_segment() -> None:
    from echo.app import _should_trigger_insight_live

    ok, reason = _should_trigger_insight_live(
        new_chars=120,
        new_finalized_segments=1,
        max_finalized_segment_chars=120,
        finalized_segment_min_chars=100,
        skip_min_chars=50,
    )
    assert ok is True
    assert reason == "new_segment"


def test_should_flush_live_pending_on_interval() -> None:
    from echo.app import _should_flush_live_pending

    ok, reason = _should_flush_live_pending(
        has_pending=True,
        pending_duration=6.0,
        idle_tick=False,
        idle_for=0.1,
        pause_seconds=2.0,
        emit_interval=5.0,
    )
    assert ok is True
    assert reason == "interval"


def test_should_flush_live_pending_immediately_when_interval_disabled() -> None:
    from echo.app import _should_flush_live_pending

    ok, reason = _should_flush_live_pending(
        has_pending=True,
        pending_duration=0.2,
        idle_tick=False,
        idle_for=0.0,
        pause_seconds=2.0,
        emit_interval=0.0,
    )
    assert ok is True
    assert reason == "immediate"


def test_should_flush_live_pending_on_pause_idle_tick_only() -> None:
    from echo.app import _should_flush_live_pending

    ok, reason = _should_flush_live_pending(
        has_pending=True,
        pending_duration=1.0,
        idle_tick=False,
        idle_for=3.0,
        pause_seconds=2.0,
        emit_interval=5.0,
    )
    assert ok is False
    assert reason == ""

    ok2, reason2 = _should_flush_live_pending(
        has_pending=True,
        pending_duration=1.0,
        idle_tick=True,
        idle_for=3.0,
        pause_seconds=2.0,
        emit_interval=5.0,
    )
    assert ok2 is True
    assert reason2 == "pause"


def test_live_config_capture_out_q_drop_from_env() -> None:
    from echo.config import LiveConfig

    c0 = LiveConfig.from_env({})
    assert c0.capture_out_q_drop_threshold == 400
    assert c0.capture_out_q_drop_batch == 30

    c1 = LiveConfig.from_env(
        {
            "ECHO_LIVE_CAPTURE_OUT_Q_DROP_THRESHOLD": "0",
            "ECHO_LIVE_CAPTURE_OUT_Q_DROP_BATCH": "12",
        },
    )
    assert c1.capture_out_q_drop_threshold == 0
    assert c1.capture_out_q_drop_batch == 12


def test_run_echo_uses_live_specific_insight_defaults(monkeypatch) -> None:
    import echo.app as app
    import echo.asr as asr

    captured: dict[str, object] = {}

    class DummyEchoApp:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def run(self) -> None:
            captured["ran"] = True

    monkeypatch.setattr(app, "_load_dotenv_files", lambda: None)
    monkeypatch.setattr(app, "echo_env_hub_local_only", lambda: False)
    monkeypatch.setattr(app, "echo_llm_online", lambda: True)
    monkeypatch.setattr(asr, "ensure_model_loaded", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(app, "EchoApp", DummyEchoApp)

    app.run_echo(live_audio=True, chunk_seconds=5.0)

    assert captured["ran"] is True
    assert captured["insight_min_chars"] == 220
    assert captured["insight_min_interval"] == 45.0
    assert captured["live_emit_interval"] == 0.0


def test_run_echo_passes_meeting_output_configuration(monkeypatch) -> None:
    from pathlib import Path

    import echo.app as app

    captured: dict[str, object] = {}

    class DummyEchoApp:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def run(self) -> None:
            captured["ran"] = True

    monkeypatch.setattr(app, "_load_dotenv_files", lambda: None)
    monkeypatch.setattr(app, "echo_env_hub_local_only", lambda: False)
    monkeypatch.setattr(app, "echo_llm_online", lambda: True)
    monkeypatch.setattr(app, "EchoApp", DummyEchoApp)

    out_dir = Path("/tmp/echo-test-output")
    app.run_echo(
        meeting_title="Weekly Sync",
        meeting_context="Review blockers and next milestones.",
        output_dir=out_dir,
    )

    assert captured["ran"] is True
    assert captured["meeting_title"] == "Weekly Sync"
    assert captured["meeting_context"] == "Review blockers and next milestones."
    assert captured["output_dir"] == out_dir


def test_meeting_title_to_folder_name_sanitizes_path_unsafe_chars() -> None:
    from echo.app import meeting_title_to_folder_name

    assert meeting_title_to_folder_name("Q2 / Planning: kickoff?") == "Q2 - Planning- kickoff-"
    assert meeting_title_to_folder_name("   ") == "untitled-meeting"


def test_pause_drain_quiescent_requires_empty_outq_and_no_pending_windows() -> None:
    from unittest.mock import MagicMock

    import numpy as np

    from echo.live_session import LiveTranscriptionSession

    app = MagicMock()
    app.live_asr_language = "ja"
    log = MagicMock()
    s = LiveTranscriptionSession(app, log)
    s._chunk_samples = 8000
    s._hop_samples = 4000
    s._cap = MagicMock()
    s._cap.out_q = MagicMock()
    s._cap.out_q.qsize = lambda: 0
    assert s._pause_drain_quiescent()

    s._cap.out_q.qsize = lambda: 2
    assert not s._pause_drain_quiescent()

    s._cap.out_q.qsize = lambda: 0
    s._audio_blocks.append(np.zeros(16000, dtype=np.float32))
    assert not s._pause_drain_quiescent()

    s._audio_blocks.clear()
    s._cap = None
    assert not s._pause_drain_quiescent()
