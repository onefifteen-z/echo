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


def test_should_trigger_insight_live_skips_without_finalized_segment() -> None:
    from echo.app import _should_trigger_insight_live

    ok, reason = _should_trigger_insight_live(
        new_chars=80,
        new_seconds=12.0,
        new_finalized_segments=0,
        max_finalized_segment_chars=0,
        idle_tick=False,
        idle_for=0.1,
        min_chars=120,
        min_seconds=10.0,
        pause_min_chars=80,
        finalized_segment_min_chars=100,
        skip_min_chars=50,
        pause_seconds=2.0,
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
        new_seconds=2.0,
        new_finalized_segments=0,
        max_finalized_segment_chars=0,
        idle_tick=True,
        idle_for=2.5,
        min_chars=220,
        min_seconds=10.0,
        pause_min_chars=80,
        finalized_segment_min_chars=100,
        skip_min_chars=50,
        pause_seconds=2.0,
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
        new_seconds=2.0,
        new_finalized_segments=0,
        max_finalized_segment_chars=0,
        idle_tick=False,
        idle_for=0.2,
        min_chars=220,
        min_seconds=10.0,
        pause_min_chars=80,
        finalized_segment_min_chars=100,
        skip_min_chars=50,
        pause_seconds=2.0,
    )
    assert ok is False
    assert reason == ""


def test_should_trigger_insight_live_stays_off_when_short_and_active() -> None:
    from echo.app import _should_trigger_insight_live

    ok, reason = _should_trigger_insight_live(
        new_chars=40,
        new_seconds=12.0,
        new_finalized_segments=1,
        max_finalized_segment_chars=200,
        idle_tick=False,
        idle_for=0.2,
        min_chars=120,
        min_seconds=10.0,
        pause_min_chars=80,
        finalized_segment_min_chars=100,
        skip_min_chars=50,
        pause_seconds=2.0,
    )
    assert ok is False
    assert reason == ""


def test_should_trigger_insight_live_on_new_finalized_segment() -> None:
    from echo.app import _should_trigger_insight_live

    ok, reason = _should_trigger_insight_live(
        new_chars=120,
        new_seconds=4.0,
        new_finalized_segments=1,
        max_finalized_segment_chars=120,
        idle_tick=False,
        idle_for=0.1,
        min_chars=220,
        min_seconds=10.0,
        pause_min_chars=80,
        finalized_segment_min_chars=100,
        skip_min_chars=50,
        pause_seconds=2.0,
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
