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


def test_should_trigger_insight_live_on_duration() -> None:
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
    assert ok is True
    assert reason == "duration"


def test_should_trigger_insight_live_on_pause() -> None:
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
    assert ok is True
    assert reason == "pause"


def test_should_trigger_insight_live_on_char_count() -> None:
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
    assert ok is True
    assert reason == "char_count"


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
