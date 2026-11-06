"""Engine-level tests: ``EchoSession`` without any UI, driven through its event stream."""

from __future__ import annotations

import asyncio
import json

import numpy as np
import pytest

from echo.engine import events as ev
from echo.engine.session import EchoSession


@pytest.fixture(autouse=True)
def _quiet_env(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.delenv("ECHO_DEBUG", raising=False)
    monkeypatch.chdir(tmp_path)


def _collect(session: EchoSession) -> list[ev.EngineEvent]:
    out: list[ev.EngineEvent] = []
    session.subscribe(out.append)
    return out


async def _wait_for(pred, timeout: float = 5.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not pred():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("timed out waiting for condition")
        await asyncio.sleep(0.01)


def test_replay_emits_final_lines_and_ends(tmp_path) -> None:
    s = EchoSession(replay_lines=["[00:01] alpha", "beta"], line_delay=0.0, llm_offline=True, output_dir=tmp_path)
    got = _collect(s)

    async def run() -> None:
        s.start()
        await _wait_for(lambda: any(isinstance(e, ev.SessionEnded) for e in got))

    asyncio.run(run())
    finals = [e for e in got if isinstance(e, ev.TranscriptFinal)]
    assert [f.text for f in finals] == ["alpha", "beta"]
    assert finals[0].line == "[00:01] alpha"
    assert s.status().state == "finished"


def test_insight_pass_applies_correction_and_emits_events(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    import echo.engine.session as session_mod
    from echo.llm import InsightRouteResult

    async def fake_route(snapshot, *, segments=None, **_kw):
        sid = segments[0][0]
        return InsightRouteResult(
            "comment", "Budget is the main topic.", "", "ok", "", corrected_segments=[(sid, "alpha corrected")]
        )

    monkeypatch.setattr(session_mod, "llm_route_insight", fake_route)
    monkeypatch.setattr(session_mod.EchoSession, "_dictionary_terms", lambda self: ())
    s = EchoSession(
        replay_lines=["[00:01] alpha"], line_delay=0.0, insight_min_chars=1, llm_offline=False, output_dir=tmp_path
    )
    got = _collect(s)

    async def run() -> None:
        s.start()
        await _wait_for(lambda: any(isinstance(e, ev.SessionEnded) for e in got))

    asyncio.run(run())
    kinds = [e.type for e in got]
    assert "insight.pending" in kinds and "insight.pass" in kinds
    passes = [e for e in got if isinstance(e, ev.InsightPass)]
    assert passes[0].body_md == "Budget is the main topic."
    corrected = [e for e in got if isinstance(e, ev.TranscriptCorrected)]
    assert corrected and corrected[-1].lines[0].text == "alpha corrected"
    assert corrected[-1].lines[0].flushed is True
    assert s.lines == ["[00:01] alpha corrected"]


def test_ask_emits_chat_reply(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    import echo.engine.session as session_mod

    async def fake_chat(transcript, question, ctx, **_kw):
        return f"answer to {question} with {len(transcript.splitlines())} lines"

    monkeypatch.setattr(session_mod, "llm_chat_answer", fake_chat)
    monkeypatch.setattr(session_mod.EchoSession, "_dictionary_terms", lambda self: ())
    s = EchoSession(llm_offline=False, output_dir=tmp_path)
    s._lines = ["a", "b"]
    got = _collect(s)
    ans = asyncio.run(s.ask("what?"))
    assert ans == "answer to what? with 2 lines"
    assert [e for e in got if isinstance(e, ev.ChatReply)][0].answer == ans


def test_reset_clears_state_and_pauses(tmp_path) -> None:
    s = EchoSession(llm_offline=True, output_dir=tmp_path)
    s._lines = ["x"]
    got = _collect(s)
    s.reset()
    assert s.lines == [] and s.paused
    assert any(isinstance(e, ev.SessionReset) for e in got)


def test_events_are_json_serializable() -> None:
    samples = [
        ev.TranscriptDraft(line_id=1, line="[00:00:01] a", text="a"),
        ev.TranscriptCorrected(lines=[ev.TranscriptLineState(1, "l", "t", True)]),
        ev.InsightPass(n=1, stamp="00:00:05", kind="comment", body_md="x"),
        ev.Meter(rms=0.1, peak=0.2, note="Listening"),
        ev.ChatReply(question="q", answer="a"),
    ]
    for e in samples:
        d = json.loads(json.dumps(e.to_dict()))
        assert d["type"] == e.type


def test_live_session_with_fake_streaming_backend(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Live capture → fake Apple streaming ASR → engine transcript events + insight queue."""
    import echo.asr_apple as asr_apple
    import echo.live_session as live_session_mod
    from echo.asr_apple import AppleStreamEvent
    from echo.asr_routing import AsrRouter, set_router

    script = [
        [AppleStreamEvent("partial", "hello", 0.0, 0.4)],
        [AppleStreamEvent("partial", "hello world", 0.0, 0.8)],
        [AppleStreamEvent("final", "Hello world.", 0.0, 1.0)],
    ]

    class FakeRecognizer:
        def __init__(self, language: str) -> None:
            self.language = language
            self.locale = "en_US"
            self.alive = True

        def start(self) -> None:
            return None

        def feed(self, samples) -> None:
            return None

        def poll(self):
            return script.pop(0) if script else []

        def finish(self, timeout: float = 0.0):
            return []

        def stop(self) -> None:
            self.alive = False

    class FakeCapture:
        def __init__(self) -> None:
            import queue

            self.out_q = queue.Queue()

        def read_block(self, _timeout: float):
            import time

            time.sleep(0.01)
            return np.zeros(1600, dtype=np.float32)

        def stop(self) -> None:
            return None

    async def fake_start_capture(self) -> bool:
        self._cap = FakeCapture()
        return True

    monkeypatch.setattr(asr_apple, "AppleStreamingRecognizer", FakeRecognizer)
    monkeypatch.setattr(live_session_mod.LiveTranscriptionSession, "_start_capture", fake_start_capture)
    monkeypatch.setattr(live_session_mod.LiveTranscriptionSession, "_emit_capture_banners", lambda self: None)
    router = AsrRouter(
        "apple", apple_status=lambda _l: "installed", api_usable=lambda: False, platform="darwin"
    )
    router.select("english")
    set_router(router)
    try:
        s = EchoSession(live_audio=True, live_emit_interval=0.0, llm_offline=True, language="english", output_dir=tmp_path)
        s.live_segment_pause_seconds = 0.2
        got = _collect(s)

        async def run() -> None:
            s.start()
            await _wait_for(lambda: any(isinstance(e, ev.TranscriptFinal) for e in got))
            await s.stop()

        asyncio.run(run())
    finally:
        set_router(None)

    drafts = [e.text for e in got if isinstance(e, ev.TranscriptDraft)]
    assert drafts[:2] == ["hello", "hello world"]
    finals = [e for e in got if isinstance(e, ev.TranscriptFinal)]
    assert finals[0].text == "Hello world."
    assert any(isinstance(e, ev.Meter) for e in got)
