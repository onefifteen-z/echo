from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from echo.engine.text import meeting_time_label, normalize_meeting_time
from echo.meetings import MeetingStore

JST = timezone(timedelta(hours=9))


def test_normalize_floors_to_minute_and_keeps_offset() -> None:
    assert normalize_meeting_time("2026-11-09T10:03:47.5+09:00") == "2026-11-09T10:03:00+09:00"
    assert normalize_meeting_time(datetime(2026, 11, 9, 10, 3, 59, tzinfo=JST)) == "2026-11-09T10:03:00+09:00"
    naive = normalize_meeting_time("2026-11-09T10:03")
    assert naive.startswith("2026-11-09T10:03:00") and datetime.fromisoformat(naive).tzinfo is not None
    with pytest.raises(ValueError):
        normalize_meeting_time("tomorrow")


def test_label() -> None:
    assert meeting_time_label("2026-11-09T10:03:00+09:00") == "2026-11-09 10:03"
    assert meeting_time_label("") == "" and meeting_time_label(None) == "" and meeting_time_label("bad") == ""


def _meeting(root: Path, mid: str, **meta: object) -> Path:
    d = root / mid
    d.mkdir()
    (d / "meeting.json").write_text(json.dumps({"version": 1, "id": mid, "title": mid, **meta}), encoding="utf-8")
    (d / "notes.txt").write_text("[00:01] hi\n", encoding="utf-8")
    return d


def test_store_default_update_and_sort(tmp_path: Path) -> None:
    _meeting(tmp_path, "a", started_at="2026-11-06T10:00:30+09:00")
    d = _meeting(tmp_path, "b", started_at="2026-11-05T10:00:00+09:00")
    store = MeetingStore(tmp_path)
    items = store.list()
    assert [m["id"] for m in items] == ["a", "b"]
    assert items[0]["meeting_time"] == "2026-11-06T10:00:00+09:00"

    out = store.update("b", meeting_time="2026-11-07T09:15:00+09:00")
    assert out["meeting_time"] == "2026-11-07T09:15:00+09:00"
    assert json.loads((d / "meeting.json").read_text())["meeting_time"] == "2026-11-07T09:15:00+09:00"
    assert [m["id"] for m in store.list()] == ["b", "a"]
    assert store.summary_inputs("b")["meeting_time"] == "2026-11-07 09:15"

    assert store.update("b", meeting_time="")["meeting_time"] == "2026-11-05T10:00:00+09:00"


def test_session_meeting_time_default_and_override(tmp_path: Path) -> None:
    from echo.engine.session import EchoSession

    s = EchoSession(llm_offline=True, output_dir=tmp_path / "m")
    s.started_at = datetime(2026, 11, 9, 10, 3, 42, tzinfo=JST)
    assert s.meeting_meta()["meeting_time"] == "2026-11-09T10:03:00+09:00"
    s.set_meeting_time("2026-11-09T09:58:10+09:00")
    assert s.meeting_meta()["meeting_time"] == "2026-11-09T09:58:00+09:00"
    s.set_meeting_time(None)
    assert s.meeting_time == "2026-11-09T10:03:00+09:00"
    with pytest.raises(ValueError):
        s.set_meeting_time("nope")

    s2 = EchoSession(llm_offline=True, output_dir=tmp_path / "n", meeting_time="2026-11-08T08:00:00+09:00")
    assert s2.meeting_time == "2026-11-08T08:00:00+09:00"


def test_server_meeting_time_arg() -> None:
    from echo.server import CommandError, EchoServer

    assert EchoServer._meeting_time_arg("") is None
    assert EchoServer._meeting_time_arg("2026-11-09T10:03:12+09:00") == "2026-11-09T10:03:00+09:00"
    with pytest.raises(CommandError):
        EchoServer._meeting_time_arg("soon")


def test_summary_prompt_includes_time(monkeypatch: pytest.MonkeyPatch) -> None:
    from echo import llm

    prompts: list[str] = []

    async def fake_text(prompt: str, *_a: object, **_k: object) -> tuple[str, str]:
        prompts.append(prompt)
        return "## Key takeaways\n", ""

    monkeypatch.setattr(llm, "_llm_is_configured", lambda: True)
    monkeypatch.setattr(llm, "_llm_user_text_with_retry", fake_text)
    asyncio.run(llm.llm_meeting_summary("[00:01] hi", meeting_title="Weekly", meeting_time="2026-11-09 10:03"))
    asyncio.run(llm.llm_meeting_summary("[00:01] hi", meeting_title="Weekly"))
    assert "- Time: 2026-11-09 10:03\n" in prompts[0]
    assert "- Time: (unknown)\n" in prompts[1]
