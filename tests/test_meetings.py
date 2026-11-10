from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pytest

from echo.meetings import MeetingNotFoundError, MeetingStore


def _new_meeting(root: Path, mid: str, *, started: str, notebook: str = "") -> Path:
    d = root / mid
    d.mkdir()
    (d / "meeting.json").write_text(
        json.dumps({"version": 1, "id": mid, "title": "Weekly", "context": "agenda", "started_at": started, "duration_s": 120}),
        encoding="utf-8",
    )
    (d / "notes.txt").write_text("[00:01] hello\n[00:02] world\n", encoding="utf-8")
    (d / "summary.md").write_text("## Key takeaways\n\n- **Ship it:** by Friday.\n", encoding="utf-8")
    (d / "insights.json").write_text(json.dumps([{"n": 1, "stamp": "00:01", "kind": "comment", "body_md": "Decide"}]), encoding="utf-8")
    (d / "notebook.md").write_text(notebook, encoding="utf-8")
    return d


def _legacy_meeting(root: Path, name: str) -> Path:
    d = root / name
    d.mkdir()
    (d / "notes.txt").write_text("[00:01] old line\n", encoding="utf-8")
    (d / "meeting_context.md").write_text("# Meeting\n\n- Title: Old\n\n## Context\n\nlegacy ctx\n", encoding="utf-8")
    os.utime(d, (1_700_000_000, 1_700_000_000))
    return d


def test_list_sorts_new_and_legacy(tmp_path: Path) -> None:
    _new_meeting(tmp_path, "2026-11-06 1000 Weekly", started="2026-11-06T10:00:00+09:00", notebook="my note")
    _new_meeting(tmp_path, "2026-11-05 1000 Weekly", started="2026-11-05T10:00:00+09:00")
    _legacy_meeting(tmp_path, "Old")
    (tmp_path / "empty-folder").mkdir()
    (tmp_path / ".hidden").mkdir()

    items = MeetingStore(tmp_path).list()
    assert [m["id"] for m in items] == ["2026-11-06 1000 Weekly", "2026-11-05 1000 Weekly", "Old"]
    assert items[0]["summary_preview"] == "Ship it: by Friday."
    assert items[0]["has_notebook"] is True and items[1]["has_notebook"] is False
    assert items[0]["line_count"] == 2
    legacy = items[2]
    assert legacy["legacy"] is True and legacy["title"] == "Old" and legacy["context"] == "legacy ctx"


def test_get_and_update(tmp_path: Path) -> None:
    mid = "2026-11-06 1000 Weekly"
    d = _new_meeting(tmp_path, mid, started="2026-11-06T10:00:00+09:00")
    store = MeetingStore(tmp_path)
    detail = store.get(mid)
    assert detail["transcript"] == ["[00:01] hello", "[00:02] world"] and detail["notebook_md"] == ""

    out = store.update(mid, title="Renamed", context="new ctx", notebook="- todo")
    assert out["title"] == "Renamed" and out["context"] == "new ctx" and out["notebook_md"] == "- todo"
    assert (d / "notebook.md").read_text(encoding="utf-8") == "- todo"
    assert "Title: Renamed" in (d / "meeting_context.md").read_text(encoding="utf-8")
    assert json.loads((d / "meeting.json").read_text(encoding="utf-8"))["duration_s"] == 120


def test_update_legacy_writes_meeting_json(tmp_path: Path) -> None:
    d = _legacy_meeting(tmp_path, "Old")
    MeetingStore(tmp_path).update("Old", notebook="n")
    meta = json.loads((d / "meeting.json").read_text(encoding="utf-8"))
    assert meta["title"] == "Old" and meta["context"] == "legacy ctx" and "legacy" not in meta
    assert MeetingStore(tmp_path).get("Old")["legacy"] is False


@pytest.mark.parametrize("bad", ["", "..", "../x", "a/b", ".hidden", "missing"])
def test_resolve_rejects_bad_ids(tmp_path: Path, bad: str) -> None:
    (tmp_path / ".hidden").mkdir()
    with pytest.raises(MeetingNotFoundError):
        MeetingStore(tmp_path).get(bad)


def test_regenerate_summary_passes_notebook_and_insights(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import echo.llm as llm

    mid = "2026-11-06 1000 Weekly"
    d = _new_meeting(tmp_path, mid, started="2026-11-06T10:00:00+09:00", notebook="Bob owns rollout")
    seen: dict = {}

    async def fake_summary(
        transcript_text, *, insight_passes=None, meeting_title="", meeting_context="", dictionary_terms=(), notebook="", assets_dir=None, meeting_time=""
    ):
        seen.update(
            transcript=transcript_text,
            insight_passes=insight_passes,
            meeting_title=meeting_title,
            notebook=notebook,
            assets_dir=assets_dir,
        )
        return "## Key takeaways\n\n- new\n"

    monkeypatch.setattr(llm, "llm_meeting_summary", fake_summary)
    md = asyncio.run(MeetingStore(tmp_path).regenerate_summary(mid))
    assert md.endswith("- new\n") and (d / "summary.md").read_text(encoding="utf-8") == md
    assert seen["notebook"] == "Bob owns rollout"
    assert seen["assets_dir"] == d / "notes-assets"
    assert seen["insight_passes"] == [("comment", "Decide")]
    assert seen["meeting_title"] == "Weekly" and "hello" in seen["transcript"]


def test_regenerate_failure_restores_previous_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import echo.llm as llm

    mid = "m"
    d = _new_meeting(tmp_path, mid, started="2026-11-06T10:00:00+09:00")
    before = (d / "summary.md").read_text(encoding="utf-8")

    async def boom(*_a, **_k):
        raise RuntimeError("llm down")

    monkeypatch.setattr(llm, "llm_meeting_summary", boom)
    with pytest.raises(RuntimeError):
        asyncio.run(MeetingStore(tmp_path).regenerate_summary(mid))
    assert (d / "summary.md").read_text(encoding="utf-8") == before


def test_summary_inputs_bind_to_real_llm_signature(tmp_path: Path) -> None:
    import inspect

    from echo.llm import llm_meeting_summary

    _new_meeting(tmp_path, "m", started="2026-11-06T10:00:00+09:00", notebook="x")
    inputs = MeetingStore(tmp_path).summary_inputs("m")
    inspect.signature(llm_meeting_summary).bind(**inputs, dictionary_terms=())


def test_regenerate_reports_deleted_meeting(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import shutil

    import echo.llm as llm

    d = _new_meeting(tmp_path, "m", started="2026-11-06T10:00:00+09:00")

    async def slow_summary(*_a, **_k):
        shutil.rmtree(d)
        return "## Key takeaways\n"

    monkeypatch.setattr(llm, "llm_meeting_summary", slow_summary)
    with pytest.raises(MeetingNotFoundError, match="deleted"):
        asyncio.run(MeetingStore(tmp_path).regenerate_summary("m"))
    assert not d.exists()


def test_list_skips_evicted_cloud_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import echo.meetings as meetings

    new = _new_meeting(tmp_path, "2026-11-06 1000 Weekly", started="2026-11-06T10:00:00+09:00", notebook="n")
    legacy = _legacy_meeting(tmp_path, "Old")
    evicted = {new / "summary.md", new / "notebook.md", legacy / "notes.txt", legacy / "meeting_context.md"}
    monkeypatch.setattr(meetings, "is_dataless", lambda p: Path(p) in evicted)
    real_read = meetings._read

    def guarded_read(p: Path) -> str:
        assert Path(p) not in evicted, f"list() downloaded {p}"
        return real_read(p)

    monkeypatch.setattr(meetings, "_read", guarded_read)
    items = {m["id"]: m for m in MeetingStore(tmp_path).list()}
    assert items["2026-11-06 1000 Weekly"]["summary_preview"] == ""
    assert items["Old"]["line_count"] is None and items["Old"]["title"] == "Old"

    # Cached values in meeting.json are used without touching the evicted files.
    from echo.engine.text import patch_meeting_json

    patch_meeting_json(new, summary_preview="Cached preview", has_notebook=True)
    items = {m["id"]: m for m in MeetingStore(tmp_path).list()}
    assert items["2026-11-06 1000 Weekly"]["summary_preview"] == "Cached preview"
    assert items["2026-11-06 1000 Weekly"]["has_notebook"] is True
    assert "version" not in items["2026-11-06 1000 Weekly"]

    # Opening a meeting still reads everything.
    monkeypatch.setattr(meetings, "_read", real_read)
    detail = MeetingStore(tmp_path).get("2026-11-06 1000 Weekly")
    assert "Ship it" in detail["summary_md"] and detail["notebook_md"] == "n"
    assert "summary_preview" not in detail


def test_cache_updated_by_notebook_edit_and_regenerate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mid = "2026-11-06 1000 Weekly"
    d = _new_meeting(tmp_path, mid, started="2026-11-06T10:00:00+09:00")
    store = MeetingStore(tmp_path)
    store.update(mid, notebook="- my note")
    assert json.loads((d / "meeting.json").read_text())["has_notebook"] is True

    async def fake_summary(**_kw: object) -> str:
        return "## Key takeaways\n\n- New takeaway\n"

    monkeypatch.setattr("echo.llm.llm_meeting_summary", fake_summary)
    asyncio.run(store.regenerate_summary(mid))
    assert json.loads((d / "meeting.json").read_text())["summary_preview"] == "New takeaway"


def test_is_dataless_on_regular_file(tmp_path: Path) -> None:
    from echo.meetings import is_dataless

    f = tmp_path / "x.txt"
    f.write_text("x")
    assert is_dataless(f) is False and is_dataless(tmp_path / "missing") is False
