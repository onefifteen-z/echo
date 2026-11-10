"""Saved-meeting archive: list, read, edit metadata/notebook, and regenerate summaries on disk.

Each meeting is a direct child folder of the output root (the folder name is the meeting id):
``meeting.json``, ``notes.txt``, ``insights.md``, ``insights.json``, ``summary.md``,
``meeting_context.md`` and ``notebook.md``. Folders written before ``meeting.json`` existed
("legacy") are listed with the folder name as title and the folder mtime as date.

Listing never reads files that a cloud provider (OneDrive, iCloud Drive, …) has evicted to
"dataless" placeholders, since each read would block on a download; it uses values cached in
``meeting.json`` (``line_count``, ``summary_preview``, ``has_notebook``) instead.
"""

from __future__ import annotations

import json
import os
import stat
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

from echo.engine.text import (
    _UNTITLED_MEETING,
    meeting_time_label,
    normalize_meeting_time,
    patch_meeting_json,
    summary_preview,
)

_MEETING_FILES = ("notes.txt", "summary.md", "insights.md", "meeting_context.md")
SUMMARY_PLACEHOLDER = "_Generating meeting summary…_\n"


class MeetingNotFoundError(LookupError):
    pass


_SF_DATALESS = getattr(stat, "SF_DATALESS", 0x40000000)


def is_dataless(path: Path) -> bool:
    """True when the file's contents are not on disk (stat does not trigger a download)."""
    try:
        return bool(getattr(os.stat(path), "st_flags", 0) & _SF_DATALESS)
    except OSError:
        return False


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (FileNotFoundError, IsADirectoryError):
        return ""


def _read_local(path: Path) -> str | None:
    """Like ``_read`` but returns None instead of downloading an evicted cloud file."""
    return None if is_dataless(path) else _read(path)


def _context_from_markdown(md: str) -> str:
    """Inverse of ``EchoSession._meeting_context_markdown`` for legacy folders."""
    marker = "## Context"
    if marker not in md:
        return ""
    ctx = md.split(marker, 1)[1].strip()
    return "" if ctx == "_(none)_" else ctx


def context_markdown(title: str, context: str) -> str:
    return f"# Meeting\n\n- Title: {title}\n\n## Context\n\n{context or '_(none)_'}\n"


class MeetingStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def _resolve(self, meeting_id: str) -> Path:
        mid = (meeting_id or "").strip()
        if not mid or mid in (".", "..") or "/" in mid or "\\" in mid or "\x00" in mid or mid.startswith("."):
            raise MeetingNotFoundError(f"Invalid meeting id: {meeting_id!r}")
        path = self.root / mid
        if path.resolve().parent != self.root.resolve() or not path.is_dir():
            raise MeetingNotFoundError(f"Meeting not found: {meeting_id}")
        return path

    def path(self, meeting_id: str) -> Path:
        return self._resolve(meeting_id)

    def _meta(self, path: Path, *, local_only: bool = False) -> dict[str, Any]:
        read = (lambda p: _read_local(p) or "") if local_only else _read
        meta: dict[str, Any] = {}
        raw = read(path / "meeting.json")
        if raw:
            try:
                loaded = json.loads(raw)
                if isinstance(loaded, dict):
                    meta = loaded
            except json.JSONDecodeError:
                meta = {}
        legacy = not meta
        if legacy:
            mtime = datetime.fromtimestamp(path.stat().st_mtime).astimezone()
            meta = {
                "title": path.name,
                "context": _context_from_markdown(read(path / "meeting_context.md")),
                "started_at": mtime.isoformat(timespec="seconds"),
                "duration_s": None,
                "mode": "",
                "language": "",
                "asr_source": "",
            }
        meta["id"] = path.name
        meta["legacy"] = legacy
        meta.setdefault("title", path.name)
        meta.setdefault("context", "")
        if not meta.get("meeting_time"):
            try:
                meta["meeting_time"] = normalize_meeting_time(str(meta.get("started_at") or ""))
            except ValueError:
                meta["meeting_time"] = meta.get("started_at") or ""
        if "line_count" not in meta:
            notes = _read_local(path / "notes.txt") if local_only else _read(path / "notes.txt")
            meta["line_count"] = None if notes is None else sum(1 for ln in notes.splitlines() if ln.strip())
        return meta

    def _is_meeting_dir(self, path: Path) -> bool:
        return path.is_dir() and not path.name.startswith(".") and any((path / f).exists() for f in ("meeting.json", *_MEETING_FILES))

    def list(self) -> list[dict[str, Any]]:
        if not self.root.is_dir():
            return []
        items = []
        for child in self.root.iterdir():
            if not self._is_meeting_dir(child):
                continue
            meta = self._meta(child, local_only=True)
            if "summary_preview" not in meta:
                summary = _read_local(child / "summary.md")
                meta["summary_preview"] = summary_preview(summary) if summary else ""
            if "has_notebook" not in meta:
                meta["has_notebook"] = bool((_read_local(child / "notebook.md") or "").strip())
            meta.pop("version", None)
            items.append(meta)
        items.sort(key=lambda m: _sort_key(m), reverse=True)
        return items

    def get(self, meeting_id: str) -> dict[str, Any]:
        path = self._resolve(meeting_id)
        names = ("notes.txt", "summary.md", "insights.md", "notebook.md")
        # Parallel so evicted cloud files download concurrently.
        with ThreadPoolExecutor(max_workers=len(names) + 1) as pool:
            meta_f = pool.submit(self._meta, path)
            notes, summary, insights, notebook = pool.map(lambda n: _read(path / n), names)
            meta = meta_f.result()
        meta.pop("summary_preview", None)
        meta.pop("has_notebook", None)
        return {
            **meta,
            "path": str(path),
            "transcript": [ln for ln in notes.splitlines() if ln.strip()],
            "summary_md": summary,
            "insights_md": insights,
            "notebook_md": notebook,
        }

    def update(
        self,
        meeting_id: str,
        *,
        title: str | None = None,
        context: str | None = None,
        notebook: str | None = None,
        meeting_time: str | None = None,
    ) -> dict[str, Any]:
        """Edit metadata / notebook. ``meeting_time`` is ISO; an empty string restores the default (start time)."""
        path = self._resolve(meeting_id)
        meta = self._meta(path)
        if title is not None:
            meta["title"] = title.strip() or _UNTITLED_MEETING
        if context is not None:
            meta["context"] = context.strip()
        if meeting_time is not None:
            raw = meeting_time or str(meta.get("started_at") or "")
            meta["meeting_time"] = normalize_meeting_time(raw) if raw else ""
        if title is not None or context is not None or meeting_time is not None or meta.pop("legacy", False):
            stored = {k: v for k, v in meta.items() if k not in ("legacy",)}
            stored.setdefault("version", 1)
            _atomic_write(path / "meeting.json", json.dumps(stored, ensure_ascii=False, indent=1))
            _atomic_write(path / "meeting_context.md", context_markdown(meta["title"], meta["context"]))
        if notebook is not None:
            _atomic_write(path / "notebook.md", notebook)
            patch_meeting_json(path, has_notebook=bool(notebook.strip()))
        return self.get(meeting_id)

    def summary_inputs(self, meeting_id: str) -> dict[str, Any]:
        path = self._resolve(meeting_id)
        meta = self._meta(path)
        passes: list[tuple[str, str]] = []
        raw = _read(path / "insights.json")
        if raw:
            try:
                passes = [(str(p.get("kind", "comment")), str(p.get("body_md", ""))) for p in json.loads(raw)]
            except (json.JSONDecodeError, AttributeError, TypeError):
                passes = []
        return {
            "transcript_text": _read(path / "notes.txt"),
            "insight_passes": passes,
            "meeting_title": meta["title"],
            "meeting_context": meta.get("context", ""),
            "meeting_time": meeting_time_label(meta.get("meeting_time")),
            "notebook": _read(path / "notebook.md"),
            "assets_dir": path / "notes-assets",
        }

    async def regenerate_summary(self, meeting_id: str, *, dictionary_terms: "object" = ()) -> str:
        from echo.llm import llm_meeting_summary

        inputs = self.summary_inputs(meeting_id)
        summary_path = self._resolve(meeting_id) / "summary.md"
        previous = _read(summary_path)
        _atomic_write(summary_path, SUMMARY_PLACEHOLDER)
        try:
            md = await llm_meeting_summary(**inputs, dictionary_terms=dictionary_terms)
        except BaseException:
            if summary_path.parent.is_dir():
                _atomic_write(summary_path, previous)
            raise
        if not summary_path.parent.is_dir():
            raise MeetingNotFoundError(f"Meeting was deleted: {meeting_id}")
        _atomic_write(summary_path, md)
        patch_meeting_json(summary_path.parent, summary_preview=summary_preview(md))
        return md


def _sort_key(meta: dict[str, Any]) -> float:
    for key in ("meeting_time", "started_at"):
        try:
            return datetime.fromisoformat(str(meta.get(key) or "")).timestamp()
        except ValueError:
            continue
    return 0.0


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
