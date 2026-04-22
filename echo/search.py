"""Web text search via the `ddgs` package (metasearch; DuckDuckGo/Bing/… backends)."""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass


@dataclass
class SearchHit:
    title: str
    body: str
    href: str


def _normalize_row(r: dict) -> SearchHit:
    return SearchHit(
        title=str(r.get("title") or ""),
        body=str(r.get("body") or ""),
        href=str(r.get("href") or r.get("url") or ""),
    )


def _ddg_text_sync(query: str, max_results: int = 5) -> list[SearchHit]:
    try:
        from ddgs import DDGS
    except ImportError:
        return []

    def run(backend: str) -> list[SearchHit]:
        try:
            rows = DDGS().text(query, max_results=max_results, backend=backend)
        except Exception:  # noqa: BLE001
            return []
        if not rows:
            return []
        hits = [_normalize_row(r) for r in rows[:max_results]]
        return [h for h in hits if h.title or h.body or h.href]

    explicit = (os.environ.get("ECHO_DDGS_BACKEND") or "").strip()
    if explicit:
        return run(explicit)

    for backend in ("auto", "bing", "duckduckgo", "yahoo"):
        hits = run(backend)
        if hits:
            return hits
    return []


async def duckduckgo_search(query: str, max_results: int = 5) -> list[SearchHit]:
    return await asyncio.to_thread(_ddg_text_sync, query, max_results)
