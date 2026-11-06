"""Web text search: Perplexity Sonar (when configured) with `ddgs` metasearch fallback."""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

_MIN_SNIPPET_CHARS = 160


@dataclass
class SearchHit:
    title: str
    body: str
    href: str


@dataclass
class SearchOutcome:
    hits: list[SearchHit]
    provider: str
    note: str = ""


class SearchError(RuntimeError):
    pass


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


# --------------------------------------------------------------------------- Perplexity


def _env(*names: str) -> str:
    for name in names:
        value = (os.environ.get(name) or "").strip()
        if value:
            return value
    return ""


def validate_perplexity_base_url(base_url: str) -> str:
    """Return a normalized base URL or raise ValueError."""
    normalized = base_url.strip().rstrip("/")
    parsed = urlparse(normalized)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError("Perplexity base URL must be an https:// URL")
    if "/v1/v1" in (parsed.path or ""):
        raise ValueError("Perplexity base URL has a duplicate /v1 segment")
    return normalized


def perplexity_config() -> tuple[str, str, str] | None:
    """`(base_url, api_key, model)` when Perplexity search is configured, else None.

    Raises ValueError when a base URL is set but invalid.
    """
    base_url = _env("ECHO_PERPLEXITY_BASE_URL", "AI_GATEWAY_PERPLEXITY_BASE_URL")
    api_key = _env("ECHO_PERPLEXITY_API_KEY", "AI_GATEWAY_API_KEY", "API_KEY")
    if not base_url or not api_key:
        return None
    model = _env("ECHO_PERPLEXITY_MODEL", "AI_GATEWAY_PERPLEXITY_MODEL") or "sonar"
    return validate_perplexity_base_url(base_url), api_key, model


def _title_from_url(url: str) -> str:
    parsed = urlparse(url)
    host = parsed.netloc.removeprefix("www.")
    segment = parsed.path.strip("/").split("/")[-1].replace("-", " ").replace("_", " ")
    return f"{segment} — {host}" if segment else (host or url)


def _valid_url(raw: str) -> str:
    url = raw.strip()
    parsed = urlparse(url)
    return url if parsed.scheme in {"http", "https"} and parsed.netloc else ""


def _dedupe_key(url: str) -> str:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower().removeprefix("www.")
    return f"{host}{parsed.path.rstrip('/') or '/'}"


def _as_dict(item: Any) -> dict[str, Any]:
    if hasattr(item, "model_dump"):
        return item.model_dump()
    return item if isinstance(item, dict) else {}


def _cited_span(content: str, citation: dict[str, Any]) -> str:
    """The sentence that cites the page is the closest grounded snippet (no page extract is returned)."""
    start, end = citation.get("start_index"), citation.get("end_index")
    if isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= len(content):
        span = content[start:end].strip()
        if len(span) >= _MIN_SNIPPET_CHARS:
            return span
        return content[start : start + _MIN_SNIPPET_CHARS].strip() or span
    return content[:_MIN_SNIPPET_CHARS].strip()


def _extra(obj: Any, name: str) -> Any:
    value = getattr(obj, name, None)
    if value is None:
        value = (getattr(obj, "model_extra", None) or {}).get(name)
    return value


def collect_perplexity_hits(response: Any, max_results: int) -> list[SearchHit]:
    """Deduped hits from a Sonar response: annotations, `search_results`, then bare `citations`."""
    message = response.choices[0].message
    content = str(getattr(message, "content", "") or "")
    lead = content[:_MIN_SNIPPET_CHARS].strip()
    hits: list[SearchHit] = []
    seen: set[str] = set()

    def add(url: str, title: str, snippet: str) -> None:
        cited = _valid_url(str(url or ""))
        if not cited or _dedupe_key(cited) in seen:
            return
        seen.add(_dedupe_key(cited))
        title = str(title or "").strip()
        if not title or _valid_url(title):
            title = _title_from_url(cited)
        hits.append(SearchHit(title=title, body=snippet.strip() or lead, href=cited))

    for ann in _extra(message, "annotations") or []:
        citation = _as_dict(_as_dict(ann).get("url_citation"))
        if citation:
            add(citation.get("url", ""), citation.get("title", ""), _cited_span(content, citation))
    for row in _extra(response, "search_results") or []:
        row = _as_dict(row)
        add(row.get("url", ""), row.get("title", ""), str(row.get("snippet") or ""))
    for raw in _extra(response, "citations") or []:
        if isinstance(raw, str):
            add(raw, "", "")
        else:
            row = _as_dict(raw)
            add(row.get("url", ""), row.get("title", ""), str(row.get("snippet") or ""))
    return hits[:max_results]


def _classify_error(exc: Exception) -> str:
    status = getattr(exc, "status_code", None)
    if status in (401, 403):
        return "authentication failed"
    if status == 402:
        return "quota exceeded"
    if status == 404:
        return "endpoint or model not found"
    if status == 429:
        return "rate limited"
    if isinstance(status, int) and status >= 500:
        return f"server error {status}"
    if "timeout" in type(exc).__name__.lower():
        return "timed out"
    return type(exc).__name__


def _perplexity_search_sync(query: str, max_results: int = 5, *, client: Any = None) -> list[SearchHit]:
    cfg = perplexity_config()
    if cfg is None:
        raise SearchError("Perplexity is not configured")
    base_url, api_key, model = cfg
    if client is None:
        from openai import OpenAI

        timeout = float(os.environ.get("ECHO_PERPLEXITY_TIMEOUT") or 30)
        client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout, max_retries=1)
    try:
        response = client.chat.completions.create(model=model, messages=[{"role": "user", "content": query}])
    except Exception as exc:  # noqa: BLE001
        raise SearchError(_classify_error(exc)) from exc
    try:
        hits = collect_perplexity_hits(response, max_results)
    except (AttributeError, IndexError) as exc:
        raise SearchError("malformed response") from exc
    if not hits:
        raise SearchError("no citations returned")
    return hits


def active_search_provider() -> str:
    try:
        return "Perplexity" if perplexity_config() else "ddgs"
    except ValueError:
        return "ddgs"


def _web_search_sync(query: str, max_results: int = 5) -> SearchOutcome:
    note = ""
    try:
        configured = perplexity_config() is not None
    except ValueError as exc:
        configured = False
        note = f"Perplexity disabled: {exc}"
    if configured:
        try:
            return SearchOutcome(_perplexity_search_sync(query, max_results), "Perplexity")
        except SearchError as exc:
            note = f"Perplexity search failed ({exc}); fell back to ddgs."
    return SearchOutcome(_ddg_text_sync(query, max_results), "ddgs", note)


async def web_search(query: str, max_results: int = 5) -> SearchOutcome:
    return await asyncio.to_thread(_web_search_sync, query, max_results)


async def duckduckgo_search(query: str, max_results: int = 5) -> list[SearchHit]:
    return await asyncio.to_thread(_ddg_text_sync, query, max_results)
