from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from echo import search
from echo.search import SearchError, SearchHit

_ENV = (
    "ECHO_PERPLEXITY_BASE_URL",
    "AI_GATEWAY_PERPLEXITY_BASE_URL",
    "ECHO_PERPLEXITY_API_KEY",
    "AI_GATEWAY_API_KEY",
    "API_KEY",
    "ECHO_PERPLEXITY_MODEL",
    "AI_GATEWAY_PERPLEXITY_MODEL",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in _ENV:
        monkeypatch.delenv(name, raising=False)


def _response(content="", annotations=None, search_results=None, citations=None):
    message = SimpleNamespace(content=content, annotations=annotations)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        search_results=search_results,
        citations=citations,
    )


class _FakeClient:
    def __init__(self, response=None, exc=None):
        self.calls: list[dict] = []
        self._response, self._exc = response, exc
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self._exc is not None:
            raise self._exc
        return self._response


def test_config_absent_without_base_url(monkeypatch) -> None:
    monkeypatch.setenv("API_KEY", "k")
    assert search.perplexity_config() is None
    assert search.active_search_provider() == "ddgs"


def test_config_precedence_and_api_key_fallback(monkeypatch) -> None:
    monkeypatch.setenv("AI_GATEWAY_PERPLEXITY_BASE_URL", "https://gw.example/v1/")
    monkeypatch.setenv("API_KEY", "shared")
    assert search.perplexity_config() == ("https://gw.example/v1", "shared", "sonar")

    monkeypatch.setenv("ECHO_PERPLEXITY_BASE_URL", "https://echo.example/v1")
    monkeypatch.setenv("AI_GATEWAY_API_KEY", "gateway")
    monkeypatch.setenv("AI_GATEWAY_PERPLEXITY_MODEL", "sonar-pro")
    assert search.perplexity_config() == ("https://echo.example/v1", "gateway", "sonar-pro")

    monkeypatch.setenv("ECHO_PERPLEXITY_API_KEY", "echo")
    monkeypatch.setenv("ECHO_PERPLEXITY_MODEL", "sonar-reasoning")
    assert search.perplexity_config() == ("https://echo.example/v1", "echo", "sonar-reasoning")
    assert search.active_search_provider() == "Perplexity"


@pytest.mark.parametrize("url", ["http://gw.example/v1", "https://gw.example/v1/v1"])
def test_invalid_base_url_rejected(monkeypatch, url) -> None:
    monkeypatch.setenv("ECHO_PERPLEXITY_BASE_URL", url)
    monkeypatch.setenv("API_KEY", "k")
    with pytest.raises(ValueError):
        search.perplexity_config()


def test_collect_hits_from_all_citation_shapes_deduped() -> None:
    content = "Revenue grew 8% in Q4 according to the filing. " * 5
    resp = _response(
        content=content,
        annotations=[
            {"type": "url_citation", "url_citation": {"url": "https://a.example/report", "title": "A report", "start_index": 0, "end_index": 40}},
            {"type": "url_citation", "url_citation": {"url": "ftp://bad.example/x"}},
        ],
        search_results=[
            {"url": "https://www.a.example/report/", "title": "dup", "snippet": "dup"},
            {"url": "https://b.example/news", "title": "B news", "snippet": "B snippet"},
        ],
        citations=["https://c.example/some-page", "https://b.example/news", {"url": "https://e.example/a-b", "title": "https://e.example/a-b"}],
    )
    hits = search.collect_perplexity_hits(resp, max_results=5)
    assert [h.href for h in hits] == [
        "https://a.example/report",
        "https://b.example/news",
        "https://c.example/some-page",
        "https://e.example/a-b",
    ]
    assert hits[3].title == "a b — e.example"
    assert hits[0].title == "A report"
    assert hits[0].body.startswith("Revenue grew 8%") and len(hits[0].body) >= 100
    assert hits[1].body == "B snippet"
    assert hits[2].title == "some page — c.example"
    assert search.collect_perplexity_hits(resp, max_results=2) == hits[:2]


def test_perplexity_search_uses_model_and_raises_without_citations(monkeypatch) -> None:
    monkeypatch.setenv("ECHO_PERPLEXITY_BASE_URL", "https://gw.example/v1")
    monkeypatch.setenv("API_KEY", "k")
    ok = _FakeClient(_response(content="x", citations=["https://a.example/"]))
    hits = search._perplexity_search_sync("q", client=ok)
    assert hits[0].href == "https://a.example/"
    assert ok.calls[0]["model"] == "sonar"
    assert ok.calls[0]["messages"] == [{"role": "user", "content": "q"}]

    with pytest.raises(SearchError, match="no citations"):
        search._perplexity_search_sync("q", client=_FakeClient(_response(content="uncited")))

    err = Exception("denied")
    err.status_code = 403  # type: ignore[attr-defined]
    with pytest.raises(SearchError, match="authentication"):
        search._perplexity_search_sync("q", client=_FakeClient(exc=err))


def _patch_ddgs(monkeypatch) -> list[str]:
    calls: list[str] = []

    def fake(query, max_results=5):
        calls.append(query)
        return [SearchHit(title="ddg", body="b", href="https://d.example/")]

    monkeypatch.setattr(search, "_ddg_text_sync", fake)
    return calls


def test_web_search_unconfigured_uses_ddgs_only(monkeypatch) -> None:
    calls = _patch_ddgs(monkeypatch)
    monkeypatch.setattr(search, "_perplexity_search_sync", lambda *a, **k: pytest.fail("perplexity called"))
    out = asyncio.run(search.web_search("q"))
    assert out.provider == "ddgs" and out.note == "" and calls == ["q"]


def test_web_search_prefers_perplexity(monkeypatch) -> None:
    monkeypatch.setenv("ECHO_PERPLEXITY_BASE_URL", "https://gw.example/v1")
    monkeypatch.setenv("API_KEY", "k")
    calls = _patch_ddgs(monkeypatch)
    monkeypatch.setattr(search, "_perplexity_search_sync", lambda q, n: [SearchHit("p", "b", "https://p.example/")])
    out = asyncio.run(search.web_search("q"))
    assert out.provider == "Perplexity" and out.hits[0].title == "p" and calls == []


def test_web_search_falls_back_with_note(monkeypatch) -> None:
    monkeypatch.setenv("ECHO_PERPLEXITY_BASE_URL", "https://gw.example/v1")
    monkeypatch.setenv("API_KEY", "k")
    calls = _patch_ddgs(monkeypatch)

    def boom(q, n):
        raise SearchError("rate limited")

    monkeypatch.setattr(search, "_perplexity_search_sync", boom)
    out = asyncio.run(search.web_search("q"))
    assert out.provider == "ddgs" and calls == ["q"]
    assert "rate limited" in out.note and "fell back to ddgs" in out.note


def test_web_search_invalid_config_notes_and_uses_ddgs(monkeypatch) -> None:
    monkeypatch.setenv("ECHO_PERPLEXITY_BASE_URL", "http://insecure.example")
    monkeypatch.setenv("API_KEY", "k")
    _patch_ddgs(monkeypatch)
    out = asyncio.run(search.web_search("q"))
    assert out.provider == "ddgs" and "Perplexity disabled" in out.note
