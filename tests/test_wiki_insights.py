"""Knowledge-vault (Lumina) routing, insight passes, chat excerpts, and server commands."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

import echo.engine.session as session_mod
from echo import llm
from echo.engine import events as ev
from echo.engine.session import EchoSession
from echo.llm import InsightRouteResult
from echo.wiki import WikiError, WikiHit, WikiPage


@pytest.fixture(autouse=True)
def _quiet_env(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.delenv("ECHO_DEBUG", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(session_mod.EchoSession, "_dictionary_terms", lambda self: ())


class FakeWiki:
    def __init__(self, *, fail: bool = False, hits: list[WikiHit] | None = None) -> None:
        self.fail = fail
        self.hits = hits if hits is not None else [WikiHit("Pricing model", "wiki/concepts/pricing.md", "Basic $10")]
        self.calls: list[tuple[str, str]] = []

    async def titles(self, vault: str, limit: int = 80) -> list[str]:
        return ["Pricing model", "Acme Corp"]

    async def search(self, vault: str, query: str, limit: int = 5) -> list[WikiHit]:
        return self.hits[:limit]

    async def lookup(self, vault: str, query: str, *, limit: int = 5, read: int = 3):
        self.calls.append((vault, query))
        if self.fail:
            raise WikiError("Lumina did not answer search_wiki within 8s.")
        pages = [WikiPage(h.title, h.filename, "Tiered pricing: Basic $10, Pro $25.") for h in self.hits if h.is_page][:read]
        return self.hits, pages


def _route_payload(action: str, **extra: str) -> str:
    return json.dumps({"action": action, "corrected_transcript": "", "corrected_segments": [], "comment": "", "search_query": "", "reason": "r", **extra})


def test_router_prompt_has_vault_block_only_with_vault(monkeypatch: pytest.MonkeyPatch) -> None:
    plain = llm._build_insight_route_prompt("hello")
    assert "wiki_lookup" not in plain and "wiki_query" not in plain
    with_vault = llm._build_insight_route_prompt(
        "hello",
        vault="agent-era",
        vault_titles=["Pricing model", "Acme Corp"],
        vault_hits=[WikiHit("Workflow vs. Agent", "wiki/concepts/workflow-vs-agent.md", "Who decides the control flow")],
    )
    assert '"action": one of "comment", "web_search", or "wiki_lookup"' in with_vault
    assert '"wiki_query"' in with_vault
    assert "Knowledge vault attached to this meeting: agent-era" in with_vault
    assert "- Pricing model" in with_vault and "- Acme Corp" in with_vault
    assert "- Workflow vs. Agent (`wiki/concepts/workflow-vs-agent.md`) — Who decides the control flow" in with_vault
    assert "no page of the knowledge vault below is relevant" in with_vault


def test_route_parses_wiki_lookup_and_coerces_without_vault(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake(_content: str, _max_tokens: int) -> str:
        return _route_payload("wiki_lookup", wiki_query="pricing tiers")

    monkeypatch.setattr(llm, "_llm_user_text", fake)
    out = asyncio.run(llm.llm_route_insight("we discussed pricing", vault="agent-era"))
    assert out.action == "wiki_lookup" and out.wiki_query == "pricing tiers"
    out = asyncio.run(llm.llm_route_insight("we discussed pricing"))
    assert out.action == "comment" and out.wiki_query == "" and "no_vault" in out.reason

    async def fake_empty(_content: str, _max_tokens: int) -> str:
        return _route_payload("wiki_lookup")

    monkeypatch.setattr(llm, "_llm_user_text", fake_empty)
    out = asyncio.run(llm.llm_route_insight("pricing tiers for Acme", vault="agent-era"))
    assert out.action == "wiki_lookup" and out.wiki_query


def test_chat_prompt_includes_wiki_excerpts_only_when_given() -> None:
    assert "knowledge-vault" not in llm._chat_prompt("t", "q")
    p = llm._chat_prompt("t", "q", wiki_excerpts='Knowledge vault "v":\n<<<\n### P (`wiki/p.md`)\nbody\n>>>')
    assert "wiki/p.md" in p and "cite its filename" in p


def test_summary_labels_wiki_passes() -> None:
    assert "Pass 1 — Wiki lookup" in llm._format_insight_history([("wiki", "From the vault")])


def _wiki_session(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, wiki: FakeWiki, *, vault: str | None = "agent-era") -> EchoSession:
    async def fake_route(snapshot, *, segments=None, vault=None, vault_titles=None, **_kw):
        assert vault_titles is not None
        if vault:
            return InsightRouteResult("wiki_lookup", "", "", "vault covers pricing", "", [], "pricing")
        return InsightRouteResult("comment", "Plain comment.", "", "ok", "", [])

    async def fake_synth(ctx, query, hits, pages, *, vault, **_kw):
        return f"Pricing per `{pages[0].filename}` ({vault})"

    monkeypatch.setattr(session_mod, "llm_route_insight", fake_route)
    monkeypatch.setattr(session_mod, "llm_synthesize_wiki_findings", fake_synth)
    return EchoSession(
        replay_lines=["[00:01] pricing for Acme"],
        line_delay=0.0,
        insight_min_chars=1,
        llm_offline=False,
        output_dir=tmp_path,
        vault=vault,
        wiki=wiki,  # type: ignore[arg-type]
    )


def _run_to_end(s: EchoSession) -> list[ev.EngineEvent]:
    got: list[ev.EngineEvent] = []
    s.subscribe(got.append)

    async def run() -> None:
        s.start()
        for _ in range(500):
            if any(isinstance(e, ev.SessionEnded) for e in got):
                break
            await asyncio.sleep(0.01)

    asyncio.run(run())
    return got


def test_wiki_lookup_pass_emits_wiki_insight(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    wiki = FakeWiki()
    s = _wiki_session(monkeypatch, tmp_path, wiki)
    got = _run_to_end(s)
    (p,) = [e for e in got if isinstance(e, ev.InsightPass)]
    assert p.kind == "wiki" and p.vault == "agent-era"
    assert "wiki/concepts/pricing.md" in p.body_md
    assert p.web_md and "`wiki/concepts/pricing.md`" in p.web_md
    assert wiki.calls == [("agent-era", "pricing")]
    assert s.status().vault == "agent-era"
    s.save_outputs(announce=False)
    assert json.loads((tmp_path / "meeting.json").read_text())["vault"] == "agent-era"
    assert "### From agent-era" in (tmp_path / "insights.md").read_text()


def test_wiki_failure_falls_back_without_crashing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    s = _wiki_session(monkeypatch, tmp_path, FakeWiki(fail=True))
    (p,) = [e for e in _run_to_end(s) if isinstance(e, ev.InsightPass)]
    assert p.kind == "wiki" and "did not answer" in p.body_md


def test_no_vault_keeps_comment_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    wiki = FakeWiki()
    s = _wiki_session(monkeypatch, tmp_path, wiki, vault=None)
    (p,) = [e for e in _run_to_end(s) if isinstance(e, ev.InsightPass)]
    assert p.kind == "comment" and wiki.calls == []


def test_chat_uses_wiki_excerpts_only_with_vault(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    seen: list[str] = []

    async def fake_chat(transcript, question, ctx, *, wiki_excerpts="", **_kw):
        seen.append(wiki_excerpts)
        return "ok"

    monkeypatch.setattr(session_mod, "llm_chat_answer", fake_chat)
    wiki = FakeWiki()
    s = EchoSession(llm_offline=False, output_dir=tmp_path, wiki=wiki)  # type: ignore[arg-type]
    asyncio.run(s.ask("what is the pricing?"))
    s.set_vault("agent-era")
    asyncio.run(s.ask("what is the pricing?"))
    s.set_vault(None)
    asyncio.run(s.ask("again?"))
    assert seen[0] == "" and seen[2] == ""
    assert 'Knowledge vault "agent-era"' in seen[1] and "wiki/concepts/pricing.md" in seen[1]
    assert wiki.calls == [("agent-era", "what is the pricing?")]


def test_server_vault_commands(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from echo.server import EchoServer

    user_data = tmp_path / "lumina"
    user_data.mkdir()
    (user_data / "llm-wiki-config.json").write_text(
        json.dumps({"vaultPath": str(tmp_path / "v1"), "vaultHistory": [{"path": str(tmp_path / "v1"), "name": "Alpha"}]})
    )
    monkeypatch.setenv("ECHO_LUMINA_USER_DATA", str(user_data))
    monkeypatch.setenv("ECHO_LUMINA_MCP_CMD", "false")
    monkeypatch.setattr("echo.connectivity.echo_llm_online", lambda: False)
    s = EchoServer("t", output_root=tmp_path / "out")

    async def call(q: asyncio.Queue[str], n: int, cmd: str, args: dict | None = None) -> dict:
        await s._dispatch(q, json.dumps({"id": n, "cmd": cmd, "args": args or {}}))
        return json.loads(q.get_nowait())

    async def go() -> None:
        q: asyncio.Queue[str] = asyncio.Queue()
        r = await call(q, 1, "wiki.vaults")
        assert r["ok"] and r["result"]["available"] and r["result"]["vaults"][0]["name"] == "Alpha"
        r = await call(q, 2, "session.start", {"source": "demo", "line_delay": 0, "vault": "nope"})
        assert r["ok"] is False and "nope" in r["error"]
        r = await call(q, 3, "session.start", {"source": "demo", "line_delay": 0, "vault": "alpha"})
        assert r["ok"] and r["result"]["status"]["vault"] == "Alpha"
        r = await call(q, 4, "session.set_vault", {"vault": None})
        assert r["ok"] and r["result"]["vault"] is None
        await call(q, 5, "session.stop")
        await s.shutdown()

    asyncio.run(go())
