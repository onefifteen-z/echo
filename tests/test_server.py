"""Sidecar protocol tests (in-process; no sockets)."""

from __future__ import annotations

import asyncio
import json

import pytest

from echo.server import EchoServer


def _run(server: EchoServer, req: dict) -> dict:
    async def go() -> dict:
        q: asyncio.Queue[str] = asyncio.Queue()
        await server._dispatch(q, json.dumps(req))
        return json.loads(q.get_nowait())

    return asyncio.run(go())


def test_token_check() -> None:
    s = EchoServer("secret")
    assert s.authorized("/?token=secret")
    assert not s.authorized("/?token=nope")
    assert not s.authorized("/")


def test_unknown_command_and_missing_session() -> None:
    s = EchoServer("t")
    r = _run(s, {"id": 7, "cmd": "nope"})
    assert r == {"type": "response", "id": 7, "ok": False, "error": "Unknown command: 'nope'"}
    r = _run(s, {"id": 8, "cmd": "session.pause"})
    assert r["ok"] is False and "session.start" in r["error"]
    assert _run(s, {"id": 9, "cmd": "ping"})["result"]["pong"] is True


def test_demo_session_broadcasts_events(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setattr("echo.connectivity.echo_llm_online", lambda: False)
    s = EchoServer("t", output_root=tmp_path)
    client_q: asyncio.Queue[str] = asyncio.Queue()
    s._clients["c"] = client_q

    async def go() -> list[dict]:
        q: asyncio.Queue[str] = asyncio.Queue()
        await s._dispatch(q, json.dumps({"id": 1, "cmd": "session.start", "args": {"source": "demo", "line_delay": 0}}))
        resp = json.loads(q.get_nowait())
        assert resp["ok"], resp
        for _ in range(200):
            if s.session is not None and s.session.status().state == "finished":
                break
            await asyncio.sleep(0.01)
        await s._dispatch(q, json.dumps({"id": 2, "cmd": "session.stop"}))
        stop = json.loads(q.get_nowait())
        assert stop["ok"] and stop["result"]["output_dir"]
        out = []
        while not client_q.empty():
            out.append(json.loads(client_q.get_nowait()))
        return out

    msgs = asyncio.run(go())
    finals = [m["event"] for m in msgs if m["event"]["type"] == "transcript.final"]
    assert len(finals) >= 1
    statuses = [m["event"] for m in msgs if m["event"]["type"] == "status"]
    assert statuses[-1]["state"] == "stopped"
    (folder,) = [p for p in tmp_path.iterdir() if p.is_dir()]
    assert folder.name.endswith(" untitled-meeting")
    assert (folder / "notes.txt").exists()
    meta = json.loads((folder / "meeting.json").read_text(encoding="utf-8"))
    assert meta["id"] == folder.name and meta["mode"] == "replay" and meta["line_count"] >= 1


def test_meetings_commands_and_live_notebook(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setattr("echo.connectivity.echo_llm_online", lambda: False)
    s = EchoServer("t", output_root=tmp_path)
    client_q: asyncio.Queue[str] = asyncio.Queue()
    s._clients["c"] = client_q

    async def call(q: asyncio.Queue[str], n: int, cmd: str, args: dict | None = None) -> dict:
        await s._dispatch(q, json.dumps({"id": n, "cmd": cmd, "args": args or {}}))
        return json.loads(q.get_nowait())

    def status_titles() -> list[str]:
        out = []
        while not client_q.empty():
            ev = json.loads(client_q.get_nowait())["event"]
            if ev["type"] == "status":
                out.append(ev["meeting_title"])
        return out

    async def go() -> None:
        q: asyncio.Queue[str] = asyncio.Queue()
        r = await call(q, 1, "session.start", {"source": "demo", "line_delay": 0, "title": "Sync"})
        assert r["ok"], r
        r = await call(q, 2, "session.notebook.set", {"text": "- decide budget"})
        assert r["ok"] and s.session is not None and s.session.notebook_md == "- decide budget"

        listing = (await call(q, 3, "meetings.list"))["result"]
        live_id = listing["live_id"]
        assert live_id and live_id.endswith(" Sync")
        assert listing["meetings"][0]["id"] == live_id and listing["meetings"][0]["live"] is True

        r = await call(q, 4, "meetings.regenerate_summary", {"id": live_id})
        assert r["ok"] is False and "End the meeting first" in r["error"]
        status_titles()
        r = await call(q, 5, "meetings.update", {"id": live_id, "title": "Budget sync", "context": "Q3 numbers"})
        assert r["ok"] and r["result"]["title"] == "Budget sync" and s.session.meeting_title == "Budget sync"
        assert r["result"]["context"] == "Q3 numbers" and s.session.meeting_context == "Q3 numbers"
        assert "Budget sync" in status_titles()
        assert s.session.output_dir.name == live_id

        await call(q, 6, "session.stop")
        detail = (await call(q, 7, "meetings.get", {"id": live_id}))["result"]
        assert detail["live"] is False and detail["title"] == "Budget sync" and detail["context"] == "Q3 numbers"
        assert detail["notebook_md"] == "- decide budget"

        r = await call(q, 8, "meetings.update", {"id": live_id, "notebook": "edited", "context": "Q3"})
        assert r["ok"] and r["result"]["notebook_md"] == "edited" and r["result"]["context"] == "Q3"
        r = await call(q, 9, "meetings.get", {"id": "../etc"})
        assert r["ok"] is False

    asyncio.run(go())


def test_config_effective_sources_and_masking(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ECHO_APP_SETTINGS_KEYS", "API_KEY,ECHO_ASR_BACKEND")
    monkeypatch.setenv("API_KEY", "sk-abcdefghijkl")
    monkeypatch.setenv("ECHO_ASR_BACKEND", "api")
    monkeypatch.setenv("MODEL", "gpt-x")
    monkeypatch.delenv("ECHO_DEBUG", raising=False)
    s = EchoServer("t")
    r = _run(s, {"id": 1, "cmd": "config.effective", "args": {"keys": ["API_KEY", "ECHO_ASR_BACKEND", "MODEL", "ECHO_DEBUG"]}})
    assert r["ok"] is True
    res = r["result"]
    assert res["API_KEY"] == {"value": "sk-…ijkl", "source": "app", "secret": True}
    assert res["ECHO_ASR_BACKEND"] == {"value": "api", "source": "app", "secret": False}
    assert res["MODEL"] == {"value": "gpt-x", "source": "env", "secret": False}
    assert res["ECHO_DEBUG"] == {"value": None, "source": "default", "secret": False}
    assert "sk-abcdefghijkl" not in json.dumps(r)
    bad = _run(s, {"id": 2, "cmd": "config.effective", "args": {"keys": "API_KEY"}})
    assert bad["ok"] is False


def test_search_probe_reports_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    from echo.server import _search_probe

    for k in ("ECHO_PERPLEXITY_BASE_URL", "AI_GATEWAY_PERPLEXITY_BASE_URL"):
        monkeypatch.delenv(k, raising=False)
    assert _search_probe() == {"provider": "ddgs", "perplexity_configured": False, "note": ""}
    monkeypatch.setenv("ECHO_PERPLEXITY_BASE_URL", "http://insecure/v1")
    monkeypatch.setenv("ECHO_PERPLEXITY_API_KEY", "k")
    p = _search_probe()
    assert p["provider"] == "ddgs" and not p["perplexity_configured"] and "https" in p["note"]
    monkeypatch.setenv("ECHO_PERPLEXITY_BASE_URL", "https://pplx.example/v1")
    assert _search_probe()["provider"] == "Perplexity"
