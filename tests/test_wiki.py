from __future__ import annotations

import asyncio
import json
import shlex
import sys
from pathlib import Path

import pytest

from echo import wiki

FAKE = Path(__file__).with_name("fake_lumina_mcp.py")


@pytest.fixture
def lumina(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    user_data = tmp_path / "lumina-data"
    user_data.mkdir()
    a, b = tmp_path / "vaults" / "alpha", tmp_path / "vaults" / "beta"
    (a / "wiki").mkdir(parents=True)
    b.mkdir(parents=True)
    (user_data / "llm-wiki-config.json").write_text(
        json.dumps({"vaultPath": str(a), "vaultHistory": [{"path": str(a), "name": "alpha"}, {"path": str(b), "name": "beta"}]})
    )
    monkeypatch.setenv("ECHO_LUMINA_USER_DATA", str(user_data))
    monkeypatch.setenv("ECHO_LUMINA_MCP_CMD", f"{shlex.quote(sys.executable)} {shlex.quote(str(FAKE))}")
    return tmp_path


def test_vault_list_from_lumina_config(lumina: Path) -> None:
    vaults = wiki.LuminaWiki().vaults()
    assert vaults["available"] is True
    assert [(v["name"], v["default"], v["initialized"]) for v in vaults["vaults"]] == [("alpha", True, True), ("beta", False, False)]
    assert wiki.resolve_vault("ALPHA")["name"] == "alpha"
    with pytest.raises(wiki.WikiError):
        wiki.resolve_vault("gamma")


def test_missing_app_is_reported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ECHO_LUMINA_MCP_CMD", raising=False)
    monkeypatch.setenv("ECHO_LUMINA_APP", str(tmp_path / "Nope.app"))
    out = wiki.LuminaWiki().vaults()
    assert out["available"] is False and "not found" in out["reason"]


def test_search_article_titles_and_lookup(lumina: Path) -> None:
    async def run() -> None:
        w = wiki.LuminaWiki()
        try:
            hits = await w.search("alpha", "pricing", limit=3)
            assert [h.filename for h in hits] == ["wiki/concepts/pricing.md", "wiki/entities/acme.md", "raw/notes.md"]
            assert hits[0].excerpt == "pricing in alpha"
            page = await w.article("alpha", "wiki/concepts/pricing.md", max_chars=10)
            assert page.title == "Pricing model" and page.body.endswith("…")
            with pytest.raises(wiki.WikiError, match="not found"):
                await w.article("alpha", "wiki/missing.md")
            assert await w.titles("alpha") == ["Pricing model", "Acme Corp"]
            hits, pages = await w.lookup("beta", "pricing", read=1)
            assert len(hits) == 3 and [p.filename for p in pages] == ["wiki/concepts/pricing.md"]
            assert hits[0].excerpt == "pricing in beta"
        finally:
            await w.close()

    asyncio.run(run())


def test_timeout_and_restart_after_crash(lumina: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(wiki, "SEARCH_TIMEOUT_S", 0.5)
    crash = lumina / "crash-once"

    async def run() -> None:
        w = wiki.LuminaWiki()
        try:
            with pytest.raises(wiki.WikiError, match="did not answer"):
                await w.search("alpha", "slow")
            crash.write_text("1")
            monkeypatch.setenv("FAKE_LUMINA_CRASH_ONCE", str(crash))
            await w.close()
            hits = await w.search("alpha", "pricing")
            assert hits and not crash.exists()
        finally:
            await w.close()

    asyncio.run(run())


def test_tool_error_payload() -> None:
    with pytest.raises(wiki.WikiError, match="boom"):
        wiki._tool_json({"content": [{"type": "text", "text": "boom"}], "isError": True})
    assert wiki._tool_json({"content": [{"type": "text", "text": "[1]"}]}) == [1]
