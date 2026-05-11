"""Glossary YAML loader + injection into LLM prompts."""

from __future__ import annotations

import asyncio
import builtins
import json
import time
from pathlib import Path

import pytest


def _write(p: Path, body: str) -> None:
    p.write_text(body, encoding="utf-8")


def test_load_dictionary_bare_list(tmp_path: Path) -> None:
    import echo.dictionary as dictionary

    f = tmp_path / "d.yml"
    _write(f, "- EchoApp\n- Kotoba-Whisper\n- 石井\n- FOMC\n")
    dictionary.reset_cache()

    assert dictionary.load_dictionary(f) == ("EchoApp", "Kotoba-Whisper", "石井", "FOMC")


def test_load_dictionary_terms_key(tmp_path: Path) -> None:
    import echo.dictionary as dictionary

    f = tmp_path / "d.yml"
    _write(f, "terms:\n  - EchoApp\n  - 石井\n")
    dictionary.reset_cache()

    assert dictionary.load_dictionary(f) == ("EchoApp", "石井")


def test_load_dictionary_grouped_form_flattens_and_dedupes(tmp_path: Path) -> None:
    import echo.dictionary as dictionary

    f = tmp_path / "d.yml"
    _write(
        f,
        "people: [石井, 田中]\n"
        "products: [EchoApp, EchoHub]\n"
        "acronyms: [FOMC, ARPU, FOMC]\n",
    )
    dictionary.reset_cache()

    out = dictionary.load_dictionary(f)
    assert out == ("石井", "田中", "EchoApp", "EchoHub", "FOMC", "ARPU")


def test_load_dictionary_missing_file_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    import echo.dictionary as dictionary

    dictionary.reset_cache()
    monkeypatch.delenv("ECHO_DICTIONARY_PATH", raising=False)

    assert dictionary.load_dictionary(Path("/no/such/file.yml")) == ()


def test_load_dictionary_env_var_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import echo.dictionary as dictionary

    f = tmp_path / "from-env.yml"
    _write(f, "- alpha\n- beta\n")
    dictionary.reset_cache()
    monkeypatch.setenv("ECHO_DICTIONARY_PATH", str(f))

    assert dictionary.load_dictionary() == ("alpha", "beta")


def test_load_dictionary_mtime_cache_invalidates_on_edit(tmp_path: Path) -> None:
    import echo.dictionary as dictionary

    f = tmp_path / "d.yml"
    _write(f, "- one\n")
    dictionary.reset_cache()
    assert dictionary.load_dictionary(f) == ("one",)

    time.sleep(0.01)
    _write(f, "- one\n- two\n")
    import os

    new_mtime = f.stat().st_mtime + 1.0
    os.utime(f, (new_mtime, new_mtime))

    assert dictionary.load_dictionary(f) == ("one", "two")


def test_load_dictionary_caps_term_count(tmp_path: Path) -> None:
    import echo.dictionary as dictionary

    f = tmp_path / "d.yml"
    body = "\n".join(f"- t{i:03d}" for i in range(dictionary.MAX_DICT_TERMS + 25))
    _write(f, body + "\n")
    dictionary.reset_cache()

    out = dictionary.load_dictionary(f)
    assert len(out) <= dictionary.MAX_DICT_TERMS


def test_load_dictionary_caps_serialized_chars(tmp_path: Path) -> None:
    import echo.dictionary as dictionary

    f = tmp_path / "d.yml"
    body = "\n".join(f"- {'x' * 30}_{i}" for i in range(dictionary.MAX_DICT_TERMS))
    _write(f, body + "\n")
    dictionary.reset_cache()

    out = dictionary.load_dictionary(f)
    assert len(", ".join(out)) <= dictionary.MAX_DICT_SERIALIZED_CHARS


def test_load_dictionary_pyyaml_missing_returns_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys

    import echo.dictionary as dictionary

    f = tmp_path / "d.yml"
    _write(f, "- a\n- b\n")
    dictionary.reset_cache()
    monkeypatch.setitem(sys.modules, "yaml", None)

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "yaml":
            raise ImportError("no yaml for you")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    notes: list[str] = []
    assert dictionary.load_dictionary(f, notify=notes.append) == ()
    assert notes and "pyyaml" in notes[0].lower()


def test_llm_route_insight_includes_glossary(monkeypatch: pytest.MonkeyPatch) -> None:
    import echo.llm as llm

    captured: dict[str, str] = {}

    async def fake_llm_user_text(content: str, _max_tokens: int) -> str:
        captured["prompt"] = content
        return json.dumps(
            {
                "action": "comment",
                "corrected_transcript": "ok",
                "comment": "noted",
                "search_query": "",
                "reason": "ok",
            }
        )

    monkeypatch.setattr(llm, "_llm_user_text", fake_llm_user_text)

    asyncio.run(
        llm.llm_route_insight(
            "transcript",
            dictionary_terms=("EchoApp", "石井", "FOMC"),
        )
    )

    prompt = captured["prompt"]
    assert "Preferred glossary" in prompt
    assert "sprinkle glossary nouns" in prompt
    assert "pronunciation-level similarity" in prompt
    # comment-side hard rule: glossary terms cannot be invented in the comment body
    assert "Glossary discipline" in prompt
    assert "spelling reference only" in prompt
    assert "EchoApp" in prompt
    assert "石井" in prompt
    assert "FOMC" in prompt


def test_llm_synthesize_web_findings_includes_glossary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import echo.llm as llm
    from echo.search import SearchHit

    captured: dict[str, str] = {}

    async def fake_llm_user_text(content: str, _max_tokens: int) -> str:
        captured["prompt"] = content
        return "summary"

    monkeypatch.setattr(llm, "_llm_user_text", fake_llm_user_text)

    asyncio.run(
        llm.llm_synthesize_web_findings(
            "transcript chunk",
            "search query",
            [SearchHit(title="t", body="b", href="https://example.com")],
            dictionary_terms=["EchoApp", "石井"],
        )
    )
    p = captured["prompt"]
    assert "Preferred glossary" in p
    assert "spelling reference only" in p
    assert "**not** a topic list" in p
    assert "EchoApp" in p
    assert "石井" in p


def test_chat_prompt_includes_glossary_line() -> None:
    from echo.llm import _chat_prompt

    out = _chat_prompt(
        "transcript text",
        "what was decided?",
        meeting_context="ctx",
        meeting_title="Weekly Sync",
        dictionary_terms=["EchoApp", "石井"],
    )
    assert "Preferred glossary" in out
    assert "pronunciation ambiguity" in out
    assert "EchoApp" in out
    assert "石井" in out


def test_chat_prompt_omits_glossary_line_when_empty() -> None:
    from echo.llm import _chat_prompt

    out = _chat_prompt("t", "q", meeting_context="", meeting_title="", dictionary_terms=())
    assert "Preferred glossary" not in out
