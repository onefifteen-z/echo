"""Meeting title + popup context flow into LLM prompts (routing, synthesis, chat)."""

from __future__ import annotations

import asyncio
import json

import pytest


def test_llm_route_insight_includes_meeting_title_and_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
            "some transcript",
            meeting_title="Weekly Sync",
            meeting_context="Review blockers and next milestones",
        )
    )

    prompt = captured["prompt"]
    assert "Title: Weekly Sync" in prompt
    assert "Review blockers and next milestones" in prompt


def test_llm_route_insight_drops_untitled_placeholder_and_date_suffix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
            "some transcript",
            meeting_title="untitled-meeting",
            meeting_context="",
        )
    )
    assert "Title: (none)" in captured["prompt"]

    asyncio.run(
        llm.llm_route_insight(
            "some transcript",
            meeting_title="Q2 Planning @2026-05-11",
        )
    )
    assert "Title: Q2 Planning" in captured["prompt"]
    assert "@2026-05-11" not in captured["prompt"].split("Title: ", 1)[1].splitlines()[0]


def test_llm_synthesize_web_findings_includes_meeting_title(
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
            meeting_context="background here",
            meeting_title="Weekly Sync",
        )
    )

    assert "Title: Weekly Sync" in captured["prompt"]
    assert "background here" in captured["prompt"]


def test_chat_prompt_includes_title_and_context() -> None:
    from echo.llm import _chat_prompt

    out = _chat_prompt(
        "transcript text",
        "what was decided?",
        meeting_context="context body",
        meeting_title="Weekly Sync",
    )
    assert "Meeting title: Weekly Sync" in out
    assert "Meeting background/context:" in out
    assert "context body" in out
    assert "Transcript:\ntranscript text" in out
    assert "Question:\nwhat was decided?" in out


def test_chat_prompt_omits_title_line_when_unset() -> None:
    from echo.llm import _chat_prompt

    out = _chat_prompt("t", "q", meeting_context="", meeting_title="")
    assert "Meeting title:" not in out
    assert "Meeting background/context:" not in out

    out2 = _chat_prompt("t", "q", meeting_context="", meeting_title="untitled-meeting")
    assert "Meeting title:" not in out2
