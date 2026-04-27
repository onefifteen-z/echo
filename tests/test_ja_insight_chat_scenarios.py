"""Japanese scenarios: insight (comment / web) + chat — replay fixture + mocked LLM."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

FIXTURE = Path(__file__).resolve().parent.parent / "samples" / "ja_meeting_insight_chat.txt"


def test_ja_fixture_utf8_and_insight_thresholds() -> None:
    """Multi-line replay so TUI shows gradual transcript; buffers still reach insight thresholds."""
    text = FIXTURE.read_text(encoding="utf-8")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    assert len(lines) >= 10, "sample should be many short lines (one write per line in replay)"
    assert lines[0].startswith("[") and "本日" in lines[0] and "四半期" in lines[0]
    assert len("\n".join(lines[:4])) >= 120, "first four lines should reach insight_min_chars when joined in buffer"
    idx = next(i for i, ln in enumerate(lines) if "日経" in ln)
    assert len("\n".join(lines[idx : idx + 6])) >= 120, "web-oriented block should reach second-pass threshold"
    tail = "\n".join(lines[-2:])
    assert "アウトソース" in tail and "いくら" in tail


def test_llm_route_insight_ja_comment_json_mock(monkeypatch: pytest.MonkeyPatch) -> None:
    import echo.llm as llm

    payload = {
        "action": "comment",
        "corrected_transcript": "本日のアジェンダは四半期レビューとキャップエックスの再配分です。（軽微な表記整理）",
        "comment": "**結論：** まず予算圧力と期限の整合を確認すべきです。\n"
        "- マーケ費用は上振れしている一方で在庫コストが利益を圧迫\n"
        "- 金曜ドラフト共有は意思決定の前提条件\n"
        "**次の一手：** 各部ドラフトの前提（在庫前提）を一枚に要約して共有してください。",
        "search_query": "",
        "reason": "社内調整とスケジュール確認が中心で外部検証は不要",
    }

    async def fake_llm_user_text(_content: str, _max_tokens: int) -> str:
        return json.dumps(payload, ensure_ascii=False)

    monkeypatch.setattr(llm, "_llm_user_text", fake_llm_user_text)

    async def _run() -> None:
        out = await llm.llm_route_insight("本日のアジェンダは四半期レビューとキャップエックスの再配分です。")
        assert out.action == "comment"
        assert "結論" in out.comment
        assert out.corrected_transcript.startswith("本日のアジェンダ")
        assert out.search_query == ""

    asyncio.run(_run())


def test_llm_route_insight_ja_web_search_json_mock(monkeypatch: pytest.MonkeyPatch) -> None:
    import echo.llm as llm

    payload = {
        "action": "web_search",
        "corrected_transcript": "昨日の日経平均株価の終値とFOMCの公表スケジュールを確認したい。",
        "comment": "",
        "search_query": "日経平均 終値 FOMC スケジュール",
        "reason": "公開市場データと公式スケジュールの確認が必要",
    }

    async def fake_llm_user_text(_content: str, _max_tokens: int) -> str:
        return json.dumps(payload, ensure_ascii=False)

    monkeypatch.setattr(llm, "_llm_user_text", fake_llm_user_text)

    async def _run() -> None:
        out = await llm.llm_route_insight("昨日の日経平均株価の終値と米連邦準備制度理事会の次回FOMC…")
        assert out.action == "web_search"
        assert "日経" in out.search_query
        assert out.corrected_transcript

    asyncio.run(_run())


def test_llm_synthesize_web_findings_ja_mock(monkeypatch: pytest.MonkeyPatch) -> None:
    import echo.llm as llm
    from echo.search import SearchHit

    async def fake_llm_user_text(content: str, _max_tokens: int) -> str:
        assert "research blurb" in content or "research summary" in content
        assert "日経" in content or "FOMC" in content
        return "要約：公開情報は限定的だが、転写の意図は金利・指数の最新値確認である。"

    monkeypatch.setattr(llm, "_llm_user_text", fake_llm_user_text)
    hits = [
        SearchHit(title="サンプル", body="本文", href="https://example.com"),
    ]

    async def _run() -> None:
        text = await llm.llm_synthesize_web_findings(
            "昨日の日経平均株価の終値と…",
            "日経平均 終値",
            hits,
        )
        assert "要約" in text

    asyncio.run(_run())


def test_llm_chat_answer_ja_mock(monkeypatch: pytest.MonkeyPatch) -> None:
    import echo.llm as llm

    lines = [ln.strip() for ln in FIXTURE.read_text(encoding="utf-8").splitlines() if ln.strip()]
    full = "\n".join(f"[00:00:0{i}] {ln}" for i, ln in enumerate(lines))

    async def fake_chat_answer(_full: str, q: str) -> str:
        assert "アウトソース" in _full
        assert "月額上限" in q or "いくら" in q
        return "議事録上の合意は税抜き月額上限が未確定のまま「金曜までに法務が数値を確定」との記載のみでした。"

    monkeypatch.setattr(llm, "llm_chat_answer", fake_chat_answer)
    ans = asyncio.run(llm.llm_chat_answer(full, "先ほど合意したアウトソース契約の月額上限は税抜きでいくらでしたか？"))
    assert "議事録" in ans


def test_load_replay_lines_matches_fixture() -> None:
    from echo.app import _load_replay_lines

    lines = list(_load_replay_lines(FIXTURE))
    assert len(lines) == 13
