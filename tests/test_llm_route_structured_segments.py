from __future__ import annotations

import asyncio
import json


def test_llm_route_insight_structured_segments(monkeypatch) -> None:
    import echo.llm as llm

    payload = {
        "action": "comment",
        "corrected_transcript": "市場環境の確認と予算再配分を進める。",
        "corrected_segments": [
            {"id": 7, "text": "市場環境の確認を進める。"},
            {"id": 8, "text": "予算再配分の結論を次回会議で決める。"},
        ],
        "comment": "要点は意思決定のタイミングを揃えること。",
        "search_query": "",
        "reason": "社内判断が中心で外部検索は不要",
    }

    async def fake_llm_user_text(content: str, _max_tokens: int) -> str:
        assert "Segment IDs (for structured correction mapping)" in content
        assert "7: 市場環境の確認を進める" in content
        return json.dumps(payload, ensure_ascii=False)

    monkeypatch.setattr(llm, "_llm_user_text", fake_llm_user_text)

    async def _run() -> None:
        out = await llm.llm_route_insight(
            "市場環境の確認を進める 予算再配分を決める",
            segments=[
                (7, "市場環境の確認を進める"),
                (8, "予算再配分を決める"),
            ],
        )
        assert out.action == "comment"
        assert out.corrected_segments == [
            (7, "市場環境の確認を進める。"),
            (8, "予算再配分の結論を次回会議で決める。"),
        ]

    asyncio.run(_run())
