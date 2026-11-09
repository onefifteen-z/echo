from __future__ import annotations

from echo.llm import _parse_insight_route_json

HALF_QUOTED_KEY = """{
  "action": "web_search",
  "corrected_transcript": "第一个阶段就是所谓的这种文章时期",
  "corrected_segments": [
    {
      "id": 16,
      "text": "许多人想要行善 ，许��人想要通过 GiveWell 去捐钱"
    }
  ],
  "comment": "",
  "search_query": "EA effective altruism TelenserPy Falantobay",
  title": "E257",
  "reason": "Garbled proper nouns related to Effective Altruism need verification."
}"""


def test_half_quoted_key_is_repaired_and_mangled_segment_dropped() -> None:
    r = _parse_insight_route_json(HALF_QUOTED_KEY)
    assert r is not None
    assert r.action == "web_search"
    assert r.search_query == "EA effective altruism TelenserPy Falantobay"
    assert r.corrected_transcript == "第一个阶段就是所谓的这种文章时期"
    assert r.corrected_segments == []


def test_unquoted_keys_trailing_commas_and_prose_around_json() -> None:
    text = 'Here you go:\n{action: "comment", comment: "Budget first.", "reason": "ok",}\nThanks'
    r = _parse_insight_route_json(text)
    assert r is not None and r.action == "comment" and r.comment == "Budget first."


def test_salvages_fields_from_unrepairable_json() -> None:
    text = '{"action": "wiki_lookup", "wiki_query": "pricing \\"tiers\\"", "corrected_segments": [{"id": 1, "text": "x"}}'
    r = _parse_insight_route_json(text)
    assert r is not None and r.action == "wiki_lookup" and r.wiki_query == 'pricing "tiers"'


def test_garbage_still_returns_none() -> None:
    assert _parse_insight_route_json("I cannot help with that.") is None
