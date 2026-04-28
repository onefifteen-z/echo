"""Optional LLM calls (Anthropic / OpenAI / OpenAI-compatible e.g. LiteLLM)."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from echo.search import SearchHit

_openai_client_cache: dict[str, object] = {}
_anthropic_client_cache: object | None = None

AGENT_MD_MARKER = "## Echo runtime digest (injected into LLM prompts)"

# Used when AGENT.md is missing or has no digest section (transcript + routing only; ``comment`` shape
# comes from ``INSIGHT_COMMENT_STYLE_RULES`` inside ``INSIGHT_ROUTE_CORE``).
_AGENT_DIGEST_FALLBACK = """**Transcript / ASR:** Treat input as noisy STT; return `corrected_transcript` with same meaning, fixing probable errors; never invent facts or new lines.

**Routing:** Choose `comment` vs `web_search` per the routing instructions below. When `action` is `comment`, the `comment` field must follow the **McKinsey-style rules** block at the top of the routed prompt (not repeated here).

**Web search:** `comment` empty; sharp `search_query`; still return `corrected_transcript`."""

_agent_digest_cache: tuple[float | None, str] = (None, "")


def _agent_md_path() -> Path:
    return Path(__file__).resolve().parent.parent / "AGENT.md"


def _agent_digest_for_prompt() -> str:
    """Load digest from AGENT.md (mtime-invalidated)."""
    global _agent_digest_cache
    path = _agent_md_path()
    mtime: float | None
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = None
    if _agent_digest_cache[0] == mtime and _agent_digest_cache[1]:
        return _agent_digest_cache[1]

    digest = _AGENT_DIGEST_FALLBACK
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        raw = ""
    if raw and AGENT_MD_MARKER in raw:
        digest = raw.split(AGENT_MD_MARKER, 1)[1].strip()
    if not digest.strip():
        digest = _AGENT_DIGEST_FALLBACK
    if len(digest) > 9000:
        digest = digest[:9000].rstrip() + "\n\n…[digest truncated]"

    _agent_digest_cache = (mtime, digest)
    return digest


def _build_insight_route_prompt(
    snippet: str,
    *,
    segments: list[tuple[int, str]] | None = None,
    meeting_context: str = "",
) -> str:
    agent = _agent_digest_for_prompt()
    core = INSIGHT_ROUTE_CORE.replace("<<<AGENT>>>", agent.strip())
    ctx_block = meeting_context.strip() if meeting_context else "(none)"
    core = core.replace("<<<MEETING_CONTEXT>>>", ctx_block)
    prompt = _embed_snippet(core, snippet)
    if not segments:
        return prompt.replace("<<<SEGMENTS>>>", "(none)")
    lines = [f"{sid}: {txt.strip()}" for sid, txt in segments if txt.strip()]
    blob = "\n".join(lines) if lines else "(none)"
    return prompt.replace("<<<SEGMENTS>>>", blob)


def _strip_code_fence(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    return t.strip()


def extract_keywords_fallback(text: str, max_kw: int = 6) -> list[str]:
    """Cheap keyword pick when no LLM is configured."""
    tokens = re.findall(r"[\w\u3040-\u30ff\u4e00-\u9fff]{2,}", text.lower())
    stop = {
        "the",
        "and",
        "for",
        "that",
        "this",
        "with",
        "from",
        "have",
        "will",
        "about",
        "です",
        "ます",
        "こと",
        "これ",
        "たち",
        "する",
        "ない",
    }
    freq: dict[str, int] = {}
    for t in tokens:
        if t in stop:
            continue
        freq[t] = freq.get(t, 0) + 1
    ranked = sorted(freq.items(), key=lambda x: (-x[1], -len(x[0])))
    out: list[str] = []
    seen: set[str] = set()
    for w, _ in ranked:
        if w in seen:
            continue
        seen.add(w)
        out.append(w)
        if len(out) >= max_kw:
            break
    return out


def configured_llm_label() -> str:
    """Which LLM backend `llm_*` will use (for diagnostics)."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if _litellm_openai_config():
        return "openai-compatible (BASE_URL + API_KEY + MODEL)"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    return "none"


def _litellm_openai_config() -> tuple[str, str, str] | None:
    """OpenAI-compatible API (e.g. LiteLLM proxy) when BASE_URL + API_KEY + MODEL are set."""
    base = (os.environ.get("BASE_URL") or "").strip()
    key = (os.environ.get("API_KEY") or "").strip()
    model = (os.environ.get("MODEL") or "").strip()
    if not (base and key and model):
        return None
    return base.rstrip("/"), key, model


def _get_openai_client(api_key: str, base_url: str | None = None) -> Any:
    global _openai_client_cache
    cache_key = f"{api_key}:{base_url or ''}"
    if cache_key not in _openai_client_cache:
        from openai import AsyncOpenAI

        kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        _openai_client_cache[cache_key] = AsyncOpenAI(**kwargs)
    return _openai_client_cache[cache_key]


def _get_anthropic_client(api_key: str) -> Any:
    global _anthropic_client_cache
    if _anthropic_client_cache is None:
        import anthropic

        _anthropic_client_cache = anthropic.AsyncAnthropic(api_key=api_key)
    return _anthropic_client_cache


async def _openai_chat_completions(
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    *,
    base_url: str | None = None,
) -> str:
    client = _get_openai_client(api_key, base_url)
    resp = await client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
    )
    return (resp.choices[0].message.content or "").strip()


# Shipped with the insight routing prompt only (do not rely on AGENT.md for this shape).
INSIGHT_COMMENT_STYLE_RULES = """**Insight `comment` — McKinsey-style message (required when `action` is `comment`):**
- **Pyramid:** Open with **one** sentence that states the **governing thought** (So what / the answer). No warm-up or “in this chunk…” framing.
- **Support:** **At most 2** bullets; each must be a **distinct** point **grounded only in the chunk**; omit bullets if the lead alone is enough. Do not repeat the lead in bullets.
- **Now what:** End with **one** line: concrete next step, decision, or owner—still declarative.
- **~90 words** maximum for `comment`; same language as the transcript; **no questions** (avoid `?`); no section headers; do not name consulting frameworks (MECE, 7S, etc.) unless the transcript already uses them."""


INSIGHT_ROUTE_CORE = f"""{INSIGHT_COMMENT_STYLE_RULES}

<<<AGENT>>>

You route one meeting **transcript chunk** (may be imperfect ASR). Apply the digest above for transcript cleanup and for routing; for `comment` body text, **always** apply the **McKinsey-style rules** block at the very top of this prompt (even if the digest repeats similar ideas).

Meeting background/context (provided by user):
<<<
<<<MEETING_CONTEXT>>>
>>>

Return ONLY valid JSON (no markdown code fences) with exactly these keys:
- "action": either "comment" or "web_search"
- "corrected_transcript": string — same content as the raw chunk with probable ASR/typo/word-boundary fixes; **same language**; **do not** add facts, numbers, or lines not clearly grounded in the chunk. If nothing needs fixing, return the raw chunk verbatim.
- "corrected_segments": array of objects with shape {{"id": number, "text": string}}. Use ONLY ids listed below. Keep line boundaries: do not merge or split segments. "text" must be content-only (no timestamps). If no segment-level edits, return [].
- "comment": string (markdown). If action is "comment": write per the **McKinsey-style rules** at the top of this prompt. If action is "web_search": empty string "".
- "search_query": string. If action is "web_search": concise keywords for a web search. If action is "comment": empty string "".
- "reason": one short line explaining the routing choice.

Routing rules:
- "comment" when the chunk is mostly coordination, status, opinions, or Q&A answerable from context alone without external facts.
- "web_search" when fresh or verifiable public information would materially help (companies, laws, markets, products, statistics, news, unfamiliar proper nouns).
- If factual claims need verification, prefer "web_search".

Raw transcript chunk:
<<<
<<<SNIPPET>>>
>>>

Segment IDs (for structured correction mapping):
<<<
<<<SEGMENTS>>>
>>>
"""


INSIGHT_SYNTH_TEMPLATE = """You write a **very brief** research blurb for people **in a live meeting** (they have seconds, not minutes).

The transcript chunk may already be ASR-corrected (cleaned wording, same facts).

Use the transcript chunk for context, the search query, and the web excerpts.
If sources are empty or weak, say so in one line and lean on the transcript only.

Meeting background/context (provided by user):
<<<
<<<MEETING_CONTEXT>>>
>>>

Transcript chunk:
<<<
<<<SNIPPET>>>
>>>

Search query: <<<QUERY>>>

Web excerpts (may be empty):
<<<
<<<SOURCES>>>
>>>

Rules: same primary language as the transcript; **McKinsey-style executive blurb**—**lead with the takeaway** (pyramid), then tight supporting lines only from excerpts + transcript; **at most 2 short paragraphs** OR **one** paragraph + **up to 3** tight bullets; **hard cap ~110 words** total; declarative; no JSON; no long lists."""


def _embed_snippet(template: str, snippet: str) -> str:
    return template.replace("<<<SNIPPET>>>", snippet)


def _embed_synth(
    template: str, snippet: str, query: str, sources: str, meeting_context: str = "",
) -> str:
    ctx = meeting_context.strip() if meeting_context else "(none)"
    return (
        template.replace("<<<MEETING_CONTEXT>>>", ctx)
        .replace("<<<SNIPPET>>>", snippet)
        .replace("<<<QUERY>>>", query)
        .replace("<<<SOURCES>>>", sources)
    )


def _hits_bullets(hits: list[SearchHit]) -> str:
    if not hits:
        return "(no results)"
    lines: list[str] = []
    for i, h in enumerate(hits, 1):
        title = h.title or "(no title)"
        body = (h.body or "").replace("\n", " ")
        href = h.href or ""
        if href:
            lines.append(f"{i}. {title} — {body}\n   URL: {href}")
        else:
            lines.append(f"{i}. {title} — {body}")
    return "\n".join(lines)


async def _llm_user_text(content: str, max_tokens: int) -> str:
    if key := os.environ.get("ANTHROPIC_API_KEY"):
        return await _anthropic_user_text(key, content, max_tokens)
    if cfg := _litellm_openai_config():
        return await _openai_compatible_user_text(cfg, content, max_tokens)
    if key := os.environ.get("OPENAI_API_KEY"):
        return await _openai_user_text(key, content, max_tokens)
    return ""


async def _anthropic_user_text(api_key: str, content: str, max_tokens: int) -> str:
    client = _get_anthropic_client(api_key)
    msg = await client.messages.create(
        model=os.environ.get("ECHO_ANTHROPIC_MODEL", "claude-3-5-haiku-20241022"),
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": content}],
    )
    return "".join(b.text for b in msg.content if getattr(b, "text", None)).strip()


async def _openai_compatible_user_text(
    cfg: tuple[str, str, str],
    content: str,
    max_tokens: int,
) -> str:
    base_url, api_key, model = cfg
    return await _openai_chat_completions(
        api_key,
        model,
        [{"role": "user", "content": content}],
        max_tokens,
        base_url=base_url,
    )


async def _openai_user_text(api_key: str, content: str, max_tokens: int) -> str:
    model = os.environ.get("ECHO_OPENAI_MODEL", "gpt-4o-mini")
    return await _openai_chat_completions(
        api_key,
        model,
        [{"role": "user", "content": content}],
        max_tokens,
    )


@dataclass
class InsightRouteResult:
    """LLM routing for one insight pass."""

    action: Literal["comment", "web_search"]
    comment: str
    search_query: str
    reason: str
    corrected_transcript: str = ""
    corrected_segments: list[tuple[int, str]] | None = None


def _parse_insight_route_json(text: str) -> InsightRouteResult | None:
    raw = _strip_code_fence(text)
    try:
        data: Any = json.loads(raw)
    except json.JSONDecodeError:
        return None
    action_raw = str(data.get("action") or "").lower().strip()
    if action_raw in ("web", "web_search", "search", "research", "lookup"):
        action: Literal["comment", "web_search"] = "web_search"
    else:
        action = "comment"
    comment = str(data.get("comment") or "").strip()
    search_query = str(data.get("search_query") or "").strip()
    reason = str(data.get("reason") or "").strip()
    corrected = str(
        data.get("corrected_transcript")
        or data.get("corrected")
        or data.get("cleaned_transcript")
        or "",
    ).strip()
    corrected_segments_raw = data.get("corrected_segments") or []
    corrected_segments: list[tuple[int, str]] = []
    if isinstance(corrected_segments_raw, list):
        for row in corrected_segments_raw:
            if not isinstance(row, dict):
                continue
            rid = row.get("id")
            txt = row.get("text")
            try:
                sid = int(rid)
            except (TypeError, ValueError):
                continue
            text_val = str(txt or "").strip()
            if not text_val:
                continue
            corrected_segments.append((sid, text_val))
    return InsightRouteResult(
        action=action,
        comment=comment,
        search_query=search_query,
        reason=reason,
        corrected_transcript=corrected,
        corrected_segments=corrected_segments,
    )


async def llm_route_insight(
    transcript_snippet: str,
    *,
    segments: list[tuple[int, str]] | None = None,
    meeting_context: str = "",
) -> InsightRouteResult:
    """Ask LLM to choose comment-only vs web_search. Without API keys, returns a comment explaining missing config."""
    snippet = transcript_snippet.strip()
    if not snippet:
        return InsightRouteResult("comment", "", "", "", "")

    prompt = _build_insight_route_prompt(snippet, segments=segments, meeting_context=meeting_context)
    raw = await _llm_user_text(prompt, 768)
    if not raw.strip():
        return InsightRouteResult(
            "comment",
            "(LLM not configured: set ANTHROPIC_API_KEY, OPENAI_API_KEY, "
            "or BASE_URL + API_KEY + MODEL; routing between comment and web search is disabled.)",
            "",
            "no_llm",
            snippet,
            [],
        )

    parsed = _parse_insight_route_json(raw)
    if parsed is None:
        return InsightRouteResult(
            "comment",
            f"_Routing JSON could not be parsed. Raw model output:_\n\n```\n{raw[:1200]}\n```",
            "",
            "parse_error",
            snippet,
            [],
        )

    if parsed.action == "comment" and not parsed.comment:
        parsed = InsightRouteResult(
            "comment",
            "_Model selected `comment` but returned no body text._",
            "",
            parsed.reason or "empty_comment",
            parsed.corrected_transcript or snippet,
            parsed.corrected_segments or [],
        )
    if parsed.action == "web_search" and not parsed.search_query:
        kws = extract_keywords_fallback(snippet)
        q = " ".join(kws[:4]) if kws else snippet[:120].strip()
        parsed = InsightRouteResult(
            "web_search",
            "",
            q,
            (parsed.reason + " · " if parsed.reason else "") + "filled_empty_query",
            parsed.corrected_transcript or snippet,
            parsed.corrected_segments or [],
        )

    final_ct = (parsed.corrected_transcript or "").strip() or snippet
    return InsightRouteResult(
        parsed.action,
        parsed.comment,
        parsed.search_query,
        parsed.reason,
        final_ct,
        parsed.corrected_segments or [],
    )


async def llm_synthesize_web_findings(
    transcript_snippet: str,
    query: str,
    hits: list[SearchHit],
    meeting_context: str = "",
) -> str:
    """Turn transcript + search hits into a short narrative summary."""
    snippet = transcript_snippet.strip()
    sources = _hits_bullets(hits)
    prompt = _embed_synth(INSIGHT_SYNTH_TEMPLATE, snippet, query.strip(), sources, meeting_context)
    text = await _llm_user_text(prompt, 512)
    if not text.strip():
        return (
            "_(No summary text returned by LLM; check configuration or retry. Search excerpts below.)_\n\n"
            f"**Query:** `{query}`\n\n{_hits_bullets(hits)}"
        )
    return text


async def llm_keywords_and_summary(transcript_snippet: str) -> tuple[list[str], str]:
    """Return (keywords, short_summary). Uses env API keys when present."""
    snippet = transcript_snippet.strip()
    if not snippet:
        return [], ""

    if key := os.environ.get("ANTHROPIC_API_KEY"):
        return await _anthropic_extract(key, snippet)
    if cfg := _litellm_openai_config():
        return await _openai_compatible_extract(cfg, snippet)
    if key := os.environ.get("OPENAI_API_KEY"):
        return await _openai_extract(key, snippet)

    kws = extract_keywords_fallback(snippet)
    summary = (
        "(LLM not configured: set ANTHROPIC_API_KEY, OPENAI_API_KEY, "
        "or LiteLLM-style BASE_URL + API_KEY + MODEL; using heuristic keywords only.)"
    )
    return kws, summary


async def llm_chat_answer(
    full_transcript: str, question: str, meeting_context: str = "",
) -> str:
    if key := os.environ.get("ANTHROPIC_API_KEY"):
        return await _anthropic_chat(key, full_transcript, question, meeting_context)
    if cfg := _litellm_openai_config():
        return await _openai_compatible_chat(cfg, full_transcript, question, meeting_context)
    if key := os.environ.get("OPENAI_API_KEY"):
        return await _openai_chat(key, full_transcript, question, meeting_context)
    return (
        "No API key detected. Set ANTHROPIC_API_KEY, OPENAI_API_KEY, "
        "or LiteLLM-style BASE_URL + API_KEY + MODEL.\n"
        f"Current transcript length: {len(full_transcript)} chars."
    )


async def _anthropic_extract(api_key: str, snippet: str) -> tuple[list[str], str]:
    client = _get_anthropic_client(api_key)
    msg = await client.messages.create(
        model=os.environ.get("ECHO_ANTHROPIC_MODEL", "claude-3-5-haiku-20241022"),
        max_tokens=512,
        messages=[
            {
                "role": "user",
                "content": (
                    "You extract meeting insights from a transcript chunk.\n"
                    "Return ONLY compact JSON: "
                    '{"keywords":["..."],"summary":"one short paragraph"}\n'
                    "Keywords: 3-8 items, same language as transcript.\n\n"
                    f"Transcript chunk:\n{snippet}"
                ),
            }
        ],
    )
    text = "".join(b.text for b in msg.content if getattr(b, "text", None))
    return _parse_kw_summary(text)


async def _openai_compatible_extract(
    cfg: tuple[str, str, str],
    snippet: str,
) -> tuple[list[str], str]:
    base_url, api_key, model = cfg
    text = await _openai_chat_completions(
        api_key,
        model,
        [
            {
                "role": "user",
                "content": (
                    "Extract meeting insights. Return ONLY JSON: "
                    '{"keywords":["..."],"summary":"one short paragraph"}\n'
                    f"Transcript chunk:\n{snippet}"
                ),
            }
        ],
        512,
        base_url=base_url,
    )
    return _parse_kw_summary(text)


async def _openai_extract(api_key: str, snippet: str) -> tuple[list[str], str]:
    model = os.environ.get("ECHO_OPENAI_MODEL", "gpt-4o-mini")
    text = await _openai_chat_completions(
        api_key,
        model,
        [
            {
                "role": "user",
                "content": (
                    "Extract meeting insights. Return ONLY JSON: "
                    '{"keywords":["..."],"summary":"one short paragraph"}\n'
                    f"Transcript chunk:\n{snippet}"
                ),
            }
        ],
        512,
    )
    return _parse_kw_summary(text)


def _parse_kw_summary(text: str) -> tuple[list[str], str]:
    raw = _strip_code_fence(text)
    try:
        data: Any = json.loads(raw)
        kws = list(data.get("keywords") or [])
        summary = str(data.get("summary") or "").strip()
        kws = [str(x).strip() for x in kws if str(x).strip()]
        return kws[:12], summary
    except json.JSONDecodeError:
        return extract_keywords_fallback(raw), raw[:400]


def _chat_prompt(full_transcript: str, question: str, meeting_context: str = "") -> str:
    ctx = f"Meeting background/context:\n{meeting_context.strip()}\n\n" if meeting_context.strip() else ""
    return (
        "You answer questions about a meeting using the transcript as context.\n"
        "Be concise. If unknown, say you do not know.\n\n"
        f"{ctx}Transcript:\n{full_transcript}\n\nQuestion:\n{question}"
    )


async def _anthropic_chat(
    api_key: str, full_transcript: str, question: str, meeting_context: str = "",
) -> str:
    client = _get_anthropic_client(api_key)
    msg = await client.messages.create(
        model=os.environ.get("ECHO_ANTHROPIC_MODEL", "claude-3-5-haiku-20241022"),
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": _chat_prompt(full_transcript, question, meeting_context),
            }
        ],
    )
    return "".join(b.text for b in msg.content if getattr(b, "text", None)).strip()


async def _openai_compatible_chat(
    cfg: tuple[str, str, str],
    full_transcript: str,
    question: str,
    meeting_context: str = "",
) -> str:
    base_url, api_key, model = cfg
    return await _openai_chat_completions(
        api_key,
        model,
        [
            {
                "role": "user",
                "content": _chat_prompt(full_transcript, question, meeting_context),
            }
        ],
        1024,
        base_url=base_url,
    )


async def _openai_chat(
    api_key: str, full_transcript: str, question: str, meeting_context: str = "",
) -> str:
    model = os.environ.get("ECHO_OPENAI_MODEL", "gpt-4o-mini")
    return await _openai_chat_completions(
        api_key,
        model,
        [
            {
                "role": "user",
                "content": _chat_prompt(full_transcript, question, meeting_context),
            }
        ],
        1024,
    )
