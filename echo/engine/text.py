"""Pure transcript / insight helpers shared by the engine, the TUI, and the sidecar (no Textual)."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterable, Literal

from echo.llm import InsightRouteResult
from echo.search import SearchHit
from echo.wiki import WikiHit


@dataclass
class InsightPassRecord:
    n: int
    stamp: str
    kind: Literal["comment", "web_search", "wiki"]
    body_md: str
    routing_md: str | None = None
    corrected_md: str | None = None
    web_md: str | None = None
    vault: str | None = None


@dataclass
class TranscriptSegment:
    line_id: int
    line: str
    spoken_text: str
    stable_after: float
    allow_correction: bool
    finalized: bool = True


_LEADING_BRACKET_TS = re.compile(r"^(\[[^\]]+\]\s*)")


_LIVE_LINE_SPAN_RE = re.compile(
    r"^\[(\d{2}):(\d{2}):(\d{2})[–-](\d{2}):(\d{2}):(\d{2})\]"
)


_LIVE_SENTENCE_END_RE = re.compile(r"[。！？!?]+$")


_LIVE_SENTENCE_END_ANYWHERE_RE = re.compile(r"[。！？!?]")


_LIVE_SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？!?])\s*")


_LIVE_CLAUSE_SPLIT_RE = re.compile(r"(?<=[、，,;；])\s*")


_UNTITLED_MEETING = "untitled-meeting"


def _default_meeting_title(title: str) -> str:
    today = f"@{date.today().isoformat()}"
    t = (title or "").strip()
    if not t or t == _UNTITLED_MEETING:
        return today
    if today not in t:
        return f"{t} {today}"
    return t


def _leading_ts_prefix(line: str) -> str:
    m = _LEADING_BRACKET_TS.match(line)
    return m.group(1) if m else ""


def _strip_leading_ts(line: str) -> str:
    return _LEADING_BRACKET_TS.sub("", line, count=1).strip()


def _line_span_seconds(line: str) -> tuple[float, float] | None:
    m = _LIVE_LINE_SPAN_RE.match(line)
    if not m:
        return None
    h0, m0, s0, h1, m1, s1 = (int(x) for x in m.groups())
    t0 = float((h0 * 3600) + (m0 * 60) + s0)
    t1 = float((h1 * 3600) + (m1 * 60) + s1)
    return t0, t1


def _normalize_spaces(text: str) -> str:
    return " ".join(text.split())


def meeting_title_to_folder_name(title: str) -> str:
    clean = " ".join((title or "").split()).strip()
    if not clean:
        return _UNTITLED_MEETING
    # Keep Unicode/spacing in folder names while avoiding path separators.
    clean = clean.replace("/", "-").replace("\\", "-")
    clean = re.sub(r'[:*?"<>|]+', "-", clean).strip().strip(".")
    return clean or _UNTITLED_MEETING


def summary_preview(md: str, limit: int = 160) -> str:
    """First content line of a summary, for the meeting list (cached in ``meeting.json``)."""
    for raw in md.splitlines():
        line = raw.strip().lstrip("-*").strip()
        if not line or line.startswith("#") or line.startswith("_Generating"):
            continue
        line = line.replace("**", "")
        return line if len(line) <= limit else line[: limit - 1] + "…"
    return ""


def normalize_meeting_time(value: str | datetime) -> str:
    """ISO meeting time floored to the minute, with a UTC offset (naive input is local time).

    Raises ``ValueError`` for unparsable strings.
    """
    dt = datetime.fromisoformat(value.strip()) if isinstance(value, str) else value
    if dt.tzinfo is None:
        dt = dt.astimezone()
    return dt.replace(second=0, microsecond=0).isoformat(timespec="seconds")


def meeting_time_label(value: str | None) -> str:
    """``YYYY-MM-DD HH:MM`` for prompts; empty when unset or unparsable."""
    if not value:
        return ""
    try:
        return f"{datetime.fromisoformat(value):%Y-%m-%d %H:%M}"
    except ValueError:
        return ""


def patch_meeting_json(folder: Path, **fields: object) -> None:
    """Merge ``fields`` into ``folder/meeting.json`` (no-op when the folder has none)."""
    path = folder / "meeting.json"
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    if not isinstance(meta, dict):
        return
    meta.update(fields)
    tmp = path.with_name(".meeting.json.tmp")
    tmp.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def meeting_folder_name(title: str, when: datetime, root: Path) -> str:
    """Unique ``"YYYY-MM-DD HHMM <title>"`` folder name under ``root`` (meetings never overwrite each other)."""
    base = f"{when:%Y-%m-%d %H%M} {meeting_title_to_folder_name(title)}"
    name, n = base, 2
    while (root / name).exists():
        name = f"{base} ({n})"
        n += 1
    return name


def _render_asr_load_source_label(source: str | None) -> str:
    if source == "apple-on-device":
        return "Apple Speech (on-device)"
    if source == "whisper-api":
        return "Whisper API (OpenAI-compatible)"
    if source == "local-whisper":
        return "local Kotoba-Whisper"
    if source == "local-cache-only":
        return "local cache only (offline mode)"
    if source == "local-cache":
        return "local cache (no network fetch needed)"
    if source == "hub-fetch":
        return "network (Hugging Face Hub fallback)"
    return "unknown"


def _short_asr_label(source: str | None) -> str:
    if source == "apple-on-device":
        return "ASR=Apple"
    if source == "whisper-api":
        return "ASR=API"
    if source in ("local-whisper", "local-cache-only", "local-cache", "hub-fetch"):
        return "ASR=local"
    return "ASR=?"


def _trim_chunk_overlap(prev: str, cur: str, *, min_chars: int = 8, max_chars: int = 80) -> str:
    p = _normalize_spaces(prev)
    c = _normalize_spaces(cur)
    if not c:
        return ""
    if not p:
        return c
    lim = min(len(p), len(c), max_chars)
    for n in range(lim, min_chars - 1, -1):
        if p.endswith(c[:n]):
            return c[n:].lstrip()
    return c


def _split_live_groups(text: str, *, max_chars: int = 36) -> list[str]:
    normalized = _normalize_spaces(text)
    if not normalized:
        return []

    units = [u.strip() for u in _LIVE_SENTENCE_SPLIT_RE.split(normalized) if u.strip()]
    groups: list[str] = []
    for unit in units:
        if len(unit) <= max_chars:
            groups.append(unit)
            continue

        clauses = [c.strip() for c in _LIVE_CLAUSE_SPLIT_RE.split(unit) if c.strip()]
        if len(clauses) <= 1:
            groups.extend(
                unit[i : i + max_chars].strip()
                for i in range(0, len(unit), max_chars)
                if unit[i : i + max_chars].strip()
            )
            continue

        cur = ""
        for clause in clauses:
            if len(clause) > max_chars:
                if cur:
                    groups.append(cur)
                    cur = ""
                groups.extend(
                    clause[i : i + max_chars].strip()
                    for i in range(0, len(clause), max_chars)
                    if clause[i : i + max_chars].strip()
                )
                continue
            if not cur:
                cur = clause
                continue
            if len(cur) + 1 + len(clause) <= max_chars:
                cur = f"{cur} {clause}"
                continue
            groups.append(cur)
            cur = clause
        if cur:
            groups.append(cur)
    return groups


def _strip_live_tail_punct(text: str) -> str:
    return text.rstrip("。！？!?.,、，,;；:： ")


def _live_hits_stop_word(text: str, stop_words: tuple[str, ...]) -> bool:
    if not stop_words:
        return False
    tail = _strip_live_tail_punct(_normalize_spaces(text)).lower()
    if not tail:
        return False
    return any(tail.endswith(w.lower()) for w in stop_words)


def _is_live_noise_token(
    text: str,
    *,
    noise_tokens: tuple[str, ...],
    max_chars: int,
) -> bool:
    if not noise_tokens:
        return False
    norm = _strip_live_tail_punct(_normalize_spaces(text)).lower()
    if not norm:
        return False
    if len(norm) > max(1, max_chars):
        return False
    return norm in {t.lower() for t in noise_tokens}


def _should_drop_live_noise_token(
    *,
    text: str,
    noise_tokens: tuple[str, ...],
    always_drop_tokens: tuple[str, ...],
    max_chars: int,
    piece_rms: float,
    min_rms: float,
    max_rms_factor: float,
    is_segment_start: bool,
    drop_at_segment_start: bool,
) -> bool:
    if not _is_live_noise_token(text, noise_tokens=noise_tokens, max_chars=max_chars):
        return False
    norm = _strip_live_tail_punct(_normalize_spaces(text)).lower()
    if norm in {t.strip().lower() for t in always_drop_tokens if t.strip()}:
        return True
    if drop_at_segment_start and is_segment_start:
        return True
    return piece_rms <= (min_rms * max_rms_factor)


def _live_chunk_metrics(
    samples: object,
    *,
    active_amp: float,
) -> tuple[float, float, float]:
    import numpy as np

    x = np.asarray(samples, dtype=np.float32).reshape(-1)
    if x.size == 0:
        return 0.0, 0.0, 0.0
    absx = np.abs(x)
    rms = float(np.sqrt(np.mean(np.square(x), dtype=np.float64)))
    peak = float(absx.max())
    active_ratio = float(np.mean(absx >= max(active_amp, 1e-8)))
    return rms, peak, active_ratio


def _passes_live_voice_gate(
    *,
    rms: float,
    peak: float,
    active_ratio: float,
    min_rms: float,
    active_amp: float,
    min_active_ratio: float,
) -> bool:
    # Gate low-energy chunks before Whisper to reduce "silence hallucinations".
    if rms >= min_rms:
        return True
    if active_ratio >= min_active_ratio and peak >= active_amp * 1.2:
        return True
    return False


def _should_trigger_insight_live(
    *,
    new_chars: int,
    new_finalized_segments: int,
    max_finalized_segment_chars: int,
    finalized_segment_min_chars: int,
    skip_min_chars: int,
) -> tuple[bool, str]:
    """Decide whether to fire an insight pass for the current live transcript buffer.

    The live insight policy is intentionally simple: only fire after the live worker has finalized
    at least one segment that crosses ``finalized_segment_min_chars``, and the total buffered text
    crosses ``skip_min_chars``. Time-based / pause-based triggers were experimented with and
    removed because they fired noisy passes mid-thought; they are kept out of the signature on
    purpose to avoid suggesting they are wired up.
    """
    if new_chars < max(1, skip_min_chars):
        return False, ""
    if (
        new_finalized_segments >= 1
        and max_finalized_segment_chars >= max(1, finalized_segment_min_chars)
    ):
        return True, "new_segment"
    return False, ""


def _should_flush_live_pending(
    *,
    has_pending: bool,
    pending_duration: float,
    idle_tick: bool,
    idle_for: float,
    pause_seconds: float,
    emit_interval: float,
) -> tuple[bool, str]:
    if not has_pending:
        return False, ""
    if emit_interval <= 0.0:
        return True, "immediate"
    if pending_duration >= emit_interval:
        return True, "interval"
    if idle_tick and idle_for >= max(0.0, pause_seconds):
        return True, "pause"
    return False, ""


def _insight_corrected_segment_md(raw: str, corrected: str) -> str | None:
    """Markdown for collapsible when routing returned a materially different cleaned segment."""
    c = corrected.strip()
    if not c:
        return None
    if " ".join(raw.split()) == " ".join(c.split()):
        return None
    safe = c.replace("```", "′′′")
    return (
        "_Routing-time segment (ASR / wording cleanup; same facts as raw chunk):_\n\n"
        f"```text\n{safe}\n```"
    )


def _insight_segments_signature(segments: list[TranscriptSegment]) -> str:
    parts = [f"{seg.line_id}:{int(seg.finalized)}:{_normalize_spaces(seg.spoken_text)}" for seg in segments]
    return "|".join(parts)


def _has_meaningful_new_content_since_comment(
    *,
    current_text: str,
    previous_comment_input: str,
    min_delta_chars: int,
) -> bool:
    cur = _normalize_spaces(current_text)
    prev = _normalize_spaces(previous_comment_input)
    if not cur:
        return False
    if not prev:
        return True
    if cur == prev:
        return False
    threshold = max(1, min_delta_chars)
    if cur.startswith(prev):
        return len(cur[len(prev) :].strip()) >= threshold
    if prev in cur:
        return len(cur.replace(prev, "", 1).strip()) >= threshold
    return True


def _transcript_lines_after_correction(source_lines: list[str], corrected: str) -> list[str] | None:
    """If ``corrected`` materially changes the chunk, return new Transcript lines; else ``None``."""
    raw = "\n".join(source_lines)
    if _insight_corrected_segment_md(raw, corrected) is None:
        return None
    c = corrected.strip()
    if not c:
        return None
    parts = [ln.strip() for ln in c.splitlines() if ln.strip()]
    if not parts:
        return None
    if len(parts) == len(source_lines):
        out: list[str] = []
        for old, p in zip(source_lines, parts, strict=True):
            if p.lstrip().startswith("["):
                out.append(p)
            else:
                out.append(_leading_ts_prefix(old) + p)
        return out
    if len(parts) == 1 and len(source_lines) > 1:
        return [_leading_ts_prefix(source_lines[0]) + parts[0]]
    out2: list[str] = []
    for i, p in enumerate(parts):
        old = source_lines[min(i, len(source_lines) - 1)]
        if p.startswith("["):
            out2.append(p)
        else:
            pref = _leading_ts_prefix(old)
            out2.append(pref + p if pref else p)
    return out2


def _spoken_lines_after_correction(source_spoken: list[str], corrected: str) -> list[str] | None:
    raw = "\n".join(source_spoken)
    c = corrected.strip()
    if not c:
        return None
    if " ".join(raw.split()) == " ".join(c.split()):
        return None
    parts = [ln.strip() for ln in c.splitlines() if ln.strip()]
    if len(parts) != len(source_spoken):
        return None
    return parts


def _insight_routing_collapsible(route: InsightRouteResult, *, web_path: bool, wiki_path: bool = False) -> str | None:
    if wiki_path:
        return f"**Choice:** knowledge vault lookup\n\n**Reason:** {route.reason or '—'}"
    if web_path:
        return f"**Choice:** web search\n\n**Reason:** {route.reason or '—'}"
    if route.reason in ("", "no_llm"):
        return None
    return f"**Choice:** comment / draft\n\n**Reason:** {route.reason}"


def _format_web_search_md(query: str, hits: list[SearchHit], search_err: str, *, provider: str = "") -> str:
    lines: list[str] = []
    if search_err:
        lines.append(f"> {search_err}")
        lines.append("")
    if query.strip():
        via = f" · _via {provider}_" if provider else ""
        lines.append(f"_Query:_ `{query}`{via}")
        lines.append("")
    if hits:
        for h in hits:
            title = h.title or "(no title)"
            body = (h.body or "").replace("\n", " ")
            if h.href:
                lines.append(f"- [{title}]({h.href}) — {body}")
            else:
                lines.append(f"- **{title}** — {body}")
    else:
        lines.append("- (no results, offline, or error above)")
    return "\n".join(lines).strip()


def _format_wiki_md(query: str, hits: list[WikiHit], err: str, *, vault: str) -> str:
    lines: list[str] = []
    if err:
        lines.extend([f"> {err}", ""])
    if query.strip():
        lines.extend([f"_Query:_ `{query}` · _vault {vault}_", ""])
    if hits:
        for h in hits:
            excerpt = " ".join((h.excerpt or "").split())
            if len(excerpt) > 160:
                excerpt = excerpt[:160].rstrip() + "…"
            lines.append(f"- **{h.title or h.filename}** (`{h.filename}`) — {excerpt}" if excerpt else f"- **{h.title or h.filename}** (`{h.filename}`)")
    else:
        lines.append("- (no matching pages)")
    return "\n".join(lines).strip()


def _initial_insights_markdown() -> str:
    return (
        "## Insights\n\n"
        "Each pass is routed by the **LLM** using **`AGENT.md`** (digest; editable). "
        "Outputs are kept **short for live meetings**: brief **Comment** or compact **Research summary**, "
        "plus optional **Cleaned segment** / **Web search** (collapsible).\n\n"
        "_Waiting for enough text…_\n"
    )


def _llm_unavailable_insights_markdown() -> str:
    return (
        "## Insights\n\n"
        "**LLM unavailable:** no `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / "
        "(`BASE_URL` + `API_KEY` + `MODEL`), or the **TCP probe** to the configured API host failed.\n\n"
        "Insights and chat require an LLM; those requests are disabled here.\n\n"
        "**Transcript** is unaffected: text replay works as usual; whether file/live ASR hits "
        "Hugging Face Hub is controlled separately via `ECHO_OFFLINE`, `ECHO_HF_LOCAL_ONLY`, "
        "`HF_HUB_OFFLINE`, etc.\n\n"
        "_If the API is reachable but the probe is blocked (e.g. corporate split DNS), set "
        "`ECHO_ASSUME_ONLINE=1` to skip LLM reachability checks._\n"
    )


def _load_replay_lines(replay_path: Path | None, default_lines: Iterable[str] = ()) -> Iterable[str]:
    if replay_path is None:
        return list(default_lines)
    text = replay_path.read_text(encoding="utf-8")
    parts = [ln.rstrip("\n") for ln in text.splitlines()]
    return [ln for ln in parts if ln.strip()]
