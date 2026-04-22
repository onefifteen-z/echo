"""CLI: quick connectivity test for LLM + web search (ddgs) (no TUI)."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path


def _load_dotenv_files() -> None:
    from dotenv import load_dotenv

    root = Path(__file__).resolve().parent.parent
    load_dotenv(root / ".env")
    load_dotenv()


async def _probe_llm(llm_timeout: float) -> tuple[bool, str]:
    from echo.llm import configured_llm_label, llm_keywords_and_summary

    snippet = (
        "Q4 budget review: marketing spend up 8%. "
        "Decision: share redistribution draft by Friday."
    )
    t0 = time.monotonic()
    try:
        kws, summary = await asyncio.wait_for(
            llm_keywords_and_summary(snippet),
            timeout=llm_timeout,
        )
    except asyncio.TimeoutError:
        return False, f"timeout after {llm_timeout:.0f}s"
    except Exception as e:  # noqa: BLE001
        return False, str(e)
    dt = time.monotonic() - t0
    route = configured_llm_label()
    lines = [
        f"route={route}  ({dt:.1f}s)",
        f"keywords ({len(kws)}): {', '.join(kws) if kws else '(empty)'}",
        f"summary: {summary[:400]}{'…' if len(summary) > 400 else ''}",
    ]
    return True, "\n".join(lines)


async def _probe_ddg(ddg_timeout: float, query: str) -> tuple[bool, str]:
    from echo.search import duckduckgo_search

    t0 = time.monotonic()
    try:
        hits = await asyncio.wait_for(
            duckduckgo_search(query, max_results=3),
            timeout=ddg_timeout,
        )
    except asyncio.TimeoutError:
        return False, f"timeout after {ddg_timeout:.0f}s"
    except Exception as e:  # noqa: BLE001
        return False, str(e)
    dt = time.monotonic() - t0
    if not hits:
        return True, (
            f"OK ({dt:.1f}s) but zero results. Try `ECHO_DDGS_BACKEND=bing` "
            f"(or another backend from the ddgs package); check network / VPN."
        )
    parts = [f"OK ({dt:.1f}s), {len(hits)} hit(s):"]
    for h in hits:
        t = (h.title or "(no title)")[:80]
        parts.append(f"  - {t}")
    return True, "\n".join(parts)


async def _run_async(*, llm_timeout: float, ddg_timeout: float, ddg_query: str) -> int:
    from echo.llm import configured_llm_label

    failed = 0
    route = configured_llm_label()
    print(f"Configured LLM route: {route}")
    if route == "none":
        print("(No API keys: LLM step uses local keyword heuristic only.)\n")

    print("=== LLM (llm_keywords_and_summary) ===")
    ok, msg = await _probe_llm(llm_timeout)
    print(msg if ok else f"FAIL: {msg}")
    if not ok:
        failed += 1

    print("\n=== Web search (ddgs) ===")
    ok, msg = await _probe_ddg(ddg_timeout, ddg_query)
    print(msg if ok else f"FAIL: {msg}")
    if not ok:
        failed += 1

    return failed


def main() -> None:
    p = argparse.ArgumentParser(
        prog="echo-probe",
        description="Test Echo LLM + ddgs web-search connectivity (loads .env like echo-tui).",
    )
    p.add_argument(
        "--llm-timeout",
        type=float,
        default=float(os.environ.get("ECHO_INSIGHT_LLM_TIMEOUT", "120")),
        help="Seconds for LLM test (default: ECHO_INSIGHT_LLM_TIMEOUT or 120).",
    )
    p.add_argument(
        "--ddg-timeout",
        type=float,
        default=float(os.environ.get("ECHO_INSIGHT_DDG_TIMEOUT", "45")),
        help="Seconds for ddgs search test (default: ECHO_INSIGHT_DDG_TIMEOUT or 45).",
    )
    p.add_argument(
        "--ddg-query",
        default="Echo meeting assistant open source",
        help="Search query for ddgs test.",
    )
    args = p.parse_args()

    _load_dotenv_files()

    failed = asyncio.run(
        _run_async(
            llm_timeout=args.llm_timeout,
            ddg_timeout=args.ddg_timeout,
            ddg_query=args.ddg_query,
        )
    )
    if failed:
        print(f"\nDone with {failed} failure(s).", file=sys.stderr)
        sys.exit(1)
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
