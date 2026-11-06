"""CLI: environment, ASR, LLM, and web search (Perplexity / ddgs) checks (no TUI)."""

from __future__ import annotations

import argparse
import asyncio
import os
import subprocess
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
    from echo.search import web_search

    t0 = time.monotonic()
    try:
        outcome = await asyncio.wait_for(
            web_search(query, max_results=3),
            timeout=ddg_timeout,
        )
    except asyncio.TimeoutError:
        return False, f"timeout after {ddg_timeout:.0f}s"
    except Exception as e:  # noqa: BLE001
        return False, str(e)
    dt = time.monotonic() - t0
    hits = outcome.hits
    head = [f"provider={outcome.provider}"] + ([f"note: {outcome.note}"] if outcome.note else [])
    if not hits:
        return True, "\n".join(head) + "\n" + (
            f"OK ({dt:.1f}s) but zero results. Try `ECHO_DDGS_BACKEND=bing` "
            f"(or another backend from the ddgs package); check network / VPN."
        )
    parts = head + [f"OK ({dt:.1f}s), {len(hits)} hit(s):"]
    for h in hits:
        t = (h.title or "(no title)")[:80]
        parts.append(f"  - {t}")
    return True, "\n".join(parts)


def _probe_environment() -> tuple[bool, str]:
    from echo.preflight import LAZY_NATIVE_MODULES, LIVE_NATIVE_MODULES, check_imports_in_child

    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    lines = [
        f"python={sys.version.split()[0]}  executable={sys.executable}",
        f"prefix={sys.prefix}  virtualenv={'yes' if in_venv else 'no (base interpreter)'}",
    ]
    ok, msg = check_imports_in_child(LAZY_NATIVE_MODULES + LIVE_NATIVE_MODULES)
    lines.append("native imports: ok" if ok else f"native imports: FAIL\n{msg}")
    return ok, "\n".join(lines)


def _codesign_arm64_failures(roots: list[Path]) -> list[str]:
    """Native modules whose arm64 slice is unsigned or has a modified signature."""
    import platform
    import shutil

    if sys.platform != "darwin" or platform.machine() != "arm64" or not shutil.which("codesign"):
        return []
    bad: list[str] = []
    for root in roots:
        if not root.is_dir():
            continue
        for f in root.rglob("*"):
            if f.suffix not in (".so", ".dylib") or not f.is_file():
                continue
            proc = subprocess.run(
                ["codesign", "-v", "--arch", "arm64", str(f)],
                capture_output=True,
                text=True,
            )
            err = (proc.stderr or "").lower()
            if proc.returncode != 0 and "format unrecognized" not in err and "unsuitable" not in err:
                first = (proc.stderr or "").strip().splitlines()
                bad.append(f"{f}: {first[0] if first else 'invalid'}")
    return bad


def _probe_codesign() -> tuple[bool, str]:
    import sysconfig

    roots = [
        Path(sysconfig.get_paths()["purelib"]),
        Path(sysconfig.get_paths()["platlib"]),
        Path(sysconfig.get_paths()["stdlib"]) / "lib-dynload",
    ]
    bad = _codesign_arm64_failures(list(dict.fromkeys(roots)))
    if not bad:
        return True, "arm64 signatures: ok (or not applicable on this platform)"
    return False, "arm64 signature problems (macOS may SIGKILL Echo when these load):\n" + "\n".join(
        f"  - {b}" for b in bad[:20]
    )


def _probe_apple_speech() -> tuple[bool, str]:
    if sys.platform != "darwin":
        return True, "skipped (not macOS)"
    from echo.asr_apple import apple_speech_status

    try:
        status = apple_speech_status(("ja_JP", "en_US", "zh_CN"))
    except Exception as e:  # noqa: BLE001
        return False, f"helper failed: {e}"
    return True, "\n".join(f"{loc}: {state}" for loc, state in status.items())


def _probe_whisper_api() -> tuple[bool, str]:
    from echo.asr_api import transcribe_mono16k_chunk_api, whisper_api_config

    cfg = whisper_api_config()
    if not cfg:
        return True, "not configured (set ECHO_WHISPER_API_BASE_URL + API_KEY, or OPENAI_API_KEY)"
    import numpy as np

    base, _key, model = cfg
    t0 = time.monotonic()
    try:
        tone = (0.1 * np.sin(np.linspace(0, 2 * np.pi * 440, 16000))).astype(np.float32)
        transcribe_mono16k_chunk_api(tone, language="english")
    except Exception as e:  # noqa: BLE001
        return False, f"{base} model={model}: {e}"
    return True, f"OK ({time.monotonic() - t0:.1f}s) base={base} model={model}"


async def _run_async(
    *,
    llm_timeout: float,
    ddg_timeout: float,
    ddg_query: str,
    skip_asr: bool = False,
) -> int:
    from echo.llm import configured_llm_label

    failed = 0
    for title, fn in (
        ("Environment", _probe_environment),
        ("Native code signatures", _probe_codesign),
    ) + (
        ()
        if skip_asr
        else (("Apple on-device speech", _probe_apple_speech), ("Whisper API", _probe_whisper_api))
    ):
        print(f"=== {title} ===")
        ok, msg = await asyncio.to_thread(fn)
        print(msg if ok else f"FAIL: {msg}")
        print()
        if not ok:
            failed += 1

    route = configured_llm_label()
    print(f"Configured LLM route: {route}")
    if route == "none":
        print("(No API keys: LLM step uses local keyword heuristic only.)\n")

    print("=== LLM (llm_keywords_and_summary) ===")
    ok, msg = await _probe_llm(llm_timeout)
    print(msg if ok else f"FAIL: {msg}")
    if not ok:
        failed += 1

    from echo.search import active_search_provider

    print(f"\n=== Web search ({active_search_provider()}, ddgs fallback) ===")
    ok, msg = await _probe_ddg(ddg_timeout, ddg_query)
    print(msg if ok else f"FAIL: {msg}")
    if not ok:
        failed += 1

    return failed


def main() -> None:
    p = argparse.ArgumentParser(
        prog="echo-probe",
        description="Test Echo LLM + web-search (Perplexity / ddgs) connectivity (loads .env like echo-tui).",
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
        help="Seconds for web search test (default: ECHO_INSIGHT_DDG_TIMEOUT or 45).",
    )
    p.add_argument(
        "--ddg-query",
        default="Echo meeting assistant open source",
        help="Search query for the web search test.",
    )
    p.add_argument(
        "--skip-asr",
        action="store_true",
        help="Skip the Apple speech and Whisper API checks.",
    )
    args = p.parse_args()

    _load_dotenv_files()

    failed = asyncio.run(
        _run_async(
            llm_timeout=args.llm_timeout,
            ddg_timeout=args.ddg_timeout,
            ddg_query=args.ddg_query,
            skip_asr=args.skip_asr,
        )
    )
    if failed:
        print(f"\nDone with {failed} failure(s).", file=sys.stderr)
        sys.exit(1)
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
