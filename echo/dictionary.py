"""Optional user-supplied glossary loaded from a single YAML file.

Terms from this file are injected into LLM prompts (transcript correction, web synthesis, chat)
so the model prefers canonical spellings for proper nouns / jargon / names when the transcript
clearly refers to them.

Supported YAML shapes (all flattened to the same deduped, order-preserving list):

  # Form 1 - bare list
  - EchoApp
  - Kotoba-Whisper
  - 石井

  # Form 2 - explicit ``terms`` key
  terms:
    - EchoApp
    - 石井

  # Form 3 - cosmetic groupings (any keys; values may be lists or strings)
  people: [石井, 田中]
  products: [EchoApp, EchoHub]
  acronyms: [FOMC, ARPU]

Path resolution: explicit arg -> ``$ECHO_DICTIONARY_PATH`` -> ``<repo_root>/dictionary.yml`` ->
empty tuple. Loads are mtime-cached so per-chunk callers stay O(1) and live edits picked up
without restart.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Iterable, Sequence

# Caps so a runaway dictionary cannot blow up the prompt budget.
MAX_DICT_TERMS = 200
MAX_DICT_SERIALIZED_CHARS = 600

_PYYAML_HINT_LOGGED = False


def _repo_root_default() -> Path:
    return Path(__file__).resolve().parent.parent / "dictionary.yml"


def _resolve_path(explicit: Path | None) -> Path | None:
    if explicit is not None:
        p = Path(explicit).expanduser()
        return p if p.is_file() else None
    env = (os.environ.get("ECHO_DICTIONARY_PATH") or "").strip()
    if env:
        p = Path(env).expanduser()
        return p if p.is_file() else None
    default = _repo_root_default()
    return default if default.is_file() else None


def _flatten(node: object) -> list[str]:
    """Walk a nested YAML node and yield string leaves (skipping blanks)."""
    out: list[str] = []
    if node is None:
        return out
    if isinstance(node, str):
        s = node.strip()
        if s:
            out.append(s)
        return out
    if isinstance(node, (int, float, bool)):
        out.append(str(node))
        return out
    if isinstance(node, dict):
        for v in node.values():
            out.extend(_flatten(v))
        return out
    if isinstance(node, Iterable):
        for v in node:
            out.extend(_flatten(v))
        return out
    return out


def _dedupe_preserve_order(items: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    out: list[str] = []
    for it in items:
        if it in seen:
            continue
        seen.add(it)
        out.append(it)
    return tuple(out)


def _maybe_warn_pyyaml_missing(notify: Callable[[str], None] | None) -> None:
    global _PYYAML_HINT_LOGGED
    if _PYYAML_HINT_LOGGED:
        return
    _PYYAML_HINT_LOGGED = True
    msg = (
        "dictionary.yml found but PyYAML is not installed; install with `pip install pyyaml` "
        "to enable the glossary feature. Continuing without it."
    )
    if notify is not None:
        try:
            notify(msg)
        except Exception:  # noqa: BLE001 - debug hook must never break loaders
            pass


# Cache: resolved_path -> (mtime, terms tuple)
_cache: dict[Path, tuple[float, tuple[str, ...]]] = {}


def load_dictionary(
    path: Path | None = None,
    *,
    notify: Callable[[str], None] | None = None,
) -> tuple[str, ...]:
    """Resolve, parse, flatten, dedupe and cap a glossary YAML.

    Returns ``()`` on every error path: missing file, missing PyYAML, malformed YAML, etc. The
    optional ``notify`` callback receives a single human-readable warning string for the
    PyYAML-missing case (logged at most once per process).
    """
    resolved = _resolve_path(path)
    if resolved is None:
        return ()

    try:
        mtime = resolved.stat().st_mtime
    except OSError:
        return ()

    cached = _cache.get(resolved)
    if cached is not None and cached[0] == mtime:
        return cached[1]

    try:
        import yaml  # type: ignore[import-not-found]
    except ImportError:
        _maybe_warn_pyyaml_missing(notify)
        return ()

    try:
        raw = resolved.read_text(encoding="utf-8")
        data = yaml.safe_load(raw)
    except (OSError, yaml.YAMLError):
        return ()

    if isinstance(data, dict) and "terms" in data and len(data) == 1:
        flat = _flatten(data["terms"])
    else:
        flat = _flatten(data)

    deduped = _dedupe_preserve_order(flat)

    if len(deduped) > MAX_DICT_TERMS:
        deduped = deduped[:MAX_DICT_TERMS]

    serialized = ", ".join(deduped)
    if len(serialized) > MAX_DICT_SERIALIZED_CHARS:
        kept: list[str] = []
        running = 0
        for term in deduped:
            extra = len(term) + (2 if kept else 0)
            if running + extra > MAX_DICT_SERIALIZED_CHARS:
                break
            kept.append(term)
            running += extra
        deduped = tuple(kept)

    _cache[resolved] = (mtime, deduped)
    return deduped


def reset_cache() -> None:
    """Test helper: drop the mtime cache and the PyYAML warning latch."""
    global _PYYAML_HINT_LOGGED
    _cache.clear()
    _PYYAML_HINT_LOGGED = False
