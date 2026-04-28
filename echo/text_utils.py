"""Small text and timestamp formatting helpers shared by ASR and the TUI."""

from __future__ import annotations


def format_timestamp_hms(seconds: float) -> str:
    """Format a non-negative number of seconds as ``HH:MM:SS``."""
    t = int(max(0.0, float(seconds)))
    h, r = divmod(t, 3600)
    m, s = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"
