"""Startup splash: Braille-style ECHO (Kiro-like) + short hints."""

from __future__ import annotations

from textual import events
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Static

_GAP_COLS = 4  # pixel columns between letters (after 2× upscale → 2 braille cells)

# 5×N “pixel” letters (# = ink); will be 2× upscaled then Braille-encoded
_LETTERS: dict[str, list[str]] = {
    "E": ["#####", "#    ", "#####", "#    ", "#####"],
    "C": [" ####", "#    ", "#    ", "#    ", " ####"],
    "H": ["#   #", "#   #", "#####", "#   #", "#   #"],
    "O": [" ### ", "#   #", "#   #", "#   #", " ### "],
}


def _upscale_2x(lines: list[str]) -> list[str]:
    """Each character becomes a 2×2 block (higher-res dot look)."""
    out: list[str] = []
    for row in lines:
        doubled = "".join(ch * 2 for ch in row)
        out.append(doubled)
        out.append(doubled)
    return out


def _join_letters(keys: str) -> list[str]:
    blocks = [_upscale_2x(_LETTERS[k]) for k in keys]
    h = len(blocks[0])
    gap = "." * _GAP_COLS
    rows: list[str] = []
    for y in range(h):
        parts: list[str] = []
        for i, b in enumerate(blocks):
            parts.append(b[y])
            if i < len(blocks) - 1:
                parts.append(gap)
        rows.append("".join(parts))
    return rows


def _pad_bitmap(rows: list[str]) -> list[str]:
    w = max(len(r) for r in rows)
    if w % 2:
        w += 1
    padded = [r.ljust(w, ".") for r in rows]
    while len(padded) % 4:
        padded.append("." * w)
    return padded


def _bitmap_to_braille(rows: list[str]) -> str:
    """2×4 True pixels → one Braille codepoint (U+2800..U+28FF)."""
    h = len(rows)
    w = len(rows[0])
    out_lines: list[str] = []
    for y0 in range(0, h, 4):
        line_chars: list[str] = []
        for x0 in range(0, w, 2):
            bits = 0
            # Unicode Braille dot → bit (standard 8-dot layout)
            mapping = (
                (0, 0, 0x01),
                (1, 0, 0x02),
                (2, 0, 0x04),
                (0, 1, 0x08),
                (1, 1, 0x10),
                (2, 1, 0x20),
                (3, 0, 0x40),
                (3, 1, 0x80),
            )
            for dy, dx, mask in mapping:
                y, x = y0 + dy, x0 + dx
                ch = rows[y][x]
                if ch not in (".", " ", "\0"):
                    bits |= mask
            line_chars.append(chr(0x2800 + bits))
        out_lines.append("".join(line_chars))
    return "\n".join(out_lines)


def braille_echo_logo() -> str:
    raw = _join_letters("ECHO")
    # '.' = empty; '#' and anything else = ink
    rows = [r.replace(" ", ".") for r in raw]
    rows = _pad_bitmap(rows)
    return _bitmap_to_braille(rows)


def splash_hint_markup() -> str:
    # Rich: use [#rrggbb]...[/] only — [dim #hex] is invalid (breaks [/dim]).
    return (
        "[#8b949e]Meeting transcript · web search · LLM chat — [/]"
        "[bold #3fb950]echo-tui[/] "
        "[#8b949e]to run · [/]"
        "[bold #3fb950]echo-probe[/] "
        "[#8b949e]to test APIs[/]\n"
        "[#6e7681]Press any key or wait — auto-dismiss in a few seconds[/]"
    )


class SplashScreen(ModalScreen[None]):
    """Braille-style ECHO (green) + Kiro-like hint lines."""

    DEFAULT_CSS = """
    SplashScreen {
        align: center middle;
        background: #0d1117;
        &:ansi {
            background: #0d1117;
        }
    }
    #splash-wrap {
        width: auto;
        height: auto;
        min-width: 28;
        max-width: 96;
        padding: 2 4;
        background: #161b22;
        border: solid #30363d;
    }
    #splash-art {
        color: #3fb950;
        text-style: bold;
        text-align: center;
        margin-bottom: 1;
    }
    #splash-hint {
        text-align: center;
        width: 1fr;
        min-width: 40;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self._gone = False

    def compose(self) -> ComposeResult:
        with Vertical(id="splash-wrap"):
            yield Static(braille_echo_logo(), id="splash-art", markup=False)
            yield Static(splash_hint_markup(), id="splash-hint", markup=True)

    def on_mount(self) -> None:
        self.set_timer(3.2, self._auto_dismiss)

    def _dismiss(self) -> None:
        if self._gone:
            return
        self._gone = True
        self.dismiss()

    def _auto_dismiss(self) -> None:
        self._dismiss()

    def on_key(self, _event: events.Key) -> None:
        self._dismiss()

    def on_click(self, _event: events.Click) -> None:
        self._dismiss()
