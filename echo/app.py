"""Textual TUI for Echo — offline transcript replay + insights + chat."""

from __future__ import annotations

import asyncio
import os
import re
import time
from dataclasses import dataclass
from datetime import timedelta
from functools import partial
from pathlib import Path
from typing import Iterable, Literal

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Collapsible, Footer, Header, Input, Markdown, RichLog, Rule

from echo.asr import MODEL_ID_DEFAULT
from echo.connectivity import echo_env_hub_local_only, echo_llm_online, hub_local_only_from_env
from echo.splash import SplashScreen
from echo.llm import (
    InsightRouteResult,
    llm_chat_answer,
    llm_route_insight,
    llm_synthesize_web_findings,
)
from echo.search import SearchHit, duckduckgo_search


@dataclass
class InsightPassRecord:
    n: int
    stamp: str
    kind: Literal["comment", "web_search"]
    body_md: str
    routing_md: str | None = None
    corrected_md: str | None = None
    web_md: str | None = None


_LEADING_BRACKET_TS = re.compile(r"^(\[[^\]]+\]\s*)")


def _leading_ts_prefix(line: str) -> str:
    m = _LEADING_BRACKET_TS.match(line)
    return m.group(1) if m else ""


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


def _insight_routing_collapsible(route: InsightRouteResult, *, web_path: bool) -> str | None:
    if web_path:
        return f"**Choice:** web search\n\n**Reason:** {route.reason or '—'}"
    if route.reason in ("", "no_llm"):
        return None
    return f"**Choice:** comment / draft\n\n**Reason:** {route.reason}"


def _format_web_search_md(query: str, hits: list[SearchHit], ddg_err: str) -> str:
    lines: list[str] = []
    if ddg_err:
        lines.append(f"> {ddg_err}")
        lines.append("")
    if query.strip():
        lines.append(f"_Query:_ `{query}`")
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


DEFAULT_DEMO_LINES: list[str] = [
    "[00:01] 今日の議題は四半期レビューと予算の見直しです。",
    "[00:12] まず予算について、マーケティングは前四半期比で 8% 増です。",
    "[00:28] Q4 の売上は計画を上回りましたが、在庫コストが圧迫しています。",
    "[00:45] 次に、来期のキャップエクスをどう配分するか決めたいです。",
    "[01:02] 予算の再配分案は金曜までにドラフトを共有しましょう。",
    "[01:18] それでは次のトピックに移ります。",
]


class EchoApp(App[None]):
    """Offline-first meeting assistant demo."""

    TITLE = "Echo"
    SUB_TITLE = "🎙 Echo"

    CSS = """
    Screen {
        layout: vertical;
    }
    #main_row {
        height: 1fr;
        min-height: 12;
    }
    #left_column {
        width: 1fr;
        height: 1fr;
        layout: vertical;
    }
    #transcript {
        height: 1fr;
        min-height: 6;
        border: tall $accent;
        margin: 0 1 0 0;
        overflow-x: hidden;
    }
    #insights {
        width: 1fr;
        height: 1fr;
        border: tall $primary;
        margin: 0 0 0 1;
    }
    #insights:focus {
        border: tall $success;
    }
    #insights Markdown {
        width: 100%;
        overflow-x: hidden;
    }
    #bottom {
        height: auto;
        max-height: 16;
        min-height: 6;
        border: tall $success;
        margin-top: 1;
    }
    #chat_log {
        height: 1fr;
        min-height: 3;
        max-height: 11;
        margin-bottom: 1;
        overflow-x: hidden;
    }
    #chat_input {
        dock: bottom;
        height: auto;
    }
    """

    BINDINGS = [
        ("p", "toggle_pause", "Pause"),
        ("s", "save_transcript", "Save"),
        ("c", "clear_chat", "Clear chat"),
        ("i", "focus_insights", "Insights"),
        ("t", "focus_transcript", "Transcript"),
        ("q", "quit", "Quit"),
    ]

    def __init__(
        self,
        replay_path: Path | None = None,
        line_delay: float = 1.0,
        insight_min_chars: int = 120,
        insight_min_interval: float = 25.0,
        audio_path: Path | None = None,
        model_id: str = MODEL_ID_DEFAULT,
        chunk_seconds: float = 30.0,
        whisper_device: str | None = None,
        llm_offline: bool = False,
    ) -> None:
        super().__init__()
        self.replay_path = replay_path
        self.line_delay = line_delay
        self.insight_min_chars = insight_min_chars
        self.insight_min_interval = insight_min_interval
        self.audio_path = audio_path
        self.model_id = model_id
        self.chunk_seconds = chunk_seconds
        self.whisper_device = whisper_device
        self._llm_offline = llm_offline

        self._paused = False
        self._t0 = time.monotonic()
        self._lines: list[str] = []
        self._transcript_log_prefix: list[str] = []
        self._transcript_queue: asyncio.Queue[str] = asyncio.Queue()
        self._insight_buffer: list[str] = []
        self._last_insight_at = time.monotonic()
        self._insight_lock = asyncio.Lock()
        self._eof_sentinel = "__ECHO_EOF__"
        self._insight_passes: list[InsightPassRecord] = []

    def _elapsed_hms(self) -> str:
        dt = timedelta(seconds=int(time.monotonic() - self._t0))
        h, r = divmod(dt.seconds, 3600)
        m, s = divmod(r, 60)
        if dt.days:
            h += dt.days * 24
        return f"{h:02d}:{m:02d}:{s:02d}"

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="main_row"):
            with Vertical(id="left_column"):
                yield RichLog(
                    id="transcript",
                    highlight=True,
                    markup=True,
                    wrap=True,
                    min_width=0,
                )
                with Vertical(id="bottom"):
                    yield RichLog(
                        id="chat_log",
                        highlight=True,
                        markup=True,
                        wrap=True,
                        min_width=0,
                    )
                    yield Input(
                        placeholder=(
                            "LLM unavailable — chat disabled"
                            if self._llm_offline
                            else "Ask about this meeting…"
                        ),
                        id="chat_input",
                        disabled=self._llm_offline,
                    )
            with VerticalScroll(id="insights"):
                yield Markdown(
                    _llm_unavailable_insights_markdown()
                    if self._llm_offline
                    else _initial_insights_markdown()
                )
        yield Footer()

    def on_mount(self) -> None:
        if self._llm_offline:
            self.query_one("#chat_log", RichLog).write(
                "[dim]LLM unavailable (no API keys or TCP probe to the configured host failed): "
                "Insights and chat are disabled. The Transcript panel still works (replay or ASR).[/]"
            )
        if os.environ.get("ECHO_NO_SPLASH", "").lower() not in ("1", "true", "yes"):
            self.push_screen(SplashScreen())
        mode = "ASR file" if self.audio_path else "Text replay"
        off = " · no LLM" if self._llm_offline else ""
        self.sub_title = f"🎙 {mode}{off}  [{self._elapsed_hms()}]  p/s/c/q"
        self.set_interval(1.0, self._tick_header)
        # Keep exclusive=False so the insight loop does not cancel transcription.
        self.run_worker(self._transcription_worker(), exclusive=False)
        self.run_worker(self._insight_worker(), exclusive=False)

    def _tick_header(self) -> None:
        if self.audio_path is not None:
            state = "Paused" if self._paused else "Transcribing"
        else:
            state = "Paused" if self._paused else "Replaying"
        mode = "ASR" if self.audio_path else "Text"
        off = " · no LLM" if self._llm_offline else ""
        self.sub_title = f"🎙 {mode} · {state}{off}  [{self._elapsed_hms()}]  p/s/c/q"

    def action_toggle_pause(self) -> None:
        self._paused = not self._paused

    def action_save_transcript(self) -> None:
        path = Path.cwd() / f"echo_transcript_{int(time.time())}.txt"
        path.write_text("\n".join(self._lines), encoding="utf-8")
        self.notify(f"Saved {path}", timeout=4)

    def action_clear_chat(self) -> None:
        chat = self.query_one("#chat_log", RichLog)
        chat.clear()
        if self._llm_offline:
            chat.write(
                "[dim]LLM unavailable: Insights and chat are disabled. Transcript still works.[/]"
            )

    def action_focus_insights(self) -> None:
        self.query_one("#insights", VerticalScroll).focus()

    def action_focus_transcript(self) -> None:
        self.query_one("#transcript", RichLog).focus()

    def _log_transcript_meta(self, log: RichLog, text: str) -> None:
        """Status / error lines above replay or ASR lines (preserved when transcript body is rewritten)."""
        self._transcript_log_prefix.append(text)
        log.write(text)

    async def _refresh_transcript_log(self) -> None:
        log = self.query_one("#transcript", RichLog)
        log.clear()
        for m in self._transcript_log_prefix:
            log.write(m)
        for line in self._lines:
            log.write(line)

    async def _transcription_worker(self) -> None:
        log = self.query_one("#transcript", RichLog)
        if self.audio_path is not None:
            await self._transcription_from_audio(log)
            await self._transcript_queue.put(self._eof_sentinel)
            return

        lines = _load_replay_lines(self.replay_path)
        for raw in lines:
            while self._paused:
                await asyncio.sleep(0.15)
            line = raw.strip()
            if not line:
                continue
            if not line.startswith("["):
                line = f"[{self._elapsed_hms()}] {line}"
            self._lines.append(line)
            log.write(line)
            await self._transcript_queue.put(line)
            await asyncio.sleep(self.line_delay)
        await self._transcript_queue.put(self._eof_sentinel)

    async def _transcription_from_audio(self, log: RichLog) -> None:
        from echo.asr import TranscribeSession, format_timestamp_hms

        path = self.audio_path
        if path is None or not path.is_file():
            self._log_transcript_meta(log, f"[red]Audio file not found:[/] {path}")
            return

        load_msg = (
            f"[dim]Loading `{self.model_id}` from local cache（不访问 Hugging Face Hub）…[/]"
            if hub_local_only_from_env()
            else f"[dim]Loading `{self.model_id}` (first run downloads ~3GB of weights)…[/]"
        )
        self._log_transcript_meta(log, load_msg)
        try:
            session = await asyncio.to_thread(
                partial(
                    TranscribeSession,
                    path,
                    model_id=self.model_id,
                    chunk_seconds=self.chunk_seconds,
                    device=self.whisper_device,
                ),
            )
        except RuntimeError as e:
            self._log_transcript_meta(log, f"[red]{e}[/]")
            return
        except Exception as e:  # noqa: BLE001
            self._log_transcript_meta(log, f"[red]ASR init failed:[/] {e!s}")
            return

        self._log_transcript_meta(log, "[dim]Transcription started (p = pause between chunks).[/]")
        while True:
            while self._paused:
                await asyncio.sleep(0.15)
            try:
                seg = await asyncio.to_thread(session.next_segment)
            except Exception as e:  # noqa: BLE001
                self._log_transcript_meta(log, f"[red]Chunk failed:[/] {e!s}")
                break
            if seg is None:
                break
            t0, t1, text = seg
            if not text.strip():
                continue
            line = f"[{format_timestamp_hms(t0)}–{format_timestamp_hms(t1)}] {text.strip()}"
            self._lines.append(line)
            log.write(line)
            await self._transcript_queue.put(line)

    async def _insight_worker(self) -> None:
        while True:
            try:
                chunk = await asyncio.wait_for(self._transcript_queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                chunk = ""
            if chunk == self._eof_sentinel:
                source_lines = list(self._insight_buffer)
                buf_text = "\n".join(self._insight_buffer).strip()
                self._insight_buffer.clear()
                if buf_text and not self._llm_offline:
                    asyncio.create_task(self._run_insight_pass(buf_text, source_lines))
                break
            if chunk and not self._llm_offline:
                self._insight_buffer.append(chunk)
            now = time.monotonic()
            buf_text = "\n".join(self._insight_buffer).strip()
            should_run = bool(buf_text) and (
                len(buf_text) >= self.insight_min_chars
                or (now - self._last_insight_at) >= self.insight_min_interval
                and len(buf_text) >= 40
            )
            if chunk == "" and not should_run:
                continue
            if not should_run:
                continue

            snapshot = buf_text
            source_lines = list(self._insight_buffer)
            self._insight_buffer.clear()
            self._last_insight_at = now
            asyncio.create_task(self._run_insight_pass(snapshot, source_lines))

    async def _refresh_insights_panel(self, *, tail: str | None = None) -> None:
        scroll = self.query_one("#insights", VerticalScroll)
        for child in list(scroll.children):
            await child.remove()
        if not self._insight_passes:
            if tail is None:
                md = (
                    _llm_unavailable_insights_markdown()
                    if self._llm_offline
                    else _initial_insights_markdown()
                )
                await scroll.mount(Markdown(md))
            else:
                await scroll.mount(Markdown(f"## Insights\n\n{tail}\n"))
            return
        await scroll.mount(Markdown("## Insights"))
        await scroll.mount(Rule())
        for i, rec in enumerate(self._insight_passes):
            if i:
                await scroll.mount(Rule())
            heading = "Comment" if rec.kind == "comment" else "Research summary"
            await scroll.mount(
                Markdown(
                    f"### Pass {rec.n} @ `{rec.stamp}`\n\n"
                    f"## {heading}\n\n{rec.body_md}\n",
                ),
            )
            if rec.routing_md:
                await scroll.mount(
                    Collapsible(
                        Markdown(rec.routing_md),
                        title="Routing",
                        collapsed=True,
                    ),
                )
            if rec.corrected_md:
                await scroll.mount(
                    Collapsible(
                        Markdown(rec.corrected_md),
                        title="Cleaned segment",
                        collapsed=True,
                    ),
                )
            if rec.web_md:
                await scroll.mount(
                    Collapsible(
                        Markdown(rec.web_md),
                        title="Web search",
                        collapsed=True,
                    ),
                )
        if tail:
            await scroll.mount(Rule())
            await scroll.mount(Markdown(tail))

    def _find_transcript_block_start(self, source_lines: list[str]) -> int:
        """Locate ``source_lines`` as a contiguous block in ``_lines`` (prefer latest match)."""
        n = len(source_lines)
        if n == 0 or len(self._lines) < n:
            return -1
        for i in range(len(self._lines) - n, -1, -1):
            if self._lines[i : i + n] == source_lines:
                return i
        return -1

    def _try_apply_corrected_transcript(self, source_lines: list[str], corrected: str) -> bool:
        """Replace the matching slice of ``_lines`` when routing returns a cleaned chunk. Returns whether applied."""
        new_lines = _transcript_lines_after_correction(source_lines, corrected)
        if not new_lines:
            return False
        start_idx = self._find_transcript_block_start(source_lines)
        if start_idx < 0:
            return False
        n = len(source_lines)
        self._lines[start_idx : start_idx + n] = new_lines
        return True

    async def _run_insight_pass(
        self,
        snapshot: str,
        source_lines: list[str] | None = None,
    ) -> None:
        if self._llm_offline:
            return
        llm_timeout = float(os.environ.get("ECHO_INSIGHT_LLM_TIMEOUT", "120"))
        ddg_timeout = float(os.environ.get("ECHO_INSIGHT_DDG_TIMEOUT", "45"))

        async with self._insight_lock:
            next_n = len(self._insight_passes) + 1
            stamp = self._elapsed_hms()
            tail = (
                "_Updating…_"
                if not self._insight_passes
                else f"_[Generating pass {next_n} @ `{stamp}`…]_\n"
            )
            await self._refresh_insights_panel(tail=tail)

            try:
                try:
                    route = await asyncio.wait_for(
                        llm_route_insight(snapshot),
                        timeout=llm_timeout,
                    )
                except asyncio.TimeoutError:
                    route = InsightRouteResult(
                        "comment",
                        f"_Insight 路由 LLM 超过 {llm_timeout:.0f}s（检查网络、`BASE_URL` / API key 或代理）。_",
                        "",
                        "timeout",
                        "",
                    )
                except Exception as e:  # noqa: BLE001
                    route = InsightRouteResult(
                        "comment",
                        f"_Insight 路由失败:_ `{e!s}`",
                        "",
                        "error",
                        "",
                    )

                ctx = route.corrected_transcript.strip() or snapshot
                raw_for_md = "\n".join(source_lines) if source_lines else snapshot
                corrected_md = _insight_corrected_segment_md(raw_for_md, route.corrected_transcript)
                applied = bool(source_lines) and self._try_apply_corrected_transcript(
                    source_lines,
                    route.corrected_transcript,
                )
                if applied:
                    await self._refresh_transcript_log()
                    corrected_md = None

                if route.action == "comment":
                    body = route.comment.strip() or "_(empty)_"
                    routing = _insight_routing_collapsible(route, web_path=False)
                    self._insight_passes.append(
                        InsightPassRecord(
                            n=next_n,
                            stamp=stamp,
                            kind="comment",
                            body_md=body,
                            routing_md=routing,
                            corrected_md=corrected_md,
                            web_md=None,
                        ),
                    )
                else:
                    query = route.search_query.strip() or ctx[:100].strip()
                    hits: list[SearchHit] = []
                    ddg_err = ""
                    if query:
                        try:
                            hits = await asyncio.wait_for(
                                duckduckgo_search(query, max_results=5),
                                timeout=ddg_timeout,
                            )
                        except asyncio.TimeoutError:
                            ddg_err = (
                                f"ddgs 搜索超过 {ddg_timeout:.0f}s 无响应（网络/限流/代理）。"
                                "可试 ECHO_DDGS_BACKEND=bing。"
                            )
                        except Exception as e:  # noqa: BLE001
                            ddg_err = f"ddgs: `{e!s}`"
                    try:
                        summary = await asyncio.wait_for(
                            llm_synthesize_web_findings(ctx, query, hits),
                            timeout=llm_timeout,
                        )
                    except asyncio.TimeoutError:
                        summary = (
                            f"_综合总结 LLM 超过 {llm_timeout:.0f}s；以下为检索摘录。_\n\n"
                            f"**Query:** `{query}`"
                        )
                    except Exception as e:  # noqa: BLE001
                        summary = f"_综合总结失败:_ `{e!s}`\n\n**Query:** `{query}`"

                    web_md = _format_web_search_md(query, hits, ddg_err)
                    routing = _insight_routing_collapsible(route, web_path=True)
                    self._insight_passes.append(
                        InsightPassRecord(
                            n=next_n,
                            stamp=stamp,
                            kind="web_search",
                            body_md=summary.strip() or "_(empty)_",
                            routing_md=routing,
                            corrected_md=corrected_md,
                            web_md=web_md,
                        ),
                    )
                await self._refresh_insights_panel(tail=None)
            except Exception as e:  # noqa: BLE001
                self.log.exception("insights panel failed")
                self._insight_passes.append(
                    InsightPassRecord(
                        n=next_n,
                        stamp=stamp,
                        kind="comment",
                        body_md=f"```\n{e!s}\n```",
                        routing_md=None,
                        corrected_md=None,
                        web_md=None,
                    ),
                )
                await self._refresh_insights_panel(tail=None)

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "chat_input":
            return
        q = event.value.strip()
        if not q:
            return
        event.input.clear()
        chat = self.query_one("#chat_log", RichLog)
        if self._llm_offline:
            chat.write(f"[bold cyan]You:[/] {q}")
            chat.write("[yellow]当前 LLM 不可用，无法使用聊天。[/]")
            return
        chat.write(f"[bold cyan]You:[/] {q}")
        full = "\n".join(self._lines)
        try:
            ans = await llm_chat_answer(full, q)
        except Exception as e:  # noqa: BLE001 — demo surface errors in UI
            ans = f"(error) {e!s}"
        chat.write(f"[bold green]AI:[/] {ans}")


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
        "当前 **LLM 不可用**：未配置 `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / "
        "（`BASE_URL` + `API_KEY` + `MODEL`），或对已配置 API 主机的 **TCP 探测失败**。\n\n"
        "Insights 与聊天依赖 LLM；此处已关闭相关请求。\n\n"
        "左侧 **Transcript** 不受影响：文本重播照常；语音转写是否访问 Hub 由 "
        "`ECHO_OFFLINE` / `ECHO_HF_LOCAL_ONLY` / `HF_HUB_OFFLINE` 等环境变量单独控制。\n\n"
        "_若 API 实际可达但探测被拦（仅内网等），可设置 `ECHO_ASSUME_ONLINE=1` 跳过 LLM 可达性检测。_\n"
    )


def _load_replay_lines(replay_path: Path | None) -> Iterable[str]:
    if replay_path is None:
        return list(DEFAULT_DEMO_LINES)
    text = replay_path.read_text(encoding="utf-8")
    parts = [ln.rstrip("\n") for ln in text.splitlines()]
    return [ln for ln in parts if ln.strip()]


def _load_dotenv_files() -> None:
    """Load `.env` from repo root, then from cwd (defaults: do not override existing os.environ)."""
    from dotenv import load_dotenv

    repo_root = Path(__file__).resolve().parent.parent
    load_dotenv(repo_root / ".env")
    load_dotenv()


def run_echo(
    replay_path: Path | None = None,
    line_delay: float = 1.0,
    audio_path: Path | None = None,
    model_id: str = MODEL_ID_DEFAULT,
    chunk_seconds: float = 30.0,
    whisper_device: str | None = None,
) -> None:
    _load_dotenv_files()
    if echo_env_hub_local_only():
        os.environ.setdefault("ECHO_HF_LOAD_LOCAL_ONLY", "1")
    llm_ok = echo_llm_online()
    if audio_path is not None:
        import sys

        ap = Path(audio_path)
        if not ap.is_file():
            raise FileNotFoundError(f"Audio file not found: {ap.resolve()}")

        from echo.asr import ensure_model_loaded

        if hub_local_only_from_env():
            print(
                "Echo: ASR loading from local Hugging Face cache only (Hub download disabled)…",
                file=sys.stderr,
                flush=True,
            )
        else:
            print(
                "Echo: loading ASR weights on the main thread (first run may download ~3GB)…",
                file=sys.stderr,
                flush=True,
            )
        # Main-thread load avoids HF subprocess + asyncio.to_thread + Textual (macOS fds_to_keep).
        ensure_model_loaded(model_id, whisper_device)
    EchoApp(
        replay_path=replay_path,
        line_delay=line_delay,
        audio_path=audio_path,
        model_id=model_id,
        chunk_seconds=chunk_seconds,
        whisper_device=whisper_device,
        llm_offline=not llm_ok,
    ).run()
