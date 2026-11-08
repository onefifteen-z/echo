"""Read-only access to Lumina vaults (LLM-maintained wikis) through Lumina's MCP stdio CLI.

Echo starts ``Lumina.app``'s bundled CLI (``ELECTRON_RUN_AS_NODE=1 Lumina cli.js --vault <path>``)
lazily, one process per vault in use, and speaks newline-delimited JSON-RPC (MCP) to it. Lumina
itself does not need to be running. The vault list comes from Lumina's own vault history.

Environment overrides:
- ``ECHO_LUMINA_APP``: path to ``Lumina.app`` (default ``/Applications/Lumina.app``).
- ``ECHO_LUMINA_MCP_CMD``: full CLI command (shell-style), e.g. a dev build of Lumina; ``--vault`` is appended.
- ``ECHO_LUMINA_USER_DATA``: Lumina's userData folder (default ``~/Library/Application Support/Lumina``).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import shlex
import subprocess
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROTOCOL_VERSION = "2025-06-18"
START_TIMEOUT_S = 90.0  # the first launch after install can be slow (Gatekeeper scan, index load)
SEARCH_TIMEOUT_S = 8.0
ARTICLE_TIMEOUT_S = 5.0
MAX_PROCESSES = 2


class WikiError(Exception):
    pass


@dataclass
class WikiHit:
    title: str
    filename: str
    excerpt: str

    @property
    def is_page(self) -> bool:
        return self.filename.startswith("wiki/")


@dataclass
class WikiPage:
    title: str
    filename: str
    body: str


def lumina_app() -> Path:
    return Path(os.environ.get("ECHO_LUMINA_APP") or "/Applications/Lumina.app").expanduser()


def lumina_user_data() -> Path:
    raw = os.environ.get("ECHO_LUMINA_USER_DATA") or "~/Library/Application Support/Lumina"
    return Path(raw).expanduser()


def cli_command() -> list[str]:
    if cmd := (os.environ.get("ECHO_LUMINA_MCP_CMD") or "").strip():
        return shlex.split(cmd)
    app = lumina_app()
    return [str(app / "Contents" / "MacOS" / "Lumina"), str(app / "Contents" / "Resources" / "app.asar" / "dist" / "main" / "mcp" / "cli.js")]


def unavailable_reason() -> str | None:
    """None when the CLI can be started; otherwise a user-facing reason."""
    if (os.environ.get("ECHO_LUMINA_MCP_CMD") or "").strip():
        return None
    exe = Path(cli_command()[0])
    if not exe.exists():
        return f"Lumina was not found at {lumina_app()}. Install Lumina or set ECHO_LUMINA_APP."
    return None


def list_vaults() -> list[dict[str, Any]]:
    """Lumina's vault history: ``[{name, path, default, initialized}]`` (default = last opened)."""
    cfg_path = lumina_user_data() / "llm-wiki-config.json"
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    current = str(cfg.get("vaultPath") or "")
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    entries = list(cfg.get("vaultHistory") or [])
    if current and not any(str(e.get("path")) == current for e in entries if isinstance(e, dict)):
        entries.insert(0, {"path": current})
    for e in entries:
        if not isinstance(e, dict) or not e.get("path"):
            continue
        p = str(e["path"])
        if p in seen:
            continue
        seen.add(p)
        out.append(
            {
                "name": str(e.get("name") or Path(p).name),
                "path": p,
                "default": p == current,
                "initialized": (Path(p) / "wiki").is_dir(),
            }
        )
    return out


def resolve_vault(vault: str) -> dict[str, Any]:
    """Match a vault by name (case-insensitive) or path among Lumina's vaults."""
    key = (vault or "").strip()
    vaults = list_vaults()
    for v in vaults:
        if v["path"] == key:
            return v
    named = [v for v in vaults if v["name"].lower() == key.lower()]
    if len(named) == 1:
        return named[0]
    if named:
        raise WikiError(f"Several Lumina vaults are named {key!r}; choose one by path.")
    raise WikiError(f"Lumina vault {key!r} is not in Lumina's vault list.")


def _tool_json(result: dict[str, Any]) -> Any:
    texts = [c.get("text", "") for c in result.get("content") or [] if c.get("type") == "text"]
    text = "\n".join(texts).strip()
    if result.get("isError"):
        raise WikiError(text or "Lumina tool error")
    try:
        return json.loads(text) if text else None
    except json.JSONDecodeError:
        return text


class _McpProcess:
    """One MCP stdio server process (one vault).

    Uses ``subprocess.Popen`` plus a reader thread rather than asyncio subprocesses: cancelling
    ``asyncio.create_subprocess_exec`` while a short-lived child exits can hang the event loop.
    """

    def __init__(self, cmd: list[str]) -> None:
        self.cmd = cmd
        self.proc: subprocess.Popen[bytes] | None = None
        self._next_id = 0
        self._pending: dict[int, asyncio.Future[dict[str, Any]]] = {}
        self._eof = True
        self._ready = False
        self._start_lock = asyncio.Lock()

    @property
    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None and not self._eof

    async def ensure_started(self) -> None:
        async with self._start_lock:
            if self.alive and self._ready:
                return
            await self.close()
            env = {**os.environ, "ELECTRON_RUN_AS_NODE": "1"}
            try:
                proc = subprocess.Popen(  # noqa: ASYNC220 — spawn is quick; see class docstring
                    self.cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=env
                )
            except OSError as e:
                raise WikiError(f"Could not start Lumina's MCP server: {e}") from e
            self.proc, self._eof = proc, False
            loop = asyncio.get_running_loop()
            threading.Thread(target=self._read_thread, args=(proc, loop), daemon=True, name="echo-lumina-mcp").start()
            await self._request(
                "initialize",
                {"protocolVersion": PROTOCOL_VERSION, "capabilities": {}, "clientInfo": {"name": "echo", "version": "0.1"}},
                START_TIMEOUT_S,
            )
            self._write({"jsonrpc": "2.0", "method": "notifications/initialized"})
            self._ready = True

    def _write(self, msg: dict[str, Any]) -> None:
        if self.proc is None or self.proc.stdin is None or self._eof:
            raise WikiError("Lumina's MCP server exited.")
        try:
            self.proc.stdin.write((json.dumps(msg, ensure_ascii=False) + "\n").encode())
            self.proc.stdin.flush()
        except (OSError, ValueError) as e:
            raise WikiError("Lumina's MCP server exited.") from e

    def _read_thread(self, proc: subprocess.Popen[bytes], loop: asyncio.AbstractEventLoop) -> None:
        assert proc.stdout
        try:
            for line in proc.stdout:
                with contextlib.suppress(RuntimeError):
                    loop.call_soon_threadsafe(self._on_line, proc, line)
        except (OSError, ValueError):
            pass
        with contextlib.suppress(RuntimeError):
            loop.call_soon_threadsafe(self._on_eof, proc)

    def _on_line(self, proc: subprocess.Popen[bytes], line: bytes) -> None:
        if proc is not self.proc:
            return
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            return
        fut = self._pending.pop(msg.get("id"), None) if isinstance(msg, dict) else None
        if fut and not fut.done():
            fut.set_result(msg)

    def _on_eof(self, proc: subprocess.Popen[bytes]) -> None:
        if proc is not self.proc:
            return
        self._eof, self._ready = True, False
        for fut in self._pending.values():
            if not fut.done():
                fut.set_exception(WikiError("Lumina's MCP server exited."))
        self._pending.clear()

    async def _request(self, method: str, params: dict[str, Any], timeout: float) -> dict[str, Any]:
        self._next_id += 1
        rid = self._next_id
        fut: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._pending[rid] = fut
        try:
            self._write({"jsonrpc": "2.0", "id": rid, "method": method, "params": params})
            msg = await asyncio.wait_for(fut, timeout)
        except asyncio.TimeoutError as e:
            raise WikiError(f"Lumina did not answer {method} within {timeout:.0f}s.") from e
        finally:
            self._pending.pop(rid, None)
        if err := msg.get("error"):
            raise WikiError(str(err.get("message") or err))
        return msg.get("result") or {}

    async def call_tool(self, name: str, arguments: dict[str, Any], timeout: float) -> Any:
        await self.ensure_started()
        return _tool_json(await self._request("tools/call", {"name": name, "arguments": arguments}, timeout))

    async def close(self) -> None:
        proc, self.proc = self.proc, None
        self._eof, self._ready = True, False
        for fut in self._pending.values():
            if not fut.done():
                fut.set_exception(WikiError("Lumina's MCP server was closed."))
        self._pending.clear()
        if proc is None:
            return
        with contextlib.suppress(OSError, ValueError):
            assert proc.stdin
            proc.stdin.close()
        for _ in range(60):
            if proc.poll() is not None:
                return
            await asyncio.sleep(0.05)
        proc.kill()
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(1)


class LuminaWiki:
    """Lazily started per-vault MCP clients (at most ``MAX_PROCESSES`` alive, LRU)."""

    def __init__(self) -> None:
        self._procs: OrderedDict[str, _McpProcess] = OrderedDict()
        self._titles: dict[str, list[str]] = {}

    def vaults(self) -> dict[str, Any]:
        reason = unavailable_reason()
        out: dict[str, Any] = {"available": reason is None, "vaults": list_vaults() if reason is None else []}
        if reason:
            out["reason"] = reason
        elif not out["vaults"]:
            out["reason"] = "Lumina has no vaults yet. Open or create one in Lumina first."
        return out

    async def _proc(self, vault: str) -> _McpProcess:
        if reason := unavailable_reason():
            raise WikiError(reason)
        path = resolve_vault(vault)["path"]
        proc = self._procs.pop(path, None) or _McpProcess([*cli_command(), "--vault", path])
        self._procs[path] = proc
        while len(self._procs) > MAX_PROCESSES:
            _old_path, old = self._procs.popitem(last=False)
            await old.close()
        return proc

    async def _call(self, vault: str, tool: str, args: dict[str, Any], timeout: float) -> Any:
        proc = await self._proc(vault)
        try:
            return await proc.call_tool(tool, args, timeout)
        except WikiError:
            if proc.alive:
                raise
            await proc.close()
            return await proc.call_tool(tool, args, timeout)

    async def search(self, vault: str, query: str, limit: int = 5) -> list[WikiHit]:
        rows = await self._call(vault, "search_wiki", {"query": query, "limit": limit}, SEARCH_TIMEOUT_S)
        hits = [
            WikiHit(title=str(r.get("title") or r.get("filename") or ""), filename=str(r.get("filename") or ""), excerpt=str(r.get("excerpt") or ""))
            for r in (rows if isinstance(rows, list) else [])
            if isinstance(r, dict) and r.get("filename")
        ]
        # Wiki pages first; older Lumina builds also return raw/ sources.
        return sorted(hits, key=lambda h: not h.is_page)[:limit]

    async def article(self, vault: str, filename: str, max_chars: int = 2500) -> WikiPage:
        data = await self._call(vault, "get_article", {"filename": filename}, ARTICLE_TIMEOUT_S)
        if not isinstance(data, dict):
            raise WikiError(f"Unexpected article payload for {filename}")
        fm = data.get("frontmatter") if isinstance(data.get("frontmatter"), dict) else {}
        body = str(data.get("body") or "").strip()
        if len(body) > max_chars:
            body = body[:max_chars].rstrip() + "\n…"
        return WikiPage(title=str(fm.get("title") or Path(filename).stem), filename=str(data.get("filename") or filename), body=body)

    async def titles(self, vault: str, limit: int = 80) -> list[str]:
        """Recently modified page titles (cached per vault) so the router can judge relevance."""
        key = resolve_vault(vault)["path"]
        if key not in self._titles:
            data = await self._call(vault, "list_articles", {"limit": limit}, SEARCH_TIMEOUT_S)
            rows = data.get("articles") if isinstance(data, dict) else data
            self._titles[key] = [str(r.get("title") or r.get("filename")) for r in (rows or []) if isinstance(r, dict)]
        return self._titles[key]

    async def lookup(self, vault: str, query: str, *, limit: int = 5, read: int = 3) -> tuple[list[WikiHit], list[WikiPage]]:
        """Search, then read the top ``read`` wiki pages (other hits keep only their excerpt)."""
        hits = await self.search(vault, query, limit)
        pages: list[WikiPage] = []
        for h in [h for h in hits if h.is_page][:read]:
            try:
                pages.append(await self.article(vault, h.filename))
            except WikiError:
                continue
        return hits, pages

    async def close(self) -> None:
        procs = list(self._procs.values())
        self._procs.clear()
        for p in procs:
            await p.close()
