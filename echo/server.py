"""``echo-server``: WebSocket sidecar that exposes ``EchoSession`` to the Electron GUI.

Startup prints exactly one JSON line on stdout — ``{"port": N, "token": "..."}`` — then serves
``ws://127.0.0.1:N/?token=...``. Connections without the token are rejected with HTTP 401.

Protocol (JSON text frames):

* client → server: ``{"id": 1, "cmd": "session.start", "args": {...}}``
* server → client: ``{"type": "response", "id": 1, "ok": true, "result": {...}}``
  or ``{"type": "response", "id": 1, "ok": false, "error": "..."}``
* server → client (broadcast): ``{"type": "event", "event": {"type": "transcript.draft", ...}}``

The sidecar exits when stdin closes (the parent Electron process went away).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import secrets
import signal
import sys
import threading
from dataclasses import asdict
from http import HTTPStatus
from pathlib import Path
from typing import Any, Awaitable, Callable
from urllib.parse import parse_qs, urlparse

from echo.engine.events import EngineEvent
from echo.engine.session import LIVE_ASR_LANGUAGES, EchoSession
from echo.wiki import LuminaWiki, WikiError, resolve_vault

PROTOCOL_VERSION = 1

_LANG_ALIASES = {"ja": "japanese", "en": "english", "zh": "chinese"}


def _norm_language(raw: object) -> str:
    s = str(raw or "").strip().lower()
    return _LANG_ALIASES.get(s, s) or "japanese"


_SECRET_HINTS = ("KEY", "TOKEN", "SECRET", "PASSWORD")
APP_SETTINGS_KEYS_ENV = "ECHO_APP_SETTINGS_KEYS"


def _mask(value: str) -> str:
    v = value.strip()
    if len(v) <= 8:
        return "•" * len(v)
    return f"{v[:3]}…{v[-4:]}"


def effective_config(keys: list[str]) -> dict[str, dict[str, Any]]:
    """Value + origin for each env key: ``app`` (desktop settings), ``env`` (shell / .env) or ``default``."""
    app_keys = {k for k in (os.environ.get(APP_SETTINGS_KEYS_ENV) or "").split(",") if k}
    out: dict[str, dict[str, Any]] = {}
    for key in keys:
        secret = any(h in key for h in _SECRET_HINTS)
        raw = os.environ.get(key)
        if raw is None or (raw == "" and key not in app_keys):
            out[key] = {"value": None, "source": "default", "secret": secret}
            continue
        out[key] = {
            "value": _mask(raw) if secret and raw else raw,
            "source": "app" if key in app_keys else "env",
            "secret": secret,
        }
    return out


def _search_probe() -> dict[str, Any]:
    from echo.search import active_search_provider, perplexity_config

    note = ""
    try:
        configured = perplexity_config() is not None
    except ValueError as e:
        configured, note = False, str(e)
    return {"provider": active_search_provider(), "perplexity_configured": configured, "note": note}


class CommandError(Exception):
    """Expected, user-facing command failure (sent back as ``ok: false``)."""


class EchoServer:
    def __init__(self, token: str, *, output_root: Path | None = None) -> None:
        self.token = token
        self.output_root = output_root
        self.session: EchoSession | None = None
        self._unsubscribe: Callable[[], None] | None = None
        self._clients: dict[Any, asyncio.Queue[str]] = {}
        self._starting = asyncio.Lock()
        self.wiki = LuminaWiki()
        self._handlers: dict[str, Callable[[dict[str, Any]], Awaitable[Any]]] = {
            "ping": self._cmd_ping,
            "probe": self._cmd_probe,
            "devices.list": self._cmd_devices_list,
            "apple.status": self._cmd_apple_status,
            "apple.install_assets": self._cmd_apple_install,
            "session.start": self._cmd_session_start,
            "session.stop": self._cmd_session_stop,
            "session.pause": self._cmd_session_pause,
            "session.resume": self._cmd_session_resume,
            "session.reset": self._cmd_session_reset,
            "session.set_language": self._cmd_session_set_language,
            "session.save": self._cmd_session_save,
            "session.snapshot": self._cmd_session_snapshot,
            "chat.ask": self._cmd_chat_ask,
            "session.notebook.set": self._cmd_session_notebook_set,
            "meetings.list": self._cmd_meetings_list,
            "meetings.get": self._cmd_meetings_get,
            "meetings.update": self._cmd_meetings_update,
            "meetings.regenerate_summary": self._cmd_meetings_regenerate_summary,
            "config.effective": self._cmd_config_effective,
            "wiki.vaults": self._cmd_wiki_vaults,
            "session.set_vault": self._cmd_session_set_vault,
        }

    # ------------------------------------------------------------------ transport

    def authorized(self, path: str) -> bool:
        qs = parse_qs(urlparse(path).query)
        supplied = (qs.get("token") or [""])[0]
        return secrets.compare_digest(supplied, self.token)

    def broadcast(self, payload: dict[str, Any]) -> None:
        msg = json.dumps(payload, ensure_ascii=False)
        is_meter = payload.get("event", {}).get("type") == "meter"
        for q in self._clients.values():
            if is_meter and q.qsize() > 50:
                continue
            q.put_nowait(msg)

    def _on_event(self, event: EngineEvent) -> None:
        self.broadcast({"type": "event", "event": event.to_dict()})

    def _emit_custom(self, event_type: str, **fields: Any) -> None:
        self.broadcast({"type": "event", "event": {"type": event_type, **fields}})

    async def handle_connection(self, ws: Any) -> None:
        q: asyncio.Queue[str] = asyncio.Queue()
        self._clients[ws] = q
        writer = asyncio.create_task(self._writer(ws, q))
        try:
            q.put_nowait(
                json.dumps(
                    {
                        "type": "hello",
                        "protocol": PROTOCOL_VERSION,
                        "status": self.session.status().to_dict() if self.session else None,
                    }
                )
            )
            async for raw in ws:
                asyncio.create_task(self._dispatch(q, raw))
        finally:
            self._clients.pop(ws, None)
            writer.cancel()

    async def _writer(self, ws: Any, q: asyncio.Queue[str]) -> None:
        while True:
            msg = await q.get()
            await ws.send(msg)

    async def _dispatch(self, q: asyncio.Queue[str], raw: str | bytes) -> None:
        req_id: Any = None
        try:
            req = json.loads(raw)
            req_id = req.get("id")
            cmd = str(req.get("cmd") or "")
            handler = self._handlers.get(cmd)
            if handler is None:
                raise CommandError(f"Unknown command: {cmd!r}")
            result = await handler(dict(req.get("args") or {}))
            resp: dict[str, Any] = {"type": "response", "id": req_id, "ok": True, "result": result}
        except CommandError as e:
            resp = {"type": "response", "id": req_id, "ok": False, "error": str(e)}
        except Exception as e:  # noqa: BLE001 — report, keep serving
            resp = {"type": "response", "id": req_id, "ok": False, "error": f"{type(e).__name__}: {e}"}
        q.put_nowait(json.dumps(resp, ensure_ascii=False))

    # ------------------------------------------------------------------ helpers

    def _require_session(self) -> EchoSession:
        if self.session is None:
            raise CommandError("No active session. Send session.start first.")
        return self.session

    async def _close_session(self) -> str | None:
        s = self.session
        if s is None:
            return None
        self.session = None
        if self._unsubscribe is not None:
            self._unsubscribe()
            self._unsubscribe = None
        await s.stop()
        self.broadcast({"type": "event", "event": s.status().to_dict()})
        out = s.save_outputs(announce=False)
        await s.wait_for_summary()
        return str(out)

    # ------------------------------------------------------------------ commands

    async def _cmd_ping(self, _args: dict[str, Any]) -> dict[str, Any]:
        return {"pong": True, "protocol": PROTOCOL_VERSION}

    async def _cmd_probe(self, _args: dict[str, Any]) -> dict[str, Any]:
        from echo.asr_api import whisper_api_config
        from echo.asr_routing import whisper_api_usable
        from echo.connectivity import echo_llm_online
        from echo.llm import configured_llm_label

        apple: dict[str, str] = {}
        if sys.platform == "darwin":
            from echo.asr_apple import apple_speech_status

            try:
                apple = await asyncio.to_thread(apple_speech_status, ("ja_JP", "en_US", "zh_CN"))
            except Exception as e:  # noqa: BLE001
                apple = {"error": str(e)}
        cfg = whisper_api_config()
        return {
            "platform": sys.platform,
            "python": sys.executable,
            "llm": {"online": await asyncio.to_thread(echo_llm_online), "label": configured_llm_label()},
            "whisper_api": {
                "configured": cfg is not None,
                "base_url": cfg[0] if cfg else None,
                "model": cfg[2] if cfg else None,
                "usable": await asyncio.to_thread(whisper_api_usable) if cfg else False,
            },
            "apple_speech": apple,
            "languages": list(LIVE_ASR_LANGUAGES),
            "search": _search_probe(),
        }

    async def _cmd_config_effective(self, args: dict[str, Any]) -> dict[str, Any]:
        keys = args.get("keys") or []
        if not isinstance(keys, list):
            raise CommandError("keys must be a list of variable names")
        return effective_config([str(k) for k in keys])

    async def _cmd_devices_list(self, _args: dict[str, Any]) -> dict[str, Any]:
        try:
            import sounddevice as sd
        except ImportError as e:
            raise CommandError("sounddevice is not installed (pip install -e '.[asr]').") from e
        devices = []
        for i, d in enumerate(sd.query_devices()):
            if d.get("max_input_channels", 0) > 0:
                devices.append({"index": i, "name": str(d.get("name", "?")), "channels": int(d["max_input_channels"])})
        try:
            default_in = int(sd.default.device[0])
        except Exception:  # noqa: BLE001
            default_in = None
        return {"inputs": devices, "default_input": default_in, "system_capture_helper": sys.platform == "darwin"}

    async def _cmd_apple_status(self, args: dict[str, Any]) -> dict[str, str]:
        from echo.asr_apple import apple_speech_status, echo_language_to_apple_locale

        langs = args.get("languages") or list(LIVE_ASR_LANGUAGES)
        locales = tuple(echo_language_to_apple_locale(_norm_language(x)) for x in langs)
        return await asyncio.to_thread(apple_speech_status, locales)

    async def _cmd_apple_install(self, args: dict[str, Any]) -> dict[str, Any]:
        from echo.asr_apple import install_apple_speech
        from echo.asr_routing import get_router

        language = _norm_language(args.get("language"))
        loop = asyncio.get_running_loop()

        def _progress(fraction: float) -> None:
            loop.call_soon_threadsafe(
                self._emit_custom, "apple.install.progress", language=language, fraction=fraction
            )

        try:
            await asyncio.to_thread(install_apple_speech, language, on_progress=_progress)
        except Exception as e:  # noqa: BLE001
            self._emit_custom("apple.install.done", language=language, ok=False, error=str(e))
            raise CommandError(f"Apple speech model download failed: {e}") from e
        get_router().invalidate_apple()
        s = self.session
        if s is not None and s.live_audio and s.live_asr_language == language:
            s.set_language(language)
        self._emit_custom("apple.install.done", language=language, ok=True)
        return {"language": language, "installed": True}

    async def _cmd_session_start(self, args: dict[str, Any]) -> dict[str, Any]:
        from echo.asr_routing import AsrUnavailableError
        from echo.connectivity import echo_llm_online
        from echo.engine.asr_setup import prepare_asr

        async with self._starting:
            await self._close_session()
            source = str(args.get("source") or "mic").strip().lower()
            if source not in ("mic", "both", "system", "demo"):
                raise CommandError("source must be one of: mic, both, demo")
            language = _norm_language(args.get("language"))
            if language not in LIVE_ASR_LANGUAGES:
                raise CommandError(f"Unsupported language: {language!r}")
            title = str(args.get("title") or "").strip()
            context = str(args.get("context") or "").strip()
            vault = self._vault_arg(args.get("vault"))
            llm_ok = await asyncio.to_thread(echo_llm_online)
            output_dir = None
            if self.output_root is not None:
                from datetime import datetime

                from echo.engine.text import meeting_folder_name

                self.output_root.mkdir(parents=True, exist_ok=True)
                output_dir = self.output_root / meeting_folder_name(
                    title or "untitled-meeting", datetime.now(), self.output_root
                )

            if source == "demo":
                from echo.app import DEFAULT_DEMO_LINES

                session = EchoSession(
                    replay_lines=DEFAULT_DEMO_LINES,
                    line_delay=float(args.get("line_delay", 1.0)),
                    llm_offline=not llm_ok,
                    meeting_title=title,
                    meeting_context=context,
                    output_dir=output_dir,
                    language=language,
                    vault=vault,
                    wiki=self.wiki,
                )
            else:
                system = source in ("both", "system")
                self._emit_custom("session.preparing", backend=args.get("backend") or "auto", language=language)
                try:
                    # Runs on the loop (main) thread on purpose: local Kotoba must load on the main thread.
                    _router, asr_source, chunk_seconds = prepare_asr(
                        args.get("backend") or None,
                        language=language,
                        live_audio=True,
                        chunk_seconds=args.get("chunk_seconds"),
                        live_system_authorized=system,
                    )
                except AsrUnavailableError as e:
                    raise CommandError(str(e)) from e
                mic_device = args.get("mic_device")
                session = EchoSession(
                    live_audio=True,
                    live_mic_device=int(mic_device) if mic_device is not None else None,
                    live_system_authorized=system,
                    live_emit_interval=0.0,
                    insight_min_chars=220,
                    insight_min_interval=45.0,
                    chunk_seconds=chunk_seconds,
                    asr_load_source=asr_source,
                    llm_offline=not llm_ok,
                    meeting_title=title,
                    meeting_context=context,
                    output_dir=output_dir,
                    language=language,
                    vault=vault,
                    wiki=self.wiki,
                )
            session.set_meeting_metadata(session.meeting_title, session.meeting_context)
            self.session = session
            self._unsubscribe = session.subscribe(self._on_event)
            session.start()
            return {"status": session.status().to_dict(), "output_dir": str(session.output_dir)}

    async def _cmd_session_stop(self, _args: dict[str, Any]) -> dict[str, Any]:
        return {"output_dir": await self._close_session()}

    async def _cmd_session_pause(self, _args: dict[str, Any]) -> dict[str, Any]:
        s = self._require_session()
        s.set_paused(True)
        return s.status().to_dict()

    async def _cmd_session_resume(self, _args: dict[str, Any]) -> dict[str, Any]:
        s = self._require_session()
        s.set_paused(False)
        return s.status().to_dict()

    async def _cmd_session_reset(self, args: dict[str, Any]) -> dict[str, Any]:
        s = self._require_session()
        s.reset()
        if "title" in args or "context" in args:
            s.set_meeting_metadata(str(args.get("title") or ""), str(args.get("context") or ""))
        if args.get("resume", True):
            s.set_paused(False)
        return s.status().to_dict()

    async def _cmd_session_set_language(self, args: dict[str, Any]) -> dict[str, Any]:
        s = self._require_session()
        ok, message = s.set_language(_norm_language(args.get("language")))
        if not ok:
            raise CommandError(message)
        return {"message": message, "status": s.status().to_dict()}

    @staticmethod
    def _vault_arg(raw: object) -> str | None:
        name = str(raw or "").strip()
        if not name:
            return None
        try:
            return str(resolve_vault(name)["name"])
        except WikiError as e:
            raise CommandError(str(e)) from e

    async def _cmd_wiki_vaults(self, _args: dict[str, Any]) -> dict[str, Any]:
        return await asyncio.to_thread(self.wiki.vaults)

    async def _cmd_session_set_vault(self, args: dict[str, Any]) -> dict[str, Any]:
        s = self._require_session()
        s.set_vault(self._vault_arg(args.get("vault")))
        return s.status().to_dict()

    async def _cmd_session_save(self, _args: dict[str, Any]) -> dict[str, Any]:
        s = self._require_session()
        return {"output_dir": str(s.save_outputs())}

    async def _cmd_session_snapshot(self, _args: dict[str, Any]) -> dict[str, Any]:
        s = self.session
        if s is None:
            return {"status": None}
        return {
            "status": s.status().to_dict(),
            "transcript": [asdict(x) for x in s.transcript_snapshot()],
            "insights": [asdict(p) for p in s.insight_passes],
            "chat": [{"question": q, "answer": a} for q, a in s._chat_history],
            "output_dir": str(s.output_dir),
            "meeting_id": s.output_dir.name,
            "notebook": s.notebook_md,
        }

    async def _cmd_chat_ask(self, args: dict[str, Any]) -> dict[str, Any]:
        s = self._require_session()
        question = str(args.get("question") or "").strip()
        if not question:
            raise CommandError("question is required")
        return {"question": question, "answer": await s.ask(question)}

    async def _cmd_session_notebook_set(self, args: dict[str, Any]) -> dict[str, Any]:
        s = self._require_session()
        s.set_notebook(str(args.get("text") or ""))
        return {"ok": True}

    # ------------------------------------------------------------------ meetings archive

    def _store(self) -> Any:
        from echo.meetings import MeetingStore

        if self.output_root is None:
            raise CommandError("Meeting archive requires --output-root.")
        return MeetingStore(self.output_root)

    def _live_id(self) -> str | None:
        s = self.session
        if s is None or self.output_root is None:
            return None
        return s.output_dir.name if s.output_dir.parent.resolve() == self.output_root.resolve() else None

    def _live_detail(self) -> dict[str, Any]:
        s = self.session
        assert s is not None
        return {
            **s.meeting_meta(),
            "legacy": False,
            "live": True,
            "path": str(s.output_dir),
            "transcript": list(s._lines),
            "summary_md": "",
            "insights_md": s._insights_export_markdown(),
            "notebook_md": s.notebook_md,
        }

    @staticmethod
    def _meeting_id(args: dict[str, Any]) -> str:
        mid = str(args.get("id") or "").strip()
        if not mid:
            raise CommandError("id is required")
        return mid

    async def _cmd_meetings_list(self, _args: dict[str, Any]) -> dict[str, Any]:
        items = await asyncio.to_thread(self._store().list)
        live = self._live_id()
        for m in items:
            m["live"] = m["id"] == live
        if live is not None and not any(m["id"] == live for m in items):
            assert self.session is not None
            meta = self.session.meeting_meta()
            items.insert(0, {**meta, "legacy": False, "live": True, "summary_preview": "", "has_notebook": bool(self.session.notebook_md.strip())})
        return {"meetings": items, "live_id": live}

    async def _cmd_meetings_get(self, args: dict[str, Any]) -> dict[str, Any]:
        from echo.meetings import MeetingNotFoundError

        mid = self._meeting_id(args)
        if mid == self._live_id():
            return self._live_detail()
        try:
            detail = await asyncio.to_thread(self._store().get, mid)
        except MeetingNotFoundError as e:
            raise CommandError(str(e)) from e
        return {**detail, "live": False}

    async def _cmd_meetings_update(self, args: dict[str, Any]) -> dict[str, Any]:
        from echo.meetings import MeetingNotFoundError

        mid = self._meeting_id(args)
        title = args.get("title")
        context = args.get("context")
        notebook = args.get("notebook")
        if mid == self._live_id():
            s = self.session
            assert s is not None
            if title is not None or context is not None:
                s.set_meeting_metadata(
                    str(title) if title is not None else s.meeting_title,
                    str(context) if context is not None else s.meeting_context,
                )
            if notebook is not None:
                s.set_notebook(str(notebook))
            return self._live_detail()
        try:
            detail = await asyncio.to_thread(
                lambda: self._store().update(
                    mid,
                    title=str(title) if title is not None else None,
                    context=str(context) if context is not None else None,
                    notebook=str(notebook) if notebook is not None else None,
                )
            )
        except MeetingNotFoundError as e:
            raise CommandError(str(e)) from e
        return {**detail, "live": False}

    async def _cmd_meetings_regenerate_summary(self, args: dict[str, Any]) -> dict[str, Any]:
        from echo.dictionary import load_dictionary
        from echo.meetings import MeetingNotFoundError

        mid = self._meeting_id(args)
        if mid == self._live_id():
            raise CommandError("This meeting is still in progress. End the meeting first.")
        store = self._store()
        try:
            md = await store.regenerate_summary(mid, dictionary_terms=load_dictionary())
        except MeetingNotFoundError as e:
            raise CommandError(str(e)) from e
        path = store.path(mid) / "summary.md"
        self._emit_custom("summary.written", path=str(path), ok=True, announce=False, id=mid)
        return {"id": mid, "summary_md": md}

    async def shutdown(self) -> None:
        await self._close_session()
        await self.wiki.close()


def _watch_stdin_eof(loop: asyncio.AbstractEventLoop, stop: asyncio.Event) -> None:
    def _run() -> None:
        try:
            while sys.stdin.buffer.read(1024):
                pass
        except Exception:  # noqa: BLE001
            pass
        loop.call_soon_threadsafe(stop.set)

    threading.Thread(target=_run, daemon=True, name="echo-stdin-watch").start()


async def _serve(args: argparse.Namespace) -> None:
    from websockets.asyncio.server import serve

    token = args.token or os.environ.get("ECHO_SERVER_TOKEN") or secrets.token_urlsafe(24)
    server = EchoServer(token, output_root=Path(args.output_root) if args.output_root else None)

    def _process_request(connection: Any, request: Any) -> Any:
        if not server.authorized(request.path):
            return connection.respond(HTTPStatus.UNAUTHORIZED, "invalid or missing token\n")
        return None

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except (NotImplementedError, RuntimeError):
            pass
    if not args.no_stdin_watch and not sys.stdin.isatty():
        _watch_stdin_eof(loop, stop)

    async with serve(
        server.handle_connection,
        args.host,
        args.port,
        process_request=_process_request,
        ping_interval=None,
        max_size=4 * 1024 * 1024,
    ) as ws_server:
        port = next(iter(ws_server.sockets)).getsockname()[1]
        print(json.dumps({"port": port, "token": token, "protocol": PROTOCOL_VERSION}), flush=True)
        from echo.preflight import preflight_modules, warm_imports

        threading.Thread(
            target=warm_imports, args=(preflight_modules(live=True),), daemon=True, name="echo-warm-imports"
        ).start()
        await stop.wait()
        await server.shutdown()


def main() -> None:
    p = argparse.ArgumentParser(prog="echo-server", description="Echo WebSocket sidecar for the desktop GUI.")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=0, help="TCP port (default: random free port).")
    p.add_argument("--token", default=None, help="Auth token (default: ECHO_SERVER_TOKEN or random).")
    p.add_argument("--output-root", default=None, help="Directory for meeting outputs (default: ./output).")
    p.add_argument("--no-stdin-watch", action="store_true", help="Do not exit when stdin closes.")
    args = p.parse_args()

    from dotenv import load_dotenv

    from echo.preflight import enable_crash_log, run_preflight

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    load_dotenv(Path.cwd() / ".env")
    enable_crash_log()
    # stdout is reserved for the single handshake line; route preflight chatter to stderr.
    real_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        run_preflight(live=True, warm=False)
    finally:
        sys.stdout = real_stdout
    asyncio.run(_serve(args))


if __name__ == "__main__":
    main()
