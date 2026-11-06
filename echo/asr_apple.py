"""macOS on-device speech recognition via the bundled Swift helper (``SpeechTranscribe.swift``).

On macOS 26+ the helper uses ``SpeechAnalyzer`` + ``SpeechTranscriber`` (model assets are
downloaded per locale through ``AssetInventory``); older systems fall back to on-device
``SFSpeechRecognizer``. Live mode keeps one helper process running and streams PCM into it,
receiving incremental ``partial`` results and ``final`` utterances as JSON lines.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

_LOCALES = {
    "japanese": "ja_JP",
    "ja": "ja_JP",
    "english": "en_US",
    "en": "en_US",
    "chinese": "zh_CN",
    "zh": "zh_CN",
}

_build_lock = threading.Lock()


def echo_language_to_apple_locale(language: str) -> str:
    return _LOCALES.get((language or "").strip().lower(), "ja_JP")


def _speech_helper_source_path() -> Path:
    return Path(__file__).resolve().parent / "macos" / "SpeechTranscribe.swift"


def _speech_helper_binary_path() -> Path:
    cache_home = Path(os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache"))
    return cache_home / "echo" / "speech_transcribe"


def ensure_macos_speech_helper() -> str:
    if sys.platform != "darwin":
        raise RuntimeError("Apple on-device ASR is available on macOS only.")

    src = _speech_helper_source_path()
    if not src.is_file():
        raise RuntimeError(f"Bundled speech helper source not found: {src}")

    out = _speech_helper_binary_path()
    with _build_lock:
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists() and src.stat().st_mtime <= out.stat().st_mtime:
            return str(out)
        cmd = [
            "xcrun",
            "swiftc",
            "-parse-as-library",
            "-O",
            "-framework",
            "Speech",
            "-framework",
            "AVFoundation",
            "-framework",
            "CoreMedia",
            str(src),
            "-o",
            str(out),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            proc2 = subprocess.run(cmd[1:], capture_output=True, text=True)
            if proc2.returncode != 0:
                err = (proc.stderr or "") + "\n" + (proc2.stderr or "")
                raise RuntimeError(
                    "Failed to build macOS speech helper with swiftc.\n"
                    f"Source: {src}\n"
                    f"Errors:\n{err.strip()}"
                )
        out.chmod(0o755)
    return str(out)


def apple_speech_status(locales: tuple[str, ...]) -> dict[str, str]:
    """Return ``{locale: "<engine>:<status>"}``; status is installed | supported | downloading | unsupported."""
    if sys.platform != "darwin":
        return {loc: "unsupported" for loc in locales}
    helper = ensure_macos_speech_helper()
    proc = subprocess.run(
        [helper, "--status", "--locales", ",".join(locales)],
        capture_output=True,
        text=True,
        timeout=60.0,
    )
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout or "status failed").strip())
    data = json.loads((proc.stdout or "{}").strip().splitlines()[-1])
    return {loc: f"{v.get('engine', '?')}:{v.get('status', 'unsupported')}" for loc, v in data.items()}


def apple_locale_status(language: str) -> str:
    """Bare status (installed | supported | downloading | unsupported) for one Echo language."""
    loc = echo_language_to_apple_locale(language)
    try:
        return apple_speech_status((loc,)).get(loc, "?:unsupported").split(":", 1)[1]
    except Exception:  # noqa: BLE001 — helper build/run failure means unusable
        return "unsupported"


def probe_apple_on_device(*, language: str = "japanese") -> bool:
    return apple_locale_status(language) == "installed"


def install_apple_speech(language: str, *, on_progress=None) -> None:
    """Download the on-device model for ``language`` (macOS 26+). Raises on failure."""
    helper = ensure_macos_speech_helper()
    loc = echo_language_to_apple_locale(language)
    proc = subprocess.Popen(
        [helper, "--install", "--locale", loc],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert proc.stdout is not None
    last_error = ""
    for line in proc.stdout:
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        if msg.get("type") == "progress" and on_progress is not None:
            on_progress(float(msg.get("fraction") or 0.0))
        elif msg.get("type") == "error":
            last_error = str(msg.get("message") or "")
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(last_error or (proc.stderr.read() if proc.stderr else "") or "install failed")


def _float32_to_s16le(samples: object) -> bytes:
    import numpy as np

    x = np.clip(np.asarray(samples, dtype=np.float32).reshape(-1), -1.0, 1.0)
    return (x * 32767.0).astype("<i2").tobytes()


@dataclass
class AppleStreamEvent:
    kind: str  # "partial" | "final" | "error" | "ready" | "exit"
    text: str = ""
    start: float = 0.0
    end: float = 0.0


class AppleStreamingRecognizer:
    """One long-running helper process per locale; feed 16 kHz float32, poll events."""

    def __init__(self, language: str) -> None:
        self.language = language
        self.locale = echo_language_to_apple_locale(language)
        self._proc: subprocess.Popen | None = None
        self._events: queue.Queue[AppleStreamEvent] = queue.Queue()
        self._stderr_tail: list[str] = []

    def start(self) -> None:
        helper = ensure_macos_speech_helper()
        self._proc = subprocess.Popen(
            [helper, "--stream", "--locale", self.locale],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        threading.Thread(target=self._read_stdout, daemon=True).start()
        threading.Thread(target=self._read_stderr, daemon=True).start()

    def _read_stdout(self) -> None:
        proc = self._proc
        assert proc is not None and proc.stdout is not None
        for raw in proc.stdout:
            try:
                msg = json.loads(raw.decode("utf-8", errors="replace"))
            except json.JSONDecodeError:
                continue
            kind = str(msg.get("type") or "")
            self._events.put(
                AppleStreamEvent(
                    kind=kind,
                    text=str(msg.get("text") or msg.get("message") or ""),
                    start=float(msg.get("start") or 0.0),
                    end=float(msg.get("end") or 0.0),
                )
            )
        proc.wait()
        self._events.put(AppleStreamEvent(kind="exit", text=" ".join(self._stderr_tail[-3:])))

    def _read_stderr(self) -> None:
        proc = self._proc
        assert proc is not None and proc.stderr is not None
        for raw in proc.stderr:
            self._stderr_tail = (self._stderr_tail + [raw.decode("utf-8", errors="replace").strip()])[-10:]

    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def feed(self, samples: object) -> None:
        proc = self._proc
        if proc is None or proc.stdin is None or proc.poll() is not None:
            raise RuntimeError("Apple speech helper is not running.")
        try:
            proc.stdin.write(_float32_to_s16le(samples))
        except (BrokenPipeError, OSError) as e:
            raise RuntimeError(f"Apple speech helper stopped: {e}") from e

    def poll(self) -> list[AppleStreamEvent]:
        out: list[AppleStreamEvent] = []
        while True:
            try:
                out.append(self._events.get_nowait())
            except queue.Empty:
                return out

    def finish(self, timeout: float = 10.0) -> list[AppleStreamEvent]:
        """Close stdin, wait for final results, and return all remaining events."""
        proc = self._proc
        if proc is None:
            return []
        try:
            if proc.stdin is not None:
                proc.stdin.close()
            proc.wait(timeout=timeout)
        except (subprocess.TimeoutExpired, OSError):
            proc.kill()
        events: list[AppleStreamEvent] = []
        while True:
            try:
                events.append(self._events.get(timeout=0.5))
            except queue.Empty:
                break
            if events[-1].kind == "exit":
                break
        return events

    def stop(self) -> None:
        proc = self._proc
        if proc is not None and proc.poll() is None:
            proc.kill()


def transcribe_mono16k_chunk_apple(samples: object, *, language: str = "japanese") -> str:
    """One-shot transcription of a buffered chunk (file mode). Live mode uses streaming."""
    import numpy as np

    x = np.asarray(samples, dtype=np.float32).reshape(-1)
    if x.size < int(0.25 * 16000):
        return ""
    rec = AppleStreamingRecognizer(language)
    rec.start()
    rec.feed(x)
    events = rec.finish(timeout=max(30.0, x.size / 16000 * 2))
    errors = [e.text for e in events if e.kind == "error"]
    finals = [e.text.strip() for e in events if e.kind == "final" and e.text.strip()]
    if errors and not finals:
        raise RuntimeError(f"Apple on-device ASR failed: {errors[-1]}")
    return " ".join(finals) if not language.startswith(("ja", "zh", "japanese", "chinese")) else "".join(finals)
