"""Startup checks that catch a broken Python environment before the TUI or server starts.

macOS kills a process outright (``SIGKILL (Code Signature Invalid)``) when it maps a native
extension whose arm64 code signature is missing or modified. Echo lazily imports several native
modules deep into a session (web search, LLM SDKs, sqlite via HTTP caches), so a broken env used
to surface as a silent mid-meeting kill. These helpers probe those imports in a child process
first and report which module is at fault.
"""

from __future__ import annotations

import faulthandler
import hashlib
import os
import subprocess
import sys
import sysconfig
import time
from pathlib import Path

LAZY_NATIVE_MODULES: tuple[str, ...] = (
    "sqlite3",
    "ssl",
    "zlib",
    "lxml.etree",
    "primp",
    "ddgs",
    "httpx",
    "openai",
    "anthropic",
    "yaml",
    "numpy",
)

LIVE_NATIVE_MODULES: tuple[str, ...] = ("sounddevice",)

_crash_log_handle = None


def enable_crash_log(log_dir: Path | None = None) -> Path | None:
    """Write Python-level tracebacks for fatal signals to ``log/echo_crash_<ts>.log``."""
    global _crash_log_handle
    try:
        d = log_dir or (Path.cwd() / "log")
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"echo_crash_{int(time.time())}.log"
        _crash_log_handle = open(path, "w", encoding="utf-8")  # noqa: SIM115 — must stay open
        faulthandler.enable(file=_crash_log_handle, all_threads=True)
        return path
    except OSError:
        faulthandler.enable()
        return None


def check_imports_in_child(modules: tuple[str, ...]) -> tuple[bool, str]:
    """Import ``modules`` one at a time in a child interpreter.

    Returns ``(ok, message)``. When the child dies, the last module it announced is reported.
    """
    script = (
        "import sys\n"
        "for name in sys.argv[1:]:\n"
        "    print(name, flush=True)\n"
        "    try:\n"
        "        __import__(name)\n"
        "    except ImportError:\n"
        "        pass\n"
        "print('__done__', flush=True)\n"
    )
    try:
        proc = subprocess.run(
            [sys.executable, "-c", script, *modules],
            capture_output=True,
            text=True,
            timeout=120,
            env={**os.environ, "PYTHONFAULTHANDLER": "1"},
        )
    except subprocess.TimeoutExpired:
        return False, "Import preflight timed out after 120s."
    lines = [ln.strip() for ln in (proc.stdout or "").splitlines() if ln.strip()]
    if proc.returncode == 0 and lines and lines[-1] == "__done__":
        return True, "ok"
    culprit = lines[-1] if lines and lines[-1] != "__done__" else "(unknown)"
    if proc.returncode in (-9, 137):
        reason = "killed by macOS (SIGKILL, usually an invalid native code signature)"
    else:
        reason = f"exited with code {proc.returncode}"
    return False, (
        f"Importing `{culprit}` {reason}.\n"
        f"Interpreter: {sys.executable}\n"
        "This Python environment is broken. Create a clean virtualenv and reinstall Echo:\n"
        "  python3 -m venv .venv && .venv/bin/pip install -e '.[asr,asr-vad]'\n"
        "Then launch with .venv/bin/echo-tui (bin/echo-live does this automatically)."
    )


def warm_imports(modules: tuple[str, ...]) -> None:
    """Import modules in-process so later lazy imports are instant and failures are early."""
    for name in modules:
        try:
            __import__(name)
        except ImportError:
            pass


def preflight_modules(*, live: bool) -> tuple[str, ...]:
    return LAZY_NATIVE_MODULES + (LIVE_NATIVE_MODULES if live else ())


def _marker_path() -> Path:
    return Path(sys.prefix) / ".echo-preflight-ok"


def _preflight_key(modules: tuple[str, ...]) -> str:
    """Changes whenever the interpreter or the installed packages change (pip touches site-packages)."""
    try:
        site_mtime = os.stat(sysconfig.get_paths()["purelib"]).st_mtime_ns
    except OSError:
        site_mtime = 0
    raw = "\n".join([sys.executable, sys.version, ",".join(modules), str(site_mtime)])
    return hashlib.sha256(raw.encode()).hexdigest()


def _passed_before(key: str) -> bool:
    try:
        return _marker_path().read_text(encoding="utf-8").strip() == key
    except OSError:
        return False


def _remember_pass(key: str) -> None:
    try:
        _marker_path().write_text(key, encoding="utf-8")
    except OSError:
        pass


def run_preflight(*, live: bool, warm: bool = True) -> None:
    """Exit with a clear message when the env would crash mid-session; otherwise warm imports.

    The child-process check runs once per environment state; a passing result is remembered in
    ``<sys.prefix>/.echo-preflight-ok``. With ``warm=False`` the caller warms imports itself.
    """
    if os.environ.get("ECHO_SKIP_PREFLIGHT", "").strip().lower() in ("1", "true", "yes"):
        return
    modules = preflight_modules(live=live)
    key = _preflight_key(modules)
    if not _passed_before(key):
        ok, msg = check_imports_in_child(modules)
        if not ok:
            print(f"Echo: environment check failed.\n{msg}", file=sys.stderr, flush=True)
            raise SystemExit(3)
        _remember_pass(key)
    if warm:
        warm_imports(modules)
