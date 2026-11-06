"""ASR backend selection: Apple on-device → Whisper API → local Kotoba-Whisper.

``auto`` mode picks the first usable tier for the current language and re-picks when the language
changes. A tier is only abandoned after several consecutive failures, then retried after an
exponential cooldown. Forced mode (``--asr-backend apple|api|local``) never falls back.

Local Kotoba weights are only ever loaded on the main thread at startup (``ensure_ready``);
loading Hugging Face weights from a worker thread breaks on macOS (``fds_to_keep``), so a
mid-session fallback to local is allowed only when the weights are already resident.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from dataclasses import dataclass
from typing import Callable, Literal

from echo.asr import MODEL_ID_DEFAULT

AsrBackend = Literal["apple_on_device", "whisper_api", "local_whisper"]

BACKEND_ORDER: tuple[AsrBackend, ...] = ("apple_on_device", "whisper_api", "local_whisper")
STREAMING_BACKENDS: frozenset[str] = frozenset({"apple_on_device"})

_ALIASES: dict[str, AsrBackend] = {
    "apple": "apple_on_device",
    "apple_on_device": "apple_on_device",
    "api": "whisper_api",
    "whisper_api": "whisper_api",
    "local": "local_whisper",
    "local_whisper": "local_whisper",
}

_LANGUAGE_SHORT = {"japanese": "ja", "english": "en", "chinese": "zh"}


class AsrUnavailableError(RuntimeError):
    """No ASR backend can serve the request; the message says what to do about it."""


def _env_truthy(name: str) -> bool:
    return (os.environ.get(name) or "").strip().lower() in ("1", "true", "yes", "on")


def parse_backend_name(raw: str | None) -> AsrBackend | Literal["auto"]:
    key = (raw or "").strip().lower()
    if not key or key == "auto":
        return "auto"
    if key not in _ALIASES:
        raise ValueError(f"Unknown ASR backend {raw!r}. Use auto, apple, api, or local.")
    return _ALIASES[key]


def whisper_api_usable() -> bool:
    """Credentials exist, Echo is not forced offline, and the API host answers a TCP probe."""
    from echo.asr_api import whisper_api_config
    from echo.connectivity import probe_whisper_api

    if not whisper_api_config() or _env_truthy("ECHO_OFFLINE"):
        return False
    if _env_truthy("ECHO_ASSUME_ONLINE"):
        return True
    return probe_whisper_api()


def _local_deps_available() -> bool:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        return False
    return True


@dataclass
class _Health:
    failures: int = 0
    cooldown: float = 0.0
    cooldown_until: float = 0.0


class AsrRouter:
    FAILURE_THRESHOLD = 3
    BASE_COOLDOWN = 30.0
    MAX_COOLDOWN = 300.0
    UPGRADE_CHECK_SECONDS = 60.0
    API_PROBE_TTL = 30.0

    def __init__(
        self,
        requested: str | None = None,
        *,
        model_id: str = MODEL_ID_DEFAULT,
        device: str | None = None,
        apple_status: Callable[[str], str] | None = None,
        api_usable: Callable[[], bool] | None = None,
        api_configured: Callable[[], bool] | None = None,
        local_available: Callable[[], bool] | None = None,
        clock: Callable[[], float] = time.monotonic,
        platform: str | None = None,
    ) -> None:
        raw = requested if requested is not None else os.environ.get("ECHO_ASR_BACKEND")
        choice = parse_backend_name(raw)
        self.mode: Literal["auto", "forced"] = "auto" if choice == "auto" else "forced"
        self.forced: AsrBackend | None = None if choice == "auto" else choice
        self.model_id = model_id
        self.device = device
        self._apple_status_fn = apple_status or _default_apple_status
        self._api_usable_fn = api_usable or whisper_api_usable
        self._api_configured_fn = api_configured or _default_api_configured
        self._local_available_fn = local_available or _local_deps_available
        self._clock = clock
        self._platform = platform or sys.platform
        self._lock = threading.RLock()
        self._apple_cache: dict[str, str] = {}
        self._api_cache: tuple[float, bool] | None = None
        self._health: dict[str, _Health] = {b: _Health() for b in BACKEND_ORDER}
        self._prompt_tail: dict[str, str] = {}
        self._last_upgrade_check = 0.0
        self.local_preloaded = False
        self.local_load_source: str | None = None
        self.active: AsrBackend | None = None
        self.language: str | None = None
        self.last_error: str = ""

    # ------------------------------------------------------------------ probes

    def apple_status(self, language: str) -> str:
        if self._platform != "darwin":
            return "unsupported"
        key = language.strip().lower()
        with self._lock:
            if key not in self._apple_cache:
                self._apple_cache[key] = self._apple_status_fn(key)
            return self._apple_cache[key]

    def invalidate_apple(self) -> None:
        with self._lock:
            self._apple_cache.clear()

    def _api_ok(self) -> bool:
        now = self._clock()
        with self._lock:
            if self._api_cache is not None and now - self._api_cache[0] < self.API_PROBE_TTL:
                return self._api_cache[1]
        ok = bool(self._api_usable_fn())
        with self._lock:
            self._api_cache = (now, ok)
        return ok

    def _usable(self, backend: AsrBackend, language: str, *, startup: bool) -> tuple[bool, str]:
        if backend == "apple_on_device":
            status = self.apple_status(language)
            if status == "installed":
                return True, ""
            if status in ("supported", "downloading"):
                short = _LANGUAGE_SHORT.get(language, language)
                return False, (
                    f"Apple on-device {language} speech model is not installed. "
                    f"Run: echo-tui --install-apple-speech {short}"
                )
            return False, f"Apple on-device speech does not support {language} on this Mac."
        if backend == "whisper_api":
            if not self._api_configured_fn():
                return False, (
                    "Whisper API is not configured. Set ECHO_WHISPER_API_BASE_URL + API_KEY "
                    "(or OPENAI_API_KEY)."
                )
            if not self._api_ok():
                return False, "Whisper API is unreachable (ECHO_OFFLINE or TCP probe failed)."
            return True, ""
        if self.local_preloaded:
            return True, ""
        if not startup:
            return False, (
                "Local Kotoba-Whisper was not loaded at startup and cannot be loaded mid-session. "
                "Restart with --asr-backend local."
            )
        if not self._local_available_fn():
            return False, "Local Kotoba-Whisper needs ASR extras: pip install -e '.[asr]'"
        return True, ""

    def _healthy(self, backend: AsrBackend) -> bool:
        return self._clock() >= self._health[backend].cooldown_until

    def _auto_allows_local(self) -> bool:
        """Prefer the API while online; local Kotoba is the tier for offline / no-API setups."""
        if _env_truthy("ECHO_OFFLINE"):
            return True
        return not (self._api_configured_fn() and self._api_ok())

    # ------------------------------------------------------------------ selection

    def select(self, language: str, *, startup: bool = False) -> AsrBackend:
        """Pick the backend for ``language``; raises ``AsrUnavailableError`` with guidance."""
        language = (language or "japanese").strip().lower()
        if self.forced is not None:
            ok, reason = self._usable(self.forced, language, startup=startup)
            if not ok:
                raise AsrUnavailableError(reason)
            return self._activate(self.forced, language)

        reasons: list[str] = []
        for backend in BACKEND_ORDER:
            if backend == "apple_on_device" and self._platform != "darwin":
                continue
            if backend == "local_whisper" and not self._auto_allows_local():
                continue
            if not self._healthy(backend):
                reasons.append(f"{backend} is cooling down after repeated failures")
                continue
            ok, reason = self._usable(backend, language, startup=startup)
            if ok:
                return self._activate(backend, language)
            reasons.append(reason)
        raise AsrUnavailableError("No ASR backend is available: " + " ".join(r for r in reasons if r))

    def _activate(self, backend: AsrBackend, language: str) -> AsrBackend:
        with self._lock:
            self.active = backend
            self.language = language
        return backend

    def ensure_ready(self) -> str:
        """Main-thread initialization for the active backend; returns the display source label."""
        if self.active is None:
            raise AsrUnavailableError("No ASR backend selected.")
        if self.active == "local_whisper" and not self.local_preloaded:
            from echo.asr import ensure_model_loaded, get_last_model_load_source

            ensure_model_loaded(self.model_id, self.device)
            self.local_preloaded = True
            self.local_load_source = get_last_model_load_source()
        elif self.active == "apple_on_device":
            from echo.asr_apple import ensure_macos_speech_helper

            ensure_macos_speech_helper()
        return self.display_source()

    def display_source(self) -> str:
        if self.active == "apple_on_device":
            return "apple-on-device"
        if self.active == "whisper_api":
            return "whisper-api"
        if self.active == "local_whisper":
            return self.local_load_source or "local-whisper"
        return "unknown"

    def is_streaming(self, backend: AsrBackend | None = None) -> bool:
        return (backend or self.active) in STREAMING_BACKENDS

    # ------------------------------------------------------------------ health

    def record_success(self, backend: AsrBackend) -> None:
        with self._lock:
            h = self._health[backend]
            h.failures = 0
            h.cooldown = 0.0
            self.last_error = ""

    def record_failure(self, backend: AsrBackend, error: BaseException | str) -> AsrBackend | None:
        """Count a failure; after ``FAILURE_THRESHOLD`` in a row, cool down and re-select (auto).

        Returns the new backend when the router switched, else ``None``.
        """
        with self._lock:
            self.last_error = str(error)
            h = self._health[backend]
            h.failures += 1
            if h.failures < self.FAILURE_THRESHOLD:
                return None
            h.failures = 0
            h.cooldown = min(self.MAX_COOLDOWN, max(self.BASE_COOLDOWN, h.cooldown * 2))
            h.cooldown_until = self._clock() + h.cooldown
            self._api_cache = None
        if self.mode == "forced" or self.language is None:
            return None
        try:
            new = self.select(self.language)
        except AsrUnavailableError:
            return None
        return new if new != backend else None

    def maybe_upgrade(self, language: str) -> AsrBackend | None:
        """In auto mode, periodically move back to a higher tier that has recovered."""
        if self.mode == "forced":
            return None
        now = self._clock()
        language = language.strip().lower()
        if language == self.language and now - self._last_upgrade_check < self.UPGRADE_CHECK_SECONDS:
            return None
        self._last_upgrade_check = now
        before = self.active
        try:
            new = self.select(language)
        except AsrUnavailableError:
            return None
        return new if new != before else None

    # ------------------------------------------------------------------ windowed decode

    def transcribe_window(self, samples: object, *, language: str) -> str:
        """Decode one buffered window with the active (non-streaming or file-mode) backend."""
        backend = self.active
        if backend is None:
            backend = self.select(language)
        try:
            text = self._decode(backend, samples, language)
        except AsrUnavailableError:
            raise
        except Exception as e:
            self.record_failure(backend, e)
            raise
        self.record_success(backend)
        if text:
            tail = (self._prompt_tail.get(language, "") + text)[-200:]
            self._prompt_tail[language] = tail
        return text

    def _decode(self, backend: AsrBackend, samples: object, language: str) -> str:
        if backend == "apple_on_device":
            from echo.asr_apple import transcribe_mono16k_chunk_apple

            return transcribe_mono16k_chunk_apple(samples, language=language)
        if backend == "whisper_api":
            from echo.asr_api import transcribe_mono16k_chunk_api

            return transcribe_mono16k_chunk_api(
                samples, language=language, prompt=self._prompt_tail.get(language) or None
            )
        if not self.local_preloaded:
            raise AsrUnavailableError(
                "Local Kotoba-Whisper is not loaded; restart with --asr-backend local."
            )
        from echo.asr import _transcribe_local_whisper_chunk

        return _transcribe_local_whisper_chunk(
            samples, model_id=self.model_id, device=self.device, language=language
        )


def _default_apple_status(language: str) -> str:
    from echo.asr_apple import apple_locale_status

    return apple_locale_status(language)


def _default_api_configured() -> bool:
    from echo.asr_api import whisper_api_config

    return whisper_api_config() is not None


_router_lock = threading.Lock()
_router: AsrRouter | None = None


def get_router() -> AsrRouter:
    """Process-wide router used by ``echo.asr.transcribe_mono16k_chunk`` call sites."""
    global _router
    with _router_lock:
        if _router is None:
            _router = AsrRouter()
        return _router


def set_router(router: AsrRouter | None) -> None:
    global _router
    with _router_lock:
        _router = router
