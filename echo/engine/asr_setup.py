"""ASR backend setup shared by the TUI (``run_echo``) and the sidecar server.

Must run on the main thread: local Kotoba-Whisper weights are loaded here when that tier is
selected (see ``echo.asr_routing``).
"""

from __future__ import annotations

from typing import Callable

from echo.asr import MODEL_ID_DEFAULT
from echo.asr_routing import AsrRouter, set_router
from echo.connectivity import hub_local_only_from_env


def prepare_asr(
    asr_backend: str | None,
    *,
    language: str,
    live_audio: bool,
    chunk_seconds: float | None,
    model_id: str = MODEL_ID_DEFAULT,
    whisper_device: str | None = None,
    live_system_authorized: bool = False,
    before_select: Callable[[AsrRouter, str], None] | None = None,
    log: Callable[[str], None] = lambda _m: None,
) -> tuple[AsrRouter, str, float]:
    """Select + initialize the ASR backend. Returns ``(router, display_source, chunk_seconds)``.

    Raises ``AsrUnavailableError`` with an actionable message when nothing can serve ``language``.
    """
    from echo.asr_api import whisper_api_chunk_seconds

    router = AsrRouter(asr_backend, model_id=model_id, device=whisper_device)
    set_router(router)
    if before_select is not None:
        before_select(router, language)
    backend = router.select(language, startup=True)

    if chunk_seconds is None:
        if not live_audio:
            chunk_seconds = 30.0
        elif backend == "whisper_api":
            chunk_seconds = whisper_api_chunk_seconds()
        else:
            chunk_seconds = 5.0

    if backend == "local_whisper":
        if hub_local_only_from_env():
            log("Echo: ASR loading from local Hugging Face cache only (Hub download disabled)…")
        elif live_audio and live_system_authorized:
            log("Echo: loading local Kotoba-Whisper for live capture (mic + system output via authorized command)…")
        elif live_audio:
            log("Echo: loading local Kotoba-Whisper on the main thread for live capture…")
        else:
            log("Echo: loading local Kotoba-Whisper on the main thread (first run may download ~3GB)…")
    elif backend == "apple_on_device":
        log("Echo: preparing Apple on-device speech recognition…")
    else:
        log("Echo: using Whisper API for speech recognition…")

    source = router.ensure_ready()
    return router, source, float(chunk_seconds)
