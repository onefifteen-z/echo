"""Whisper transcription via OpenAI-compatible HTTP API."""

from __future__ import annotations

import io
import os
import wave
from typing import Any

from echo.llm import _litellm_openai_config

_openai_sync_client_cache: dict[str, Any] = {}


def whisper_api_model_name() -> str:
    for name in ("ECHO_WHISPER_API_MODEL", "ECHO_OPENAI_WHISPER_MODEL"):
        v = (os.environ.get(name) or "").strip()
        if v:
            return v
    return "whisper-1"


def whisper_api_config() -> tuple[str, str, str] | None:
    """Return ``(base_url, api_key, whisper_model)`` when API ASR is configured."""
    model = whisper_api_model_name()
    whisper_base = (
        (os.environ.get("ECHO_WHISPER_API_BASE_URL") or os.environ.get("ECHO_WHISPER_BASE_URL") or "")
        .strip()
        .rstrip("/")
    )
    shared_key = (os.environ.get("API_KEY") or os.environ.get("ECHO_WHISPER_API_KEY") or "").strip()
    if whisper_base and shared_key:
        return whisper_base, shared_key, model

    litellm = _litellm_openai_config()
    if litellm:
        base, key, _chat_model = litellm
        if base and key:
            return base.rstrip("/"), key, model
    key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if key:
        return "https://api.openai.com/v1", key, model
    return None


def _get_sync_openai_client(base_url: str, api_key: str) -> Any:
    cache_key = f"{api_key}:{base_url}"
    if cache_key not in _openai_sync_client_cache:
        from openai import OpenAI

        timeout = float(os.environ.get("ECHO_WHISPER_API_TIMEOUT") or 20.0)
        _openai_sync_client_cache[cache_key] = OpenAI(
            api_key=api_key, base_url=base_url, timeout=timeout, max_retries=1
        )
    return _openai_sync_client_cache[cache_key]


def whisper_api_chunk_seconds(default: float = 8.0) -> float:
    try:
        return max(2.0, float(os.environ.get("ECHO_WHISPER_API_CHUNK_SECONDS") or default))
    except ValueError:
        return default


# Phrases Whisper emits on near-silent input (YouTube-caption training artifacts).
_SILENCE_HALLUCINATIONS = (
    "ご視聴ありがとうございました",
    "ご視聴ありがとうございます",
    "チャンネル登録",
    "おやすみなさい",
    "thank you for watching",
    "thanks for watching",
    "please subscribe",
    "subtitles by",
    "字幕",
    "请不吝点赞",
    "谢谢观看",
    "明镜与点点",
)

_QUIET_RMS = 0.01


def is_silence_hallucination(text: str, rms: float) -> bool:
    """True when ``text`` is a known Whisper filler phrase and the audio was quiet."""
    if rms >= _QUIET_RMS:
        return False
    t = (text or "").strip().lower()
    if not t:
        return False
    return any(p in t for p in _SILENCE_HALLUCINATIONS) or len(set(t)) <= 2


def echo_language_to_iso639(language: str) -> str | None:
    """Map Echo live language names to ISO-639-1 for the Whisper API."""
    key = (language or "").strip().lower()
    mapping = {
        "japanese": "ja",
        "ja": "ja",
        "english": "en",
        "en": "en",
        "chinese": "zh",
        "zh": "zh",
    }
    return mapping.get(key)


def _float32_mono_to_wav_bytes(samples: object) -> bytes:
    import numpy as np

    x = np.asarray(samples, dtype=np.float32).reshape(-1)
    if x.size == 0:
        x = np.zeros((1,), dtype=np.float32)
    clipped = np.clip(x, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def transcribe_mono16k_chunk_api(
    samples: object,
    *,
    language: str = "japanese",
    prompt: str | None = None,
) -> str:
    """Transcribe one mono 16 kHz float32 chunk via ``/audio/transcriptions``."""
    cfg = whisper_api_config()
    if not cfg:
        raise RuntimeError(
            "Whisper API ASR is not configured. Set BASE_URL+API_KEY (and ECHO_WHISPER_API_MODEL), "
            "or OPENAI_API_KEY."
        )

    import numpy as np

    x = np.asarray(samples, dtype=np.float32).reshape(-1)
    if x.size < int(0.25 * 16000):
        return ""

    base_url, api_key, model = cfg
    client = _get_sync_openai_client(base_url, api_key)
    wav_bytes = _float32_mono_to_wav_bytes(x)
    file_tuple = ("chunk.wav", io.BytesIO(wav_bytes), "audio/wav")

    iso = echo_language_to_iso639(language)
    kwargs: dict[str, Any] = {"model": model, "file": file_tuple}
    if iso:
        kwargs["language"] = iso
    if prompt:
        kwargs["prompt"] = prompt[-200:]

    resp = client.audio.transcriptions.create(**kwargs)
    text = str(getattr(resp, "text", None) or "").strip()
    rms = float(np.sqrt(np.mean(np.square(x)))) if x.size else 0.0
    if is_silence_hallucination(text, rms):
        return ""
    return text
