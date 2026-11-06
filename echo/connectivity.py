"""Connectivity: LLM endpoint reachability (Insights / Chat) and optional Hub-local ASR."""

from __future__ import annotations

import os
import socket
from urllib.parse import urlparse

from echo.llm import _litellm_openai_config, configured_llm_label


def _env_truthy(name: str) -> bool:
    v = os.environ.get(name)
    if v is None:
        return False
    return v.strip().lower() in ("1", "true", "yes", "on")


def hub_local_only_from_env() -> bool:
    """When True, ASR ``from_pretrained`` must use only on-disk Hub cache."""
    if _env_truthy("ECHO_HF_LOAD_LOCAL_ONLY"):
        return True
    if _env_truthy("HF_HUB_OFFLINE"):
        return True
    v = (os.environ.get("TRANSFORMERS_OFFLINE") or "").strip().upper()
    if v in ("1", "TRUE", "YES", "ON"):
        return True
    if _env_truthy("ECHO_OFFLINE"):
        return True
    if _env_truthy("ECHO_HF_LOCAL_ONLY"):
        return True
    return False


def echo_env_hub_local_only() -> bool:
    """User explicitly wants Hugging Face weights loaded from cache only (no Hub download)."""
    return _env_truthy("ECHO_OFFLINE") or _env_truthy("ECHO_HF_LOCAL_ONLY")


def _tcp_probe(host: str, port: int, *, timeout: float = 1.25) -> bool:
    try:
        with socket.create_connection((host, int(port)), timeout=timeout):
            return True
    except OSError:
        return False


def _probe_openai_compatible_base(base: str) -> bool:
    raw = base.strip()
    if not raw:
        return False
    url = raw if "://" in raw else f"https://{raw}"
    parsed = urlparse(url)
    host = parsed.hostname
    if not host:
        return False
    port = parsed.port
    if port is None:
        port = 443 if (parsed.scheme or "https") == "https" else 80
    return _tcp_probe(host, int(port))


def echo_llm_online() -> bool:
    """Whether the **configured** LLM backend looks reachable (TCP to API host).

    Used to gate Insights and Chat. Does **not** imply Hugging Face Hub or web search reachability.
    """
    if _env_truthy("ECHO_OFFLINE"):
        return False
    if _env_truthy("ECHO_ASSUME_ONLINE"):
        return True

    label = configured_llm_label()
    if label == "none":
        return False
    if label == "anthropic":
        return _tcp_probe("api.anthropic.com", 443)
    if label == "openai":
        return _tcp_probe("api.openai.com", 443)
    if label.startswith("openai-compatible"):
        cfg = _litellm_openai_config()
        if not cfg:
            return False
        base_url, _, _ = cfg
        return _probe_openai_compatible_base(base_url)
    return False


def echo_connectivity_online() -> bool:
    """Deprecated alias: use :func:`echo_llm_online`."""
    return echo_llm_online()


def probe_whisper_api() -> bool:
    """Whether Whisper API credentials are set and the API host looks reachable."""
    if _env_truthy("ECHO_OFFLINE"):
        return False
    if _env_truthy("ECHO_ASSUME_ONLINE"):
        return True
    from echo.asr_api import whisper_api_config

    cfg = whisper_api_config()
    if not cfg:
        return False
    base_url, _, _ = cfg
    return _probe_openai_compatible_base(base_url)
