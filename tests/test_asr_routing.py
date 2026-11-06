"""Tests for ASR backend routing (Apple on-device → Whisper API → local Kotoba)."""

from __future__ import annotations

import pytest

from echo.asr_routing import AsrRouter, AsrUnavailableError


class _Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def _router(
    requested: str | None = "auto",
    *,
    apple: dict[str, str] | None = None,
    api: bool = True,
    api_configured: bool = True,
    local: bool = True,
    platform: str = "darwin",
    clock: _Clock | None = None,
) -> AsrRouter:
    apple = apple or {}
    return AsrRouter(
        requested,
        apple_status=lambda lang: apple.get(lang, "unsupported"),
        api_usable=lambda: api,
        api_configured=lambda: api_configured,
        local_available=lambda: local,
        platform=platform,
        clock=clock or _Clock(),
    )


@pytest.fixture(autouse=True)
def _clear_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ECHO_OFFLINE", raising=False)
    monkeypatch.delenv("ECHO_ASR_BACKEND", raising=False)


def test_auto_prefers_installed_apple_model() -> None:
    r = _router(apple={"japanese": "installed"})
    assert r.select("japanese", startup=True) == "apple_on_device"
    assert r.is_streaming()


def test_auto_uses_api_when_apple_not_installed_for_language() -> None:
    r = _router(apple={"english": "installed", "japanese": "supported"})
    assert r.select("japanese", startup=True) == "whisper_api"


def test_auto_skips_apple_on_linux() -> None:
    r = _router(apple={"japanese": "installed"}, platform="linux")
    assert r.select("japanese", startup=True) == "whisper_api"


def test_auto_prefers_api_over_local_when_online() -> None:
    r = _router(api=True)
    assert r.select("japanese", startup=True) == "whisper_api"


def test_auto_offline_uses_local(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ECHO_OFFLINE", "1")
    r = _router(api=False)
    assert r.select("japanese", startup=True) == "local_whisper"


def test_auto_without_api_config_uses_local() -> None:
    r = _router(api=False, api_configured=False)
    assert r.select("japanese", startup=True) == "local_whisper"


def test_auto_never_cold_loads_local_mid_session() -> None:
    r = _router(api=False, api_configured=False)
    with pytest.raises(AsrUnavailableError, match="cannot be loaded mid-session"):
        r.select("japanese")
    r.local_preloaded = True
    assert r.select("japanese") == "local_whisper"


def test_forced_apple_errors_with_install_hint() -> None:
    r = _router("apple", apple={"japanese": "supported"})
    with pytest.raises(AsrUnavailableError, match="--install-apple-speech ja"):
        r.select("japanese", startup=True)


def test_forced_api_does_not_fall_back() -> None:
    r = _router("api", api=False, apple={"japanese": "installed"})
    with pytest.raises(AsrUnavailableError, match="unreachable"):
        r.select("japanese", startup=True)


def test_per_language_reselect_in_auto() -> None:
    r = _router(apple={"english": "installed"})
    assert r.select("english") == "apple_on_device"
    assert r.select("japanese") == "whisper_api"
    assert r.select("english") == "apple_on_device"


def test_single_failure_does_not_downgrade() -> None:
    r = _router(apple={"japanese": "installed"})
    r.select("japanese")
    assert r.record_failure("apple_on_device", "boom") is None
    assert r.record_failure("apple_on_device", "boom") is None
    assert r.active == "apple_on_device"


def test_repeated_failures_cool_down_then_recover() -> None:
    clock = _Clock()
    r = _router(apple={"japanese": "installed"}, clock=clock)
    r.select("japanese")
    for _ in range(AsrRouter.FAILURE_THRESHOLD - 1):
        r.record_failure("apple_on_device", "boom")
    assert r.record_failure("apple_on_device", "boom") == "whisper_api"
    assert r.active == "whisper_api"

    clock.t += AsrRouter.BASE_COOLDOWN - 1
    assert r.maybe_upgrade("japanese") is None
    clock.t += AsrRouter.UPGRADE_CHECK_SECONDS
    assert r.maybe_upgrade("japanese") == "apple_on_device"


def test_forced_mode_never_switches_on_failure() -> None:
    r = _router("apple", apple={"japanese": "installed"})
    r.select("japanese")
    for _ in range(AsrRouter.FAILURE_THRESHOLD * 2):
        assert r.record_failure("apple_on_device", "boom") is None
    assert r.active == "apple_on_device"


def test_ensure_ready_local_loads_on_caller_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, str | None]] = []

    def fake_ensure(model_id: str, device: str | None) -> tuple[str, object, object]:
        calls.append((model_id, device))
        return "cpu", object(), object()

    monkeypatch.setattr("echo.asr.ensure_model_loaded", fake_ensure)
    monkeypatch.setattr("echo.asr.get_last_model_load_source", lambda: "local-cache")
    r = AsrRouter("local", model_id="test-model", device="cpu", local_available=lambda: True)
    r.select("japanese", startup=True)
    assert r.ensure_ready() == "local-cache"
    assert calls == [("test-model", "cpu")]
    assert r.local_preloaded


def test_transcribe_window_passes_prompt_tail(monkeypatch: pytest.MonkeyPatch) -> None:
    import numpy as np

    prompts: list[str | None] = []

    def fake_api(samples: object, *, language: str, prompt: str | None = None) -> str:
        prompts.append(prompt)
        return "こんにちは"

    monkeypatch.setattr("echo.asr_api.transcribe_mono16k_chunk_api", fake_api)
    r = _router("api")
    r.select("japanese")
    x = np.zeros(16000, dtype=np.float32)
    r.transcribe_window(x, language="japanese")
    r.transcribe_window(x, language="japanese")
    assert prompts == [None, "こんにちは"]


def test_silence_hallucination_filter() -> None:
    from echo.asr_api import is_silence_hallucination

    assert is_silence_hallucination("ご視聴ありがとうございました", rms=0.001)
    assert not is_silence_hallucination("ご視聴ありがとうございました", rms=0.05)
    assert not is_silence_hallucination("本日の議題です", rms=0.001)


def test_whisper_api_config_separate_base_same_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    from echo.asr_api import whisper_api_config

    monkeypatch.setenv("API_KEY", "shared-secret")
    monkeypatch.setenv("BASE_URL", "https://chat-proxy.example/v1")
    monkeypatch.setenv("MODEL", "gpt-4o-mini")
    monkeypatch.setenv("ECHO_WHISPER_API_BASE_URL", "https://whisper-proxy.example/v1")
    monkeypatch.setenv("ECHO_WHISPER_API_MODEL", "whisper-large-v3")
    cfg = whisper_api_config()
    assert cfg == ("https://whisper-proxy.example/v1", "shared-secret", "whisper-large-v3")


def test_whisper_api_config_uses_litellm_base_and_separate_model(monkeypatch: pytest.MonkeyPatch) -> None:
    from echo.asr_api import whisper_api_config

    monkeypatch.delenv("ECHO_WHISPER_API_BASE_URL", raising=False)
    monkeypatch.delenv("ECHO_WHISPER_BASE_URL", raising=False)
    monkeypatch.setenv("BASE_URL", "https://proxy.example/v1")
    monkeypatch.setenv("API_KEY", "secret")
    monkeypatch.setenv("MODEL", "gpt-4o-mini")
    monkeypatch.setenv("ECHO_WHISPER_API_MODEL", "whisper-1")
    cfg = whisper_api_config()
    assert cfg == ("https://proxy.example/v1", "secret", "whisper-1")
