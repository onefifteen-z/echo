"""LiveConfig.from_env parsing + bounds for the new VAD / denoise knobs."""

from __future__ import annotations


def test_live_config_defaults_are_off() -> None:
    from echo.config import LiveConfig

    cfg = LiveConfig.from_env({})
    assert cfg.vad_backend == "off"
    assert cfg.denoise_backend == "off"
    assert cfg.vad_min_speech_ratio == 0.10
    assert cfg.vad_speech_prob_threshold == 0.5
    assert cfg.vad_aggressiveness == 2
    assert cfg.denoise_strength == 0.85


def test_live_config_accepts_silero_and_noisereduce() -> None:
    from echo.config import LiveConfig

    cfg = LiveConfig.from_env(
        {
            "ECHO_LIVE_VAD": "Silero",
            "ECHO_LIVE_VAD_MIN_SPEECH_RATIO": "0.25",
            "ECHO_LIVE_VAD_SPEECH_PROB_THRESHOLD": "0.7",
            "ECHO_LIVE_VAD_AGGRESSIVENESS": "1",
            "ECHO_LIVE_DENOISE": "noisereduce",
            "ECHO_LIVE_DENOISE_STRENGTH": "0.6",
        }
    )
    assert cfg.vad_backend == "silero"
    assert cfg.vad_min_speech_ratio == 0.25
    assert cfg.vad_speech_prob_threshold == 0.7
    assert cfg.vad_aggressiveness == 1
    assert cfg.denoise_backend == "noisereduce"
    assert cfg.denoise_strength == 0.6


def test_live_config_unknown_backends_fall_back_to_off() -> None:
    from echo.config import LiveConfig

    cfg = LiveConfig.from_env(
        {
            "ECHO_LIVE_VAD": "deepfilter",
            "ECHO_LIVE_DENOISE": "magic",
        }
    )
    assert cfg.vad_backend == "off"
    assert cfg.denoise_backend == "off"


def test_live_config_clamps_numeric_bounds() -> None:
    from echo.config import LiveConfig

    cfg = LiveConfig.from_env(
        {
            "ECHO_LIVE_VAD": "webrtc",
            "ECHO_LIVE_VAD_MIN_SPEECH_RATIO": "5.0",        # above 1.0 -> 1.0
            "ECHO_LIVE_VAD_SPEECH_PROB_THRESHOLD": "-3.0",  # below 0.0 -> 0.0
            "ECHO_LIVE_VAD_AGGRESSIVENESS": "9",            # above 3 -> 3
            "ECHO_LIVE_DENOISE": "noisereduce",
            "ECHO_LIVE_DENOISE_STRENGTH": "1.5",            # above 1.0 -> 1.0
        }
    )
    assert cfg.vad_min_speech_ratio == 1.0
    assert cfg.vad_speech_prob_threshold == 0.0
    assert cfg.vad_aggressiveness == 3
    assert cfg.denoise_strength == 1.0


def test_live_config_invalid_numbers_fall_back_to_defaults() -> None:
    from echo.config import LiveConfig

    cfg = LiveConfig.from_env(
        {
            "ECHO_LIVE_VAD": "silero",
            "ECHO_LIVE_VAD_MIN_SPEECH_RATIO": "not-a-float",
            "ECHO_LIVE_VAD_AGGRESSIVENESS": "abc",
            "ECHO_LIVE_DENOISE_STRENGTH": "",
        }
    )
    assert cfg.vad_min_speech_ratio == 0.10
    assert cfg.vad_aggressiveness == 2
    assert cfg.denoise_strength == 0.85
