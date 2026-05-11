"""Environment-driven configuration for the live ASR pipeline and insight scheduling.

Centralizes all ``ECHO_LIVE_*``, ``ECHO_UI_SEGMENT_*`` and ``ECHO_INSIGHT_*`` knobs so the values
live in one place, can be dumped at startup, and are easy to override in tests.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Mapping

_LIVE_STOP_WORDS_DEFAULT: tuple[str, ...] = (
    "that's all",
    "thank you",
    "over",
    "next topic",
    "以上",
    "以上です",
    "谢谢",
)


def _csv(value: str) -> tuple[str, ...]:
    return tuple(w.strip() for w in value.split(",") if w.strip())


def _truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in ("1", "true", "yes", "on")


def _normalize_choice(value: str | None, *, allowed: tuple[str, ...], default: str) -> str:
    """Lowercase + whitelist guard for env-driven enum knobs."""
    if value is None:
        return default
    v = value.strip().lower()
    return v if v in allowed else default


def _get_int(
    env: Mapping[str, str],
    key: str,
    default: int,
    *,
    low: int | None = None,
    high: int | None = None,
) -> int:
    try:
        v = int(env.get(key, str(default)))
    except (TypeError, ValueError):
        v = default
    if low is not None:
        v = max(low, v)
    if high is not None:
        v = min(high, v)
    return v


def _get_float(
    env: Mapping[str, str],
    key: str,
    default: float,
    *,
    low: float | None = None,
    high: float | None = None,
) -> float:
    try:
        v = float(env.get(key, str(default)))
    except (TypeError, ValueError):
        v = default
    if low is not None:
        v = max(low, v)
    if high is not None:
        v = min(high, v)
    return v


@dataclass(frozen=True)
class LiveConfig:
    """Live ASR audio gating, segmentation, and noise-token policy."""

    # Voice gate (decides whether a chunk is even sent to Whisper).
    min_rms: float = 0.0035
    active_amp: float = 0.012
    min_active_ratio: float = 0.015

    # Chunking and overlap dedup.
    chunk_overlap_seconds: float = 0.3
    overlap_min_chars: int = 2

    # Live segmentation cadence for the draft transcript line.
    segment_pause_seconds: float = 2.0
    rewrite_stable_seconds: float = 10.0
    asr_log_interval_seconds: float = 2.0

    # Noise-token suppression (very short tokens often hallucinated from low-energy chunks).
    noise_tokens: tuple[str, ...] = ("ごめん",)
    always_drop_noise_tokens: tuple[str, ...] = ("ごめん",)
    noise_max_chars: int = 4
    noise_max_rms_factor: float = 1.3
    noise_drop_at_segment_start: bool = True

    # Stop words used to indicate end-of-talk in legacy interval mode.
    stop_words: tuple[str, ...] = _LIVE_STOP_WORDS_DEFAULT

    # Meter / chart UI history width (samples).
    meter_history_width: int = 52

    # Live capture ``queue.Queue`` capacities (PortAudio callbacks vs ASR consumer).
    capture_out_queue_max: int = 512
    capture_mix_queue_max: int = 256
    # Live only: when ``out_q`` depth exceeds the threshold, discard the oldest ``batch`` blocks
    # before draining (0 threshold disables). Reduces ASR lag at the cost of lost audio.
    capture_out_q_drop_threshold: int = 400
    capture_out_q_drop_batch: int = 30

    # VAD (real per-frame model layered on top of the energy gate).
    vad_backend: str = "off"  # "off" | "silero" | "webrtc"
    vad_min_speech_ratio: float = 0.10
    vad_speech_prob_threshold: float = 0.5  # Silero per-frame probability cutoff
    vad_aggressiveness: int = 2  # WebRTC 0..3

    # Denoise stage (spectral gating before Whisper).
    denoise_backend: str = "off"  # "off" | "noisereduce"
    denoise_strength: float = 0.85  # noisereduce ``prop_decrease`` 0..1

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "LiveConfig":
        e = env if env is not None else os.environ
        stop_env = e.get("ECHO_LIVE_STOP_WORDS", "")
        stop_words = _csv(stop_env) if stop_env.strip() else _LIVE_STOP_WORDS_DEFAULT
        return cls(
            min_rms=_get_float(e, "ECHO_LIVE_MIN_RMS", 0.0035, low=0.0),
            active_amp=_get_float(e, "ECHO_LIVE_ACTIVE_AMP", 0.012, low=1e-4),
            min_active_ratio=_get_float(e, "ECHO_LIVE_MIN_ACTIVE_RATIO", 0.015, low=0.0, high=1.0),
            chunk_overlap_seconds=_get_float(e, "ECHO_LIVE_CHUNK_OVERLAP_SECONDS", 0.3, low=0.0),
            overlap_min_chars=_get_int(e, "ECHO_LIVE_OVERLAP_MIN_CHARS", 2, low=1),
            segment_pause_seconds=_get_float(e, "ECHO_LIVE_SEGMENT_PAUSE_SECONDS", 2.0, low=0.3),
            rewrite_stable_seconds=_get_float(e, "ECHO_LIVE_REWRITE_STABLE_SECONDS", 10.0, low=0.0),
            asr_log_interval_seconds=_get_float(e, "ECHO_LIVE_ASR_LOG_INTERVAL_SECONDS", 2.0, low=0.0),
            noise_tokens=_csv(e.get("ECHO_LIVE_NOISE_TOKENS", "ごめん")),
            always_drop_noise_tokens=_csv(e.get("ECHO_LIVE_ALWAYS_DROP_NOISE_TOKENS", "ごめん")),
            noise_max_chars=_get_int(e, "ECHO_LIVE_NOISE_MAX_CHARS", 4, low=1),
            noise_max_rms_factor=_get_float(e, "ECHO_LIVE_NOISE_MAX_RMS_FACTOR", 1.3, low=0.1),
            noise_drop_at_segment_start=_truthy(e.get("ECHO_LIVE_NOISE_DROP_AT_SEGMENT_START", "1")),
            stop_words=stop_words,
            meter_history_width=_get_int(e, "ECHO_LIVE_METER_HISTORY_WIDTH", 52, low=16),
            capture_out_queue_max=_get_int(e, "ECHO_LIVE_CAPTURE_OUT_QUEUE_MAX", 512, low=32),
            capture_mix_queue_max=_get_int(e, "ECHO_LIVE_CAPTURE_MIX_QUEUE_MAX", 256, low=16),
            capture_out_q_drop_threshold=_get_int(
                e, "ECHO_LIVE_CAPTURE_OUT_Q_DROP_THRESHOLD", 400, low=0,
            ),
            capture_out_q_drop_batch=_get_int(e, "ECHO_LIVE_CAPTURE_OUT_Q_DROP_BATCH", 30, low=1),
            vad_backend=_normalize_choice(
                e.get("ECHO_LIVE_VAD"),
                allowed=("off", "silero", "webrtc"),
                default="off",
            ),
            vad_min_speech_ratio=_get_float(
                e, "ECHO_LIVE_VAD_MIN_SPEECH_RATIO", 0.10, low=0.0, high=1.0,
            ),
            vad_speech_prob_threshold=_get_float(
                e, "ECHO_LIVE_VAD_SPEECH_PROB_THRESHOLD", 0.5, low=0.0, high=1.0,
            ),
            vad_aggressiveness=_get_int(
                e, "ECHO_LIVE_VAD_AGGRESSIVENESS", 2, low=0, high=3,
            ),
            denoise_backend=_normalize_choice(
                e.get("ECHO_LIVE_DENOISE"),
                allowed=("off", "noisereduce"),
                default="off",
            ),
            denoise_strength=_get_float(
                e, "ECHO_LIVE_DENOISE_STRENGTH", 0.85, low=0.0, high=1.0,
            ),
        )


@dataclass(frozen=True)
class UISegmentConfig:
    """UI-side caps that decide when a live draft line should auto-finalize."""

    semantic_min_chars: int = 80
    max_seconds: float = 25.0
    max_chars: int = 460

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "UISegmentConfig":
        e = env if env is not None else os.environ
        return cls(
            semantic_min_chars=_get_int(e, "ECHO_UI_SEGMENT_SEMANTIC_MIN_CHARS", 80, low=1),
            max_seconds=_get_float(e, "ECHO_UI_SEGMENT_MAX_SECONDS", 25.0, low=4.0),
            max_chars=_get_int(e, "ECHO_UI_SEGMENT_MAX_CHARS", 460, low=80),
        )


@dataclass(frozen=True)
class InsightConfig:
    """Triggers / dedup thresholds for insight passes.

    Fields other than ``enable_transcript_rewrite`` are read in the TUI for diagnostics, but the
    worker applies finalized-segment rules, rate limits, capture backlog skips, and duplicate-input
    dedup only when ``EchoApp.live_audio`` is true (``--live``).
    """

    live_min_seconds: float = 10.0
    pause_min_chars: int = 80
    finalized_segment_min_chars: int = 100
    skip_min_chars: int = 50
    pause_seconds: float = 3.0
    live_min_interval: float = 45.0
    # Live only: skip starting an insight pass when capture ``out_q`` depth >= this (0=off).
    pause_capture_q: int = 400
    enable_transcript_rewrite: bool = True

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "InsightConfig":
        e = env if env is not None else os.environ
        return cls(
            live_min_seconds=_get_float(e, "ECHO_INSIGHT_LIVE_MIN_SECONDS", 10.0, low=0.5),
            pause_min_chars=_get_int(e, "ECHO_INSIGHT_PAUSE_MIN_CHARS", 80, low=1),
            finalized_segment_min_chars=_get_int(e, "ECHO_INSIGHT_FINALIZED_SEGMENT_MIN_CHARS", 100, low=1),
            skip_min_chars=_get_int(e, "ECHO_INSIGHT_SKIP_MIN_CHARS", 50, low=1),
            pause_seconds=_get_float(e, "ECHO_INSIGHT_PAUSE_SECONDS", 3.0, low=0.2),
            live_min_interval=_get_float(e, "ECHO_INSIGHT_LIVE_MIN_INTERVAL", 45.0, low=0.5),
            pause_capture_q=_get_int(e, "ECHO_INSIGHT_PAUSE_CAPTURE_Q", 400, low=0),
            enable_transcript_rewrite=_truthy(e.get("ECHO_ENABLE_TRANSCRIPT_REWRITE", "1")),
        )


@dataclass(frozen=True)
class EchoConfig:
    """Top-level config bundle so callers / debug logs can dump everything at once."""

    live: LiveConfig = field(default_factory=LiveConfig)
    ui_segment: UISegmentConfig = field(default_factory=UISegmentConfig)
    insight: InsightConfig = field(default_factory=InsightConfig)
    # Max transcript lines sent as context per in-app chat submit (tail window).
    chat_context_max_lines: int = 600

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "EchoConfig":
        e = env if env is not None else os.environ
        return cls(
            live=LiveConfig.from_env(e),
            ui_segment=UISegmentConfig.from_env(e),
            insight=InsightConfig.from_env(e),
            chat_context_max_lines=_get_int(e, "ECHO_CHAT_CONTEXT_MAX_LINES", 600, low=50),
        )
