"""Optional VAD + denoise stages for the live ASR pipeline.

Both stages are opt-in and degrade safely: missing optional dependencies, model load
failures, and per-call exceptions are swallowed (with a single one-time ``notify`` warning)
so the live pipeline always keeps running.

Public surface:

- :func:`vad_speech_ratio` -- fraction (0..1) of a chunk classified as speech, with a hybrid
  ``silero -> webrtc -> off`` fallback chain.
- :func:`denoise` -- spectral-gating noise reduction via ``noisereduce`` (stationary mode).
"""

from __future__ import annotations

from typing import Callable

# ---------------------------------------------------------------------------
# State / cache
# ---------------------------------------------------------------------------

# Backend that ``vad_speech_ratio`` should actually use after fallback resolution; key is the
# requested backend, value is the resolved one ("off" means "give up, no real VAD available").
_vad_resolved: dict[str, str] = {}
_vad_warned: set[str] = set()
_silero_model: object | None = None
_silero_torch: object | None = None
_webrtc_instances: dict[int, object] = {}

_denoise_resolved: dict[str, str] = {}
_denoise_warned: set[str] = set()
_noisereduce_module: object | None = None


def _warn_once(notify: Callable[[str], None] | None, key: str, bucket: set[str], msg: str) -> None:
    if key in bucket:
        return
    bucket.add(key)
    if notify is None:
        return
    try:
        notify(msg)
    except Exception:  # noqa: BLE001 -- debug hook must never break the pipeline
        pass


# ---------------------------------------------------------------------------
# VAD
# ---------------------------------------------------------------------------


def _load_silero(notify: Callable[[str], None] | None) -> tuple[object, object] | None:
    """Lazy-load the Silero VAD model + torch handle. Returns ``None`` on failure."""
    global _silero_model, _silero_torch
    if _silero_model is not None and _silero_torch is not None:
        return _silero_model, _silero_torch
    try:
        import torch
        from silero_vad import load_silero_vad  # type: ignore[import-not-found]
    except Exception as e:  # noqa: BLE001 -- ImportError or transitive issues
        _warn_once(
            notify,
            "silero-import",
            _vad_warned,
            f"Silero VAD unavailable ({e!s}); install with `pip install silero-vad torch`. "
            "Falling back to next VAD backend.",
        )
        return None
    try:
        model = load_silero_vad()
    except Exception as e:  # noqa: BLE001
        _warn_once(
            notify,
            "silero-load",
            _vad_warned,
            f"Silero VAD model failed to load ({e!s}); falling back to next VAD backend.",
        )
        return None
    _silero_model = model
    _silero_torch = torch
    return model, torch


def _silero_speech_ratio(
    samples: object,
    *,
    sr: int,
    threshold: float,
    notify: Callable[[str], None] | None,
) -> float | None:
    """Return per-frame speech ratio for Silero, or ``None`` to trigger the next fallback."""
    loaded = _load_silero(notify)
    if loaded is None:
        return None
    model, torch = loaded
    import numpy as np

    arr = np.asarray(samples, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return 0.0

    # Silero v5 supports 512 samples @ 16k (32 ms) and 256 samples @ 8k. We force 16k upstream.
    frame_len = 512 if sr == 16000 else 256
    if arr.size < frame_len:
        return 0.0

    n_frames = arr.size // frame_len
    truncated = arr[: n_frames * frame_len]
    try:
        tensor = torch.from_numpy(truncated).reshape(n_frames, frame_len)
        positives = 0
        with torch.inference_mode():
            for i in range(n_frames):
                prob = float(model(tensor[i], sr).item())
                if prob >= threshold:
                    positives += 1
    except Exception as e:  # noqa: BLE001 -- model may raise on weird inputs
        _warn_once(
            notify,
            "silero-call",
            _vad_warned,
            f"Silero VAD call failed ({e!s}); falling back to next VAD backend.",
        )
        return None
    return positives / float(n_frames)


def _load_webrtc(aggressiveness: int, notify: Callable[[str], None] | None) -> object | None:
    agg = max(0, min(3, int(aggressiveness)))
    cached = _webrtc_instances.get(agg)
    if cached is not None:
        return cached
    try:
        import webrtcvad  # type: ignore[import-not-found]
    except Exception as e:  # noqa: BLE001
        _warn_once(
            notify,
            "webrtc-import",
            _vad_warned,
            f"WebRTC VAD unavailable ({e!s}); install with `pip install webrtcvad-wheels`. "
            "Disabling VAD for this session.",
        )
        return None
    try:
        inst = webrtcvad.Vad(agg)
    except Exception as e:  # noqa: BLE001
        _warn_once(
            notify,
            "webrtc-init",
            _vad_warned,
            f"WebRTC VAD init failed ({e!s}); disabling VAD for this session.",
        )
        return None
    _webrtc_instances[agg] = inst
    return inst


def _webrtc_speech_ratio(
    samples: object,
    *,
    sr: int,
    aggressiveness: int,
    notify: Callable[[str], None] | None,
) -> float | None:
    """30 ms frames at 16 kHz int16. Returns ``None`` to trigger fallback to ``off``."""
    inst = _load_webrtc(aggressiveness, notify)
    if inst is None:
        return None
    import numpy as np

    arr = np.asarray(samples, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return 0.0

    # WebRTC VAD only accepts 8/16/32k and 10/20/30 ms frames.
    if sr not in (8000, 16000, 32000):
        _warn_once(
            notify,
            f"webrtc-sr-{sr}",
            _vad_warned,
            f"WebRTC VAD does not support {sr} Hz; skipping VAD.",
        )
        return None
    frame_len = (sr * 30) // 1000  # 30 ms frame
    if arr.size < frame_len:
        return 0.0

    n_frames = arr.size // frame_len
    pcm16 = np.clip(arr[: n_frames * frame_len] * 32767.0, -32768.0, 32767.0).astype(np.int16)
    blob = pcm16.tobytes()
    bytes_per_frame = frame_len * 2
    positives = 0
    try:
        for i in range(n_frames):
            frame = blob[i * bytes_per_frame : (i + 1) * bytes_per_frame]
            if inst.is_speech(frame, sr):
                positives += 1
    except Exception as e:  # noqa: BLE001
        _warn_once(
            notify,
            "webrtc-call",
            _vad_warned,
            f"WebRTC VAD call failed ({e!s}); skipping VAD for this chunk.",
        )
        return None
    return positives / float(n_frames)


def vad_speech_ratio(
    samples: object,
    *,
    backend: str,
    sr: int = 16000,
    speech_prob_threshold: float = 0.5,
    aggressiveness: int = 2,
    notify: Callable[[str], None] | None = None,
) -> float:
    """Return the fraction of ``samples`` classified as speech (0.0..1.0).

    ``backend == "off"`` always returns ``1.0`` (treat the chunk as fully voiced).

    For ``backend == "silero"`` we silently fall back to ``webrtc`` if Silero is unavailable,
    and to ``off`` if both fail. ``backend == "webrtc"`` falls back directly to ``off``.
    """
    backend = (backend or "off").strip().lower()
    if backend == "off":
        return 1.0
    if backend not in ("silero", "webrtc"):
        return 1.0

    resolved = _vad_resolved.get(backend, backend)

    if resolved == "silero":
        ratio = _silero_speech_ratio(
            samples, sr=sr, threshold=speech_prob_threshold, notify=notify,
        )
        if ratio is not None:
            return ratio
        # Silero declined -> step down to webrtc on subsequent calls too.
        resolved = "webrtc"
        _vad_resolved[backend] = resolved

    if resolved == "webrtc":
        ratio = _webrtc_speech_ratio(
            samples, sr=sr, aggressiveness=aggressiveness, notify=notify,
        )
        if ratio is not None:
            return ratio
        _vad_resolved[backend] = "off"

    return 1.0


# ---------------------------------------------------------------------------
# Denoise
# ---------------------------------------------------------------------------


def _load_noisereduce(notify: Callable[[str], None] | None) -> object | None:
    global _noisereduce_module
    if _noisereduce_module is not None:
        return _noisereduce_module
    try:
        import noisereduce as nr  # type: ignore[import-not-found]
    except Exception as e:  # noqa: BLE001
        _warn_once(
            notify,
            "noisereduce-import",
            _denoise_warned,
            f"noisereduce unavailable ({e!s}); install with `pip install noisereduce scipy`. "
            "Denoising disabled for this session.",
        )
        return None
    _noisereduce_module = nr
    return nr


def denoise(
    samples: object,
    *,
    backend: str,
    sr: int = 16000,
    prop_decrease: float = 0.85,
    notify: Callable[[str], None] | None = None,
) -> object:
    """Return a denoised float32 array of the same shape (or the input on any failure)."""
    backend = (backend or "off").strip().lower()
    if backend == "off" or backend not in ("noisereduce",):
        return samples

    resolved = _denoise_resolved.get(backend, backend)
    if resolved == "off":
        return samples

    nr = _load_noisereduce(notify)
    if nr is None:
        _denoise_resolved[backend] = "off"
        return samples

    import numpy as np

    arr = np.asarray(samples, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return arr

    try:
        out = nr.reduce_noise(
            y=arr,
            sr=sr,
            stationary=True,
            prop_decrease=float(max(0.0, min(1.0, prop_decrease))),
        )
    except Exception as e:  # noqa: BLE001 -- shape/dtype issues, scipy edge cases, etc.
        _warn_once(
            notify,
            "noisereduce-call",
            _denoise_warned,
            f"noisereduce call failed ({e!s}); passing audio through unchanged.",
        )
        return arr

    out_arr = np.asarray(out, dtype=np.float32).reshape(-1)
    if out_arr.size != arr.size:
        # Very unlikely but keeps downstream ASR window math intact.
        out_arr = np.resize(out_arr, arr.size).astype(np.float32, copy=False)
    return np.ascontiguousarray(out_arr, dtype=np.float32)


# ---------------------------------------------------------------------------
# Test helper
# ---------------------------------------------------------------------------


def reset_state() -> None:
    """Clear caches + one-time warning latches. Tests should call this in a fixture."""
    global _silero_model, _silero_torch, _noisereduce_module
    _vad_resolved.clear()
    _vad_warned.clear()
    _silero_model = None
    _silero_torch = None
    _webrtc_instances.clear()
    _denoise_resolved.clear()
    _denoise_warned.clear()
    _noisereduce_module = None
