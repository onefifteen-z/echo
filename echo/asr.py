"""Local ASR using Kotoba-Whisper weights (standard Whisper stack, no diarization deps)."""

from __future__ import annotations

import logging
import os
import threading
import warnings
from dataclasses import dataclass
from pathlib import Path

# Re-exported for backwards compatibility (existing call sites import from echo.asr).
from echo.text_utils import format_timestamp_hms

__all__ = [
    "MODEL_ID_DEFAULT",
    "TranscribeSession",
    "ensure_model_loaded",
    "format_timestamp_hms",
    "get_last_model_load_source",
    "transcribe_mono16k_chunk",
]

MODEL_ID_DEFAULT = "kotoba-tech/kotoba-whisper-v2.2"


@dataclass
class _LoadedAsr:
    """Snapshot of the currently-loaded Whisper assets and inference metadata."""

    model_id: str
    device: str
    processor: object
    model: object
    dtype: object
    max_length: int
    source: str  # "local-cache-only" | "local-cache" | "hub-fetch"


_lock = threading.Lock()
_state: _LoadedAsr | None = None
_runtime_configured = False
_hf_http_configured = False


def _configure_runtime_once() -> None:
    """Idempotently configure environment variables / loggers that only matter once ASR runs."""
    global _runtime_configured
    if _runtime_configured:
        return
    # Avoid tokenizer/fork interactions when HF/transformers run from thread pools (macOS fds issues).
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    logging.getLogger("transformers.generation.utils").setLevel(logging.ERROR)
    _runtime_configured = True


def _require_asr_deps() -> None:
    try:
        import torch  # noqa: F401
        import torchaudio  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "ASR dependencies missing. Install: pip install -e '.[asr]'"
        ) from e


def _env_truthy(name: str) -> bool:
    v = os.environ.get(name)
    if v is None:
        return False
    return v.strip().lower() in ("1", "true", "yes", "on")


def _exception_chain_mentions_tls(exc: BaseException) -> bool:
    blob: list[str] = []
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        blob.append(str(cur).lower())
        cur = cur.__cause__ or cur.__context__
    text = " ".join(blob)
    return any(k in text for k in ("ssl", "certificate", "cert verify", "tls", "self-signed"))


def _configure_hf_hub_http() -> None:
    """Point Hugging Face Hub's httpx client at a custom CA bundle or disable TLS verify (last resort).

    Corporate proxies often use a private CA; set ``ECHO_HF_CA_BUNDLE=/path/to/corp.pem`` or export a
    PEM bundle via ``SSL_CERT_FILE`` / ``REQUESTS_CA_BUNDLE``. If you must bypass verification (not
    recommended), set ``ECHO_HF_INSECURE_SSL=1``.
    """
    global _hf_http_configured
    if _hf_http_configured:
        return

    ca = (
        os.environ.get("ECHO_HF_CA_BUNDLE")
        or os.environ.get("SSL_CERT_FILE")
        or os.environ.get("REQUESTS_CA_BUNDLE")
    )
    ca_path = Path(ca).expanduser() if ca else None
    ca_ok = ca_path is not None and ca_path.is_file()
    insecure = _env_truthy("ECHO_HF_INSECURE_SSL")

    if not ca_ok and not insecure:
        _hf_http_configured = True
        return

    import httpx
    from huggingface_hub import set_async_client_factory, set_client_factory
    from huggingface_hub.utils import _http as hf_http

    verify: bool | str = True
    if ca_ok:
        verify = str(ca_path)
    elif insecure:
        verify = False
        warnings.warn(
            "ECHO_HF_INSECURE_SSL is set: TLS certificate verification is disabled for Hugging Face Hub "
            "downloads. Use only on trusted networks.",
            UserWarning,
            stacklevel=2,
        )

    def _client_factory() -> httpx.Client:
        return httpx.Client(
            event_hooks={"request": [hf_http.hf_request_event_hook]},
            follow_redirects=True,
            timeout=None,
            verify=verify,
        )

    def _async_client_factory() -> httpx.AsyncClient:
        return httpx.AsyncClient(
            event_hooks={
                "request": [hf_http.async_hf_request_event_hook],
                "response": [hf_http.async_hf_response_event_hook],
            },
            follow_redirects=True,
            timeout=None,
            verify=verify,
        )

    set_client_factory(_client_factory)
    set_async_client_factory(_async_client_factory)
    _hf_http_configured = True


def _load_local_only() -> bool:
    """When True, ``from_pretrained`` uses only the on-disk Hub cache (no download)."""
    from echo.connectivity import hub_local_only_from_env

    return hub_local_only_from_env()


def _resolve_available_device(name: str) -> str:
    """Return ``name`` if usable, else fall back to a working device (cpu)."""
    import torch

    if name == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if name == "mps":
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            return "mps"
        return "cpu"
    return name


def _pick_device(explicit: str | None) -> str:
    """Resolve the preferred torch device from explicit arg / env / availability."""
    import torch

    if explicit:
        return _resolve_available_device(explicit)

    env = os.environ.get("ECHO_WHISPER_DEVICE")
    if env in ("cuda", "mps", "cpu"):
        return _resolve_available_device(env)

    if torch.cuda.is_available():
        return "cuda"
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return "mps"
    return "cpu"


def _load_whisper_assets(model_id: str, dtype: object, *, local_only: bool) -> tuple[object, object]:
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    proc = WhisperProcessor.from_pretrained(model_id, local_files_only=local_only)
    mdl = WhisperForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=dtype,
        local_files_only=local_only,
    )
    return proc, mdl


def _whisper_max_length(model: object) -> int:
    """Decoder length cap for Whisper generation (typically ``max_target_positions`` = 448)."""
    cfg = getattr(model, "config", None)
    cap = int(getattr(cfg, "max_target_positions", None) or getattr(cfg, "max_length", None) or 448)
    return max(32, cap)


def _load_state(model_id: str, device: str | None) -> _LoadedAsr:
    """Load (or reuse) a Whisper model snapshot. Caller must hold ``_lock``."""
    import torch

    dev = _pick_device(device)
    dtype = torch.float16 if dev == "cuda" else torch.float32

    _configure_hf_hub_http()
    local_only = _load_local_only()
    try:
        if local_only:
            proc, mdl = _load_whisper_assets(model_id, dtype, local_only=True)
            source = "local-cache-only"
        else:
            # Avoid Hub HEAD/etag checks when a complete snapshot already exists locally.
            try:
                proc, mdl = _load_whisper_assets(model_id, dtype, local_only=True)
                source = "local-cache"
            except OSError:
                proc, mdl = _load_whisper_assets(model_id, dtype, local_only=False)
                source = "hub-fetch"
    except OSError as e:
        if _exception_chain_mentions_tls(e):
            raise OSError(
                "TLS / certificate error while contacting Hugging Face (common behind corporate proxies).\n"
                "Options:\n"
                "  1) Merge your corporate root CA into a PEM bundle and set one of:\n"
                "       export ECHO_HF_CA_BUNDLE=/path/to/combined.pem\n"
                "       export SSL_CERT_FILE=...  or  REQUESTS_CA_BUNDLE=...\n"
                "  2) As a last resort on a trusted network:\n"
                "       export ECHO_HF_INSECURE_SSL=1\n"
                "If the model is already cached locally, set `ECHO_HF_LOCAL_ONLY=1` (or `HF_HUB_OFFLINE=1`) "
                "to force offline cache loading.\n"
                f"Underlying error: {e}"
            ) from e
        if local_only:
            raise OSError(
                "Local-only Hub load is enabled (ECHO_HF_LOAD_LOCAL_ONLY / HF_HUB_OFFLINE / ECHO_OFFLINE), "
                "but the full model snapshot is not present in the cache.\n"
                "Run once online to download weights, or point HF_HOME at a cache that already contains this model.\n"
                f"Underlying error: {e}"
            ) from e
        raise

    mdl.eval()
    mdl.to(dev)
    gc = getattr(mdl, "generation_config", None)
    if gc is not None:
        if getattr(gc, "forced_decoder_ids", None):
            gc.forced_decoder_ids = None
        if getattr(gc, "suppress_tokens", None) is not None:
            gc.suppress_tokens = None
        if getattr(gc, "begin_suppress_tokens", None) is not None:
            gc.begin_suppress_tokens = None

    actual_dtype = next(mdl.parameters()).dtype
    return _LoadedAsr(
        model_id=model_id,
        device=dev,
        processor=proc,
        model=mdl,
        dtype=actual_dtype,
        max_length=_whisper_max_length(mdl),
        source=source,
    )


def ensure_model_loaded(model_id: str = MODEL_ID_DEFAULT, device: str | None = None) -> tuple[str, object, object]:
    """Load WhisperProcessor + WhisperForConditionalGeneration once (thread-safe).

    Returns ``(device, processor, model)`` for the loaded snapshot. When a model is already loaded,
    the previously-resolved device is returned regardless of the ``device`` argument so callers do
    not silently move tensors to a different backend than the model lives on.
    """
    global _state
    _configure_runtime_once()
    _require_asr_deps()

    with _lock:
        if _state is not None and _state.model_id == model_id:
            return _state.device, _state.processor, _state.model
        _state = _load_state(model_id, device)
        return _state.device, _state.processor, _state.model


def get_last_model_load_source() -> str | None:
    """Load source for the most recent successful model initialization."""
    return _state.source if _state is not None else None


def _load_audio_mono_16k(path: Path):
    """Decode to mono 16 kHz float32. Prefer torchaudio; fall back to librosa if TorchCodec/ffmpeg chain fails."""
    import numpy as np

    def _from_torchaudio() -> tuple[np.ndarray, int]:
        import torchaudio

        wav, sr = torchaudio.load(str(path))
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        wav = wav.squeeze(0)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        audio = wav.detach().cpu().numpy().astype(np.float32)
        return audio.reshape(-1), 16000

    def _from_librosa() -> tuple[np.ndarray, int]:
        import librosa

        audio, _sr = librosa.load(str(path), sr=16000, mono=True)
        return np.asarray(audio, dtype=np.float32).reshape(-1), 16000

    def _looks_like_torchcodec_failure(exc: BaseException) -> bool:
        """torchaudio decode failures that should fall back to librosa.

        ``ImportError`` typically means torchcodec is not installed; otherwise we sniff the message
        because torchaudio raises various subclasses across versions.
        """
        if isinstance(exc, ImportError):
            return True
        err = str(exc).lower()
        return "torchcodec" in err or "torch codec" in err

    try:
        return _from_torchaudio()
    except Exception as e:  # noqa: BLE001 — decode stack varies by torchaudio/torchcodec version
        if not _looks_like_torchcodec_failure(e):
            raise
        try:
            return _from_librosa()
        except ImportError as ie:
            raise RuntimeError(
                "This torchaudio build needs TorchCodec for this file. Install: pip install torchcodec\n"
                "and ensure ffmpeg is on your PATH (needed for MP3 and many formats).\n"
                "Alternatively: pip install librosa\n"
                f"Original error: {e}"
            ) from ie
        except Exception as e2:  # noqa: BLE001
            raise RuntimeError(
                "torchaudio decode failed; librosa fallback also failed.\n"
                "Install torchcodec + ffmpeg, or verify librosa can load this file standalone.\n"
                f"torchaudio: {e}\nlibrosa: {e2}"
            ) from e2


def _default_asr_language_from_env() -> str:
    env_lang = (os.environ.get("ECHO_WHISPER_LANGUAGE") or "japanese").strip().lower() or "japanese"
    if env_lang in ("japanese", "english", "chinese", "ja", "en", "zh"):
        return env_lang
    return "japanese"


def _transcribe_local_whisper_chunk(
    samples: "object",
    *,
    model_id: str = MODEL_ID_DEFAULT,
    device: str | None = None,
    language: str = "japanese",
) -> str:
    """Decode one mono float32 chunk at 16 kHz (local Kotoba-Whisper)."""
    import numpy as np
    import torch

    if isinstance(samples, np.ndarray):
        x = samples.reshape(-1)
        if x.dtype != np.float32:
            x = x.astype(np.float32, copy=False)
        elif not x.flags.c_contiguous:
            x = np.ascontiguousarray(x, dtype=np.float32)
    else:
        x = np.asarray(samples, dtype=np.float32).reshape(-1)
    if x.size < int(0.25 * 16000):
        return ""

    ensure_model_loaded(model_id, device)
    state = _state
    assert state is not None

    inputs = state.processor(x, sampling_rate=16000, return_tensors="pt")
    feats = inputs.input_features.to(state.device, dtype=state.dtype)
    attn_mask = torch.ones(feats.shape[:-1], dtype=torch.long, device=state.device)
    with torch.inference_mode():
        pred_ids = state.model.generate(
            feats,
            attention_mask=attn_mask,
            language=language,
            task="transcribe",
            max_length=state.max_length,
        )
    return state.processor.batch_decode(pred_ids, skip_special_tokens=True)[0].strip()


def transcribe_mono16k_chunk(
    samples: "object",
    *,
    model_id: str = MODEL_ID_DEFAULT,
    device: str | None = None,
    language: str | None = None,
) -> str:
    """Decode one mono float32 chunk at 16 kHz via the active ASR backend."""
    from echo.asr_routing import get_router

    lang = language if language is not None else _default_asr_language_from_env()
    return get_router().transcribe_window(samples, language=lang)


class TranscribeSession:
    """Incremental transcription for one file (one segment per `next_segment()` call)."""

    def __init__(
        self,
        path: Path,
        *,
        model_id: str = MODEL_ID_DEFAULT,
        chunk_seconds: float = 30.0,
        device: str | None = None,
    ) -> None:
        _require_asr_deps()
        self._model_id = model_id
        self._chunk_seconds = chunk_seconds
        self._device = device
        self._audio, self._sr = _load_audio_mono_16k(path)
        if self._sr != 16000:
            raise RuntimeError("internal resample error")
        self._chunk_len = int(chunk_seconds * self._sr)
        self._index = 0
        from echo.asr_routing import get_router

        router = get_router()
        if router.active == "local_whisper" and not router.local_preloaded:
            router.ensure_ready()

    def next_segment(self) -> tuple[float, float, str] | None:
        start = self._index * self._chunk_len
        if start >= len(self._audio):
            return None
        end = min(start + self._chunk_len, len(self._audio))
        chunk = self._audio[start:end]
        if chunk.size < int(0.25 * self._sr):
            return None

        text = transcribe_mono16k_chunk(
            chunk,
            model_id=self._model_id,
            device=self._device,
        )
        self._index += 1
        return (start / self._sr, end / self._sr, text)
