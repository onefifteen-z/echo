"""Local ASR using Kotoba-Whisper weights (standard Whisper stack, no diarization deps)."""

from __future__ import annotations

import logging
import os
import warnings

# Avoid tokenizer/fork interactions when HF/transformers run from thread pools (macOS fds issues).
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import threading

logging.getLogger("transformers.generation.utils").setLevel(logging.ERROR)
from pathlib import Path
from typing import Iterator

MODEL_ID_DEFAULT = "kotoba-tech/kotoba-whisper-v2.2"

_hf_http_configured = False

_lock = threading.Lock()
_processor = None
_model = None
_loaded_model_id: str | None = None
_last_model_load_source: str | None = None
_cached_dtype: object | None = None
_cached_max_length: int | None = None


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


def _pick_device(explicit: str | None) -> str:
    import torch

    if explicit:
        if explicit == "cuda" and not torch.cuda.is_available():
            return "cpu"
        if explicit == "mps" and not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
            return "cpu"
        return explicit

    env = os.environ.get("ECHO_WHISPER_DEVICE")
    if env in ("cuda", "mps", "cpu"):
        return _pick_device(env)

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
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


def ensure_model_loaded(model_id: str = MODEL_ID_DEFAULT, device: str | None = None) -> tuple[str, object, object]:
    """Load WhisperProcessor + WhisperForConditionalGeneration once (thread-safe)."""
    global _processor, _model, _loaded_model_id, _last_model_load_source
    _require_asr_deps()
    import torch

    dev = _pick_device(device)
    with _lock:
        if _processor is not None and _model is not None and _loaded_model_id == model_id:
            return dev, _processor, _model

        _cached_dtype = None
        _cached_max_length = None
        _configure_hf_hub_http()
        dtype = torch.float16 if dev == "cuda" else torch.float32
        local_only = _load_local_only()
        try:
            if local_only:
                proc, mdl = _load_whisper_assets(model_id, dtype, local_only=True)
                _last_model_load_source = "local-cache-only"
            else:
                # Avoid Hub HEAD/etag checks when a complete snapshot already exists locally.
                try:
                    proc, mdl = _load_whisper_assets(model_id, dtype, local_only=True)
                    _last_model_load_source = "local-cache"
                except OSError:
                    proc, mdl = _load_whisper_assets(model_id, dtype, local_only=False)
                    _last_model_load_source = "hub-fetch"
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

        _processor = proc
        _model = mdl
        _loaded_model_id = model_id
        return dev, proc, mdl


def get_last_model_load_source() -> str | None:
    """Load source for the most recent successful model initialization."""
    return _last_model_load_source


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

    try:
        return _from_torchaudio()
    except Exception as e:  # noqa: BLE001 — decode stack varies by torchaudio/torchcodec version
        err = str(e).lower()
        if "torchcodec" in err or "torch codec" in err:
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
        raise


def format_timestamp_hms(seconds: float) -> str:
    t = int(seconds)
    h, r = divmod(t, 3600)
    m, s = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def transcribe_mono16k_chunk(
    samples: "object",
    *,
    model_id: str = MODEL_ID_DEFAULT,
    device: str | None = None,
    language: str = "japanese",
) -> str:
    """Decode one mono float32 chunk at 16 kHz (Whisper). Returns empty string if too short."""
    global _cached_dtype, _cached_max_length
    import numpy as np
    import torch

    x = np.asarray(samples, dtype=np.float32).reshape(-1)
    if x.size < int(0.25 * 16000):
        return ""

    dev, processor, model = ensure_model_loaded(model_id, device)

    if _cached_dtype is None:
        _cached_dtype = next(model.parameters()).dtype
    if _cached_max_length is None:
        _cached_max_length = _whisper_max_length(model)

    inputs = processor(x, sampling_rate=16000, return_tensors="pt")
    feats = inputs.input_features.to(dev, dtype=_cached_dtype)
    attn_mask = torch.ones(feats.shape[:-1], dtype=torch.long, device=dev)
    with torch.inference_mode():
        pred_ids = model.generate(
            feats,
            attention_mask=attn_mask,
            language=language,
            task="transcribe",
            max_length=_cached_max_length,
        )
    return processor.batch_decode(pred_ids, skip_special_tokens=True)[0].strip()


def _whisper_max_length(model: object) -> int:
    """Decoder length cap for Whisper generation (typically ``max_target_positions`` = 448)."""
    cfg = getattr(model, "config", None)
    cap = int(getattr(cfg, "max_target_positions", None) or getattr(cfg, "max_length", None) or 448)
    return max(32, cap)


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
        ensure_model_loaded(model_id, device)

    def next_segment(self) -> tuple[float, float, str] | None:
        import torch
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
            language="japanese",
        )
        self._index += 1
        return (start / self._sr, end / self._sr, text)


def transcribe_audio_chunks(
    path: Path,
    *,
    model_id: str = MODEL_ID_DEFAULT,
    chunk_seconds: float = 30.0,
    device: str | None = None,
) -> Iterator[tuple[float, float, str]]:
    sess = TranscribeSession(path, model_id=model_id, chunk_seconds=chunk_seconds, device=device)
    while True:
        seg = sess.next_segment()
        if seg is None:
            break
        yield seg


def transcribe_audio_file(
    path: Path,
    *,
    model_id: str = MODEL_ID_DEFAULT,
    chunk_seconds: float = 30.0,
    device: str | None = None,
) -> list[tuple[float, float, str]]:
    return list(transcribe_audio_chunks(path, model_id=model_id, chunk_seconds=chunk_seconds, device=device))
