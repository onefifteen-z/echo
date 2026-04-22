"""Local ASR using Kotoba-Whisper weights (standard Whisper stack, no diarization deps)."""

from __future__ import annotations

import os
import warnings

# Avoid tokenizer/fork interactions when HF/transformers run from thread pools (macOS fds issues).
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import threading
from pathlib import Path
from typing import Iterator

MODEL_ID_DEFAULT = "kotoba-tech/kotoba-whisper-v2.2"

_hf_http_configured = False

_lock = threading.Lock()
_processor = None
_model = None
_loaded_model_id: str | None = None


def _require_asr_deps() -> None:
    try:
        import torch  # noqa: F401
        import torchaudio  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "缺少 ASR 依赖。请安装: pip install -e '.[asr]'"
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


def ensure_model_loaded(model_id: str = MODEL_ID_DEFAULT, device: str | None = None) -> tuple[str, object, object]:
    """Load WhisperProcessor + WhisperForConditionalGeneration once (thread-safe)."""
    global _processor, _model, _loaded_model_id
    _require_asr_deps()
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    dev = _pick_device(device)
    with _lock:
        if _processor is not None and _model is not None and _loaded_model_id == model_id:
            return dev, _processor, _model

        _configure_hf_hub_http()
        dtype = torch.float16 if dev == "cuda" else torch.float32
        local_only = _load_local_only()
        try:
            proc = WhisperProcessor.from_pretrained(model_id, local_files_only=local_only)
            mdl = WhisperForConditionalGeneration.from_pretrained(
                model_id, torch_dtype=dtype, local_files_only=local_only
            )
        except OSError as e:
            if _exception_chain_mentions_tls(e):
                raise OSError(
                    "从 Hugging Face 拉取模型时出现 TLS/证书错误（常见于公司代理或自签证书）。\n"
                    "可选处理方式：\n"
                    "  1) 将企业根 CA 合并为 PEM，并设置其一：\n"
                    "       export ECHO_HF_CA_BUNDLE=/path/to/combined.pem\n"
                    "     或 export SSL_CERT_FILE=... / REQUESTS_CA_BUNDLE=...\n"
                    "  2) 仅在可信网络下临时跳过校验：\n"
                    "       export ECHO_HF_INSECURE_SSL=1\n"
                    f"原始错误: {e}"
                ) from e
            if local_only:
                raise OSError(
                    "当前为仅本地加载（无网络或已设置 ECHO_HF_LOAD_LOCAL_ONLY / HF_HUB_OFFLINE），"
                    "但缓存中未找到完整模型文件。\n"
                    "请先在有网络时运行一次以下载权重，或设置 HF_HOME 指向已缓存该模型的目录。\n"
                    f"原始错误: {e}"
                ) from e
            raise
        mdl.eval()
        mdl.to(dev)

        _processor = proc
        _model = mdl
        _loaded_model_id = model_id
        return dev, proc, mdl


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
                    "当前 torchaudio 解码该文件需要 TorchCodec。请安装：pip install torchcodec\n"
                    "并确保系统 PATH 中有 ffmpeg（MP3 等格式需要）。\n"
                    "若不便安装 TorchCodec，可改用：pip install librosa\n"
                    f"原始错误: {e}"
                ) from ie
            except Exception as e2:  # noqa: BLE001
                raise RuntimeError(
                    "torchaudio 解码失败；已尝试 librosa 回退仍失败。\n"
                    "请安装 torchcodec + ffmpeg，或单独用 librosa 能成功加载该文件后再试。\n"
                    f"torchaudio: {e}\nlibrosa: {e2}"
                ) from e2
        raise


def format_timestamp_hms(seconds: float) -> str:
    t = int(seconds)
    h, r = divmod(t, 3600)
    m, s = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _whisper_max_new_tokens(model: object, forced_decoder_ids: list | None) -> int:
    """Cap new tokens so decoder prompt + generation fits ``max_target_positions`` (Whisper = 448)."""
    cfg = getattr(model, "config", None)
    cap = int(getattr(cfg, "max_target_positions", None) or getattr(cfg, "max_length", None) or 448)
    n_forced = len(forced_decoder_ids or [])
    # Initial decoder rows include forced prompt tokens plus start/special slots (empirically ~4 for JP transcribe).
    decoder_prompt_slots = max(4, n_forced + 1)
    return max(32, cap - decoder_prompt_slots)


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

        dev, processor, model = ensure_model_loaded(self._model_id, self._device)

        forced_decoder_ids = processor.get_decoder_prompt_ids(language="japanese", task="transcribe")
        dtype = next(model.parameters()).dtype
        inputs = processor(chunk, sampling_rate=self._sr, return_tensors="pt")
        feats = inputs.input_features.to(dev, dtype=dtype)
        with torch.no_grad():
            pred_ids = model.generate(
                feats,
                forced_decoder_ids=forced_decoder_ids,
                max_new_tokens=_whisper_max_new_tokens(model, forced_decoder_ids),
            )
        text = processor.batch_decode(pred_ids, skip_special_tokens=True)[0].strip()
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
