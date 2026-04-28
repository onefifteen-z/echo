from __future__ import annotations

import sys
import types


class _DummyModel:
    def eval(self) -> None:
        return None

    def to(self, _device: str) -> None:
        return None

    def parameters(self):
        # `next(model.parameters())` is used to read dtype after load; supply a stub.
        yield types.SimpleNamespace(dtype="fp32")


class _DummyConfig:
    max_target_positions = 448


def _reset_asr_globals(asr_module: object) -> None:
    asr_module._state = None


def _make_dummy_model() -> _DummyModel:
    m = _DummyModel()
    m.config = _DummyConfig()
    m.generation_config = types.SimpleNamespace(
        forced_decoder_ids=None,
        suppress_tokens=None,
        begin_suppress_tokens=None,
    )
    return m


def test_ensure_model_loaded_prefers_local_cache_before_network(monkeypatch) -> None:
    import echo.asr as asr

    _reset_asr_globals(asr)
    monkeypatch.setattr(asr, "_require_asr_deps", lambda: None)
    monkeypatch.setattr(asr, "_pick_device", lambda _device: "cpu")
    monkeypatch.setattr(asr, "_configure_hf_hub_http", lambda: None)
    monkeypatch.setattr(asr, "_load_local_only", lambda: False)
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(float16="fp16", float32="fp32"))

    calls: list[bool] = []

    def _fake_load(model_id: str, dtype: object, *, local_only: bool):
        calls.append(local_only)
        assert model_id == asr.MODEL_ID_DEFAULT
        assert dtype == "fp32"
        return object(), _make_dummy_model()

    monkeypatch.setattr(asr, "_load_whisper_assets", _fake_load)

    asr.ensure_model_loaded()

    assert calls == [True]
    assert asr.get_last_model_load_source() == "local-cache"


def test_ensure_model_loaded_falls_back_to_network_when_cache_incomplete(monkeypatch) -> None:
    import echo.asr as asr

    _reset_asr_globals(asr)
    monkeypatch.setattr(asr, "_require_asr_deps", lambda: None)
    monkeypatch.setattr(asr, "_pick_device", lambda _device: "cpu")
    monkeypatch.setattr(asr, "_configure_hf_hub_http", lambda: None)
    monkeypatch.setattr(asr, "_load_local_only", lambda: False)
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(float16="fp16", float32="fp32"))

    calls: list[bool] = []

    def _fake_load(_model_id: str, _dtype: object, *, local_only: bool):
        calls.append(local_only)
        if local_only:
            raise OSError("cache miss")
        return object(), _make_dummy_model()

    monkeypatch.setattr(asr, "_load_whisper_assets", _fake_load)

    asr.ensure_model_loaded()

    assert calls == [True, False]
    assert asr.get_last_model_load_source() == "hub-fetch"


def test_ensure_model_loaded_returns_loaded_device_on_fast_path(monkeypatch) -> None:
    """Reusing a loaded model must not switch device based on the requested arg."""
    import echo.asr as asr

    _reset_asr_globals(asr)
    monkeypatch.setattr(asr, "_require_asr_deps", lambda: None)
    monkeypatch.setattr(asr, "_configure_hf_hub_http", lambda: None)
    monkeypatch.setattr(asr, "_load_local_only", lambda: True)
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(float16="fp16", float32="fp32"))
    monkeypatch.setattr(asr, "_load_whisper_assets", lambda *_a, **_k: (object(), _make_dummy_model()))

    devices = iter(["mps", "cuda"])
    monkeypatch.setattr(asr, "_pick_device", lambda _device: next(devices))

    dev1, _, _ = asr.ensure_model_loaded()
    dev2, _, _ = asr.ensure_model_loaded(device="cuda")

    assert dev1 == "mps"
    assert dev2 == "mps"  # fast path returns the loaded device, not the new request
