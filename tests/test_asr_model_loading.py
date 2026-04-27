from __future__ import annotations

import sys
import types


class _DummyModel:
    def eval(self) -> None:
        return None

    def to(self, _device: str) -> None:
        return None


def _reset_asr_globals(asr_module: object) -> None:
    asr_module._processor = None
    asr_module._model = None
    asr_module._loaded_model_id = None
    asr_module._last_model_load_source = None


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
        return object(), _DummyModel()

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
        return object(), _DummyModel()

    monkeypatch.setattr(asr, "_load_whisper_assets", _fake_load)

    asr.ensure_model_loaded()

    assert calls == [True, False]
    assert asr.get_last_model_load_source() == "hub-fetch"
