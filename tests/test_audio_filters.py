"""Optional VAD + denoise stages for live ASR (echo/audio_filters.py)."""

from __future__ import annotations

import builtins
import sys
import types

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _reset_filters_state() -> None:
    import echo.audio_filters as af

    af.reset_state()
    yield
    af.reset_state()


# ---------------------------------------------------------------------------
# vad_speech_ratio
# ---------------------------------------------------------------------------


def test_vad_off_returns_full_speech() -> None:
    from echo.audio_filters import vad_speech_ratio

    samples = np.zeros(16000, dtype=np.float32)
    assert vad_speech_ratio(samples, backend="off") == 1.0


def test_vad_unknown_backend_treated_as_off() -> None:
    from echo.audio_filters import vad_speech_ratio

    samples = np.zeros(16000, dtype=np.float32)
    assert vad_speech_ratio(samples, backend="bogus") == 1.0


def test_vad_silero_path_uses_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub the Silero model to return canned per-frame probabilities."""
    import echo.audio_filters as af

    probs = iter([0.9, 0.1, 0.8, 0.2, 0.6])

    class _StubTensorElement:
        def __init__(self, value: float) -> None:
            self._value = value

        def item(self) -> float:
            return self._value

    def fake_model(_frame, _sr):
        return _StubTensorElement(next(probs))

    class _StubTensor:
        def __init__(self, n_frames: int) -> None:
            self._n = n_frames

        def reshape(self, n_frames, _frame_len):
            self._n = n_frames
            return self

        def __getitem__(self, _i):
            return object()

    class _NullCtx:
        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

    fake_torch = types.SimpleNamespace(
        from_numpy=lambda arr: _StubTensor(arr.size // 512),
        inference_mode=lambda: _NullCtx(),
    )

    monkeypatch.setattr(af, "_load_silero", lambda _notify: (fake_model, fake_torch))

    samples = np.zeros(512 * 5, dtype=np.float32)
    ratio = af.vad_speech_ratio(
        samples, backend="silero", speech_prob_threshold=0.5,
    )
    assert ratio == pytest.approx(3 / 5)


def test_vad_silero_falls_back_to_webrtc_on_import_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import echo.audio_filters as af

    notes: list[str] = []

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "silero_vad" or name.startswith("silero_vad."):
            raise ImportError("nope")
        return real_import(name, *args, **kwargs)

    monkeypatch.setitem(sys.modules, "silero_vad", None)
    monkeypatch.setattr(builtins, "__import__", fake_import)

    class _StubVad:
        def is_speech(self, _frame: bytes, _sr: int) -> bool:
            return True

    fake_webrtc = types.SimpleNamespace(Vad=lambda _agg: _StubVad())
    monkeypatch.setitem(sys.modules, "webrtcvad", fake_webrtc)

    samples = np.full(16000, 0.5, dtype=np.float32)
    ratio = af.vad_speech_ratio(samples, backend="silero", notify=notes.append)
    assert ratio == 1.0
    assert any("silero" in n.lower() for n in notes)

    # Subsequent calls take the webrtc fast path (silero is not retried).
    notes.clear()
    ratio2 = af.vad_speech_ratio(samples, backend="silero", notify=notes.append)
    assert ratio2 == 1.0
    assert not notes  # no repeated warnings


def test_vad_webrtc_returns_zero_when_no_speech(monkeypatch: pytest.MonkeyPatch) -> None:
    import echo.audio_filters as af

    class _StubVad:
        def is_speech(self, _frame: bytes, _sr: int) -> bool:
            return False

    fake_webrtc = types.SimpleNamespace(Vad=lambda _agg: _StubVad())
    monkeypatch.setitem(sys.modules, "webrtcvad", fake_webrtc)

    samples = np.full(16000, 0.5, dtype=np.float32)
    assert af.vad_speech_ratio(samples, backend="webrtc") == 0.0


def test_vad_webrtc_alternating_frames(monkeypatch: pytest.MonkeyPatch) -> None:
    import echo.audio_filters as af

    class _StubVad:
        def __init__(self) -> None:
            self.calls = 0

        def is_speech(self, _frame: bytes, _sr: int) -> bool:
            self.calls += 1
            return self.calls % 2 == 1  # odd-indexed frames -> speech

    fake_webrtc = types.SimpleNamespace(Vad=lambda _agg: _StubVad())
    monkeypatch.setitem(sys.modules, "webrtcvad", fake_webrtc)

    # 1 second @ 16k -> 33 full 30ms frames (480 samples each); ratio rounds to 17/33.
    samples = np.full(16000, 0.5, dtype=np.float32)
    ratio = af.vad_speech_ratio(samples, backend="webrtc")
    assert ratio == pytest.approx(17 / 33)


def test_vad_webrtc_falls_back_to_off_when_unimportable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import echo.audio_filters as af

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "webrtcvad":
            raise ImportError("missing wheel")
        return real_import(name, *args, **kwargs)

    monkeypatch.setitem(sys.modules, "webrtcvad", None)
    monkeypatch.setattr(builtins, "__import__", fake_import)

    notes: list[str] = []
    samples = np.full(16000, 0.5, dtype=np.float32)
    assert af.vad_speech_ratio(samples, backend="webrtc", notify=notes.append) == 1.0
    assert any("webrtc" in n.lower() for n in notes)


# ---------------------------------------------------------------------------
# denoise
# ---------------------------------------------------------------------------


def test_denoise_off_returns_input() -> None:
    from echo.audio_filters import denoise

    samples = np.full(8000, 0.25, dtype=np.float32)
    out = denoise(samples, backend="off")
    assert out is samples


def test_denoise_unknown_backend_returns_input() -> None:
    from echo.audio_filters import denoise

    samples = np.full(8000, 0.25, dtype=np.float32)
    out = denoise(samples, backend="bogus")
    assert out is samples


def test_denoise_noisereduce_passes_through_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    import echo.audio_filters as af

    captured: dict[str, object] = {}

    def fake_reduce_noise(*, y, sr, stationary, prop_decrease):
        captured.update(sr=sr, stationary=stationary, prop_decrease=prop_decrease)
        return (np.asarray(y, dtype=np.float32) * 0.5).astype(np.float32)

    fake_nr = types.SimpleNamespace(reduce_noise=fake_reduce_noise)
    monkeypatch.setitem(sys.modules, "noisereduce", fake_nr)

    samples = np.full(8000, 0.4, dtype=np.float32)
    out = af.denoise(samples, backend="noisereduce", sr=16000, prop_decrease=0.7)
    assert out.shape == samples.shape
    assert out.dtype == np.float32
    assert captured == {"sr": 16000, "stationary": True, "prop_decrease": pytest.approx(0.7)}
    assert float(out.mean()) == pytest.approx(0.2)


def test_denoise_clamps_prop_decrease(monkeypatch: pytest.MonkeyPatch) -> None:
    import echo.audio_filters as af

    captured: dict[str, float] = {}

    def fake_reduce_noise(*, y, sr, stationary, prop_decrease):
        captured["prop_decrease"] = prop_decrease
        return np.asarray(y, dtype=np.float32)

    fake_nr = types.SimpleNamespace(reduce_noise=fake_reduce_noise)
    monkeypatch.setitem(sys.modules, "noisereduce", fake_nr)

    af.denoise(
        np.zeros(2000, dtype=np.float32),
        backend="noisereduce",
        prop_decrease=2.5,
    )
    assert captured["prop_decrease"] == 1.0

    af.reset_state()
    monkeypatch.setitem(sys.modules, "noisereduce", fake_nr)
    af.denoise(
        np.zeros(2000, dtype=np.float32),
        backend="noisereduce",
        prop_decrease=-0.5,
    )
    assert captured["prop_decrease"] == 0.0


def test_denoise_passthrough_when_noisereduce_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import echo.audio_filters as af

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "noisereduce":
            raise ImportError("not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setitem(sys.modules, "noisereduce", None)
    monkeypatch.setattr(builtins, "__import__", fake_import)

    notes: list[str] = []
    samples = np.full(8000, 0.3, dtype=np.float32)
    out = af.denoise(samples, backend="noisereduce", notify=notes.append)
    assert out is samples
    assert any("noisereduce" in n.lower() for n in notes)

    # Once resolved to off, subsequent calls take the fast path with no warning.
    notes.clear()
    out2 = af.denoise(samples, backend="noisereduce", notify=notes.append)
    assert out2 is samples
    assert not notes


def test_denoise_swallows_call_exceptions(monkeypatch: pytest.MonkeyPatch) -> None:
    import echo.audio_filters as af

    def boom(**_kwargs):
        raise RuntimeError("scipy bad day")

    fake_nr = types.SimpleNamespace(reduce_noise=boom)
    monkeypatch.setitem(sys.modules, "noisereduce", fake_nr)

    notes: list[str] = []
    samples = np.full(8000, 0.3, dtype=np.float32)
    out = af.denoise(samples, backend="noisereduce", notify=notes.append)
    assert out.shape == samples.shape
    assert out.dtype == np.float32
    assert any("noisereduce" in n.lower() for n in notes)
