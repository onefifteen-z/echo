"""Meeting title + popup context flow into ASR initial prompt and all LLM prompts."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest


def test_build_asr_prompt_joins_title_and_context() -> None:
    from echo.asr import build_asr_prompt

    out = build_asr_prompt("Weekly Sync", "Review blockers and next milestones.")
    assert "Weekly Sync" in out
    assert "Review blockers" in out


def test_build_asr_prompt_drops_blank_parts() -> None:
    from echo.asr import build_asr_prompt

    assert build_asr_prompt("", "") == ""
    assert build_asr_prompt("  ", "context only") == "context only"
    assert build_asr_prompt("title only", "  ") == "title only"


def test_build_asr_prompt_truncates_to_char_budget() -> None:
    import echo.asr as asr

    long_ctx = "あ" * (asr._ASR_PROMPT_CHAR_BUDGET * 3)
    out = asr.build_asr_prompt("Title", long_ctx)
    assert len(out) <= asr._ASR_PROMPT_CHAR_BUDGET


def test_transcribe_chunk_passes_prompt_ids_to_generate(monkeypatch: pytest.MonkeyPatch) -> None:
    """`prompt_text` must reach `model.generate` as `prompt_ids` via the processor."""
    import sys
    import types

    import numpy as np
    import torch

    import echo.asr as asr

    class _StubProcessor:
        def __call__(self, _x, sampling_rate: int, return_tensors: str):
            assert sampling_rate == 16000
            return types.SimpleNamespace(input_features=torch.zeros(1, 80, 3000))

        def get_prompt_ids(self, text: str, return_tensors: str = "pt"):
            assert text == "Weekly Sync. Review blockers."
            assert return_tensors == "pt"
            return torch.tensor([[101, 102, 103]])

        def batch_decode(self, _ids, skip_special_tokens: bool = True):
            return ["hello"]

    class _StubModel:
        def __init__(self) -> None:
            self.last_kwargs: dict[str, object] = {}

        def generate(self, _feats, **kwargs):
            self.last_kwargs = kwargs
            return torch.tensor([[1, 2, 3]])

    stub_model = _StubModel()
    asr._state = asr._LoadedAsr(
        model_id=asr.MODEL_ID_DEFAULT,
        device="cpu",
        processor=_StubProcessor(),
        model=stub_model,
        dtype=torch.float32,
        max_length=448,
        source="local-cache",
    )
    asr._prompt_ids_cache.clear()
    monkeypatch.setattr(asr, "ensure_model_loaded", lambda *_a, **_k: ("cpu", None, None))

    samples = np.zeros(16000, dtype=np.float32)
    text = asr.transcribe_mono16k_chunk(
        samples, prompt_text="Weekly Sync. Review blockers.",
    )

    assert text == "hello"
    assert "prompt_ids" in stub_model.last_kwargs
    sent = stub_model.last_kwargs["prompt_ids"]
    assert isinstance(sent, torch.Tensor)
    assert sent.tolist() in ([101, 102, 103], [[101, 102, 103]])


def test_transcribe_chunk_omits_prompt_ids_when_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    import types

    import numpy as np
    import torch

    import echo.asr as asr

    class _StubProcessor:
        def __call__(self, _x, sampling_rate: int, return_tensors: str):
            return types.SimpleNamespace(input_features=torch.zeros(1, 80, 3000))

        def batch_decode(self, _ids, skip_special_tokens: bool = True):
            return ["hi"]

    class _StubModel:
        def __init__(self) -> None:
            self.last_kwargs: dict[str, object] = {}

        def generate(self, _feats, **kwargs):
            self.last_kwargs = kwargs
            return torch.tensor([[1]])

    stub_model = _StubModel()
    asr._state = asr._LoadedAsr(
        model_id=asr.MODEL_ID_DEFAULT,
        device="cpu",
        processor=_StubProcessor(),
        model=stub_model,
        dtype=torch.float32,
        max_length=448,
        source="local-cache",
    )
    asr._prompt_ids_cache.clear()
    monkeypatch.setattr(asr, "ensure_model_loaded", lambda *_a, **_k: ("cpu", None, None))

    asr.transcribe_mono16k_chunk(np.zeros(16000, dtype=np.float32), prompt_text=None)
    assert "prompt_ids" not in stub_model.last_kwargs

    asr.transcribe_mono16k_chunk(np.zeros(16000, dtype=np.float32), prompt_text="   ")
    assert "prompt_ids" not in stub_model.last_kwargs


def test_transcribe_session_forwards_prompt_text(monkeypatch: pytest.MonkeyPatch) -> None:
    import numpy as np

    import echo.asr as asr

    captured: dict[str, object] = {}

    monkeypatch.setattr(asr, "_require_asr_deps", lambda: None)
    monkeypatch.setattr(asr, "ensure_model_loaded", lambda *_a, **_k: ("cpu", None, None))
    monkeypatch.setattr(
        asr,
        "_load_audio_mono_16k",
        lambda _path: (np.zeros(16000 * 2, dtype=np.float32), 16000),
    )

    def fake_transcribe(samples, *, model_id, device, language, prompt_text):
        captured.update(
            model_id=model_id,
            device=device,
            language=language,
            prompt_text=prompt_text,
            n_samples=int(samples.size),
        )
        return "decoded"

    monkeypatch.setattr(asr, "transcribe_mono16k_chunk", fake_transcribe)

    session = asr.TranscribeSession(
        Path("/dev/null"),
        prompt_text="  Weekly Sync. Review blockers.  ",
    )
    seg = session.next_segment()
    assert seg is not None
    assert seg[2] == "decoded"
    assert captured["prompt_text"] == "Weekly Sync. Review blockers."
    assert captured["language"] == "japanese"


def test_llm_route_insight_includes_meeting_title_and_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import echo.llm as llm

    captured: dict[str, str] = {}

    async def fake_llm_user_text(content: str, _max_tokens: int) -> str:
        captured["prompt"] = content
        return json.dumps(
            {
                "action": "comment",
                "corrected_transcript": "ok",
                "comment": "noted",
                "search_query": "",
                "reason": "ok",
            }
        )

    monkeypatch.setattr(llm, "_llm_user_text", fake_llm_user_text)

    asyncio.run(
        llm.llm_route_insight(
            "some transcript",
            meeting_title="Weekly Sync",
            meeting_context="Review blockers and next milestones",
        )
    )

    prompt = captured["prompt"]
    assert "Title: Weekly Sync" in prompt
    assert "Review blockers and next milestones" in prompt


def test_llm_route_insight_drops_untitled_placeholder_and_date_suffix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import echo.llm as llm

    captured: dict[str, str] = {}

    async def fake_llm_user_text(content: str, _max_tokens: int) -> str:
        captured["prompt"] = content
        return json.dumps(
            {
                "action": "comment",
                "corrected_transcript": "ok",
                "comment": "noted",
                "search_query": "",
                "reason": "ok",
            }
        )

    monkeypatch.setattr(llm, "_llm_user_text", fake_llm_user_text)

    asyncio.run(
        llm.llm_route_insight(
            "some transcript",
            meeting_title="untitled-meeting",
            meeting_context="",
        )
    )
    assert "Title: (none)" in captured["prompt"]

    asyncio.run(
        llm.llm_route_insight(
            "some transcript",
            meeting_title="Q2 Planning @2026-05-11",
        )
    )
    assert "Title: Q2 Planning" in captured["prompt"]
    assert "@2026-05-11" not in captured["prompt"].split("Title: ", 1)[1].splitlines()[0]


def test_llm_synthesize_web_findings_includes_meeting_title(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import echo.llm as llm
    from echo.search import SearchHit

    captured: dict[str, str] = {}

    async def fake_llm_user_text(content: str, _max_tokens: int) -> str:
        captured["prompt"] = content
        return "summary"

    monkeypatch.setattr(llm, "_llm_user_text", fake_llm_user_text)

    asyncio.run(
        llm.llm_synthesize_web_findings(
            "transcript chunk",
            "search query",
            [SearchHit(title="t", body="b", href="https://example.com")],
            meeting_context="background here",
            meeting_title="Weekly Sync",
        )
    )

    assert "Title: Weekly Sync" in captured["prompt"]
    assert "background here" in captured["prompt"]


def test_chat_prompt_includes_title_and_context() -> None:
    from echo.llm import _chat_prompt

    out = _chat_prompt(
        "transcript text",
        "what was decided?",
        meeting_context="context body",
        meeting_title="Weekly Sync",
    )
    assert "Meeting title: Weekly Sync" in out
    assert "Meeting background/context:" in out
    assert "context body" in out
    assert "Transcript:\ntranscript text" in out
    assert "Question:\nwhat was decided?" in out


def test_chat_prompt_omits_title_line_when_unset() -> None:
    from echo.llm import _chat_prompt

    out = _chat_prompt("t", "q", meeting_context="", meeting_title="")
    assert "Meeting title:" not in out
    assert "Meeting background/context:" not in out

    out2 = _chat_prompt("t", "q", meeting_context="", meeting_title="untitled-meeting")
    assert "Meeting title:" not in out2
