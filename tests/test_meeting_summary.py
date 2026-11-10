"""Post-meeting `summary.md` generation: prompt shape, fallbacks, and Save wiring."""

from __future__ import annotations

import asyncio

import pytest


def test_llm_meeting_summary_offline_returns_static_markdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No backend configured -> deterministic offline markdown (still has both sections)."""
    import echo.llm as llm

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("BASE_URL", raising=False)
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.delenv("MODEL", raising=False)

    out = asyncio.run(llm.llm_meeting_summary("some transcript text"))
    assert "## Key takeaways" in out
    assert "## Next actions" in out
    assert out == llm.MEETING_SUMMARY_OFFLINE_MARKDOWN


def test_llm_meeting_summary_empty_transcript_short_circuits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import echo.llm as llm

    called = {"n": 0}

    async def fake_llm_user_text(_content: str, _max_tokens: int) -> str:
        called["n"] += 1
        return "should not be called"

    monkeypatch.setattr(llm, "_llm_user_text", fake_llm_user_text)

    out = asyncio.run(llm.llm_meeting_summary("   "))
    assert called["n"] == 0
    assert "## Key takeaways" in out
    assert "## Next actions" in out
    assert "No transcript was captured" in out


def test_llm_meeting_summary_prompt_includes_title_context_glossary_insights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import echo.llm as llm

    captured: dict[str, str] = {}

    async def fake_llm_user_text(content: str, _max_tokens: int) -> str:
        captured["prompt"] = content
        return "## Key takeaways\n\n- All good\n\n## Next actions\n\n- Ship it\n"

    monkeypatch.setattr(llm, "_llm_user_text", fake_llm_user_text)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    out = asyncio.run(
        llm.llm_meeting_summary(
            "Alice: we should ship by Friday.\nBob: I'll own the rollout.",
            insight_passes=[
                ("comment", "Decision: ship by Friday."),
                ("web_search", "External: no comparable rollouts found."),
            ],
            meeting_title="Launch Sync",
            meeting_context="Pre-launch coordination",
            dictionary_terms=("EchoApp", "石井"),
        )
    )

    prompt = captured["prompt"]
    assert "Title: Launch Sync" in prompt
    assert "Pre-launch coordination" in prompt
    assert "EchoApp" in prompt and "石井" in prompt
    assert "Decision: ship by Friday." in prompt
    assert "External: no comparable rollouts found." in prompt
    assert "Alice: we should ship by Friday." in prompt
    assert "## Key takeaways" in prompt
    assert "## Next actions" in prompt
    assert "spelling reference only" in prompt

    assert out.startswith("## Key takeaways")
    assert "## Next actions" in out


def test_llm_meeting_summary_includes_notebook(monkeypatch: pytest.MonkeyPatch) -> None:
    import echo.llm as llm

    prompts: list[str] = []

    async def fake_llm_user_text(content: str, _max_tokens: int) -> str:
        prompts.append(content)
        return "## Key takeaways\n\n- x\n\n## Next actions\n\n- y\n"

    monkeypatch.setattr(llm, "_llm_user_text", fake_llm_user_text)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    asyncio.run(llm.llm_meeting_summary("A: hello", notebook="Owner: Bob, due Friday"))
    asyncio.run(llm.llm_meeting_summary("A: hello"))
    asyncio.run(llm.llm_meeting_summary("", notebook="Only my notes"))

    assert "Owner: Bob, due Friday" in prompts[0] and "authoritative" in prompts[0]
    assert "notebook" in prompts[1] and "<<<\n(none)\n>>>" in prompts[1]
    assert "Only my notes" in prompts[2] and "(no transcript captured)" in prompts[2]


def test_notebook_images_are_replaced_in_summary_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    import echo.llm as llm

    assert llm.notebook_for_prompt("a ![Whiteboard](notes-assets/img-1.png) b") == "a [image: Whiteboard] b"
    assert llm.notebook_for_prompt("![](notes-assets/img-2.jpg)") == "[image]"
    assert llm.notebook_for_prompt("![logo](https://x.test/a.png)") == "![logo](https://x.test/a.png)"

    prompts: list[str] = []

    async def fake_llm_user_text(content: str, _max_tokens: int) -> str:
        prompts.append(content)
        return "## Key takeaways\n\n- x\n\n## Next actions\n\n- y\n"

    monkeypatch.setattr(llm, "_llm_user_text", fake_llm_user_text)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    asyncio.run(llm.llm_meeting_summary("A: hi", notebook="Plan\n\n![Board](notes-assets/img-3.png)"))
    assert "[image: Board]" in prompts[0] and "notes-assets/" not in prompts[0]


PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16


def test_notebook_with_images_loads_numbers_and_skips(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    import echo.llm as llm

    assets = tmp_path / "notes-assets"
    assets.mkdir()
    for name in ("a.png", "b.jpg", "big.png"):
        (assets / name).write_bytes(PNG_BYTES)
    monkeypatch.setattr(llm, "_SUMMARY_MAX_IMAGE_BYTES", len(PNG_BYTES))
    (assets / "big.png").write_bytes(PNG_BYTES + b"x")
    md = "![Slide](notes-assets/a.png)\n![](notes-assets/missing.png)\n![Big](notes-assets/big.png)\n![Board](notes-assets/b.jpg)\n![Esc](notes-assets/../x.png)"
    text, images = llm.notebook_with_images(md, assets)
    assert text.splitlines() == ["[image 1: Slide]", "[image]", "[image: Big]", "[image 2: Board]", "[image: Esc]"]
    assert [(i.label, i.media_type) for i in images] == [("[image 1: Slide]", "image/png"), ("[image 2: Board]", "image/jpeg")]
    assert llm.notebook_with_images(md, None) == (llm.notebook_for_prompt(md), [])

    monkeypatch.setattr(llm, "_SUMMARY_MAX_IMAGES", 1)
    text, images = llm.notebook_with_images(md, assets)
    assert [i.label for i in images] == ["[image 1: Board]"] and "[image: Slide]" in text


def test_image_content_blocks_per_provider() -> None:
    import echo.llm as llm

    img = llm.PromptImage(label="[image 1: Slide]", media_type="image/png", data_b64="QUJD")
    assert llm._anthropic_content("prompt", None) == "prompt"
    blocks = llm._anthropic_content("prompt", [img])
    assert blocks[1] == {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "QUJD"}}
    assert blocks[-1] == {"type": "text", "text": "prompt"}
    parts = llm._openai_content("prompt", [img])
    assert parts[0] == {"type": "text", "text": "prompt"}
    assert parts[2] == {"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJD"}}


def test_meeting_summary_attaches_screenshots_and_falls_back(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    import echo.llm as llm

    assets = tmp_path / "notes-assets"
    assets.mkdir()
    (assets / "s.png").write_bytes(PNG_BYTES)
    calls: list[tuple[str, int]] = []
    reject = {"on": False}

    async def fake_llm_user_text(content: str, _max_tokens: int, images=None) -> str:
        calls.append((content, len(images or [])))
        if images and reject["on"]:
            raise RuntimeError("model does not support images")
        return "## Key takeaways\n\n- x\n\n## Next actions\n\n- y\n"

    monkeypatch.setattr(llm, "_llm_user_text", fake_llm_user_text)
    monkeypatch.setattr(llm, "_LLM_USER_TEXT_BACKOFF_S", 0)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    notebook = "Revenue slide:\n\n![Q3](notes-assets/s.png)"

    md = asyncio.run(llm.llm_meeting_summary("A: hi", notebook=notebook, assets_dir=assets))
    assert calls[0][1] == 1 and "[image 1: Q3]" in calls[0][0] and "1 screenshot(s)" in calls[0][0]
    assert "text only" not in md

    calls.clear()
    reject["on"] = True
    md = asyncio.run(llm.llm_meeting_summary("A: hi", notebook=notebook, assets_dir=assets))
    assert [n for _c, n in calls] == [1, 1, 0]
    assert "[image: Q3]" in calls[-1][0] and "screenshot(s)" not in calls[-1][0]
    assert md.startswith("## Key takeaways") and "text only" in md


def test_llm_meeting_summary_truncates_long_transcript(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Very long transcripts are tail-truncated with a marker so prompts stay bounded."""
    import echo.llm as llm

    captured: dict[str, str] = {}

    async def fake_llm_user_text(content: str, _max_tokens: int) -> str:
        captured["prompt"] = content
        return "## Key takeaways\n\n- ok\n\n## Next actions\n\n- ok\n"

    monkeypatch.setattr(llm, "_llm_user_text", fake_llm_user_text)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    head_marker = "FIRST_LINE_SHOULD_BE_DROPPED"
    tail_marker = "LAST_LINE_KEPT"
    body = head_marker + "\n" + ("padding line. " * 5000) + "\n" + tail_marker
    asyncio.run(llm.llm_meeting_summary(body, meeting_title="Long Meeting"))

    prompt = captured["prompt"]
    assert tail_marker in prompt
    assert head_marker not in prompt
    assert llm.MEETING_SUMMARY_TRUNCATION_MARKER in prompt


def test_llm_meeting_summary_falls_back_when_llm_returns_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import echo.llm as llm

    async def empty_llm(_content: str, _max_tokens: int) -> str:
        return ""

    monkeypatch.setattr(llm, "_llm_user_text", empty_llm)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    out = asyncio.run(llm.llm_meeting_summary("hello world"))
    assert out.startswith(llm.MEETING_SUMMARY_OFFLINE_MARKDOWN)
    assert "empty summary" in out


def test_app_save_writes_summary_md_in_offline_mode(tmp_path) -> None:
    """When LLM is offline, Save writes the static fallback `summary.md` synchronously."""
    from echo.app import EchoApp
    from echo.llm import MEETING_SUMMARY_OFFLINE_MARKDOWN

    app = EchoApp(llm_offline=True, output_dir=tmp_path / "meeting-out")
    app._lines = ["alpha", "beta"]
    app._insight_passes = []

    app._save_meeting_outputs(show_notification=False)

    summary_path = tmp_path / "meeting-out" / "summary.md"
    assert summary_path.exists()
    assert summary_path.read_text(encoding="utf-8") == MEETING_SUMMARY_OFFLINE_MARKDOWN

    notes_path = tmp_path / "meeting-out" / "notes.txt"
    assert notes_path.read_text(encoding="utf-8") == "alpha\nbeta"


def test_app_save_writes_summary_md_when_no_event_loop(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """LLM-configured but no running asyncio loop -> placeholder file is written, no crash."""
    from echo.app import EchoApp
    from echo.llm import MEETING_SUMMARY_OFFLINE_MARKDOWN

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    app = EchoApp(llm_offline=False, output_dir=tmp_path / "meeting-out")
    app._lines = ["one", "two"]
    app._insight_passes = []

    app._save_meeting_outputs(show_notification=False)

    summary_path = tmp_path / "meeting-out" / "summary.md"
    assert summary_path.exists()
    assert summary_path.read_text(encoding="utf-8") == MEETING_SUMMARY_OFFLINE_MARKDOWN


def test_app_save_schedules_async_summary_when_loop_running(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """LLM-configured + running loop -> placeholder first, then real summary written by task."""
    import echo.engine.session as app_mod
    from echo.app import EchoApp

    captured: dict[str, str] = {}

    async def fake_llm_meeting_summary(
        transcript_text: str,
        *,
        insight_passes=None,
        meeting_title: str = "",
        meeting_context: str = "",
        dictionary_terms=(),
        notebook: str = "",
        assets_dir=None,
        meeting_time: str = "",
    ) -> str:
        captured["transcript"] = transcript_text
        captured["title"] = meeting_title
        captured["context"] = meeting_context
        captured["dict"] = ",".join(dictionary_terms or ())
        captured["passes"] = ";".join(k for k, _b in (insight_passes or []))
        return "## Key takeaways\n\n- LIVE_SUMMARY\n\n## Next actions\n\n- DONE\n"

    monkeypatch.setattr(app_mod, "llm_meeting_summary", fake_llm_meeting_summary)

    app = EchoApp(
        llm_offline=False,
        output_dir=tmp_path / "meeting-out",
        meeting_title="Sprint Review",
        meeting_context="ctx",
    )
    app._lines = ["gamma", "delta"]
    app._insight_passes = []

    async def run() -> None:
        app._save_meeting_outputs(show_notification=False)
        # Drain any pending tasks scheduled by the save flow.
        await asyncio.sleep(0)
        for _ in range(20):
            pending = [t for t in asyncio.all_tasks() if not t.done()]
            pending = [t for t in pending if t is not asyncio.current_task()]
            if not pending:
                break
            await asyncio.gather(*pending, return_exceptions=True)

    asyncio.run(run())

    summary_path = tmp_path / "meeting-out" / "summary.md"
    body = summary_path.read_text(encoding="utf-8")
    assert "LIVE_SUMMARY" in body
    assert "## Key takeaways" in body
    assert "## Next actions" in body
    assert captured["transcript"] == "gamma\ndelta"
    assert captured["title"] == "Sprint Review"
    assert captured["context"] == "ctx"
