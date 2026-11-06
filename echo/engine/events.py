"""Engine → UI events. Every event serializes to a JSON-safe dict via ``to_dict()``."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


@dataclass
class EngineEvent:
    type: str = field(init=False, default="event")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TranscriptDraft(EngineEvent):
    """A transcript line that is still growing / may still change (live draft or partial)."""

    line_id: int
    line: str
    text: str
    type: str = field(init=False, default="transcript.draft")


@dataclass
class TranscriptFinal(EngineEvent):
    """A transcript line that was committed (segment finalized)."""

    line_id: int
    line: str
    text: str
    type: str = field(init=False, default="transcript.final")


@dataclass
class TranscriptLineState:
    line_id: int
    line: str
    text: str
    flushed: bool


@dataclass
class TranscriptCorrected(EngineEvent):
    """Full transcript snapshot after LLM corrections or insight-flush state changes."""

    lines: list[TranscriptLineState]
    type: str = field(init=False, default="transcript.corrected")


@dataclass
class InsightPending(EngineEvent):
    n: int
    stamp: str
    type: str = field(init=False, default="insight.pending")


@dataclass
class InsightPass(EngineEvent):
    n: int
    stamp: str
    kind: Literal["comment", "web_search"]
    body_md: str
    routing_md: str | None = None
    corrected_md: str | None = None
    web_md: str | None = None
    type: str = field(init=False, default="insight.pass")


@dataclass
class Meter(EngineEvent):
    rms: float
    peak: float
    note: str
    type: str = field(init=False, default="meter")


@dataclass
class Status(EngineEvent):
    mode: Literal["live", "audio", "replay"]
    state: Literal["idle", "running", "paused", "finished", "stopped"]
    language: str
    languages: list[str]
    asr_source: str | None
    asr_label: str
    mic_only: bool
    llm_offline: bool
    meeting_title: str
    elapsed: str
    type: str = field(init=False, default="status")


@dataclass
class Notice(EngineEvent):
    """Status / error line for the transcript header area (``text`` may contain Rich markup)."""

    level: Literal["info", "warning", "error", "debug"]
    text: str
    type: str = field(init=False, default="notice")


@dataclass
class Error(EngineEvent):
    message: str
    type: str = field(init=False, default="error")


@dataclass
class ChatReply(EngineEvent):
    question: str
    answer: str
    type: str = field(init=False, default="chat.reply")


@dataclass
class SessionReset(EngineEvent):
    type: str = field(init=False, default="session.reset")


@dataclass
class SessionEnded(EngineEvent):
    type: str = field(init=False, default="session.ended")


@dataclass
class SummaryWritten(EngineEvent):
    path: str
    ok: bool
    announce: bool = True
    type: str = field(init=False, default="summary.written")
