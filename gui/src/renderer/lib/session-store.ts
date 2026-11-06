import type { EchoEvent, InsightPassEvent, Language, SessionSnapshot, Status } from "@shared/types";

export interface LineView {
  id: number;
  ts: string;
  text: string;
  final: boolean;
  flushed: boolean;
}

export interface NoticeView {
  id: number;
  level: "info" | "warning" | "error";
  text: string;
}

export interface ChatItem {
  id: number;
  question: string;
  answer: string | null;
  error?: boolean;
}

export interface AppleInstallView {
  language: Language;
  fraction: number;
  done: boolean;
  error?: string;
}

export interface SessionState {
  status: Status | null;
  lines: LineView[];
  insights: Omit<InsightPassEvent, "type">[];
  pendingInsight: { n: number; stamp: string } | null;
  notices: NoticeView[];
  meter: { rms: number; peak: number; note: string };
  meterHistory: number[];
  chat: ChatItem[];
  preparing: boolean;
  /** Audio source finished; the session can still be chatted with and saved. */
  ended: boolean;
  /** Meeting explicitly ended via session.stop. */
  stopped: boolean;
  appleInstall: AppleInstallView | null;
  lastSummaryPath: string | null;
  /** Output folder name of the live meeting (its id in the Meetings archive). */
  meetingId: string | null;
  notebook: string;
}

export const METER_HISTORY = 96;

export const initialState: SessionState = {
  status: null,
  lines: [],
  insights: [],
  pendingInsight: null,
  notices: [],
  meter: { rms: 0, peak: 0, note: "" },
  meterHistory: Array(METER_HISTORY).fill(0),
  chat: [],
  preparing: false,
  ended: false,
  stopped: false,
  appleInstall: null,
  lastSummaryPath: null,
  meetingId: null,
  notebook: "",
};

export type Action =
  | { kind: "event"; event: EchoEvent }
  | { kind: "snapshot"; snapshot: SessionSnapshot }
  | { kind: "chat.ask"; id: number; question: string }
  | { kind: "chat.answer"; id: number; answer: string; error?: boolean }
  | { kind: "stopped" }
  | { kind: "meeting"; id: string; notebook: string }
  | { kind: "clear" };

const TS_RE = /^\[([^\]]+)\]\s*/;

export function splitLine(line: string): { ts: string; rest: string } {
  const m = TS_RE.exec(line);
  return m ? { ts: m[1], rest: line.slice(m[0].length) } : { ts: "", rest: line };
}

/** Engine notices carry Rich markup (`[red]…[/]`); the GUI shows plain text. */
export function stripRichMarkup(text: string): string {
  return text.replace(/\[\/?(?:[a-z]+(?:[ #][a-z0-9]+)*)?\]/gi, "").replace(/`/g, "").trim();
}

/** Map linear amplitude to 0..1 on a -60..0 dBFS scale. */
export function amplitudeToUnit(amp: number): number {
  if (amp <= 1e-8) return 0;
  const db = 20 * Math.log10(amp);
  return Math.max(0, Math.min(1, (db + 60) / 60));
}

function upsertLine(lines: LineView[], next: LineView): LineView[] {
  const idx = lines.findIndex((l) => l.id === next.id);
  if (idx === -1) return [...lines, next];
  const copy = lines.slice();
  copy[idx] = { ...copy[idx], ...next, flushed: copy[idx].flushed || next.flushed };
  return copy;
}

let noticeSeq = 0;

export function reduce(state: SessionState, action: Action): SessionState {
  switch (action.kind) {
    case "clear":
      return { ...initialState };
    case "meeting":
      return { ...state, meetingId: action.id, notebook: action.notebook };
    case "stopped":
      return { ...state, ended: true, stopped: true, pendingInsight: null };
    case "snapshot": {
      const s = action.snapshot;
      if (!s.status) return state;
      return {
        ...state,
        status: s.status,
        ended: s.status.state === "finished" || s.status.state === "stopped",
        stopped: s.status.state === "stopped",
        meetingId: s.meeting_id ?? state.meetingId,
        notebook: s.notebook ?? state.notebook,
        lines: (s.transcript ?? []).map((l) => ({
          id: l.line_id,
          ts: splitLine(l.line).ts,
          text: l.text || splitLine(l.line).rest,
          final: true,
          flushed: l.flushed,
        })),
        insights: s.insights ?? [],
        chat: (s.chat ?? []).map((c, i) => ({ id: -(i + 1), question: c.question, answer: c.answer })),
      };
    }
    case "chat.ask":
      return { ...state, chat: [...state.chat, { id: action.id, question: action.question, answer: null }] };
    case "chat.answer":
      return {
        ...state,
        chat: state.chat.map((c) => (c.id === action.id ? { ...c, answer: action.answer, error: action.error } : c)),
      };
    case "event":
      return applyEvent(state, action.event);
  }
}

function applyEvent(state: SessionState, ev: EchoEvent): SessionState {
  switch (ev.type) {
    case "transcript.draft":
    case "transcript.final": {
      const { ts, rest } = splitLine(ev.line);
      return {
        ...state,
        lines: upsertLine(state.lines, {
          id: ev.line_id,
          ts,
          text: ev.text || rest,
          final: ev.type === "transcript.final",
          flushed: false,
        }),
      };
    }
    case "transcript.corrected": {
      const prevFinal = new Map(state.lines.map((l) => [l.id, l.final]));
      return {
        ...state,
        lines: ev.lines.map((l) => ({
          id: l.line_id,
          ts: splitLine(l.line).ts,
          text: l.text || splitLine(l.line).rest,
          final: prevFinal.get(l.line_id) ?? true,
          flushed: l.flushed,
        })),
      };
    }
    case "insight.pending":
      return { ...state, pendingInsight: { n: ev.n, stamp: ev.stamp } };
    case "insight.pass": {
      const { type: _t, ...rest } = ev;
      return { ...state, insights: [...state.insights, rest], pendingInsight: null };
    }
    case "meter": {
      const unit = Math.max(amplitudeToUnit(ev.peak) * 0.75 + amplitudeToUnit(ev.rms) * 0.25, 0);
      const meterHistory = [...state.meterHistory.slice(-(METER_HISTORY - 1)), unit];
      return { ...state, meter: { rms: ev.rms, peak: ev.peak, note: ev.note }, meterHistory };
    }
    case "status":
      return { ...state, status: ev, preparing: false, ended: state.ended || ev.state === "finished" || ev.state === "stopped", stopped: state.stopped || ev.state === "stopped" };
    case "notice": {
      if (ev.level === "debug") return state;
      const notice: NoticeView = { id: ++noticeSeq, level: ev.level, text: stripRichMarkup(ev.text) };
      return { ...state, notices: [...state.notices.slice(-49), notice] };
    }
    case "error": {
      const notice: NoticeView = { id: ++noticeSeq, level: "error", text: ev.message };
      return { ...state, notices: [...state.notices.slice(-49), notice] };
    }
    case "session.reset":
      return {
        ...state,
        lines: [],
        insights: [],
        pendingInsight: null,
        notices: [],
        chat: [],
        meterHistory: Array(METER_HISTORY).fill(0),
      };
    case "session.ended":
      return { ...state, ended: true };
    case "session.preparing":
      return { ...state, preparing: true };
    case "summary.written":
      return { ...state, lastSummaryPath: ev.path };
    case "apple.install.progress":
      return { ...state, appleInstall: { language: ev.language, fraction: ev.fraction, done: false } };
    case "apple.install.done":
      return {
        ...state,
        appleInstall: { language: ev.language, fraction: ev.ok ? 1 : 0, done: true, error: ev.error },
      };
    case "chat.reply":
      return state;
    default:
      return state;
  }
}
