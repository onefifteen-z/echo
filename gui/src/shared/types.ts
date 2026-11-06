/**
 * Wire types shared by main, preload, and renderer. They mirror `echo/engine/events.py` and the
 * command table in `echo/server.py` — keep the three in sync.
 */

export type Language = "japanese" | "english" | "chinese";
export type AsrBackendChoice = "auto" | "apple" | "api" | "local";
export type AudioSource = "mic" | "both" | "demo";

export interface Status {
  type: "status";
  mode: "live" | "audio" | "replay";
  state: "idle" | "running" | "paused" | "finished" | "stopped";
  language: Language;
  languages: Language[];
  asr_source: string | null;
  asr_label: string;
  mic_only: boolean;
  llm_offline: boolean;
  meeting_title: string;
  elapsed: string;
}

export interface TranscriptLineState {
  line_id: number;
  line: string;
  text: string;
  flushed: boolean;
}

export interface InsightPassEvent {
  type: "insight.pass";
  n: number;
  stamp: string;
  kind: "comment" | "web_search";
  body_md: string;
  routing_md: string | null;
  corrected_md: string | null;
  web_md: string | null;
}

export type EchoEvent =
  | { type: "transcript.draft"; line_id: number; line: string; text: string }
  | { type: "transcript.final"; line_id: number; line: string; text: string }
  | { type: "transcript.corrected"; lines: TranscriptLineState[] }
  | { type: "insight.pending"; n: number; stamp: string }
  | InsightPassEvent
  | { type: "meter"; rms: number; peak: number; note: string }
  | Status
  | { type: "notice"; level: "info" | "warning" | "error" | "debug"; text: string }
  | { type: "error"; message: string }
  | { type: "chat.reply"; question: string; answer: string }
  | { type: "session.reset" }
  | { type: "session.ended" }
  | { type: "session.preparing"; backend: string; language: Language }
  | { type: "summary.written"; path: string; ok: boolean; announce: boolean; id?: string }
  | { type: "apple.install.progress"; language: Language; fraction: number }
  | { type: "apple.install.done"; language: Language; ok: boolean; error?: string };

export interface SessionStartArgs {
  title: string;
  context: string;
  source: AudioSource;
  backend: AsrBackendChoice;
  language: Language;
  mic_device?: number | null;
}

export interface SessionSnapshot {
  status: Status | null;
  transcript?: TranscriptLineState[];
  insights?: Omit<InsightPassEvent, "type">[];
  chat?: { question: string; answer: string }[];
  output_dir?: string;
  meeting_id?: string;
  notebook?: string;
}

export interface MeetingSummary {
  id: string;
  title: string;
  context: string;
  started_at: string;
  ended_at?: string;
  duration_s: number | null;
  mode: string;
  language: string;
  asr_source: string;
  /** null when the transcript is only in the cloud (evicted by OneDrive / iCloud) and was not read. */
  line_count: number | null;
  legacy: boolean;
  live: boolean;
  summary_preview: string;
  has_notebook: boolean;
}

export interface MeetingDetail extends Omit<MeetingSummary, "summary_preview" | "has_notebook"> {
  path: string;
  transcript: string[];
  summary_md: string;
  insights_md: string;
  notebook_md: string;
}

export interface MeetingUpdate {
  id: string;
  title?: string;
  context?: string;
  notebook?: string;
}

export interface SettingsSnapshot {
  workspaceDir: string;
  defaultWorkspaceDir: string;
  customWorkspace: boolean;
  /** Non-secret overrides stored in the app. */
  env: Record<string, string>;
  /** Secret keys that have a stored (encrypted) value; values are never sent to the renderer. */
  secretsSet: string[];
  encryptionAvailable: boolean;
}

export interface SettingsPatch {
  /** `null` resets to the default folder. */
  workspaceDir?: string | null;
  /** `null` removes the override (falls back to `.env` / default). Empty string clears a secret. */
  env?: Record<string, string | null>;
}

export interface EffectiveValue {
  value: string | null;
  source: "app" | "env" | "default";
  secret: boolean;
}

export interface InputDevice {
  index: number;
  name: string;
  channels: number;
}

export interface ProbeResult {
  platform: string;
  python: string;
  llm: { online: boolean; label: string };
  whisper_api: { configured: boolean; base_url: string | null; model: string | null; usable: boolean };
  search: { provider: string; perplexity_configured: boolean; note: string };
  apple_speech: Record<string, string>;
  languages: Language[];
}

export type SidecarState =
  | { phase: "starting" }
  | { phase: "ready"; port: number }
  | { phase: "crashed"; message: string; restarting: boolean }
  | { phase: "failed"; message: string };

/** Commands the renderer may send (subset of `echo/server.py` handlers). */
export interface CommandMap {
  ping: [Record<string, never>, { pong: boolean }];
  probe: [Record<string, never>, ProbeResult];
  "devices.list": [Record<string, never>, { inputs: InputDevice[]; default_input: number | null }];
  "apple.status": [{ languages: Language[] }, Record<string, string>];
  "apple.install_assets": [{ language: Language }, { language: Language; installed: boolean }];
  "session.start": [SessionStartArgs, { status: Status; output_dir: string }];
  "session.stop": [Record<string, never>, { output_dir: string | null }];
  "session.pause": [Record<string, never>, Status];
  "session.resume": [Record<string, never>, Status];
  "session.reset": [{ title?: string; context?: string; resume?: boolean }, Status];
  "session.set_language": [{ language: Language }, { message: string; status: Status }];
  "session.save": [Record<string, never>, { output_dir: string }];
  "session.snapshot": [Record<string, never>, SessionSnapshot];
  "chat.ask": [{ question: string }, { question: string; answer: string }];
  "session.notebook.set": [{ text: string }, { ok: boolean }];
  "meetings.list": [Record<string, never>, { meetings: MeetingSummary[]; live_id: string | null }];
  "meetings.get": [{ id: string }, MeetingDetail];
  "meetings.update": [MeetingUpdate, MeetingDetail];
  "meetings.regenerate_summary": [{ id: string }, { id: string; summary_md: string }];
  "config.effective": [{ keys: string[] }, Record<string, EffectiveValue>];
}

export type CommandName = keyof CommandMap;

export const IPC = {
  invoke: "echo:invoke",
  event: "echo:event",
  sidecarState: "echo:sidecar-state",
  getSidecarState: "echo:get-sidecar-state",
  restartSidecar: "echo:restart-sidecar",
  openPath: "echo:open-path",
  trashMeeting: "echo:trash-meeting",
  getSettings: "echo:settings-get",
  saveSettings: "echo:settings-save",
  chooseDir: "echo:choose-dir",
  chooseFile: "echo:choose-file",
} as const;

export interface EchoAPI {
  invoke<K extends CommandName>(cmd: K, args: CommandMap[K][0]): Promise<CommandMap[K][1]>;
  startSession(args: SessionStartArgs): Promise<CommandMap["session.start"][1]>;
  stopSession(): Promise<CommandMap["session.stop"][1]>;
  pause(): Promise<Status>;
  resume(): Promise<Status>;
  reset(args?: CommandMap["session.reset"][0]): Promise<Status>;
  setLanguage(language: Language): Promise<CommandMap["session.set_language"][1]>;
  ask(question: string): Promise<CommandMap["chat.ask"][1]>;
  onEvent(cb: (event: EchoEvent) => void): () => void;
  onSidecarState(cb: (state: SidecarState) => void): () => void;
  getSidecarState(): Promise<SidecarState>;
  restartSidecar(): Promise<void>;
  openPath(path: string): Promise<void>;
  /** Move a saved meeting folder to the system Trash (refused for the meeting in progress). */
  trashMeeting(id: string): Promise<void>;
  getSettings(): Promise<SettingsSnapshot>;
  /** Persist settings; with `restart`, the engine restarts so the new environment takes effect. */
  saveSettings(patch: SettingsPatch, opts: { restart: boolean }): Promise<SettingsSnapshot>;
  chooseDir(current?: string): Promise<string | null>;
  chooseFile(current?: string): Promise<string | null>;
}
