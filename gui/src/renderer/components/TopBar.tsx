import type { Language, Status } from "@shared/types";
import { Button, Chip, LANGUAGE_LABELS } from "./ui";

interface Props {
  status: Status;
  ended: boolean;
  stopped: boolean;
  onPauseToggle: () => void;
  onLanguage: (language: Language) => void;
  onSave: () => void;
  onReset: () => void;
  onStop: () => void;
  onNew: () => void;
}

function shortAsr(source: string | null): string {
  if (source === "apple-on-device") return "Apple on-device";
  if (source === "whisper-api") return "Whisper API";
  if (!source) return "—";
  return "Local Kotoba";
}

export function TopBar({ status, ended, stopped, onPauseToggle, onLanguage, onSave, onReset, onStop, onNew }: Props) {
  const live = status.mode === "live";
  const paused = status.state === "paused";
  return (
    <div className="drag flex h-14 items-center gap-3 border-b border-line px-4">
      <div className="min-w-0">
        <div className="truncate text-sm font-semibold">{status.meeting_title}</div>
        <div className="font-mono text-xs text-zinc-500">{status.elapsed}</div>
      </div>
      <div className="no-drag ml-2 flex flex-wrap items-center gap-2">
        {stopped ? (
          <Chip>Ended</Chip>
        ) : ended ? (
          <Chip title="The audio source finished. You can still ask questions, save, or end the meeting.">Finished</Chip>
        ) : paused ? (
          <Chip tone="warn">Paused</Chip>
        ) : (
          <Chip tone="ok">● {live ? "Recording" : "Replaying"}</Chip>
        )}
        {live && (
          <select
            className="rounded-full border border-line bg-panel-2 px-2 py-0.5 text-xs text-zinc-200 outline-none"
            value={status.language}
            disabled={ended || stopped}
            onChange={(e) => onLanguage(e.target.value as Language)}
          >
            {status.languages.map((l) => (
              <option key={l} value={l}>
                {LANGUAGE_LABELS[l]}
              </option>
            ))}
          </select>
        )}
        {live && (
          <Chip tone="accent" title={status.asr_label}>
            ASR: {shortAsr(status.asr_source)}
          </Chip>
        )}
        {status.mic_only && (
          <Chip tone="warn" title="Start with 'Mic + system audio' to transcribe calls and videos.">
            Mic only — system audio not captured
          </Chip>
        )}
        {status.llm_offline && <Chip tone="error">LLM unavailable</Chip>}
      </div>
      <div className="ml-auto flex items-center gap-2">
        {stopped ? (
          <Button variant="primary" onClick={onNew}>
            New meeting
          </Button>
        ) : (
          <>
            {!ended && <Button onClick={onPauseToggle}>{paused ? "Resume" : "Pause"}</Button>}
            <Button onClick={onSave}>Save</Button>
            {!ended && <Button onClick={onReset}>Reset</Button>}
            <Button variant="danger" onClick={onStop}>
              End meeting
            </Button>
          </>
        )}
      </div>
    </div>
  );
}
