import { useEffect, useRef, useState } from "react";
import type { Language, Status, WikiVaults } from "@shared/types";
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
  /** Live meeting edits; omitted when the meeting can no longer be edited here. */
  onRename?: (title: string) => Promise<void>;
  loadContext?: () => Promise<string>;
  onContext?: (context: string) => Promise<void>;
  onVault?: (vault: string | null) => Promise<void>;
}

function VaultSelect({ vault, disabled, onVault }: { vault: string | null; disabled: boolean; onVault?: (vault: string | null) => Promise<void> }) {
  const [list, setList] = useState<WikiVaults | null>(null);
  const [saving, setSaving] = useState(false);
  useEffect(() => {
    window.echo
      .invoke("wiki.vaults", {})
      .then(setList)
      .catch(() => setList({ available: false, vaults: [] }));
  }, []);
  if (!vault && (!list?.available || list.vaults.length === 0)) return null;
  const names = list?.vaults.map((v) => v.name) ?? [];
  if (vault && !names.includes(vault)) names.unshift(vault);
  return (
    <select
      className={`max-w-[11rem] truncate rounded-full border px-2 py-0.5 text-xs outline-none ${vault ? "border-accent/60 bg-accent/10 text-zinc-100" : "border-line bg-panel-2 text-zinc-400"}`}
      title="Lumina knowledge vault consulted by insights and chat"
      value={vault ?? ""}
      disabled={disabled || saving || !onVault}
      onChange={async (e) => {
        if (!onVault) return;
        setSaving(true);
        try {
          await onVault(e.target.value || null);
        } finally {
          setSaving(false);
        }
      }}
    >
      <option value="">No vault</option>
      {names.map((n) => (
        <option key={n} value={n}>
          Vault: {n}
        </option>
      ))}
    </select>
  );
}

function TitleEditor({ title, onRename }: { title: string; onRename?: (title: string) => Promise<void> }) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(title);
  const cancelled = useRef(false);
  if (!onRename || !editing) {
    return (
      <button
        className={`no-drag block max-w-full truncate text-left text-sm font-semibold ${onRename ? "rounded hover:bg-panel-2" : "cursor-default"}`}
        title={onRename ? "Rename meeting" : undefined}
        onClick={() => {
          if (!onRename) return;
          setDraft(title);
          cancelled.current = false;
          setEditing(true);
        }}
      >
        {title}
      </button>
    );
  }
  const finish = (save: boolean) => {
    if (cancelled.current) return;
    cancelled.current = true;
    setEditing(false);
    const t = draft.trim();
    if (save && t && t !== title) void onRename(t);
  };
  return (
    <input
      autoFocus
      className="no-drag w-72 rounded border border-accent bg-panel-2 px-1.5 py-0.5 text-sm font-semibold outline-none"
      value={draft}
      onChange={(e) => setDraft(e.target.value)}
      onBlur={() => finish(true)}
      onKeyDown={(e) => {
        if (e.key === "Enter") finish(true);
        if (e.key === "Escape") finish(false);
      }}
    />
  );
}

function ContextPopover({
  load,
  save,
  onClose,
}: {
  load: () => Promise<string>;
  save: (context: string) => Promise<void>;
  onClose: () => void;
}) {
  const [text, setText] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  useEffect(() => {
    load()
      .then(setText)
      .catch(() => setText(""));
  }, [load]);
  return (
    <div className="no-drag absolute top-12 left-4 z-30 w-[28rem] rounded-lg border border-line bg-panel p-4 shadow-2xl">
      <div className="mb-1 text-sm font-semibold">Meeting context</div>
      <p className="mb-2 text-xs text-zinc-500">Agenda, participants, background. Used by upcoming insights and the summary.</p>
      <textarea
        autoFocus
        disabled={text === null}
        className="selectable h-36 w-full resize-none rounded-md border border-line bg-panel-2 px-3 py-2 text-sm text-zinc-100 outline-none focus:border-accent"
        value={text ?? "Loading…"}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={(e) => e.key === "Escape" && onClose()}
      />
      <div className="mt-3 flex justify-end gap-2">
        <Button onClick={onClose}>Cancel</Button>
        <Button
          variant="primary"
          disabled={text === null || saving}
          onClick={async () => {
            setSaving(true);
            try {
              await save(text ?? "");
              onClose();
            } finally {
              setSaving(false);
            }
          }}
        >
          {saving ? "Saving…" : "Save"}
        </Button>
      </div>
    </div>
  );
}

function shortAsr(source: string | null): string {
  if (source === "apple-on-device") return "Apple on-device";
  if (source === "whisper-api") return "Whisper API";
  if (!source) return "—";
  return "Local Kotoba";
}

export function TopBar({
  status,
  ended,
  stopped,
  onPauseToggle,
  onLanguage,
  onSave,
  onReset,
  onStop,
  onNew,
  onRename,
  loadContext,
  onContext,
  onVault,
}: Props) {
  const live = status.mode === "live";
  const paused = status.state === "paused";
  const [contextOpen, setContextOpen] = useState(false);
  const editable = !stopped;
  return (
    <div className="drag relative flex h-14 items-center gap-3 border-b border-line px-4">
      <div className="min-w-0">
        <TitleEditor title={status.meeting_title} onRename={editable ? onRename : undefined} />
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
        <VaultSelect vault={status.vault ?? null} disabled={stopped} onVault={editable ? onVault : undefined} />
        {editable && loadContext && onContext && (
          <button
            className="rounded-full border border-line px-2.5 py-0.5 text-xs text-zinc-300 hover:bg-panel-2"
            onClick={() => setContextOpen((o) => !o)}
          >
            Context…
          </button>
        )}
      </div>
      {contextOpen && loadContext && onContext && (
        <ContextPopover load={loadContext} save={onContext} onClose={() => setContextOpen(false)} />
      )}
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
