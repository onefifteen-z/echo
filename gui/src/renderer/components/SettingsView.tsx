import { useCallback, useEffect, useMemo, useState } from "react";
import type {
  EffectiveValue,
  Language,
  MeetingApp,
  MeetingWatchStatus,
  ProbeResult,
  SettingsPatch,
  SettingsSnapshot,
  WikiVaults,
} from "@shared/types";
import type { AppleInstallView } from "../lib/session-store";
import {
  CURATED_KEYS,
  type FieldDef,
  isSecretKey,
  keyError,
  SETTINGS_GROUPS,
  valueError,
} from "../lib/settings-schema";
import { appleLocaleState, Button, Chip, LANGUAGE_LABELS } from "./ui";

interface Props {
  engineReady: boolean;
  liveRunning: boolean;
  appleInstall: AppleInstallView | null;
  /** Ends the live meeting (saving it) before the engine restarts. */
  endLive: () => Promise<void>;
  onSaved: (restarted: boolean) => void;
  toast: (text: string, tone?: "info" | "error") => void;
}

/** Pending change per key: a string sets the app value, `null` removes the app override. */
type Edits = Record<string, string | null>;

const field = "w-full rounded-md border border-line bg-panel-2 px-3 py-1.5 text-sm text-zinc-100 outline-none focus:border-accent";

function errorText(e: unknown): string {
  return (e instanceof Error ? e.message : String(e)).replace(/^Error invoking remote method '[^']+': (Error: )?/, "");
}

const MEETING_APP_NAMES: Record<MeetingApp, string> = { zoom: "Zoom", teams: "Microsoft Teams" };

function MeetingDetectionSection({
  snap,
  onChange,
  toast,
}: {
  snap: SettingsSnapshot;
  onChange: (s: SettingsSnapshot) => void;
  toast: (text: string, tone?: "info" | "error") => void;
}) {
  const [watch, setWatch] = useState<MeetingWatchStatus | null>(null);
  useEffect(() => {
    void window.echo.getMeetingWatch().then(setWatch);
    return window.echo.onMeetingWatchStatus(setWatch);
  }, []);
  const prefs = snap.prefs;
  const set = (patch: Parameters<typeof window.echo.setPrefs>[0]) =>
    window.echo
      .setPrefs(patch)
      .then(onChange)
      .catch((e) => toast(errorText(e), "error"));
  const watchLabel: Record<MeetingWatchStatus["state"], string> = {
    running: "Watching",
    starting: "Starting…",
    stopped: "Off",
    unavailable: "Unavailable",
  };

  return (
    <section className="rounded-lg border border-line bg-panel px-5 py-4">
      <h2 className="text-xs font-semibold tracking-wide text-zinc-400 uppercase">Meeting detection</h2>
      <p className="mt-1 text-xs text-zinc-500">
        When Zoom or Microsoft Teams starts using the microphone, Echo shows a prompt to start transcribing with your
        last-used capture settings. Meeting names are read from window titles when Screen Recording access is granted.
      </p>
      <div className="mt-3 space-y-3">
        <label className="flex items-center gap-3 text-sm text-zinc-200">
          <input
            type="checkbox"
            className="h-4 w-4 accent-[var(--color-accent)]"
            checked={prefs.meetingDetection}
            onChange={(e) => void set({ meetingDetection: e.target.checked })}
          />
          Offer to transcribe Zoom and Teams meetings
          {watch && (
            <Chip tone={watch.state === "running" ? "ok" : watch.state === "unavailable" ? "warn" : "neutral"} title={watch.message}>
              {watchLabel[watch.state]}
            </Chip>
          )}
        </label>
        {watch?.state === "unavailable" && watch.message && <p className="text-xs text-amber-300">{watch.message}</p>}
        {prefs.mutedApps.length > 0 && (
          <div className="flex flex-wrap items-center gap-2 text-sm text-zinc-400">
            Not asking for:
            {prefs.mutedApps.map((app) => (
              <span key={app} className="flex items-center gap-1.5 rounded-full border border-line px-2.5 py-0.5 text-xs text-zinc-300">
                {MEETING_APP_NAMES[app]}
                <button
                  className="no-drag text-zinc-500 underline hover:text-zinc-200"
                  onClick={() => void set({ mutedApps: prefs.mutedApps.filter((a) => a !== app) })}
                >
                  Ask again
                </button>
              </span>
            ))}
          </div>
        )}
      </div>
    </section>
  );
}

function SourceBadge({ source }: { source: "app" | "env" | "default" | "pending" }) {
  if (source === "pending") return <Chip tone="warn">Unsaved</Chip>;
  if (source === "app") return <Chip tone="accent">Echo settings</Chip>;
  if (source === "env") return <Chip title="Set in the shell environment or a .env file">.env</Chip>;
  return null;
}

export function SettingsView({ engineReady, liveRunning, appleInstall, endLive, onSaved, toast }: Props) {
  const [snap, setSnap] = useState<SettingsSnapshot | null>(null);
  const [effective, setEffective] = useState<Record<string, EffectiveValue>>({});
  const [probe, setProbe] = useState<ProbeResult | null>(null);
  const [wiki, setWiki] = useState<WikiVaults | null>(null);
  const [apple, setApple] = useState<Record<string, string>>({});
  const [installing, setInstalling] = useState<Language | null>(null);
  const [edits, setEdits] = useState<Edits>({});
  const [workspace, setWorkspace] = useState<string | null>(null);
  const [newKey, setNewKey] = useState("");
  const [newValue, setNewValue] = useState("");
  const [saving, setSaving] = useState(false);
  const [confirmLive, setConfirmLive] = useState(false);

  const load = useCallback(async () => {
    const s = await window.echo.getSettings();
    setSnap(s);
    if (!engineReady) return;
    const keys = [...new Set([...CURATED_KEYS, ...Object.keys(s.env), ...s.secretsSet])];
    window.echo.invoke("config.effective", { keys }).then(setEffective).catch(() => undefined);
    window.echo.invoke("probe", {}).then(setProbe).catch(() => undefined);
    window.echo.invoke("wiki.vaults", {}).then(setWiki).catch(() => undefined);
  }, [engineReady]);

  useEffect(() => {
    void load();
  }, [load]);

  useEffect(() => {
    if (!engineReady) return;
    window.echo
      .invoke("apple.status", { languages: ["japanese", "english", "chinese"] })
      .then(setApple)
      .catch(() => setApple({}));
  }, [engineReady, appleInstall?.done]);

  const appValue = (key: string): string | undefined => snap?.env[key];
  const appSecret = (key: string): boolean => !!snap?.secretsSet.includes(key);
  const hasApp = (key: string) => appValue(key) !== undefined || appSecret(key);

  const dirty = Object.keys(edits).length > 0 || workspace !== null;

  const errors = useMemo(() => {
    const out: Record<string, string> = {};
    const defs = new Map(SETTINGS_GROUPS.flatMap((g) => g.fields.map((f) => [f.key, f] as const)));
    for (const [key, value] of Object.entries(edits)) {
      if (value === null) continue;
      const err = keyError(key) ?? (defs.has(key) ? valueError(defs.get(key)!, value) : null);
      if (err) out[key] = err;
    }
    return out;
  }, [edits]);

  function setEdit(key: string, value: string | null) {
    setEdits((e) => {
      const next = { ...e };
      const unchanged = value === null ? !hasApp(key) : !isSecretKey(key) && appValue(key) === value;
      if (unchanged) delete next[key];
      else next[key] = value;
      return next;
    });
  }

  function sourceOf(key: string): "app" | "env" | "default" | "pending" {
    if (key in edits) return "pending";
    if (hasApp(key)) return "app";
    return effective[key]?.source === "env" ? "env" : "default";
  }

  async function save(mode: "restart" | "later" | "end-and-restart") {
    setConfirmLive(false);
    setSaving(true);
    try {
      if (mode === "end-and-restart") await endLive();
      const env = Object.fromEntries(Object.entries(edits).filter(([k, v]) => !(isSecretKey(k) && v === "")));
      const patch: SettingsPatch = { env };
      if (workspace !== null) patch.workspaceDir = workspace;
      const s = await window.echo.saveSettings(patch, { restart: mode !== "later" });
      setSnap(s);
      setEdits({});
      setWorkspace(null);
      onSaved(mode !== "later");
      toast(mode === "later" ? "Settings saved. They apply the next time the engine restarts." : "Settings saved; engine restarted.");
    } catch (e) {
      toast(errorText(e), "error");
    } finally {
      setSaving(false);
    }
  }

  function requestSave() {
    if (Object.keys(errors).length) {
      toast("Fix the highlighted fields first.", "error");
      return;
    }
    if (liveRunning) setConfirmLive(true);
    else void save("restart");
  }

  async function installApple(language: Language) {
    setInstalling(language);
    try {
      await window.echo.invoke("apple.install_assets", { language });
      toast(`Apple ${LANGUAGE_LABELS[language]} model installed.`);
    } catch (e) {
      toast(errorText(e), "error");
    } finally {
      setInstalling(null);
    }
  }

  if (!snap) return <div className="pt-24 text-center text-sm text-zinc-500">Loading settings…</div>;

  const renderField = (f: FieldDef) => {
    const pending = edits[f.key];
    const eff = effective[f.key];
    const envHint = eff?.source === "env" && eff.value ? eff.value : undefined;
    const placeholder = envHint ?? f.placeholder ?? "";
    const err = errors[f.key];
    let input;
    if (f.kind === "secret") {
      const stored = appSecret(f.key) && pending === undefined;
      input = stored ? (
        <div className="flex items-center gap-2">
          <span className="text-sm text-zinc-400">•••••••• stored in Keychain</span>
          <Button onClick={() => setEdit(f.key, "")}>Replace</Button>
        </div>
      ) : (
        <input
          type="password"
          autoComplete="off"
          className={field}
          value={pending ?? ""}
          placeholder={envHint ? `${envHint} (from .env)` : "Not set"}
          onChange={(e) => setEdit(f.key, e.target.value)}
        />
      );
    } else if (f.kind === "select" || f.kind === "bool") {
      const options = f.kind === "bool" ? [{ value: "1", label: "On" }, { value: "0", label: "Off" }] : f.options!;
      const current = pending ?? appValue(f.key) ?? "";
      const envLabel = envHint ? ` (.env: ${envHint})` : "";
      input = (
        <select className={field} value={current} onChange={(e) => setEdit(f.key, e.target.value === "" ? null : e.target.value)}>
          <option value="">{`Default${f.placeholder ? ` — ${f.placeholder}` : ""}${envLabel}`}</option>
          {options.map((o) => (
            <option key={o.value} value={o.value}>
              {o.label}
            </option>
          ))}
        </select>
      );
    } else {
      const current = pending ?? appValue(f.key) ?? "";
      input = (
        <div className="flex gap-2">
          <input
            className={`${field} ${err ? "border-red-500/60" : ""}`}
            inputMode={f.kind === "number" ? "decimal" : undefined}
            value={current}
            placeholder={placeholder}
            onChange={(e) => setEdit(f.key, e.target.value === "" ? null : e.target.value)}
          />
          {f.kind === "file" && (
            <Button
              onClick={async () => {
                const p = await window.echo.chooseFile(current || undefined, f.fileKind);
                if (p) setEdit(f.key, p);
              }}
            >
              Choose…
            </Button>
          )}
        </div>
      );
    }
    return (
      <div key={f.key} className="grid grid-cols-[220px_minmax(0,1fr)] items-start gap-4 py-2.5">
        <div>
          <div className="text-sm text-zinc-200">{f.label}</div>
          <div className="font-mono text-[10px] text-zinc-600 selectable">{f.key}</div>
        </div>
        <div className="space-y-1">
          {input}
          <div className="flex flex-wrap items-center gap-2 text-xs text-zinc-500">
            <SourceBadge source={sourceOf(f.key)} />
            {hasApp(f.key) && pending !== null && (
              <button className="no-drag text-zinc-400 underline hover:text-zinc-200" onClick={() => setEdit(f.key, null)}>
                Remove override
              </button>
            )}
            {pending !== undefined && (
              <button
                className="no-drag text-zinc-400 underline hover:text-zinc-200"
                onClick={() =>
                  setEdits((e) => {
                    const next = { ...e };
                    delete next[f.key];
                    return next;
                  })
                }
              >
                Undo
              </button>
            )}
            {err ? <span className="text-red-300">{err}</span> : f.help && <span>{f.help}</span>}
          </div>
        </div>
      </div>
    );
  };

  const advancedKeys = [...new Set([...Object.keys(snap.env), ...snap.secretsSet, ...Object.keys(edits)])]
    .filter((k) => !CURATED_KEYS.includes(k))
    .sort();
  const newKeyErr = newKey ? keyError(newKey) ?? (CURATED_KEYS.includes(newKey) ? "Set this in its section above." : null) : null;
  const workspaceShown = workspace ?? snap.workspaceDir;

  return (
    <div className="relative flex min-h-0 flex-1 flex-col">
      <div className="min-h-0 flex-1 overflow-y-auto">
        <div className="no-drag mx-auto max-w-3xl space-y-6 px-6 pt-2 pb-24">
          <div>
            <h1 className="text-lg font-semibold">Settings</h1>
            <p className="text-sm text-zinc-400">
              Values set here override your shell and <code>.env</code> for the desktop app only; the terminal app is
              unaffected. Saving restarts the Echo engine.
            </p>
            {!snap.encryptionAvailable && (
              <p className="mt-2 rounded-md bg-amber-500/10 p-2 text-xs text-amber-200">
                System encryption is unavailable, so API keys are stored unencrypted in the settings file.
              </p>
            )}
          </div>

          {SETTINGS_GROUPS.map((g) => (
            <section key={g.id} className="rounded-lg border border-line bg-panel px-5 py-4">
              <h2 className="text-xs font-semibold tracking-wide text-zinc-400 uppercase">{g.title}</h2>
              {g.description && <p className="mt-1 text-xs text-zinc-500">{g.description}</p>}
              <div className="mt-2 divide-y divide-line">
                {g.id === "workspace" && (
                  <div className="grid grid-cols-[220px_minmax(0,1fr)] items-start gap-4 py-2.5">
                    <div className="text-sm text-zinc-200">Meetings folder</div>
                    <div className="space-y-1">
                      <div className="flex gap-2">
                        <input className={`${field} selectable`} readOnly value={workspaceShown} />
                        <Button
                          onClick={async () => {
                            const p = await window.echo.chooseDir(workspaceShown);
                            if (p) setWorkspace(p === snap.workspaceDir ? null : p);
                          }}
                        >
                          Choose…
                        </Button>
                      </div>
                      <div className="flex flex-wrap items-center gap-2 text-xs text-zinc-500">
                        {workspace !== null ? (
                          <Chip tone="warn">Unsaved</Chip>
                        ) : (
                          snap.customWorkspace && <Chip tone="accent">Echo settings</Chip>
                        )}
                        {(snap.customWorkspace || workspace !== null) && workspaceShown !== snap.defaultWorkspaceDir && (
                          <button
                            className="no-drag text-zinc-400 underline hover:text-zinc-200"
                            onClick={() => setWorkspace(snap.customWorkspace ? snap.defaultWorkspaceDir : null)}
                          >
                            Use default ({snap.defaultWorkspaceDir})
                          </button>
                        )}
                        <span>New meetings are saved here, and the Meetings list reads from it. Existing folders are not moved.</span>
                      </div>
                    </div>
                  </div>
                )}
                {g.fields.map(renderField)}
                {g.id === "speech" && probe?.platform === "darwin" && (
                  <div className="grid grid-cols-[220px_minmax(0,1fr)] items-start gap-4 py-2.5">
                    <div className="text-sm text-zinc-200">Apple on-device models</div>
                    <div className="space-y-1.5">
                      {(Object.keys(LANGUAGE_LABELS) as Language[]).map((lang) => {
                        const st = appleLocaleState(apple, lang);
                        const busy = installing === lang;
                        const pct = Math.round((appleInstall?.fraction ?? 0) * 100);
                        return (
                          <div key={lang} className="flex items-center gap-3 text-sm">
                            <span className="w-20 text-zinc-300">{LANGUAGE_LABELS[lang]}</span>
                            <Chip tone={st === "installed" ? "ok" : st === "supported" ? "warn" : "neutral"}>
                              {st === "installed" ? "Installed" : st === "supported" ? "Not downloaded" : "Unavailable"}
                            </Chip>
                            {st === "supported" && (
                              <Button disabled={installing !== null} onClick={() => void installApple(lang)}>
                                {busy ? `Downloading… ${pct}%` : "Download"}
                              </Button>
                            )}
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}
                {g.id === "search" && probe?.search && (
                  <p className="pt-2.5 text-xs text-zinc-500">
                    Engine currently uses: <span className="text-zinc-300">{probe.search.provider}</span>
                    {probe.search.note && <span className="text-amber-300"> — {probe.search.note}</span>}
                  </p>
                )}
                {g.id === "knowledge" && wiki && (
                  <div className="flex flex-wrap items-center gap-2 pt-2.5 text-xs text-zinc-500">
                    <Chip tone={wiki.available && wiki.vaults.length > 0 ? "ok" : "warn"}>
                      {wiki.available ? `${wiki.vaults.length} vault${wiki.vaults.length === 1 ? "" : "s"}` : "Not found"}
                    </Chip>
                    <span className="selectable">
                      {wiki.reason ?? wiki.vaults.map((v) => v.name).join(", ")}
                    </span>
                  </div>
                )}
                {g.id === "llm" && probe && (
                  <p className="pt-2.5 text-xs text-zinc-500">
                    Engine currently uses:{" "}
                    <span className="text-zinc-300">{probe.llm.online ? probe.llm.label : "no reachable LLM"}</span>
                  </p>
                )}
              </div>
            </section>
          ))}

          <MeetingDetectionSection snap={snap} onChange={setSnap} toast={toast} />

          <section className="rounded-lg border border-line bg-panel px-5 py-4">
            <h2 className="text-xs font-semibold tracking-wide text-zinc-400 uppercase">Advanced</h2>
            <p className="mt-1 text-xs text-zinc-500">
              Any other Echo environment variable (see the README). Keys containing KEY, TOKEN, SECRET or PASSWORD are
              stored encrypted.
            </p>
            <div className="mt-3 space-y-2">
              {advancedKeys.length === 0 && <p className="text-sm text-zinc-500">No extra variables.</p>}
              {advancedKeys.map((key) => {
                const secret = isSecretKey(key);
                const pending = edits[key];
                const removed = pending === null;
                return (
                  <div key={key} className="flex items-center gap-2">
                    <code className={`w-72 truncate text-xs selectable ${removed ? "text-zinc-600 line-through" : "text-zinc-300"}`}>
                      {key}
                    </code>
                    {secret && pending === undefined ? (
                      <span className="flex-1 text-sm text-zinc-500">•••••••• stored in Keychain</span>
                    ) : (
                      <input
                        type={secret ? "password" : "text"}
                        disabled={removed}
                        className={`${field} flex-1 ${errors[key] ? "border-red-500/60" : ""}`}
                        value={pending ?? appValue(key) ?? ""}
                        onChange={(e) => setEdit(key, e.target.value)}
                      />
                    )}
                    {removed ? (
                      <Button onClick={() => setEdits(({ [key]: _, ...rest }) => rest)}>Undo</Button>
                    ) : (
                      <Button
                        variant="danger"
                        onClick={() => (hasApp(key) ? setEdit(key, null) : setEdits(({ [key]: _, ...rest }) => rest))}
                      >
                        Remove
                      </Button>
                    )}
                  </div>
                );
              })}
              <div className="flex items-start gap-2 pt-2">
                <div className="w-72">
                  <input
                    className={`${field} font-mono text-xs ${newKeyErr ? "border-red-500/60" : ""}`}
                    placeholder="ECHO_…"
                    value={newKey}
                    onChange={(e) => setNewKey(e.target.value.toUpperCase().replace(/\s/g, "_"))}
                  />
                  {newKeyErr && <p className="mt-1 text-xs text-red-300">{newKeyErr}</p>}
                </div>
                <input
                  type={isSecretKey(newKey) ? "password" : "text"}
                  className={`${field} flex-1`}
                  placeholder="value"
                  value={newValue}
                  onChange={(e) => setNewValue(e.target.value)}
                />
                <Button
                  disabled={!newKey || !!newKeyErr}
                  onClick={() => {
                    setEdits((e) => ({ ...e, [newKey]: newValue }));
                    setNewKey("");
                    setNewValue("");
                  }}
                >
                  Add
                </Button>
              </div>
            </div>
          </section>
        </div>
      </div>

      {dirty && (
        <div className="no-drag absolute inset-x-0 bottom-0 flex items-center justify-end gap-3 border-t border-line bg-panel/95 px-6 py-3 backdrop-blur">
          <span className="mr-auto text-sm text-zinc-400">
            {Object.keys(edits).length + (workspace !== null ? 1 : 0)} unsaved change(s)
          </span>
          <Button
            disabled={saving}
            onClick={() => {
              setEdits({});
              setWorkspace(null);
            }}
          >
            Discard
          </Button>
          <Button variant="primary" disabled={saving} onClick={requestSave}>
            {saving ? "Saving…" : "Save & restart engine"}
          </Button>
        </div>
      )}

      {confirmLive && (
        <div className="no-drag fixed inset-0 z-40 flex items-center justify-center bg-black/60">
          <div className="w-[28rem] rounded-xl border border-line bg-panel p-5 shadow-2xl">
            <h2 className="font-semibold">A meeting is in progress</h2>
            <p className="mt-2 text-sm text-zinc-400">
              Applying settings restarts the Echo engine, which ends the current meeting (it is saved first). You can also
              save now and apply the settings later.
            </p>
            <div className="mt-5 flex justify-end gap-2">
              <Button onClick={() => setConfirmLive(false)}>Cancel</Button>
              <Button onClick={() => void save("later")}>Save, apply later</Button>
              <Button variant="danger" onClick={() => void save("end-and-restart")}>
                End meeting & restart
              </Button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
