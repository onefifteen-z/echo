import { useEffect, useMemo, useState } from "react";
import type { AsrBackendChoice, AudioSource, InputDevice, Language, ProbeResult, SessionStartArgs } from "@shared/types";
import type { AppleInstallView } from "../lib/session-store";
import { appleLocaleState, Button, LANGUAGE_LABELS } from "./ui";

interface Props {
  busy: boolean;
  initialTitle?: string;
  appleInstall: AppleInstallView | null;
  onStart: (args: SessionStartArgs) => Promise<void>;
}

export function SetupDialog({ busy, initialTitle = "", appleInstall, onStart }: Props) {
  const [title, setTitle] = useState(initialTitle);
  const [context, setContext] = useState("");
  const [source, setSource] = useState<AudioSource>("both");
  const [backend, setBackend] = useState<AsrBackendChoice>("auto");
  const [language, setLanguage] = useState<Language>("japanese");
  const [micDevice, setMicDevice] = useState<number | null>(null);
  const [devices, setDevices] = useState<InputDevice[]>([]);
  const [probe, setProbe] = useState<ProbeResult | null>(null);
  const [appleStatus, setAppleStatus] = useState<Record<string, string>>({});
  const [error, setError] = useState<string | null>(null);
  const [installing, setInstalling] = useState(false);
  const [workspace, setWorkspace] = useState("Documents/Echo");

  useEffect(() => {
    window.echo.invoke("probe", {}).then(setProbe).catch(() => undefined);
    window.echo
      .getSettings()
      .then(async (s) => {
        setWorkspace(s.workspaceDir.replace(/^\/Users\/[^/]+/, "~"));
        const last = s.prefs.lastStart;
        if (last) {
          setSource(last.source);
          setBackend(last.backend);
          setLanguage(last.language);
          setMicDevice(last.mic_device ?? null);
          return;
        }
        const r = await window.echo.invoke("config.effective", { keys: ["ECHO_ASR_BACKEND"] });
        const v = (r.ECHO_ASR_BACKEND?.value ?? "").trim().toLowerCase();
        if (v === "auto" || v === "apple" || v === "api" || v === "local") setBackend(v);
      })
      .catch(() => undefined);
    window.echo
      .invoke("devices.list", {})
      .then((r) => setDevices(r.inputs))
      .catch(() => undefined);
  }, []);

  useEffect(() => {
    window.echo
      .invoke("apple.status", { languages: [language] })
      .then(setAppleStatus)
      .catch(() => setAppleStatus({}));
  }, [language, appleInstall?.done]);

  const appleState = useMemo(() => appleLocaleState(appleStatus, language), [appleStatus, language]);

  const showAppleInstall =
    probe?.platform === "darwin" && (backend === "auto" || backend === "apple") && appleState === "supported";

  async function installApple() {
    setInstalling(true);
    setError(null);
    try {
      await window.echo.invoke("apple.install_assets", { language });
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setInstalling(false);
    }
  }

  async function submit() {
    setError(null);
    try {
      await onStart({ title, context, source, backend, language, mic_device: micDevice });
    } catch (e) {
      setError((e instanceof Error ? e.message : String(e)).replace(/^Error invoking remote method '[^']+': (Error: )?/, ""));
    }
  }

  const field = "w-full rounded-md border border-line bg-panel-2 px-3 py-2 text-sm text-zinc-100 outline-none focus:border-accent";
  const label = "mb-1 block text-xs font-medium text-zinc-400";
  const pct = Math.round((appleInstall?.fraction ?? 0) * 100);

  return (
    <div className="flex h-full items-center justify-center p-8">
      <div className="no-drag w-full max-w-xl rounded-xl border border-line bg-panel p-6 shadow-2xl">
        <h1 className="mb-1 text-lg font-semibold">New meeting</h1>
        <p className="mb-5 text-sm text-zinc-400">Live transcription with insights. Outputs are saved under {workspace}.</p>

        <div className="space-y-4">
          <div>
            <label className={label}>Title</label>
            <input className={field} value={title} onChange={(e) => setTitle(e.target.value)} placeholder="Weekly sync" />
          </div>
          <div>
            <label className={label}>Context (agenda, participants, background)</label>
            <textarea className={`${field} h-20 resize-none`} value={context} onChange={(e) => setContext(e.target.value)} />
          </div>
          <div className="grid grid-cols-3 gap-3">
            <div>
              <label className={label}>Audio source</label>
              <select className={field} value={source} onChange={(e) => setSource(e.target.value as AudioSource)}>
                <option value="both">Mic + system audio</option>
                <option value="mic">Microphone only</option>
                <option value="demo">Demo transcript</option>
              </select>
            </div>
            <div>
              <label className={label}>Language</label>
              <select className={field} value={language} onChange={(e) => setLanguage(e.target.value as Language)}>
                {(Object.keys(LANGUAGE_LABELS) as Language[]).map((l) => (
                  <option key={l} value={l}>
                    {LANGUAGE_LABELS[l]}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label className={label}>Speech engine</label>
              <select className={field} value={backend} onChange={(e) => setBackend(e.target.value as AsrBackendChoice)}>
                <option value="auto">Auto (Apple → API → local)</option>
                <option value="apple">Apple on-device</option>
                <option value="api">Whisper API</option>
                <option value="local">Local Kotoba-Whisper</option>
              </select>
            </div>
          </div>
          {source !== "demo" && devices.length > 0 && (
            <div>
              <label className={label}>Microphone</label>
              <select
                className={field}
                value={micDevice ?? ""}
                onChange={(e) => setMicDevice(e.target.value === "" ? null : Number(e.target.value))}
              >
                <option value="">System default</option>
                {devices.map((d) => (
                  <option key={d.index} value={d.index}>
                    {d.name}
                  </option>
                ))}
              </select>
            </div>
          )}

          {showAppleInstall && (
            <div className="rounded-md border border-amber-500/40 bg-amber-500/10 p-3 text-sm text-amber-200">
              <p>
                Apple's on-device {LANGUAGE_LABELS[language]} speech model is not installed. It is a one-time system
                download and runs offline afterwards. Without it, Echo uses the Whisper API.
              </p>
              <div className="mt-2 flex items-center gap-3">
                <Button variant="primary" disabled={installing} onClick={installApple}>
                  {installing ? `Downloading… ${pct}%` : "Download model"}
                </Button>
                {installing && (
                  <div className="h-1.5 flex-1 overflow-hidden rounded bg-black/30">
                    <div className="h-full bg-accent transition-all" style={{ width: `${pct}%` }} />
                  </div>
                )}
              </div>
            </div>
          )}

          {probe && (
            <p className="text-xs text-zinc-500">
              LLM: {probe.llm.online ? probe.llm.label : "unavailable (insights and chat disabled)"} · Whisper API:{" "}
              {probe.whisper_api.configured ? (probe.whisper_api.usable ? "ready" : "unreachable") : "not configured"}
            </p>
          )}
          {error && <p className="rounded-md bg-red-500/10 p-3 text-sm whitespace-pre-wrap text-red-300 selectable">{error}</p>}
        </div>

        <div className="mt-6 flex justify-end">
          <Button variant="primary" disabled={busy || installing} onClick={submit}>
            {busy ? "Preparing speech recognition…" : "Start meeting"}
          </Button>
        </div>
      </div>
    </div>
  );
}
