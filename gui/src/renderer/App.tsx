import { useCallback, useEffect, useReducer, useRef, useState } from "react";
import type { Language, SessionStartArgs, SidecarState, TrayCommand } from "@shared/types";
import { ChatNotebookPanel } from "./components/Chat";
import { Insights } from "./components/Insights";
import { MeetingsView } from "./components/MeetingsView";
import { Meter } from "./components/Meter";
import { SettingsView } from "./components/SettingsView";
import { SetupDialog } from "./components/SetupDialog";
import { SidecarBanner } from "./components/SidecarBanner";
import { TopBar } from "./components/TopBar";
import { Transcript } from "./components/Transcript";
import { appleLocaleState, Button, LANGUAGE_LABELS } from "./components/ui";
import { initialState, reduce } from "./lib/session-store";

interface Toast {
  id: number;
  text: string;
  tone: "info" | "error";
  path?: string;
  action?: { label: string; run: () => void };
}

type View = "live" | "meetings" | "settings";

function folderName(p: string): string {
  return p.split(/[\\/]/).filter(Boolean).pop() ?? p;
}

function errorText(e: unknown): string {
  return (e instanceof Error ? e.message : String(e)).replace(/^Error invoking remote method '[^']+': (Error: )?/, "");
}

export default function App() {
  const [state, dispatch] = useReducer(reduce, initialState);
  const [sidecar, setSidecar] = useState<SidecarState>({ phase: "starting" });
  const [starting, setStarting] = useState(false);
  const [showSetup, setShowSetup] = useState(true);
  const [toasts, setToasts] = useState<Toast[]>([]);
  const [view, setView] = useState<View>("live");
  const [meetingsRefresh, setMeetingsRefresh] = useState(0);
  const [selectMeeting, setSelectMeeting] = useState<string | null>(null);
  const [settingsRev, setSettingsRev] = useState(0);
  const [setupTitle, setSetupTitle] = useState("");
  const [setupRev, setSetupRev] = useState(0);
  const chatSeq = useRef(0);
  const toastSeq = useRef(0);

  const dismiss = useCallback((id: number) => setToasts((t) => t.filter((x) => x.id !== id)), []);
  const toast = useCallback(
    (text: string, tone: Toast["tone"] = "info", path?: string, action?: Toast["action"]) => {
      const id = ++toastSeq.current;
      setToasts((t) => [...t, { id, text, tone, path, action }]);
      setTimeout(() => dismiss(id), action ? 20000 : tone === "error" ? 9000 : 5000);
      return id;
    },
    [dismiss],
  );

  const offerAppleInstall = useCallback(
    async (language: Language) => {
      const status = await window.echo.invoke("apple.status", { languages: [language] }).catch(() => null);
      if (!status || appleLocaleState(status, language) !== "supported") return;
      const name = LANGUAGE_LABELS[language];
      const id = toast(
        `Apple's on-device ${name} model is not installed, so the Whisper API is used. Download it once to transcribe ${name} on-device.`,
        "info",
        undefined,
        {
          label: "Download model",
          run: () => {
            dismiss(id);
            toast(`Downloading Apple ${name} speech model…`);
            window.echo
              .invoke("apple.install_assets", { language })
              .then(() => toast(`Apple ${name} model installed; switched to on-device recognition.`))
              .catch((e) => toast(errorText(e), "error"));
          },
        },
      );
    },
    [toast, dismiss],
  );

  useEffect(() => {
    const offEvent = window.echo.onEvent((event) => {
      dispatch({ kind: "event", event });
      if (event.type === "summary.written" && event.announce && event.ok) toast("Meeting summary generated.");
      if (event.type === "summary.written" && !event.id) setMeetingsRefresh((n) => n + 1);
    });
    const offState = window.echo.onSidecarState(setSidecar);
    void window.echo.getSidecarState().then(setSidecar);
    return () => {
      offEvent();
      offState();
    };
  }, [toast]);

  // Re-attach to a running session after a renderer reload or sidecar restart.
  useEffect(() => {
    if (sidecar.phase !== "ready") return;
    window.echo
      .invoke("session.snapshot", {})
      .then((snap) => {
        if (snap.status && snap.status.state !== "stopped") {
          dispatch({ kind: "snapshot", snapshot: snap });
          setShowSetup(false);
        }
      })
      .catch(() => undefined);
  }, [sidecar.phase]);

  async function start(args: SessionStartArgs) {
    setStarting(true);
    try {
      dispatch({ kind: "clear" });
      const res = await window.echo.startSession(args);
      dispatch({ kind: "event", event: res.status });
      dispatch({ kind: "meeting", id: folderName(res.output_dir), notebook: "" });
      setShowSetup(false);
      setView("live");
    } finally {
      setStarting(false);
    }
  }

  const stopLive = useCallback(async () => {
    const r = await window.echo.stopSession();
    dispatch({ kind: "stopped" });
    setMeetingsRefresh((n) => n + 1);
    return r;
  }, []);

  const stopMeeting = useCallback(async () => {
    try {
      const r = await stopLive();
      if (r.output_dir) {
        const id = folderName(r.output_dir);
        toast("Meeting saved.", "info", r.output_dir, {
          label: "View in Meetings",
          run: () => {
            setSelectMeeting(id);
            setView("meetings");
          },
        });
      }
    } catch (e) {
      toast(errorText(e), "error");
    }
  }, [stopLive, toast]);

  useEffect(() => {
    const handle = (cmd: TrayCommand) => {
      if (cmd.kind === "stop") void stopMeeting();
      if (cmd.kind === "new") {
        setSetupTitle(cmd.title ?? "");
        setSetupRev((n) => n + 1);
        setShowSetup(true);
        setView("live");
      }
    };
    const off = window.echo.onTrayCommand(handle);
    void window.echo.takePendingCommand().then((cmd) => cmd && handle(cmd));
    return off;
  }, [stopMeeting]);

  // A session started from the meeting prompt (main process): show it live.
  useEffect(
    () =>
      window.echo.onSessionAttached(() => {
        window.echo
          .invoke("session.snapshot", {})
          .then((snap) => {
            if (!snap.status) return;
            dispatch({ kind: "clear" });
            dispatch({ kind: "snapshot", snapshot: snap });
            setShowSetup(false);
            setView("live");
          })
          .catch(() => undefined);
      }),
    [],
  );

  const liveId = state.meetingId;
  const loadLiveContext = useCallback(
    async () => (liveId ? (await window.echo.invoke("meetings.get", { id: liveId })).context : ""),
    [liveId],
  );

  async function run(action: () => Promise<unknown>, okText?: string) {
    try {
      await action();
      if (okText) toast(okText);
    } catch (e) {
      toast(errorText(e), "error");
    }
  }

  const status = state.status;
  const paused = status?.state === "paused";

  const NAV_ICONS: Record<View, string> = { live: "●", meetings: "≡", settings: "⚙" };
  const navItem = (id: View, label: string, badge?: boolean) => (
    <button
      onClick={() => setView(id)}
      className={`no-drag relative flex w-14 flex-col items-center rounded-md py-2 text-[11px] font-medium ${
        view === id ? "bg-accent-soft text-accent" : "text-zinc-500 hover:bg-panel-2 hover:text-zinc-300"
      }`}
    >
      <span className="text-base leading-none">{NAV_ICONS[id]}</span>
      <span className="mt-1">{label}</span>
      {badge && <span className="absolute top-1.5 right-2 h-1.5 w-1.5 rounded-full bg-emerald-400" />}
    </button>
  );
  const liveRunning = !!status && !state.stopped;

  return (
    <div className="flex h-full">
      <nav className="drag flex w-[72px] shrink-0 flex-col items-center gap-1 border-r border-line pt-12">
        {navItem("live", "Live", liveRunning && view !== "live")}
        {navItem("meetings", "Meetings")}
        <div className="flex-1" />
        <div className="mb-3">{navItem("settings", "Settings")}</div>
      </nav>
    <div className="flex min-w-0 flex-1 flex-col">
      {(view !== "live" || showSetup || !status) && <div className="drag h-10 shrink-0" />}
      <SidecarBanner state={sidecar} />
      {view === "settings" ? (
        <SettingsView
          engineReady={sidecar.phase === "ready"}
          liveRunning={liveRunning}
          appleInstall={state.appleInstall}
          endLive={async () => {
            await stopLive();
          }}
          onSaved={(restarted) => {
            if (restarted) {
              setShowSetup(true);
              setSettingsRev((n) => n + 1);
            }
          }}
          toast={toast}
        />
      ) : view === "meetings" ? (
        sidecar.phase === "ready" ? (
          <MeetingsView
            refreshKey={meetingsRefresh + settingsRev}
            selectId={selectMeeting}
            onOpenLive={() => setView("live")}
            toast={toast}
          />
        ) : (
          <div className="pt-24 text-center text-sm text-zinc-500">Waiting for the Echo engine…</div>
        )
      ) : showSetup || !status ? (
        <div className="min-h-0 flex-1">
          {sidecar.phase === "ready" ? (
            <SetupDialog key={`${settingsRev}-${setupRev}`} initialTitle={setupTitle} busy={starting} appleInstall={state.appleInstall} onStart={start} />
          ) : (
            <div className="pt-24 text-center text-sm text-zinc-500">Waiting for the Echo engine…</div>
          )}
        </div>
      ) : (
        <>
          <TopBar
            status={status}
            ended={state.ended}
            stopped={state.stopped}
            onPauseToggle={() => void run(() => (paused ? window.echo.resume() : window.echo.pause()))}
            onLanguage={(language: Language) =>
              void run(async () => {
                const r = await window.echo.setLanguage(language);
                toast(r.message);
                if (r.status.asr_source !== "apple-on-device") void offerAppleInstall(language);
              })
            }
            onSave={() =>
              void run(async () => {
                const r = await window.echo.invoke("session.save", {});
                toast("Saved notes and insights. Summary generating…", "info", r.output_dir);
              })
            }
            onReset={() => {
              if (window.confirm("Clear the transcript and insights and start over?")) {
                void run(() => window.echo.reset({ resume: true }), "Session reset");
              }
            }}
            onStop={() => void stopMeeting()}
            onNew={() => setShowSetup(true)}
            onRename={
              state.meetingId
                ? (title) =>
                    run(() => window.echo.invoke("meetings.update", { id: state.meetingId!, title }), "Meeting renamed")
                : undefined
            }
            loadContext={loadLiveContext}
            onContext={
              state.meetingId
                ? (context) =>
                    run(() => window.echo.invoke("meetings.update", { id: state.meetingId!, context }), "Context updated")
                : undefined
            }
          />
          <main className="grid min-h-0 flex-1 grid-cols-[minmax(0,3fr)_minmax(0,2fr)] gap-3 p-3">
            <div className="flex min-h-0 flex-col gap-3">
              {status.mode === "live" && (
                <Meter history={state.meterHistory} peak={state.meter.peak} note={state.meter.note} paused={paused} />
              )}
              <Transcript lines={state.lines} notices={state.notices} />
            </div>
            <div className="flex min-h-0 flex-col gap-3">
              <Insights insights={state.insights} pending={state.pendingInsight} llmOffline={status.llm_offline} />
              <ChatNotebookPanel
                notebook={{
                  docKey: state.meetingId ?? "live",
                  meetingId: state.meetingId,
                  initial: state.notebook,
                  onSave: (text) => window.echo.invoke("session.notebook.set", { text }),
                  onError: (message) => toast(message, "error"),
                }}
                items={state.chat}
                disabled={status.llm_offline || state.stopped}
                onAsk={(question) => {
                  const id = ++chatSeq.current;
                  dispatch({ kind: "chat.ask", id, question });
                  window.echo
                    .ask(question)
                    .then((r) => dispatch({ kind: "chat.answer", id, answer: r.answer }))
                    .catch((e) => dispatch({ kind: "chat.answer", id, answer: errorText(e), error: true }));
                }}
              />
            </div>
          </main>
        </>
      )}
    </div>
      <div className="pointer-events-none fixed right-4 bottom-4 z-50 flex w-96 flex-col gap-2">
        {toasts.map((t) => (
          <div
            key={t.id}
            className={`pointer-events-auto rounded-md border px-3 py-2 text-sm shadow-lg ${
              t.tone === "error" ? "border-red-500/40 bg-red-950 text-red-200" : "border-line bg-panel-2 text-zinc-200"
            }`}
          >
            <div className="selectable whitespace-pre-wrap">{t.text}</div>
            {t.path && (
              <Button className="mt-2" onClick={() => void window.echo.openPath(t.path!)}>
                Open folder
              </Button>
            )}
            {t.action && (
              <Button variant="primary" className="mt-2" onClick={t.action.run}>
                {t.action.label}
              </Button>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
