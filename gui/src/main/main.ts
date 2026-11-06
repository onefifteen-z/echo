import { app, BrowserWindow, dialog, ipcMain, net, protocol, safeStorage, shell, systemPreferences } from "electron";
import * as fs from "fs";
import * as os from "os";
import * as path from "path";
import { pathToFileURL } from "url";
import {
  IPC,
  type AppPrefs,
  type CommandName,
  type MeetingPromptData,
  type SessionStartArgs,
  type SettingsPatch,
  type SidecarState,
  type TrayCommand,
} from "../shared/types";
import { ensureEchoVenv, getEchoHome } from "./services/echo-venv";
import { SettingsStore } from "./services/settings-store";
import { SidecarService, type PythonLaunch } from "./services/sidecar-service";
import { TrayController } from "./services/tray-service";
import { appForBundle, KNOWN_APPS, MeetingDetector, WATCHED_PREFIXES } from "./services/meeting-detector";
import { MeetingPrompt, promptPreloadPath } from "./services/meeting-prompt";
import { MeetingWatchService, type WatchReport } from "./services/meeting-watch-service";
import {
  ASSET_DIR,
  ASSET_SCHEME,
  MAX_NOTE_IMAGE_BYTES,
  meetingDir,
  noteImageName,
  parseAssetUrl,
  resolveAsset,
} from "./services/note-assets";

protocol.registerSchemesAsPrivileged([{ scheme: ASSET_SCHEME, privileges: { standard: true, secure: true } }]);

let mainWindow: BrowserWindow | null = null;
let sidecar: SidecarService | null = null;
let tray: TrayController | null = null;
let meetingWatch: MeetingWatchService | null = null;
let meetingPrompt: MeetingPrompt | null = null;
/** Command for the main window that waits until its renderer asks for it (window may be loading). */
let pendingCommand: TrayCommand | null = null;
/** True while the sidecar has a session (running, paused or finished but not stopped). */
let sessionBusy = false;

const LOG_FILE = path.join(app.getPath("userData"), "echo-sidecar.log");
const DEFAULT_OUTPUT_ROOT = path.join(app.getPath("documents"), "Echo");
let settings: SettingsStore | null = null;

function outputRoot(): string {
  return path.resolve(settings?.workspaceDir() ?? DEFAULT_OUTPUT_ROOT);
}

const ALLOWED_COMMANDS: ReadonlySet<CommandName> = new Set<CommandName>([
  "ping",
  "probe",
  "devices.list",
  "apple.status",
  "apple.install_assets",
  "session.start",
  "session.stop",
  "session.pause",
  "session.resume",
  "session.reset",
  "session.set_language",
  "session.save",
  "session.snapshot",
  "chat.ask",
  "session.notebook.set",
  "meetings.list",
  "meetings.get",
  "meetings.update",
  "meetings.regenerate_summary",
  "config.effective",
]);

function logLine(line: string): void {
  try {
    fs.appendFileSync(LOG_FILE, `[${new Date().toISOString()}] ${line}\n`, "utf-8");
  } catch {
    console.error(line);
  }
}

/** Dev: the repo's `.venv` (gui/ lives inside the repo). Packaged: `~/.echo/venv` from bundled source. */
async function resolvePython(): Promise<PythonLaunch> {
  const override = process.env.ECHO_PYTHON;
  if (!app.isPackaged) {
    const repoRoot = path.resolve(app.getAppPath(), "..");
    const venvPython = path.join(repoRoot, ".venv", "bin", "python");
    const pythonBin = override || (fs.existsSync(venvPython) ? venvPython : "python3");
    return { pythonBin, cwd: repoRoot };
  }
  fs.mkdirSync(getEchoHome(), { recursive: true });
  if (override) return { pythonBin: override, cwd: getEchoHome() };
  const res = await ensureEchoVenv(path.join(process.resourcesPath, "python"), app.getVersion(), logLine);
  if (!res.ok) throw new Error(res.message);
  // `.env` for API keys is read from the working directory: ~/.echo/.env
  return { pythonBin: res.pythonBin, cwd: getEchoHome() };
}

function broadcast(channel: string, payload: unknown): void {
  if (mainWindow && !mainWindow.isDestroyed()) {
    mainWindow.webContents.send(channel, payload);
  }
}

function rendererLoad(win: BrowserWindow, page: "index" | "prompt"): Promise<void> {
  if (process.env.NODE_ENV === "development") {
    return win.loadURL(`http://localhost:5174/${page === "index" ? "" : "prompt.html"}`);
  }
  return win.loadFile(path.join(__dirname, "..", "..", "renderer", `${page}.html`));
}

function createWindow(opts: { inactive?: boolean } = {}): void {
  mainWindow = new BrowserWindow({
    show: !opts.inactive,
    width: 1440,
    height: 900,
    minWidth: 960,
    minHeight: 600,
    titleBarStyle: "hiddenInset",
    backgroundColor: "#0f1115",
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      sandbox: true,
      preload: path.join(__dirname, "..", "..", "preload", "preload", "preload.js"),
    },
  });

  if (opts.inactive) {
    const win = mainWindow;
    win.once("ready-to-show", () => win.showInactive());
  }
  void rendererLoad(mainWindow, "index");
  if (process.env.NODE_ENV === "development") mainWindow.webContents.openDevTools({ mode: "detach" });
  // Insight links open in the default browser, never in an app window.
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    if (/^https?:\/\//.test(url)) void shell.openExternal(url);
    return { action: "deny" };
  });
  // Only the app's own pages; a file dropped outside the notes editor must not replace the UI.
  const appPage = pathToFileURL(path.join(__dirname, "..", "..", "renderer", "index.html")).toString();
  mainWindow.webContents.on("will-navigate", (e, url) => {
    if (!url.startsWith("http://localhost:5174") && url.split("#")[0] !== appPage) e.preventDefault();
  });
  mainWindow.on("closed", () => {
    mainWindow = null;
  });
}

/** Show the window (focused unless `inactive`), recreating it if it was closed. */
function showWindow(opts: { inactive?: boolean } = {}): void {
  const existed = !!mainWindow && !mainWindow.isDestroyed();
  if (!existed) {
    createWindow(opts);
    if (opts.inactive) return;
  }
  const win = mainWindow!;
  if (win.isMinimized()) win.restore();
  if (opts.inactive) {
    win.showInactive();
    return;
  }
  win.show();
  win.focus();
  app.focus({ steal: true });
}

/** Deliver a command to the window, or queue it until a (re)loading renderer asks for it. */
function sendToWindow(cmd: TrayCommand): void {
  if (mainWindow && !mainWindow.isDestroyed() && !mainWindow.webContents.isLoading()) {
    broadcast(IPC.trayCommand, cmd);
  } else {
    pendingCommand = cmd;
  }
}

const DEFAULT_START: NonNullable<AppPrefs["lastStart"]> = { source: "both", backend: "auto", language: "japanese", mic_device: null };

async function startFromPrompt(service: SidecarService, data: MeetingPromptData): Promise<void> {
  const last = settings!.prefs().lastStart ?? DEFAULT_START;
  try {
    await service.invoke("session.start", { ...last, title: data.title, context: "" });
    logLine(`meeting prompt: started transcription for ${data.appName}`);
    showWindow({ inactive: true });
    broadcast(IPC.sessionAttached, null);
  } catch (err) {
    logLine(`meeting prompt: start failed: ${String(err)}`);
    showWindow();
    sendToWindow({ kind: "new", title: data.title });
    void dialog.showMessageBox(mainWindow!, {
      type: "error",
      message: "Echo could not start transcribing",
      detail: err instanceof Error ? err.message : String(err),
    });
  }
}

function setupMeetingDetection(service: SidecarService): void {
  if (process.platform !== "darwin") return;
  const sourcePath = app.isPackaged
    ? path.join(process.resourcesPath, "native", "MeetingWatch.swift")
    : path.join(app.getAppPath(), "native", "MeetingWatch.swift");
  const fakeZoom = !app.isPackaged ? process.env.ECHO_DEV_FAKE_ZOOM_BUNDLE : undefined;
  if (fakeZoom) {
    // Development only: treat another app's mic capture as a Zoom meeting to exercise the prompt.
    KNOWN_APPS.find((a) => a.id === "zoom")!.prefixes.push(fakeZoom);
    WATCHED_PREFIXES.push(fakeZoom);
  }
  const watch = new MeetingWatchService(sourcePath, path.join(app.getPath("userData"), "bin"), WATCHED_PREFIXES, logLine);
  meetingWatch = watch;
  const prompt = new MeetingPrompt((win) => rendererLoad(win, "prompt"), promptPreloadPath(__dirname), {
    start: (data) => void startFromPrompt(service, data),
    setup: (data) => {
      showWindow();
      sendToWindow({ kind: "new", title: data.title });
    },
    mute: (data) => {
      const prefs = settings!.prefs();
      settings!.setPrefs({ mutedApps: [...prefs.mutedApps, data.app] });
    },
    disable: () => applyPrefs({ meetingDetection: false }),
  });
  meetingPrompt = prompt;
  const detector = new MeetingDetector();
  let last: WatchReport | null = null;
  const ctx = () => {
    const p = settings!.prefs();
    return { enabled: p.meetingDetection, mutedApps: p.mutedApps, busy: sessionBusy };
  };
  const handle = (prompts: ReturnType<MeetingDetector["update"]>) => {
    const p = prompts[0];
    if (p) {
      logLine(`meeting detected: ${p.appName} (${p.title})`);
      void prompt.show(p);
      return;
    }
    const shown = prompt.currentApp();
    if (shown && !detector.active().includes(shown)) prompt.hide();
  };
  let lastApps = "";
  watch.on("report", (r) => {
    last = r;
    const apps = [...new Set(r.inputs.map((p) => appForBundle(p.bundle)?.id).filter(Boolean))].join(",");
    if (apps !== lastApps) logLine(`meeting watch: mic in use by [${apps}]`);
    lastApps = apps;
    handle(detector.update(r, ctx()));
  });
  watch.on("status", (s) => broadcast(IPC.meetingWatchStatus, s));
  setInterval(() => handle(detector.tick(last, ctx())), 2000);
  service.on("event", (ev) => {
    if (ev.type !== "status") return;
    sessionBusy = ev.state !== "stopped" && ev.state !== "idle";
    if (sessionBusy) prompt.hide();
  });
  service.on("state", (s) => {
    if (s.phase !== "ready") sessionBusy = false;
  });
  if (settings!.prefs().meetingDetection) void watch.start();
}

function applyPrefs(patch: Partial<AppPrefs>) {
  const snap = settings!.setPrefs(patch);
  if (meetingWatch) {
    if (snap.prefs.meetingDetection) void meetingWatch.start();
    else {
      meetingWatch.stop();
      meetingPrompt?.hide();
    }
  }
  return snap;
}

function createTray(service: SidecarService): void {
  if (process.platform !== "darwin") return;
  tray = new TrayController(path.join(app.getAppPath(), "assets/icons/tray"), service, {
    open: () => void showWindow(),
    newMeeting: () => {
      // A freshly created window already opens on the setup screen.
      showWindow();
      sendToWindow({ kind: "new" });
    },
    stop: () => {
      if (mainWindow && !mainWindow.isDestroyed() && !mainWindow.webContents.isLoading()) {
        broadcast(IPC.trayCommand, { kind: "stop" });
      } else {
        service.invoke("session.stop", {}).catch((err) => logLine(`tray stop failed: ${String(err)}`));
      }
    },
    quit: () => app.quit(),
  });
}

function registerIpc(service: SidecarService): void {
  ipcMain.handle(IPC.invoke, async (_e, cmd: CommandName, args: unknown) => {
    if (!ALLOWED_COMMANDS.has(cmd)) throw new Error(`Command not allowed: ${String(cmd)}`);
    const result = await service.invoke(cmd, (args ?? {}) as never);
    if (cmd === "session.start") {
      const a = args as SessionStartArgs;
      if (a.source !== "demo") {
        settings!.setPrefs({ lastStart: { source: a.source, backend: a.backend, language: a.language, mic_device: a.mic_device ?? null } });
      }
    }
    return result;
  });
  ipcMain.handle(IPC.getSidecarState, () => service.getState());
  ipcMain.handle(IPC.restartSidecar, () => service.restart());
  ipcMain.handle(IPC.openPath, async (_e, target: string) => {
    const resolved = path.resolve(String(target));
    const root = outputRoot();
    if (!resolved.startsWith(root + path.sep) && resolved !== root) {
      throw new Error("Only meeting output folders can be opened.");
    }
    const err = await shell.openPath(resolved);
    if (err) throw new Error(err);
  });
  ipcMain.handle(IPC.saveNoteImage, async (_e, id: string, bytes: Uint8Array, mime: string) => {
    const dir = meetingDir(outputRoot(), id);
    if (!fs.existsSync(dir)) throw new Error("Images can be added once the meeting is saved.");
    if (!(bytes instanceof Uint8Array) || bytes.byteLength === 0) throw new Error("The image is empty.");
    if (bytes.byteLength > MAX_NOTE_IMAGE_BYTES) throw new Error("Images in notes must be 20 MB or smaller.");
    const name = noteImageName(String(mime));
    await fs.promises.mkdir(path.join(dir, ASSET_DIR), { recursive: true });
    await fs.promises.writeFile(path.join(dir, ASSET_DIR, name), bytes);
    return { src: `${ASSET_DIR}/${name}` };
  });
  ipcMain.handle(IPC.trashMeeting, async (_e, id: string) => {
    const name = String(id ?? "");
    const target = meetingDir(outputRoot(), name);
    if (!fs.existsSync(target) || !fs.statSync(target).isDirectory()) throw new Error("Meeting not found.");
    const snap = await service.invoke("session.snapshot", {});
    if (snap.status && snap.meeting_id === name) {
      throw new Error("This meeting is still in progress. End the meeting before deleting it.");
    }
    await shell.trashItem(target);
  });
  ipcMain.handle(IPC.getSettings, () => settings!.snapshot());
  ipcMain.handle(IPC.setPrefs, (_e, patch: Partial<AppPrefs>) => applyPrefs(patch ?? {}));
  ipcMain.handle(IPC.getMeetingWatch, () => meetingWatch?.getStatus() ?? { state: "unavailable", message: "Meeting detection is available on macOS only." });
  ipcMain.handle(IPC.takePendingCommand, () => {
    const cmd = pendingCommand;
    pendingCommand = null;
    return cmd;
  });
  ipcMain.handle(IPC.saveSettings, async (_e, patch: SettingsPatch, opts: { restart?: boolean }) => {
    const snap = settings!.apply(patch ?? {});
    fs.mkdirSync(snap.workspaceDir, { recursive: true });
    logLine(`settings saved (restart=${Boolean(opts?.restart)})`);
    if (opts?.restart) await service.restart();
    return snap;
  });
  ipcMain.handle(IPC.chooseDir, async (_e, current?: string) => {
    const r = await dialog.showOpenDialog(mainWindow!, {
      title: "Choose meetings folder",
      defaultPath: current || outputRoot(),
      properties: ["openDirectory", "createDirectory"],
    });
    return r.canceled || !r.filePaths[0] ? null : r.filePaths[0];
  });
  ipcMain.handle(IPC.chooseFile, async (_e, current?: string) => {
    const r = await dialog.showOpenDialog(mainWindow!, {
      title: "Choose glossary file",
      defaultPath: current || os.homedir(),
      properties: ["openFile"],
      filters: [{ name: "YAML", extensions: ["yml", "yaml"] }],
    });
    return r.canceled || !r.filePaths[0] ? null : r.filePaths[0];
  });
}

app.whenReady().then(async () => {
  if (process.platform === "darwin" && !app.isPackaged) {
    // Packaged builds get the icon from the bundle; in dev the Dock would show Electron's.
    const dockIcon = path.join(app.getAppPath(), "assets/icons/mac/dock.png");
    if (fs.existsSync(dockIcon)) app.dock?.setIcon(dockIcon);
  }
  settings = new SettingsStore(
    path.join(app.getPath("userData"), "settings.json"),
    {
      available: () => safeStorage.isEncryptionAvailable(),
      encrypt: (plain) => safeStorage.encryptString(plain),
      decrypt: (data) => safeStorage.decryptString(data),
    },
    DEFAULT_OUTPUT_ROOT,
  );
  fs.mkdirSync(outputRoot(), { recursive: true });
  sidecar = new SidecarService(resolvePython, outputRoot, () => settings!.toSpawnEnv());
  sidecar.on("event", (ev) => broadcast(IPC.event, ev));
  sidecar.on("state", (state: SidecarState) => {
    logLine(`sidecar state: ${state.phase}${"message" in state ? ` — ${state.message}` : ""}`);
    broadcast(IPC.sidecarState, state);
  });
  sidecar.on("log", (line) => logLine(line));
  protocol.handle(ASSET_SCHEME, (req) => {
    const parsed = parseAssetUrl(req.url);
    const file = parsed && resolveAsset(outputRoot(), parsed.id, parsed.rel);
    if (!file || !fs.existsSync(file)) return new Response("Not found", { status: 404 });
    return net.fetch(pathToFileURL(file).toString());
  });
  registerIpc(sidecar);
  createWindow();
  createTray(sidecar);
  setupMeetingDetection(sidecar);

  if (process.platform === "darwin") {
    // The sidecar inherits this app's TCC identity, so ask for the mic here before capture starts.
    systemPreferences.askForMediaAccess("microphone").catch(() => undefined);
  }
  await sidecar.start();

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

let quitting = false;
app.on("before-quit", (e) => {
  if (quitting || !sidecar) return;
  e.preventDefault();
  quitting = true;
  sidecar
    .stop()
    .catch((err) => logLine(`sidecar stop failed: ${String(err)}`))
    .finally(() => app.quit());
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});

process.on("uncaughtException", (err) => {
  logLine(`uncaughtException: ${err.stack ?? err.message}`);
  dialog.showErrorBox("Echo — unexpected error", `${err.message}\n\nLog: ${LOG_FILE}\nHome: ${os.homedir()}`);
});
