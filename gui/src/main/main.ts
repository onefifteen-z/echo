import { app, BrowserWindow, dialog, ipcMain, safeStorage, shell, systemPreferences } from "electron";
import * as fs from "fs";
import * as os from "os";
import * as path from "path";
import { IPC, type CommandName, type SettingsPatch, type SidecarState } from "../shared/types";
import { ensureEchoVenv, getEchoHome } from "./services/echo-venv";
import { SettingsStore } from "./services/settings-store";
import { SidecarService, type PythonLaunch } from "./services/sidecar-service";

let mainWindow: BrowserWindow | null = null;
let sidecar: SidecarService | null = null;

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

function createWindow(): void {
  mainWindow = new BrowserWindow({
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

  if (process.env.NODE_ENV === "development") {
    void mainWindow.loadURL("http://localhost:5174");
    mainWindow.webContents.openDevTools({ mode: "detach" });
  } else {
    void mainWindow.loadFile(path.join(__dirname, "..", "..", "renderer", "index.html"));
  }
  // Insight links open in the default browser, never in an app window.
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    if (/^https?:\/\//.test(url)) void shell.openExternal(url);
    return { action: "deny" };
  });
  mainWindow.webContents.on("will-navigate", (e, url) => {
    if (!url.startsWith("http://localhost:5174") && !url.startsWith("file://")) e.preventDefault();
  });
  mainWindow.on("closed", () => {
    mainWindow = null;
  });
}

function registerIpc(service: SidecarService): void {
  ipcMain.handle(IPC.invoke, async (_e, cmd: CommandName, args: unknown) => {
    if (!ALLOWED_COMMANDS.has(cmd)) throw new Error(`Command not allowed: ${String(cmd)}`);
    return service.invoke(cmd, (args ?? {}) as never);
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
  ipcMain.handle(IPC.trashMeeting, async (_e, id: string) => {
    const name = String(id ?? "");
    const root = outputRoot();
    const target = path.join(root, name);
    if (!name || name.startsWith(".") || /[\\/]/.test(name) || path.dirname(target) !== root) {
      throw new Error("Invalid meeting id.");
    }
    if (!fs.existsSync(target) || !fs.statSync(target).isDirectory()) throw new Error("Meeting not found.");
    const snap = await service.invoke("session.snapshot", {});
    if (snap.status && snap.meeting_id === name) {
      throw new Error("This meeting is still in progress. End the meeting before deleting it.");
    }
    await shell.trashItem(target);
  });
  ipcMain.handle(IPC.getSettings, () => settings!.snapshot());
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
  registerIpc(sidecar);
  createWindow();

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
