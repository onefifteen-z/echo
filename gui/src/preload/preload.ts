import { contextBridge, ipcRenderer, type IpcRendererEvent } from "electron";
import type { EchoAPI, EchoEvent, MeetingWatchStatus, SidecarState, TrayCommand } from "../shared/types";

// Sandboxed preloads cannot require local modules, so the channel names are inlined here.
// Keep in sync with `IPC` in src/shared/types.ts.
const IPC = {
  invoke: "echo:invoke",
  event: "echo:event",
  sidecarState: "echo:sidecar-state",
  getSidecarState: "echo:get-sidecar-state",
  restartSidecar: "echo:restart-sidecar",
  openPath: "echo:open-path",
  trashMeeting: "echo:trash-meeting",
  saveNoteImage: "echo:save-note-image",
  getSettings: "echo:settings-get",
  saveSettings: "echo:settings-save",
  chooseDir: "echo:choose-dir",
  chooseFile: "echo:choose-file",
  trayCommand: "echo:tray-command",
  sessionAttached: "echo:session-attached",
  setPrefs: "echo:settings-set-prefs",
  getMeetingWatch: "echo:meeting-watch-get",
  meetingWatchStatus: "echo:meeting-watch-status",
  takePendingCommand: "echo:take-pending-command",
} as const;

const api: EchoAPI = {
  invoke: (cmd, args) => ipcRenderer.invoke(IPC.invoke, cmd, args),
  startSession: (args) => ipcRenderer.invoke(IPC.invoke, "session.start", args),
  stopSession: () => ipcRenderer.invoke(IPC.invoke, "session.stop", {}),
  pause: () => ipcRenderer.invoke(IPC.invoke, "session.pause", {}),
  resume: () => ipcRenderer.invoke(IPC.invoke, "session.resume", {}),
  reset: (args = {}) => ipcRenderer.invoke(IPC.invoke, "session.reset", args),
  setLanguage: (language) => ipcRenderer.invoke(IPC.invoke, "session.set_language", { language }),
  ask: (question) => ipcRenderer.invoke(IPC.invoke, "chat.ask", { question }),
  onEvent: (cb) => {
    const listener = (_e: IpcRendererEvent, ev: EchoEvent) => cb(ev);
    ipcRenderer.on(IPC.event, listener);
    return () => ipcRenderer.removeListener(IPC.event, listener);
  },
  onSidecarState: (cb) => {
    const listener = (_e: IpcRendererEvent, st: SidecarState) => cb(st);
    ipcRenderer.on(IPC.sidecarState, listener);
    return () => ipcRenderer.removeListener(IPC.sidecarState, listener);
  },
  onTrayCommand: (cb) => {
    const listener = (_e: IpcRendererEvent, cmd: TrayCommand) => cb(cmd);
    ipcRenderer.on(IPC.trayCommand, listener);
    return () => ipcRenderer.removeListener(IPC.trayCommand, listener);
  },
  onSessionAttached: (cb) => {
    const listener = () => cb();
    ipcRenderer.on(IPC.sessionAttached, listener);
    return () => ipcRenderer.removeListener(IPC.sessionAttached, listener);
  },
  setPrefs: (patch) => ipcRenderer.invoke(IPC.setPrefs, patch),
  takePendingCommand: () => ipcRenderer.invoke(IPC.takePendingCommand),
  getMeetingWatch: () => ipcRenderer.invoke(IPC.getMeetingWatch),
  onMeetingWatchStatus: (cb) => {
    const listener = (_e: IpcRendererEvent, s: MeetingWatchStatus) => cb(s);
    ipcRenderer.on(IPC.meetingWatchStatus, listener);
    return () => ipcRenderer.removeListener(IPC.meetingWatchStatus, listener);
  },
  getSidecarState: () => ipcRenderer.invoke(IPC.getSidecarState),
  restartSidecar: () => ipcRenderer.invoke(IPC.restartSidecar),
  openPath: (p) => ipcRenderer.invoke(IPC.openPath, p),
  trashMeeting: (id) => ipcRenderer.invoke(IPC.trashMeeting, id),
  saveNoteImage: (meetingId, bytes, mime) => ipcRenderer.invoke(IPC.saveNoteImage, meetingId, bytes, mime),
  getSettings: () => ipcRenderer.invoke(IPC.getSettings),
  saveSettings: (patch, opts) => ipcRenderer.invoke(IPC.saveSettings, patch, opts),
  chooseDir: (current) => ipcRenderer.invoke(IPC.chooseDir, current),
  chooseFile: (current, kind) => ipcRenderer.invoke(IPC.chooseFile, current, kind),
};

contextBridge.exposeInMainWorld("echo", api);
