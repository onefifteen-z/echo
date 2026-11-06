import { contextBridge, ipcRenderer, type IpcRendererEvent } from "electron";
import type { EchoAPI, EchoEvent, SidecarState } from "../shared/types";

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
  getSettings: "echo:settings-get",
  saveSettings: "echo:settings-save",
  chooseDir: "echo:choose-dir",
  chooseFile: "echo:choose-file",
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
  getSidecarState: () => ipcRenderer.invoke(IPC.getSidecarState),
  restartSidecar: () => ipcRenderer.invoke(IPC.restartSidecar),
  openPath: (p) => ipcRenderer.invoke(IPC.openPath, p),
  trashMeeting: (id) => ipcRenderer.invoke(IPC.trashMeeting, id),
  getSettings: () => ipcRenderer.invoke(IPC.getSettings),
  saveSettings: (patch, opts) => ipcRenderer.invoke(IPC.saveSettings, patch, opts),
  chooseDir: (current) => ipcRenderer.invoke(IPC.chooseDir, current),
  chooseFile: (current) => ipcRenderer.invoke(IPC.chooseFile, current),
};

contextBridge.exposeInMainWorld("echo", api);
