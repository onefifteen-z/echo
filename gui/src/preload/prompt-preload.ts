import { contextBridge, ipcRenderer, type IpcRendererEvent } from "electron";
import type { EchoPromptAPI, MeetingPromptData } from "../shared/types";

// Sandboxed preloads cannot require local modules; keep in sync with `PROMPT_IPC` in src/shared/types.ts.
const PROMPT_IPC = {
  show: "echo-prompt:show",
  act: "echo-prompt:act",
} as const;

const api: EchoPromptAPI = {
  onShow: (cb) => {
    const listener = (_e: IpcRendererEvent, data: MeetingPromptData) => cb(data);
    ipcRenderer.on(PROMPT_IPC.show, listener);
    return () => ipcRenderer.removeListener(PROMPT_IPC.show, listener);
  },
  act: (action) => ipcRenderer.send(PROMPT_IPC.act, action),
};

contextBridge.exposeInMainWorld("echoPrompt", api);
