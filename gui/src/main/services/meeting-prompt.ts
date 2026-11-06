/** The floating "meeting detected" prompt window (Notion style, top-right, never takes focus). */

import { BrowserWindow, ipcMain, screen, type IpcMainEvent } from "electron";
import * as path from "path";
import { PROMPT_IPC, type MeetingPromptAction, type MeetingPromptData } from "../../shared/types";

const WIDTH = 460;
const HEIGHT = 112;
/** Extra window height while the in-page options menu is open; must fit the menu in prompt.tsx. */
const MENU_HEIGHT = 124;
const AUTO_DISMISS_MS = 45_000;

export interface PromptHandlers {
  start: (data: MeetingPromptData) => void;
  setup: (data: MeetingPromptData) => void;
  mute: (data: MeetingPromptData) => void;
  disable: () => void;
}

export class MeetingPrompt {
  private win: BrowserWindow | null = null;
  private data: MeetingPromptData | null = null;
  private timer: NodeJS.Timeout | null = null;

  constructor(
    private readonly load: (win: BrowserWindow) => Promise<void>,
    private readonly preloadPath: string,
    private readonly handlers: PromptHandlers,
  ) {
    ipcMain.on(PROMPT_IPC.act, (e: IpcMainEvent, action: MeetingPromptAction) => {
      if (!this.win || e.sender !== this.win.webContents) return;
      this.onAction(action);
    });
  }

  isShowing(): boolean {
    return !!this.win && !this.win.isDestroyed() && this.win.isVisible();
  }

  currentApp(): MeetingPromptData["app"] | null {
    return this.isShowing() ? (this.data?.app ?? null) : null;
  }

  async show(data: MeetingPromptData): Promise<void> {
    this.data = data;
    const win = await this.ensureWindow();
    const display = screen.getDisplayNearestPoint(screen.getCursorScreenPoint());
    const { x, y, width } = display.workArea;
    win.setBounds({ x: x + width - WIDTH - 12, y: y + 8, width: WIDTH, height: HEIGHT });
    win.webContents.send(PROMPT_IPC.show, data);
    win.showInactive();
    this.armTimer();
  }

  hide(): void {
    this.clearTimer();
    this.data = null;
    if (this.win && !this.win.isDestroyed()) this.win.hide();
  }

  private async ensureWindow(): Promise<BrowserWindow> {
    if (this.win && !this.win.isDestroyed()) return this.win;
    const win = new BrowserWindow({
      width: WIDTH,
      height: HEIGHT,
      show: false,
      frame: false,
      transparent: true,
      resizable: false,
      movable: false,
      minimizable: false,
      maximizable: false,
      fullscreenable: false,
      skipTaskbar: true,
      focusable: false,
      hasShadow: false,
      backgroundColor: "#00000000",
      webPreferences: {
        nodeIntegration: false,
        contextIsolation: true,
        sandbox: true,
        preload: this.preloadPath,
      },
    });
    win.setAlwaysOnTop(true, "floating");
    win.setVisibleOnAllWorkspaces(true, { visibleOnFullScreen: true });
    win.webContents.setWindowOpenHandler(() => ({ action: "deny" }));
    win.webContents.on("will-navigate", (e) => e.preventDefault());
    win.on("closed", () => {
      if (this.win === win) this.win = null;
    });
    this.win = win;
    await this.load(win);
    return win;
  }

  private onAction(action: MeetingPromptAction): void {
    const data = this.data;
    if (!data) return;
    switch (action) {
      case "start":
        this.hide();
        this.handlers.start(data);
        break;
      case "dismiss":
        this.hide();
        break;
      case "hover-start":
        this.clearTimer();
        break;
      case "hover-end":
        this.armTimer();
        break;
      case "setup":
        this.hide();
        this.handlers.setup(data);
        break;
      case "mute":
        this.hide();
        this.handlers.mute(data);
        break;
      case "disable":
        this.hide();
        this.handlers.disable();
        break;
      case "menu-open":
        this.clearTimer();
        this.resize(HEIGHT + MENU_HEIGHT);
        break;
      case "menu-close":
        this.resize(HEIGHT);
        this.armTimer();
        break;
    }
  }

  private resize(height: number): void {
    if (!this.win || this.win.isDestroyed()) return;
    const b = this.win.getBounds();
    this.win.setBounds({ ...b, height });
  }

  private armTimer(): void {
    this.clearTimer();
    this.timer = setTimeout(() => this.hide(), AUTO_DISMISS_MS);
  }

  private clearTimer(): void {
    if (this.timer) clearTimeout(this.timer);
    this.timer = null;
  }
}

export function promptPreloadPath(mainDir: string): string {
  return path.join(mainDir, "..", "..", "preload", "preload", "prompt-preload.js");
}
