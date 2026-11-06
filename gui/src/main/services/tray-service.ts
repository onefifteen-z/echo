import { Menu, nativeImage, Tray } from "electron";
import * as path from "path";
import type { EchoEvent, SidecarState, Status } from "../../shared/types";
import type { SidecarService } from "./sidecar-service";
import { menuKey, trayModel, type TrayModel } from "./tray-model";

export interface TrayActions {
  open: () => void;
  newMeeting: () => void;
  stop: () => void;
  quit: () => void;
}

/** macOS menu bar item: recording state + elapsed time, with meeting controls. */
export class TrayController {
  private readonly tray: Tray;
  private readonly icons: { idle: Electron.NativeImage; recording: Electron.NativeImage };
  private status: Status | null = null;
  private phase: SidecarState["phase"];
  private lastMenu = "";
  private lastRecording: boolean | null = null;

  constructor(
    iconDir: string,
    private readonly sidecar: SidecarService,
    private readonly actions: TrayActions,
  ) {
    const load = (name: string) => {
      const img = nativeImage.createFromPath(path.join(iconDir, `${name}.png`));
      img.setTemplateImage(true);
      return img;
    };
    this.icons = { idle: load("trayTemplate"), recording: load("trayRecordingTemplate") };
    this.tray = new Tray(this.icons.idle);
    this.tray.setToolTip("Echo");
    this.phase = sidecar.getState().phase;
    sidecar.on("event", (ev: EchoEvent) => {
      if (ev.type === "status") this.status = ev;
      else if (ev.type === "session.ended") this.status = this.status && { ...this.status, state: "finished" };
      else return;
      this.render();
    });
    sidecar.on("state", (state: SidecarState) => {
      this.phase = state.phase;
      if (state.phase !== "ready") this.status = null;
      this.render();
    });
    this.render();
  }

  private render(): void {
    const model = trayModel(this.status, this.phase);
    this.tray.setTitle(model.title ? ` ${model.title}` : "", { fontType: "monospacedDigit" });
    if (model.recording !== this.lastRecording) {
      this.tray.setImage(model.recording ? this.icons.recording : this.icons.idle);
      this.lastRecording = model.recording;
    }
    const key = menuKey(model);
    if (key !== this.lastMenu) {
      this.tray.setContextMenu(this.buildMenu(model));
      this.lastMenu = key;
    }
  }

  private buildMenu(m: TrayModel): Menu {
    const run = (cmd: "session.pause" | "session.resume") => () => {
      void this.sidecar.invoke(cmd, {}).catch(() => undefined);
    };
    const items: Electron.MenuItemConstructorOptions[] = [
      { label: m.statusLabel, enabled: false },
      { type: "separator" },
      { label: "Open Echo", click: this.actions.open },
    ];
    if (m.canStartNew) items.push({ label: "New meeting…", click: this.actions.newMeeting });
    if (m.canPause) items.push({ label: "Pause", click: run("session.pause") });
    if (m.canResume) items.push({ label: "Resume", click: run("session.resume") });
    if (m.canStop) items.push({ label: "Stop meeting", click: this.actions.stop });
    items.push({ type: "separator" }, { label: "Quit Echo", click: this.actions.quit });
    return Menu.buildFromTemplate(items);
  }

  destroy(): void {
    this.tray.destroy();
  }
}
