import type { SidecarState, Status } from "../../shared/types";

export interface TrayModel {
  /** Text next to the menu bar icon. */
  title: string;
  recording: boolean;
  statusLabel: string;
  canPause: boolean;
  canResume: boolean;
  canStop: boolean;
  canStartNew: boolean;
}

/** `"00:12:34"` -> `"12:34"`; hours are kept once a meeting passes one hour. */
export function compactElapsed(elapsed: string): string {
  return /^00:\d{2}:\d{2}$/.test(elapsed) ? elapsed.slice(3) : elapsed;
}

export function trayModel(status: Status | null, phase: SidecarState["phase"]): TrayModel {
  const idle: TrayModel = {
    title: "",
    recording: false,
    statusLabel: "No meeting in progress",
    canPause: false,
    canResume: false,
    canStop: false,
    canStartNew: true,
  };
  if (phase === "starting" || phase === "crashed") {
    return { ...idle, statusLabel: "Engine starting…", canStartNew: false };
  }
  if (phase === "failed") return { ...idle, statusLabel: "Engine failed — open Echo for details", canStartNew: false };
  if (!status || status.state === "stopped" || status.state === "idle") return idle;

  const name = status.meeting_title || "Untitled meeting";
  if (status.state === "finished") {
    return { ...idle, statusLabel: `Finished: ${name}`, canStop: true, canStartNew: false };
  }
  const paused = status.state === "paused";
  return {
    title: paused ? "Paused" : compactElapsed(status.elapsed),
    recording: !paused,
    statusLabel: `${paused ? "Paused" : "Recording"}: ${name}`,
    canPause: !paused,
    canResume: paused,
    canStop: true,
    canStartNew: false,
  };
}

/** Fields that change the menu (not just the ticking title). */
export function menuKey(m: TrayModel): string {
  return [m.statusLabel, m.canPause, m.canResume, m.canStop, m.canStartNew].join("|");
}
