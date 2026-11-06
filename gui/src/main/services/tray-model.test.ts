import { describe, expect, it } from "vitest";
import type { Status } from "../../shared/types";
import { compactElapsed, menuKey, trayModel } from "./tray-model";

const status = (state: Status["state"], elapsed = "00:12:34", meeting_title = "Weekly"): Status => ({
  type: "status",
  mode: "live",
  state,
  language: "japanese",
  languages: ["japanese"],
  asr_source: "apple-on-device",
  asr_label: "Apple",
  mic_only: false,
  llm_offline: false,
  meeting_title,
  elapsed,
});

describe("trayModel", () => {
  it("is idle without a meeting", () => {
    for (const s of [null, status("idle"), status("stopped")]) {
      const m = trayModel(s, "ready");
      expect(m).toMatchObject({ title: "", recording: false, canStartNew: true, canStop: false });
      expect(m.statusLabel).toBe("No meeting in progress");
    }
  });

  it("shows elapsed time and controls while recording", () => {
    const m = trayModel(status("running"), "ready");
    expect(m).toMatchObject({ title: "12:34", recording: true, canPause: true, canResume: false, canStop: true, canStartNew: false });
    expect(m.statusLabel).toBe("Recording: Weekly");
  });

  it("shows Paused and offers Resume", () => {
    const m = trayModel(status("paused", "00:00:05", ""), "ready");
    expect(m).toMatchObject({ title: "Paused", recording: false, canPause: false, canResume: true, canStop: true });
    expect(m.statusLabel).toBe("Paused: Untitled meeting");
  });

  it("offers Stop for a finished replay", () => {
    const m = trayModel(status("finished"), "ready");
    expect(m).toMatchObject({ title: "", recording: false, canStop: true, canStartNew: false });
    expect(m.statusLabel).toBe("Finished: Weekly");
  });

  it("reports engine state when the sidecar is not ready", () => {
    expect(trayModel(status("running"), "starting")).toMatchObject({ statusLabel: "Engine starting…", canStop: false, canStartNew: false });
    expect(trayModel(null, "crashed").statusLabel).toBe("Engine starting…");
    expect(trayModel(null, "failed").statusLabel).toMatch(/failed/);
  });

  it("keeps the menu key stable while only the timer ticks", () => {
    expect(menuKey(trayModel(status("running", "00:00:01"), "ready"))).toBe(menuKey(trayModel(status("running", "00:00:02"), "ready")));
    expect(menuKey(trayModel(status("running"), "ready"))).not.toBe(menuKey(trayModel(status("paused"), "ready")));
  });
});

describe("compactElapsed", () => {
  it("drops a zero hour", () => {
    expect(compactElapsed("00:00:07")).toBe("00:07");
    expect(compactElapsed("01:02:03")).toBe("01:02:03");
    expect(compactElapsed("")).toBe("");
  });
});
