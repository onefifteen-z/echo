import { describe, expect, it } from "vitest";
import { KNOWN_APPS, MeetingDetector, appForBundle, meetingTitle } from "./meeting-detector";
import type { WatchReport } from "./meeting-watch-service";
import { localIsoMinute } from "../../shared/time";

const ctx = { enabled: true, mutedApps: [] as string[], busy: false };
const zoom = (windows: Record<string, string[]> = {}): WatchReport => ({ inputs: [{ pid: 10, bundle: "us.zoom.xos" }], windows });
const none: WatchReport = { inputs: [], windows: {} };
const at = new Date(2026, 10, 6, 9, 5).getTime();

describe("MeetingDetector", () => {
  it("prompts after two consecutive polls, once per meeting", () => {
    const d = new MeetingDetector();
    expect(d.update(zoom(), ctx, at)).toEqual([]);
    const [p] = d.tick(zoom(), ctx, at + 2000);
    expect(p).toEqual({ app: "zoom", appName: "Zoom", title: "Zoom meeting 09:05", detectedAt: localIsoMinute(new Date(at)) });
    expect(d.tick(zoom(), ctx, at + 4000)).toEqual([]);
  });

  it("ignores short blips and ends a meeting after the grace period", () => {
    const d = new MeetingDetector({ endGraceMs: 20_000 });
    d.update(zoom(), ctx, at);
    d.update(none, ctx, at + 1000);
    expect(d.update(zoom(), ctx, at + 2000)).toHaveLength(1);
    d.update(none, ctx, at + 3000);
    // Back within the grace period: same meeting, no new prompt.
    expect(d.update(zoom(), ctx, at + 10_000)).toEqual([]);
    d.update(none, ctx, at + 11_000);
    d.tick(none, ctx, at + 40_000);
    expect(d.active()).toEqual([]);
    d.update(zoom(), ctx, at + 50_000);
    expect(d.tick(zoom(), ctx, at + 52_000)).toHaveLength(1);
  });

  it("does not prompt when disabled, muted, or Echo is busy — and not later in the same meeting", () => {
    for (const c of [{ ...ctx, enabled: false }, { ...ctx, mutedApps: ["zoom"] }, { ...ctx, busy: true }]) {
      const d = new MeetingDetector();
      d.update(zoom(), c, at);
      expect(d.tick(zoom(), c, at + 2000)).toEqual([]);
      expect(d.tick(zoom(), ctx, at + 4000)).toEqual([]);
    }
  });

  it("reports when capture was first seen, not when the prompt fires", () => {
    const d = new MeetingDetector({ startPolls: 3 });
    d.update(zoom(), ctx, at + 30_000);
    d.tick(zoom(), ctx, at + 70_000);
    const [p] = d.tick(zoom(), ctx, at + 130_000);
    expect(p.detectedAt).toBe(localIsoMinute(new Date(at)));
    expect(p.detectedAt).toMatch(/^2026-11-06T09:05:00[+-]\d{2}:\d{2}$/);
  });

  it("matches Teams helper processes by prefix and ignores other apps", () => {
    const d = new MeetingDetector({ startPolls: 1 });
    const r: WatchReport = {
      inputs: [
        { pid: 1, bundle: "org.python.python" },
        { pid: 2, bundle: "com.microsoft.teams2.helper" },
      ],
      windows: { "com.microsoft.teams2": ["Calendar | Microsoft Teams", "Weekly sync | Microsoft Teams"] },
    };
    expect(d.update(r, ctx, at)).toEqual([{ app: "teams", appName: "Teams", title: "Weekly sync", detectedAt: localIsoMinute(new Date(at)) }]);
    expect(appForBundle("org.python.python")).toBeUndefined();
  });
});

describe("meetingTitle", () => {
  const zoomApp = KNOWN_APPS.find((a) => a.id === "zoom")!;
  it("skips generic window names and falls back to app + time", () => {
    expect(meetingTitle(zoomApp, { "us.zoom.xos": ["Zoom Meeting", "Zoom Workplace"] }, new Date(at))).toBe("Zoom meeting 09:05");
    expect(meetingTitle(zoomApp, { "us.zoom.xos": ["Zoom Meeting", "Design review"] }, new Date(at))).toBe("Design review");
  });
});
