/** Turns MeetingWatch reports into "meeting started" prompts (pure; no Electron). */

import { localIsoMinute } from "../../shared/time";
import type { MeetingApp } from "../../shared/types";
import type { WatchReport } from "./meeting-watch-service";

export interface KnownApp {
  id: MeetingApp;
  name: string;
  prefixes: string[];
  /** Window titles that carry no meeting name. */
  genericTitles: RegExp;
  titleSuffix?: RegExp;
}

export const KNOWN_APPS: KnownApp[] = [
  {
    id: "zoom",
    name: "Zoom",
    prefixes: ["us.zoom.xos"],
    genericTitles: /^(zoom( meeting| webinar| workplace)?|meeting|zoom cloud meetings)$/i,
  },
  {
    id: "teams",
    name: "Teams",
    prefixes: ["com.microsoft.teams2", "com.microsoft.teams"],
    genericTitles: /^(microsoft teams|teams|chat|calendar|activity|meeting|meeting compact view)$/i,
    titleSuffix: /\s*[|\-–]\s*Microsoft Teams.*$/i,
  },
];

export const WATCHED_PREFIXES = KNOWN_APPS.flatMap((a) => a.prefixes);

export interface PromptRequest {
  app: MeetingApp;
  appName: string;
  title: string;
  /** When capture was first seen (ISO, local offset, minute precision); becomes the meeting time. */
  detectedAt: string;
}

export interface DetectorOptions {
  /** Consecutive reports with capture before a meeting counts as started. */
  startPolls?: number;
  /** Milliseconds without capture before a meeting counts as ended. */
  endGraceMs?: number;
}

export interface DetectorContext {
  enabled: boolean;
  mutedApps: string[];
  /** Echo is already recording a meeting. */
  busy: boolean;
}

interface Episode {
  polls: number;
  firstSeen: number;
  lastSeen: number;
  prompted: boolean;
}

export function appForBundle(bundle: string): KnownApp | undefined {
  return KNOWN_APPS.find((a) => a.prefixes.some((p) => bundle.startsWith(p)));
}

export function meetingTitle(app: KnownApp, windows: Record<string, string[]>, now: Date): string {
  const titles = Object.entries(windows)
    .filter(([bundle]) => app.prefixes.some((p) => bundle.startsWith(p)))
    .flatMap(([, list]) => list);
  for (const raw of titles) {
    const t = (app.titleSuffix ? raw.replace(app.titleSuffix, "") : raw).trim();
    if (t && !app.genericTitles.test(t) && t.length <= 120) return t;
  }
  const hh = String(now.getHours()).padStart(2, "0");
  const mm = String(now.getMinutes()).padStart(2, "0");
  return `${app.name} meeting ${hh}:${mm}`;
}

export class MeetingDetector {
  private episodes = new Map<MeetingApp, Episode>();
  private readonly startPolls: number;
  private readonly endGraceMs: number;

  constructor(opts: DetectorOptions = {}) {
    this.startPolls = opts.startPolls ?? 2;
    this.endGraceMs = opts.endGraceMs ?? 20_000;
  }

  /** Apps currently in a meeting (for status display). */
  active(): MeetingApp[] {
    return [...this.episodes.keys()];
  }

  /**
   * Feed one report. Returns prompts to show (at most one per meeting). Reports only arrive when
   * something changes, so call `tick(now)` periodically to expire ended meetings.
   */
  update(report: WatchReport, ctx: DetectorContext, now: number = Date.now()): PromptRequest[] {
    const capturing = new Set<MeetingApp>();
    for (const p of report.inputs) {
      const app = appForBundle(p.bundle);
      if (app) capturing.add(app.id);
    }
    this.expire(capturing, now);
    const out: PromptRequest[] = [];
    for (const id of capturing) {
      const ep = this.episodes.get(id) ?? { polls: 0, firstSeen: now, lastSeen: now, prompted: false };
      ep.polls += 1;
      ep.lastSeen = now;
      this.episodes.set(id, ep);
      if (ep.prompted || ep.polls < this.startPolls) continue;
      if (!ctx.enabled || ctx.busy || ctx.mutedApps.includes(id)) {
        ep.prompted = true;
        continue;
      }
      ep.prompted = true;
      const app = KNOWN_APPS.find((a) => a.id === id)!;
      out.push({
        app: id,
        appName: app.name,
        title: meetingTitle(app, report.windows, new Date(now)),
        detectedAt: localIsoMinute(new Date(ep.firstSeen)),
      });
    }
    return out;
  }

  /**
   * Re-evaluate without a new report (reports only arrive on change, so a steady capture
   * produces no new lines). Counts as another poll for apps still capturing.
   */
  tick(last: WatchReport | null, ctx: DetectorContext, now: number = Date.now()): PromptRequest[] {
    return last ? this.update(last, ctx, now) : (this.expire(new Set(), now), []);
  }

  private expire(capturing: Set<MeetingApp>, now: number): void {
    for (const [id, ep] of this.episodes) {
      if (!capturing.has(id) && now - ep.lastSeen >= this.endGraceMs) this.episodes.delete(id);
    }
  }
}
