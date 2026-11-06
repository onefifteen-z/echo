/**
 * Builds (once, with `xcrun swiftc`) and supervises the MeetingWatch helper, which reports the
 * processes capturing audio input plus window titles of watched apps.
 */

import { execFile, spawn, type ChildProcess } from "child_process";
import { createHash } from "crypto";
import { EventEmitter } from "events";
import * as fs from "fs";
import * as path from "path";
import * as readline from "readline";
import type { MeetingWatchStatus } from "../../shared/types";

export interface WatchReport {
  inputs: { pid: number; bundle: string }[];
  windows: Record<string, string[]>;
}

export declare interface MeetingWatchService {
  on(event: "report", listener: (r: WatchReport) => void): this;
  on(event: "status", listener: (s: MeetingWatchStatus) => void): this;
}

export class MeetingWatchService extends EventEmitter {
  private proc: ChildProcess | null = null;
  private status: MeetingWatchStatus = { state: "stopped" };
  private wanted = false;
  private restarts = 0;

  constructor(
    private readonly sourcePath: string,
    private readonly binDir: string,
    private readonly bundlePrefixes: string[],
    private readonly log: (line: string) => void,
  ) {
    super();
  }

  getStatus(): MeetingWatchStatus {
    return this.status;
  }

  private setStatus(s: MeetingWatchStatus): void {
    this.status = s;
    this.emit("status", s);
  }

  async start(): Promise<void> {
    if (process.platform !== "darwin") {
      this.setStatus({ state: "unavailable", message: "Meeting detection is available on macOS only." });
      return;
    }
    if (this.proc) return;
    this.wanted = true;
    this.setStatus({ state: "starting" });
    let bin: string;
    try {
      bin = await this.ensureBinary();
    } catch (e) {
      this.setStatus({ state: "unavailable", message: e instanceof Error ? e.message : String(e) });
      return;
    }
    if (!this.wanted) return;
    this.spawn(bin);
  }

  stop(): void {
    this.wanted = false;
    const p = this.proc;
    this.proc = null;
    if (p) {
      p.stdin?.end();
      p.kill();
    }
    this.setStatus({ state: "stopped" });
  }

  private spawn(bin: string): void {
    const proc = spawn(bin, ["--interval", "2", ...this.bundlePrefixes], { stdio: ["pipe", "pipe", "pipe"] });
    this.proc = proc;
    this.setStatus({ state: "running" });
    readline.createInterface({ input: proc.stdout! }).on("line", (line) => {
      try {
        const msg = JSON.parse(line) as WatchReport | { error: string };
        if ("error" in msg) {
          this.setStatus({ state: "unavailable", message: msg.error });
          return;
        }
        this.restarts = 0;
        this.emit("report", msg);
      } catch {
        this.log(`meeting-watch: unparsable line: ${line.slice(0, 200)}`);
      }
    });
    readline.createInterface({ input: proc.stderr! }).on("line", (line) => this.log(`meeting-watch: ${line}`));
    proc.on("exit", (code, signal) => {
      if (this.proc !== proc) return;
      this.proc = null;
      if (!this.wanted) return;
      this.log(`meeting-watch exited (code=${code}, signal=${signal})`);
      if (++this.restarts > 3) {
        this.setStatus({ state: "unavailable", message: "The meeting watcher keeps exiting; see the Echo log." });
        return;
      }
      setTimeout(() => this.wanted && this.spawn(bin), 2000 * this.restarts);
    });
  }

  private async ensureBinary(): Promise<string> {
    if (!fs.existsSync(this.sourcePath)) throw new Error(`Meeting watcher source not found: ${this.sourcePath}`);
    const src = fs.readFileSync(this.sourcePath);
    const hash = createHash("sha256").update(src).digest("hex").slice(0, 16);
    const bin = path.join(this.binDir, `meeting-watch-${hash}`);
    if (fs.existsSync(bin)) return bin;
    fs.mkdirSync(this.binDir, { recursive: true });
    for (const f of fs.readdirSync(this.binDir)) {
      if (f.startsWith("meeting-watch-")) fs.rmSync(path.join(this.binDir, f), { force: true });
    }
    const tmp = `${bin}.tmp`;
    this.log("building meeting watcher (xcrun swiftc)…");
    await new Promise<void>((resolve, reject) => {
      execFile(
        "xcrun",
        ["swiftc", "-parse-as-library", "-O", "-framework", "CoreAudio", "-framework", "AppKit", this.sourcePath, "-o", tmp],
        { timeout: 180_000 },
        (err, _stdout, stderr) => {
          if (!err) return resolve();
          const missing = /not found|ENOENT|xcode-select|invalid active developer path/i.test(`${err.message}\n${stderr}`);
          reject(
            new Error(
              missing
                ? "Meeting detection needs the Xcode Command Line Tools (run: xcode-select --install)."
                : `Could not build the meeting watcher: ${(stderr || err.message).trim().slice(0, 400)}`,
            ),
          );
        },
      );
    });
    fs.renameSync(tmp, bin);
    fs.chmodSync(bin, 0o755);
    return bin;
  }
}
