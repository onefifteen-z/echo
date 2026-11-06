/**
 * Owns the Python `echo-server` sidecar: spawn, handshake, WebSocket, request/response, restarts.
 *
 * Handshake: the sidecar prints one JSON line `{"port":N,"token":"..."}` on stdout, then serves
 * `ws://127.0.0.1:N/?token=...`. Closing the sidecar's stdin asks it to save outputs and exit.
 */

import { ChildProcess, spawn } from "child_process";
import { EventEmitter } from "events";
import * as readline from "readline";
import WebSocket from "ws";
import type { CommandMap, CommandName, EchoEvent, SidecarState } from "../../shared/types";

export interface PythonLaunch {
  pythonBin: string;
  cwd: string;
  env?: NodeJS.ProcessEnv;
}

interface Pending {
  resolve: (value: unknown) => void;
  reject: (err: Error) => void;
  timer: NodeJS.Timeout;
}

const HANDSHAKE_TIMEOUT_MS = 180_000;
const MAX_RESTARTS_PER_MINUTE = 3;

/** Commands that can legitimately run for minutes (model load / asset download / LLM). */
const LONG_COMMAND_TIMEOUT_MS: Partial<Record<CommandName, number>> = {
  "session.start": 15 * 60_000,
  "session.stop": 5 * 60_000,
  "apple.install_assets": 30 * 60_000,
  "chat.ask": 5 * 60_000,
  probe: 60_000,
  "meetings.regenerate_summary": 5 * 60_000,
};
const DEFAULT_COMMAND_TIMEOUT_MS = 30_000;

export function parseHandshake(line: string): { port: number; token: string } | null {
  try {
    const obj = JSON.parse(line) as { port?: unknown; token?: unknown };
    if (typeof obj.port === "number" && typeof obj.token === "string" && obj.port > 0) {
      return { port: obj.port, token: obj.token };
    }
  } catch {
    // not JSON
  }
  return null;
}

export declare interface SidecarService {
  on(event: "event", listener: (ev: EchoEvent) => void): this;
  on(event: "state", listener: (state: SidecarState) => void): this;
  on(event: "log", listener: (line: string) => void): this;
}

export class SidecarService extends EventEmitter {
  private proc: ChildProcess | null = null;
  private ws: WebSocket | null = null;
  private nextId = 1;
  private pending = new Map<number, Pending>();
  private state: SidecarState = { phase: "starting" };
  private stderrTail: string[] = [];
  private stopping = false;
  private restartTimes: number[] = [];

  constructor(
    private readonly resolvePython: () => Promise<PythonLaunch>,
    private readonly getOutputRoot: () => string,
    private readonly getEnvOverrides: () => Record<string, string> = () => ({}),
  ) {
    super();
  }

  getState(): SidecarState {
    return this.state;
  }

  private setState(state: SidecarState): void {
    this.state = state;
    this.emit("state", state);
  }

  async start(): Promise<void> {
    this.stopping = false;
    this.setState({ phase: "starting" });
    let launch: PythonLaunch;
    try {
      launch = await this.resolvePython();
    } catch (err) {
      this.setState({ phase: "failed", message: err instanceof Error ? err.message : String(err) });
      return;
    }

    const proc = spawn(launch.pythonBin, ["-m", "echo.server", "--output-root", this.getOutputRoot()], {
      cwd: launch.cwd,
      env: { ...process.env, ...launch.env, ...this.getEnvOverrides(), PYTHONUNBUFFERED: "1" },
      stdio: ["pipe", "pipe", "pipe"],
    });
    this.proc = proc;
    this.stderrTail = [];
    proc.on("exit", (code, signal) => this.onExit(proc, code, signal));
    proc.on("error", (err) => {
      this.stderrTail.push(`Failed to spawn ${launch.pythonBin}: ${err.message}`);
      if (proc.pid === undefined) this.onExit(proc, -1, null);
    });

    readline.createInterface({ input: proc.stderr! }).on("line", (line) => {
      this.stderrTail = [...this.stderrTail, line].slice(-40);
      this.emit("log", line);
    });

    const handshake = await new Promise<{ port: number; token: string } | null>((resolve) => {
      const timer = setTimeout(() => resolve(null), HANDSHAKE_TIMEOUT_MS);
      const rl = readline.createInterface({ input: proc.stdout! });
      rl.once("line", (line) => {
        clearTimeout(timer);
        resolve(parseHandshake(line));
      });
      proc.once("exit", () => {
        clearTimeout(timer);
        resolve(null);
      });
    });

    if (!handshake) {
      if (proc.exitCode === null) proc.kill("SIGKILL");
      return;
    }

    try {
      await this.connect(handshake.port, handshake.token);
      this.setState({ phase: "ready", port: handshake.port });
    } catch (err) {
      this.emit("log", `WebSocket connect failed: ${String(err)}`);
      proc.kill("SIGTERM");
    }
  }

  private connect(port: number, token: string): Promise<void> {
    return new Promise((resolve, reject) => {
      const ws = new WebSocket(`ws://127.0.0.1:${port}/?token=${encodeURIComponent(token)}`);
      ws.once("open", () => {
        this.ws = ws;
        resolve();
      });
      ws.once("error", reject);
      ws.on("message", (data) => this.onMessage(String(data)));
      ws.on("close", () => {
        if (this.ws === ws) this.ws = null;
        if (!this.stopping && this.proc && this.proc.exitCode === null) {
          this.emit("log", "Sidecar WebSocket closed unexpectedly; restarting sidecar.");
          this.proc.kill("SIGTERM");
        }
      });
    });
  }

  private onMessage(raw: string): void {
    let msg: { type?: string; id?: number; ok?: boolean; result?: unknown; error?: string; event?: EchoEvent };
    try {
      msg = JSON.parse(raw);
    } catch {
      return;
    }
    if (msg.type === "response" && typeof msg.id === "number") {
      const p = this.pending.get(msg.id);
      if (!p) return;
      this.pending.delete(msg.id);
      clearTimeout(p.timer);
      if (msg.ok) p.resolve(msg.result);
      else p.reject(new Error(msg.error || "Command failed"));
    } else if (msg.type === "event" && msg.event) {
      this.emit("event", msg.event);
    }
  }

  private onExit(proc: ChildProcess, code: number | null, signal: NodeJS.Signals | null): void {
    if (this.proc !== proc) return;
    this.proc = null;
    this.ws = null;
    for (const [, p] of this.pending) {
      clearTimeout(p.timer);
      p.reject(new Error("Echo engine stopped"));
    }
    this.pending.clear();
    if (this.stopping) return;

    const tail = this.stderrTail.slice(-12).join("\n");
    const reason = signal === "SIGKILL" || code === 137 || code === -9
      ? "killed by the OS (SIGKILL). If this repeats, recreate the Python venv."
      : `exited (code ${code ?? "?"}${signal ? `, ${signal}` : ""})`;
    const message = `Echo engine ${reason}${tail ? `\n\n${tail}` : ""}`;

    const now = Date.now();
    this.restartTimes = this.restartTimes.filter((t) => now - t < 60_000);
    if (this.restartTimes.length >= MAX_RESTARTS_PER_MINUTE) {
      this.setState({ phase: "failed", message });
      return;
    }
    this.restartTimes.push(now);
    this.setState({ phase: "crashed", message, restarting: true });
    setTimeout(() => void this.start(), 1_500);
  }

  invoke<K extends CommandName>(cmd: K, args: CommandMap[K][0]): Promise<CommandMap[K][1]> {
    const ws = this.ws;
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      return Promise.reject(new Error("Echo engine is not running"));
    }
    const id = this.nextId++;
    const timeoutMs = LONG_COMMAND_TIMEOUT_MS[cmd] ?? DEFAULT_COMMAND_TIMEOUT_MS;
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id);
        reject(new Error(`${cmd} timed out after ${Math.round(timeoutMs / 1000)}s`));
      }, timeoutMs);
      this.pending.set(id, { resolve: resolve as (v: unknown) => void, reject, timer });
      ws.send(JSON.stringify({ id, cmd, args: args ?? {} }));
    });
  }

  /** Ask the sidecar to save + exit (stdin EOF), escalating to signals if it hangs. */
  async stop(graceMs = 20_000): Promise<void> {
    this.stopping = true;
    const proc = this.proc;
    if (!proc) return;
    await new Promise<void>((resolve) => {
      const killTimer = setTimeout(() => {
        proc.kill("SIGTERM");
        setTimeout(() => {
          if (proc.exitCode === null) proc.kill("SIGKILL");
        }, 3_000);
      }, graceMs);
      proc.once("exit", () => {
        clearTimeout(killTimer);
        resolve();
      });
      proc.stdin?.end();
    });
  }

  async restart(): Promise<void> {
    await this.stop(5_000);
    this.restartTimes = [];
    await this.start();
  }
}
