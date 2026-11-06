/**
 * Packaged builds keep Echo's Python engine in a per-user virtualenv at `~/.echo/venv`
 * (modeled on Lumina's `lumina-venv.ts`). The app bundle ships the `echo` package source under
 * `Resources/python`; the venv is (re)installed from it when missing or when the app version or
 * the bundled source changes. Dev builds skip all of this and use the repo's `.venv` (see `main.ts`).
 *
 * A dedicated venv also avoids the conda-base `SIGKILL (Code Signature Invalid)` failure mode:
 * Echo never imports native extensions from a shared interpreter.
 */

import * as os from "os";
import * as path from "path";
import * as fs from "fs/promises";
import { exec } from "child_process";
import { createHash } from "crypto";
import { promisify } from "util";

const execAsync = promisify(exec);

export const MIN_PYTHON_MAJOR = 3;
/** Keep in sync with `requires-python` in the repo's pyproject.toml. */
export const MIN_PYTHON_MINOR = 10;

export function getEchoHome(): string {
  return path.join(os.homedir(), ".echo");
}

export function getEchoVenvDir(): string {
  return path.join(getEchoHome(), "venv");
}

export function getEchoPythonBin(venvDir: string = getEchoVenvDir()): string {
  return process.platform === "win32"
    ? path.join(venvDir, "Scripts", "python.exe")
    : path.join(venvDir, "bin", "python");
}

export function shQuote(p: string): string {
  return `'${p.replace(/'/g, `'\\''`)}'`;
}

export function parsePythonVersion(stdout: string): { major: number; minor: number } | null {
  const m = /^Python\s+(\d+)\.(\d+)/.exec(stdout.trim());
  return m ? { major: Number(m[1]), minor: Number(m[2]) } : null;
}

export function isPythonVersionOk(v: { major: number; minor: number } | null): boolean {
  if (!v) return false;
  return v.major > MIN_PYTHON_MAJOR || (v.major === MIN_PYTHON_MAJOR && v.minor >= MIN_PYTHON_MINOR);
}

/** Prefer Homebrew / python.org interpreters over a conda base env on PATH. */
export async function findSystemPython(): Promise<string | null> {
  const candidates = [
    "/opt/homebrew/bin/python3",
    "/usr/local/bin/python3",
    "/Library/Frameworks/Python.framework/Versions/Current/bin/python3",
    "python3",
  ];
  for (const candidate of candidates) {
    try {
      const { stdout } = await execAsync(`${shQuote(candidate)} --version`, { timeout: 5_000 });
      if (isPythonVersionOk(parsePythonVersion(stdout))) return candidate;
    } catch {
      // try next
    }
  }
  return null;
}

export type EnsureVenvResult =
  | { ok: true; pythonBin: string; venvDir: string }
  | { ok: false; reason: "no-python" | "venv-failed" | "install-failed"; message: string };

let inflight: Promise<EnsureVenvResult> | null = null;

async function listFiles(dir: string): Promise<string[]> {
  const out: string[] = [];
  for (const entry of await fs.readdir(dir, { withFileTypes: true })) {
    if (entry.name === "__pycache__") continue;
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) out.push(...(await listFiles(full)));
    else if (entry.isFile()) out.push(full);
  }
  return out;
}

/** App version plus a content hash of the bundled engine source, so rebuilds with an unchanged version still reinstall. */
export async function sourceStamp(sourceDir: string, appVersion: string): Promise<string> {
  const hash = createHash("sha256");
  const files = [...(await listFiles(path.join(sourceDir, "echo"))), path.join(sourceDir, "pyproject.toml")].sort();
  for (const file of files) {
    hash.update(path.relative(sourceDir, file));
    hash.update("\0");
    hash.update(await fs.readFile(file));
    hash.update("\0");
  }
  return `${appVersion}+${hash.digest("hex").slice(0, 16)}`;
}

/** Create / refresh `~/.echo/venv` from the bundled source. Memoised per process. */
export function ensureEchoVenv(sourceDir: string, appVersion: string, onLog?: (line: string) => void): Promise<EnsureVenvResult> {
  if (!inflight) {
    inflight = doEnsure(sourceDir, appVersion, onLog).finally(() => {
      inflight = null;
    });
  }
  return inflight;
}

async function doEnsure(sourceDir: string, appVersion: string, onLog?: (line: string) => void): Promise<EnsureVenvResult> {
  const venvDir = getEchoVenvDir();
  const pythonBin = getEchoPythonBin(venvDir);
  const stampPath = path.join(venvDir, ".echo-app-version");
  const stamp = await fs.readFile(stampPath, "utf-8").catch(() => "");
  const wanted = await sourceStamp(sourceDir, appVersion);
  if (stamp.trim() === wanted) {
    try {
      await execAsync(`${shQuote(pythonBin)} -c "import echo.server"`, { timeout: 30_000 });
      return { ok: true, pythonBin, venvDir };
    } catch {
      // fall through to reinstall
    }
  }

  const systemPython = await findSystemPython();
  if (!systemPython) {
    return {
      ok: false,
      reason: "no-python",
      message: `Python ${MIN_PYTHON_MAJOR}.${MIN_PYTHON_MINOR}+ was not found. Install it (for example \`brew install python\`) and restart Echo.`,
    };
  }
  try {
    await fs.mkdir(getEchoHome(), { recursive: true });
    await fs.access(pythonBin).catch(async () => {
      onLog?.(`Creating ${venvDir} …`);
      await execAsync(`${shQuote(systemPython)} -m venv ${shQuote(venvDir)}`, { timeout: 120_000 });
    });
  } catch (err) {
    return { ok: false, reason: "venv-failed", message: `Could not create ${venvDir}: ${String(err)}` };
  }
  // setuptools writes build/ and *.egg-info next to the source; build from a copy so the app bundle stays untouched.
  let buildDir: string | null = null;
  try {
    onLog?.("Installing Echo's Python engine (first run can take several minutes) …");
    buildDir = await fs.mkdtemp(path.join(os.tmpdir(), "echo-src-"));
    await fs.cp(sourceDir, buildDir, { recursive: true });
    await execAsync(
      `${shQuote(pythonBin)} -m pip install --upgrade --quiet ${shQuote(`${buildDir}[asr]`)}`,
      { timeout: 30 * 60_000, maxBuffer: 16 * 1024 * 1024 },
    );
    // pip skips a local package whose version number is unchanged, so replace Echo's own files explicitly.
    await execAsync(`${shQuote(pythonBin)} -m pip install --no-deps --force-reinstall --quiet ${shQuote(buildDir)}`, {
      timeout: 10 * 60_000,
      maxBuffer: 16 * 1024 * 1024,
    });
    await fs.writeFile(stampPath, wanted, "utf-8");
  } catch (err) {
    const e = err as { stderr?: string; message?: string };
    return { ok: false, reason: "install-failed", message: (e.stderr || e.message || String(err)).slice(-4000) };
  } finally {
    if (buildDir) await fs.rm(buildDir, { recursive: true, force: true }).catch(() => undefined);
  }
  return { ok: true, pythonBin, venvDir };
}
