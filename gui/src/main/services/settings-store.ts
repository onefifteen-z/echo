/**
 * App settings for the desktop app: workspace folder + environment overrides for the sidecar.
 *
 * Stored as `settings.json` in Electron's userData. Values for secret keys are encrypted with the
 * injected cipher (Electron `safeStorage`, i.e. the macOS Keychain) and never sent to the renderer.
 * At sidecar spawn the overrides are merged over `process.env`; `.env` is loaded by Python without
 * override, so these values win over `.env`.
 */

import * as fs from "fs";
import * as path from "path";
import type { AppPrefs, MeetingApp, SettingsPatch, SettingsSnapshot } from "../../shared/types";

export interface Cipher {
  available(): boolean;
  encrypt(plain: string): Buffer;
  decrypt(data: Buffer): string;
}

interface SettingsFile {
  version: 1;
  workspaceDir: string | null;
  env: Record<string, string>;
  /** base64 ciphertext, or `plain:` + base64 when encryption is unavailable. */
  secrets: Record<string, string>;
  prefs: AppPrefs;
}

const MEETING_APPS: MeetingApp[] = ["zoom", "teams"];

function defaultPrefs(): AppPrefs {
  return { meetingDetection: true, mutedApps: [], lastStart: null };
}

function normalizePrefs(raw: Partial<AppPrefs> | undefined): AppPrefs {
  const p = { ...defaultPrefs(), ...(raw ?? {}) };
  return {
    meetingDetection: p.meetingDetection !== false,
    mutedApps: [...new Set((Array.isArray(p.mutedApps) ? p.mutedApps : []).filter((a) => MEETING_APPS.includes(a)))],
    lastStart: p.lastStart && typeof p.lastStart === "object" ? p.lastStart : null,
  };
}

const KEY_RE = /^[A-Z][A-Z0-9_]*$/;
const KNOWN_SECRETS = new Set(["API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "ECHO_PERPLEXITY_API_KEY", "AI_GATEWAY_API_KEY"]);
const SECRET_RE = /(KEY|TOKEN|SECRET|PASSWORD)/;
const PROTECTED = new Set(["PATH", "HOME", "USER", "SHELL", "ECHO_SERVER_TOKEN", "ECHO_APP_SETTINGS_KEYS", "PYTHONUNBUFFERED"]);
/** Injected by main with the names of keys coming from app settings (for "source" badges). */
export const APP_KEYS_ENV = "ECHO_APP_SETTINGS_KEYS";

export function isSecretKey(key: string): boolean {
  return KNOWN_SECRETS.has(key) || SECRET_RE.test(key);
}

/** Returns an error message, or null when the key may be set from the app. */
export function validateKey(key: string): string | null {
  if (!KEY_RE.test(key)) return `Invalid variable name: ${key || "(empty)"} (use A-Z, 0-9 and _)`;
  if (PROTECTED.has(key) || key.startsWith("PYTHON") || key.startsWith("DYLD_") || key.startsWith("ELECTRON_")) {
    return `${key} cannot be set from Echo settings.`;
  }
  return null;
}

function empty(): SettingsFile {
  return { version: 1, workspaceDir: null, env: {}, secrets: {}, prefs: defaultPrefs() };
}

export class SettingsStore {
  private data: SettingsFile;

  constructor(
    private readonly filePath: string,
    private readonly cipher: Cipher,
    private readonly defaultWorkspaceDir: string,
  ) {
    this.data = this.load();
  }

  private load(): SettingsFile {
    try {
      const raw = JSON.parse(fs.readFileSync(this.filePath, "utf-8")) as Partial<SettingsFile>;
      return {
        version: 1,
        workspaceDir: typeof raw.workspaceDir === "string" && raw.workspaceDir ? raw.workspaceDir : null,
        env: { ...(raw.env ?? {}) },
        secrets: { ...(raw.secrets ?? {}) },
        prefs: normalizePrefs(raw.prefs),
      };
    } catch {
      return empty();
    }
  }

  private persist(): void {
    fs.mkdirSync(path.dirname(this.filePath), { recursive: true });
    const tmp = `${this.filePath}.tmp`;
    fs.writeFileSync(tmp, JSON.stringify(this.data, null, 2), { encoding: "utf-8", mode: 0o600 });
    fs.renameSync(tmp, this.filePath);
  }

  workspaceDir(): string {
    return this.data.workspaceDir ?? this.defaultWorkspaceDir;
  }

  snapshot(): SettingsSnapshot {
    return {
      workspaceDir: this.workspaceDir(),
      defaultWorkspaceDir: this.defaultWorkspaceDir,
      customWorkspace: this.data.workspaceDir !== null,
      env: { ...this.data.env },
      secretsSet: Object.keys(this.data.secrets).sort(),
      encryptionAvailable: this.cipher.available(),
      prefs: { ...this.data.prefs, mutedApps: [...this.data.prefs.mutedApps] },
    };
  }

  /** Apply a patch: `null` removes a key (falls back to `.env` / default); absent keys are kept. */
  apply(patch: SettingsPatch): SettingsSnapshot {
    const next: SettingsFile = {
      ...this.data,
      env: { ...this.data.env },
      secrets: { ...this.data.secrets },
    };
    if (patch.workspaceDir !== undefined) {
      const dir = patch.workspaceDir?.trim() || null;
      if (dir !== null && !path.isAbsolute(dir)) throw new Error("Workspace folder must be an absolute path.");
      next.workspaceDir = dir === this.defaultWorkspaceDir ? null : dir;
    }
    for (const [key, value] of Object.entries(patch.env ?? {})) {
      const err = validateKey(key);
      if (err) throw new Error(err);
      delete next.env[key];
      delete next.secrets[key];
      if (value === null) continue;
      if (isSecretKey(key)) {
        if (value === "") continue;
        next.secrets[key] = this.seal(value);
      } else {
        next.env[key] = value;
      }
    }
    this.data = next;
    this.persist();
    return this.snapshot();
  }

  prefs(): AppPrefs {
    return this.snapshot().prefs;
  }

  /** Update app preferences (applied immediately; no engine restart). */
  setPrefs(patch: Partial<AppPrefs>): SettingsSnapshot {
    this.data = { ...this.data, prefs: normalizePrefs({ ...this.data.prefs, ...patch }) };
    this.persist();
    return this.snapshot();
  }

  /** Environment overrides for the sidecar (secrets decrypted), plus the list of app-provided keys. */
  toSpawnEnv(): Record<string, string> {
    const env: Record<string, string> = { ...this.data.env };
    for (const [key, sealed] of Object.entries(this.data.secrets)) {
      const value = this.unseal(sealed);
      if (value !== null) env[key] = value;
    }
    env[APP_KEYS_ENV] = Object.keys(env).sort().join(",");
    return env;
  }

  private seal(value: string): string {
    if (this.cipher.available()) return this.cipher.encrypt(value).toString("base64");
    return `plain:${Buffer.from(value, "utf-8").toString("base64")}`;
  }

  private unseal(sealed: string): string | null {
    try {
      if (sealed.startsWith("plain:")) return Buffer.from(sealed.slice(6), "base64").toString("utf-8");
      return this.cipher.decrypt(Buffer.from(sealed, "base64"));
    } catch {
      return null;
    }
  }
}
