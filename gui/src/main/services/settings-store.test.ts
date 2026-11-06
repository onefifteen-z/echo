import * as fs from "fs";
import * as os from "os";
import * as path from "path";
import { describe, expect, it } from "vitest";
import { APP_KEYS_ENV, type Cipher, isSecretKey, SettingsStore, validateKey } from "./settings-store";

const fakeCipher = (available = true): Cipher => ({
  available: () => available,
  encrypt: (s) => Buffer.from(`enc:${[...s].reverse().join("")}`, "utf-8"),
  decrypt: (b) => [...b.toString("utf-8").replace(/^enc:/, "")].reverse().join(""),
});

function tmpStore(cipher = fakeCipher()) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "echo-settings-"));
  const file = path.join(dir, "settings.json");
  return { file, make: () => new SettingsStore(file, cipher, "/Users/me/Documents/Echo") };
}

describe("SettingsStore", () => {
  it("defaults when no file exists", () => {
    const { make } = tmpStore();
    const s = make().snapshot();
    expect(s.workspaceDir).toBe("/Users/me/Documents/Echo");
    expect(s.customWorkspace).toBe(false);
    expect(s.env).toEqual({});
    expect(s.secretsSet).toEqual([]);
  });

  it("stores plain values and encrypted secrets, and never exposes secret values", () => {
    const { file, make } = tmpStore();
    const store = make();
    const snap = store.apply({ env: { ECHO_ASR_BACKEND: "api", API_KEY: "sk-123", ECHO_DEBUG: "1" } });
    expect(snap.env).toEqual({ ECHO_ASR_BACKEND: "api", ECHO_DEBUG: "1" });
    expect(snap.secretsSet).toEqual(["API_KEY"]);
    expect(JSON.stringify(snap)).not.toContain("sk-123");
    const raw = fs.readFileSync(file, "utf-8");
    expect(raw).not.toContain("sk-123");
    expect(fs.statSync(file).mode & 0o777).toBe(0o600);

    const env = make().toSpawnEnv();
    expect(env.API_KEY).toBe("sk-123");
    expect(env.ECHO_ASR_BACKEND).toBe("api");
    expect(env[APP_KEYS_ENV]).toBe("API_KEY,ECHO_ASR_BACKEND,ECHO_DEBUG");
  });

  it("keeps absent keys, removes null keys, and clears secrets with empty strings", () => {
    const { make } = tmpStore();
    const store = make();
    store.apply({ env: { MODEL: "m1", API_KEY: "k", OPENAI_API_KEY: "o" } });
    const snap = store.apply({ env: { MODEL: null, API_KEY: "" } });
    expect(snap.env).toEqual({});
    expect(snap.secretsSet).toEqual(["OPENAI_API_KEY"]);
  });

  it("rejects invalid or reserved keys without writing anything", () => {
    const { make } = tmpStore();
    const store = make();
    store.apply({ env: { MODEL: "m1" } });
    expect(() => store.apply({ env: { MODEL: "m2", "bad-key": "x" } })).toThrow(/Invalid variable name/);
    expect(() => store.apply({ env: { PATH: "/tmp" } })).toThrow(/cannot be set/);
    expect(() => store.apply({ env: { PYTHONPATH: "/tmp" } })).toThrow(/cannot be set/);
    expect(make().snapshot().env).toEqual({ MODEL: "m1" });
  });

  it("validates and resets the workspace folder", () => {
    const { make } = tmpStore();
    const store = make();
    expect(() => store.apply({ workspaceDir: "relative/dir" })).toThrow(/absolute/);
    expect(store.apply({ workspaceDir: "/tmp/meetings" }).workspaceDir).toBe("/tmp/meetings");
    expect(make().snapshot().customWorkspace).toBe(true);
    expect(store.apply({ workspaceDir: null }).customWorkspace).toBe(false);
    expect(store.apply({ workspaceDir: "/Users/me/Documents/Echo" }).customWorkspace).toBe(false);
  });

  it("falls back to an unencrypted encoding when encryption is unavailable", () => {
    const { file, make } = tmpStore(fakeCipher(false));
    const store = make();
    store.apply({ env: { ANTHROPIC_API_KEY: "secret-value" } });
    expect(fs.readFileSync(file, "utf-8")).not.toContain("secret-value");
    expect(store.toSpawnEnv().ANTHROPIC_API_KEY).toBe("secret-value");
    expect(store.snapshot().encryptionAvailable).toBe(false);
  });

  it("survives a corrupt settings file", () => {
    const { file, make } = tmpStore();
    fs.writeFileSync(file, "{not json");
    expect(make().snapshot().env).toEqual({});
  });
});

describe("key helpers", () => {
  it("classifies secrets and validates names", () => {
    expect(isSecretKey("API_KEY")).toBe(true);
    expect(isSecretKey("HF_TOKEN")).toBe(true);
    expect(isSecretKey("MODEL")).toBe(false);
    expect(validateKey("ECHO_DEBUG")).toBeNull();
    expect(validateKey("echo_debug")).not.toBeNull();
    expect(validateKey("ECHO_SERVER_TOKEN")).not.toBeNull();
    expect(validateKey("DYLD_INSERT_LIBRARIES")).not.toBeNull();
  });
});

describe("prefs", () => {
  it("defaults, persists and normalizes preferences", () => {
    const { file, make } = tmpStore();
    const store = make();
    expect(store.prefs()).toEqual({ meetingDetection: true, mutedApps: [], lastStart: null });
    store.setPrefs({ mutedApps: ["zoom", "zoom", "bogus" as never] });
    store.setPrefs({ lastStart: { source: "both", backend: "auto", language: "english", mic_device: null } });
    const again = make().prefs();
    expect(again.mutedApps).toEqual(["zoom"]);
    expect(again.lastStart?.language).toBe("english");
    expect(again.meetingDetection).toBe(true);
    store.setPrefs({ meetingDetection: false });
    expect(make().snapshot().prefs.meetingDetection).toBe(false);
    // Env changes keep prefs.
    store.apply({ env: { MODEL: "m" } });
    expect(make().prefs().mutedApps).toEqual(["zoom"]);
    expect(JSON.parse(fs.readFileSync(file, "utf-8")).prefs.meetingDetection).toBe(false);
  });
});
