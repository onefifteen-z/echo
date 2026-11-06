import { describe, expect, it } from "vitest";
import { CURATED_KEYS, keyError, SETTINGS_GROUPS, valueError } from "./settings-schema";

const fieldOf = (key: string) => SETTINGS_GROUPS.flatMap((g) => g.fields).find((f) => f.key === key)!;

describe("settings schema", () => {
  it("has unique, valid keys", () => {
    expect(new Set(CURATED_KEYS).size).toBe(CURATED_KEYS.length);
    for (const key of CURATED_KEYS) expect(keyError(key)).toBeNull();
  });

  it("validates numbers and URLs", () => {
    expect(valueError(fieldOf("ECHO_WHISPER_API_TIMEOUT"), "abc")).toMatch(/number/);
    expect(valueError(fieldOf("ECHO_WHISPER_API_TIMEOUT"), "0")).toMatch(/at least/);
    expect(valueError(fieldOf("ECHO_WHISPER_API_TIMEOUT"), "15")).toBeNull();
    expect(valueError(fieldOf("BASE_URL"), "example.com")).toMatch(/URL/);
    expect(valueError(fieldOf("ECHO_PERPLEXITY_BASE_URL"), "http://x/v1")).toMatch(/https/);
    expect(valueError(fieldOf("ECHO_PERPLEXITY_BASE_URL"), "https://x/v1")).toBeNull();
    expect(valueError(fieldOf("MODEL"), "")).toBeNull();
  });

  it("rejects reserved keys", () => {
    expect(keyError("PYTHONHOME")).not.toBeNull();
    expect(keyError("ECHO_APP_SETTINGS_KEYS")).not.toBeNull();
    expect(keyError("lower")).not.toBeNull();
  });
});
