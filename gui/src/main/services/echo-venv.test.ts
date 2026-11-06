import { mkdtempSync, mkdirSync, writeFileSync } from "fs";
import * as os from "os";
import * as path from "path";
import { describe, expect, it } from "vitest";
import { sourceStamp } from "./echo-venv";

function makeSource(): string {
  const dir = mkdtempSync(path.join(os.tmpdir(), "echo-src-"));
  mkdirSync(path.join(dir, "echo", "__pycache__"), { recursive: true });
  writeFileSync(path.join(dir, "pyproject.toml"), "[project]\n");
  writeFileSync(path.join(dir, "echo", "server.py"), "print('a')\n");
  writeFileSync(path.join(dir, "echo", "__pycache__", "server.pyc"), "x");
  return dir;
}

describe("sourceStamp", () => {
  it("is stable and prefixed with the app version", async () => {
    const dir = makeSource();
    const a = await sourceStamp(dir, "0.1.0");
    expect(a).toMatch(/^0\.1\.0\+[0-9a-f]{16}$/);
    expect(await sourceStamp(dir, "0.1.0")).toBe(a);
  });

  it("changes with source content but ignores __pycache__ and build leftovers", async () => {
    const dir = makeSource();
    const before = await sourceStamp(dir, "0.1.0");
    writeFileSync(path.join(dir, "echo", "__pycache__", "server.pyc"), "y");
    mkdirSync(path.join(dir, "echo.egg-info"));
    writeFileSync(path.join(dir, "echo.egg-info", "PKG-INFO"), "z");
    expect(await sourceStamp(dir, "0.1.0")).toBe(before);
    writeFileSync(path.join(dir, "echo", "server.py"), "print('b')\n");
    expect(await sourceStamp(dir, "0.1.0")).not.toBe(before);
  });
});
