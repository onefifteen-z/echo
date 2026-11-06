import * as path from "path";
import { describe, expect, it } from "vitest";
import { meetingDir, noteImageName, parseAssetUrl, resolveAsset } from "./note-assets";

const root = path.resolve("/tmp/echo-root");

describe("meetingDir", () => {
  it("accepts a folder name and rejects escapes", () => {
    expect(meetingDir(root, "2026-11-06 2203 Sync")).toBe(path.join(root, "2026-11-06 2203 Sync"));
    for (const bad of ["", ".", "..", "../x", "a/b", "a\\b", ".hidden"]) {
      expect(() => meetingDir(root, bad)).toThrow();
    }
  });
});

describe("noteImageName", () => {
  it("maps mime types and formats the timestamp", () => {
    const now = new Date(2026, 10, 7, 0, 15, 0);
    expect(noteImageName("image/png", now, "a3f2")).toBe("img-20261107-001500-a3f2.png");
    expect(noteImageName("image/jpeg", now, "0000")).toMatch(/\.jpg$/);
    expect(() => noteImageName("image/svg+xml", now)).toThrow();
    expect(() => noteImageName("text/html", now)).toThrow();
  });
});

describe("resolveAsset", () => {
  it("only resolves files directly inside notes-assets", () => {
    expect(resolveAsset(root, "M", "notes-assets/img.png")).toBe(path.join(root, "M", "notes-assets", "img.png"));
    expect(resolveAsset(root, "M", "notes-assets/../notebook.md")).toBeNull();
    expect(resolveAsset(root, "M", "notebook.md")).toBeNull();
    expect(resolveAsset(root, "M", "notes-assets/.secret")).toBeNull();
    expect(resolveAsset(root, "..", "notes-assets/img.png")).toBeNull();
  });
});

describe("parseAssetUrl", () => {
  it("decodes the meeting id and asset path", () => {
    expect(parseAssetUrl("echo-asset://meeting/2026-11-06%202203%20Sync/notes-assets/a.png")).toEqual({
      id: "2026-11-06 2203 Sync",
      rel: "notes-assets/a.png",
    });
    expect(parseAssetUrl("echo-asset://other/x/notes-assets/a.png")).toBeNull();
    expect(parseAssetUrl("file:///etc/passwd")).toBeNull();
    expect(parseAssetUrl("echo-asset://meeting/M/notes-assets/a/b.png")).toBeNull();
  });
});
