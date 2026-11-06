/** Images pasted into meeting notes: stored under `<meeting folder>/notes-assets/`, served via `echo-asset://`. */

import * as path from "path";

export const ASSET_DIR = "notes-assets";
export const ASSET_SCHEME = "echo-asset";
export const MAX_NOTE_IMAGE_BYTES = 20 * 1024 * 1024;

const IMAGE_EXT: Record<string, string> = {
  "image/png": "png",
  "image/jpeg": "jpg",
  "image/gif": "gif",
  "image/webp": "webp",
};

/** Absolute folder of a saved meeting; throws for ids that could escape the output root. */
export function meetingDir(root: string, id: string): string {
  const name = String(id ?? "");
  const target = path.join(root, name);
  if (!name || name.startsWith(".") || /[\\/]/.test(name) || path.dirname(target) !== root) {
    throw new Error("Invalid meeting id.");
  }
  return target;
}

export function noteImageName(mime: string, now: Date = new Date(), rand: string = randomHex()): string {
  const ext = IMAGE_EXT[mime];
  if (!ext) throw new Error("Only PNG, JPEG, GIF and WebP images can be added to notes.");
  const p = (n: number) => String(n).padStart(2, "0");
  const stamp = `${now.getFullYear()}${p(now.getMonth() + 1)}${p(now.getDate())}-${p(now.getHours())}${p(now.getMinutes())}${p(now.getSeconds())}`;
  return `img-${stamp}-${rand}.${ext}`;
}

/** Resolve `notes-assets/<file>` inside a meeting folder, or null when the path is not an asset. */
export function resolveAsset(root: string, id: string, rel: string): string | null {
  let dir: string;
  try {
    dir = meetingDir(root, id);
  } catch {
    return null;
  }
  const parts = rel.split("/");
  if (parts.length !== 2 || parts[0] !== ASSET_DIR) return null;
  const file = parts[1];
  if (!file || file.startsWith(".") || file.includes("\\")) return null;
  return path.join(dir, ASSET_DIR, file);
}

/** Parse `echo-asset://meeting/<id>/notes-assets/<file>`. */
export function parseAssetUrl(url: string): { id: string; rel: string } | null {
  let u: URL;
  try {
    u = new URL(url);
  } catch {
    return null;
  }
  if (u.protocol !== `${ASSET_SCHEME}:` || u.hostname !== "meeting") return null;
  const segs = u.pathname.split("/").filter(Boolean).map((s) => decodeURIComponent(s));
  if (segs.length !== 3) return null;
  return { id: segs[0], rel: `${segs[1]}/${segs[2]}` };
}

function randomHex(): string {
  return Math.floor(Math.random() * 0x10000)
    .toString(16)
    .padStart(4, "0");
}
