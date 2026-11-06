// Generates app icons from ../echo-icon.svg: assets/icons/{mac/icon.icns, win/icon.ico, png/*.png}.
// Usage: npm run icons
import { Resvg } from "@resvg/resvg-js";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const here = path.dirname(fileURLToPath(import.meta.url));
const gui = path.resolve(here, "..");
const source = path.resolve(gui, "../echo-icon.svg");
const out = path.join(gui, "assets/icons");

const svg = fs.readFileSync(source, "utf-8");
const inner = svg.replace(/<\?xml[^>]*>/, "").trim();

// macOS icons sit on Apple's 1024 grid with an 824 px body, so the Dock shows them at the same
// visual size as other apps. Windows / Linux icons use the full canvas.
const macSvg = `<svg xmlns="http://www.w3.org/2000/svg" width="1024" height="1024" viewBox="0 0 1024 1024">
  <svg x="100" y="100" width="824" height="824" viewBox="0 0 512 512">${inner}</svg>
</svg>`;

function render(markup, size) {
  return new Resvg(markup, { fitTo: { mode: "width", value: size } }).render().asPng();
}

// ICNS: big-endian container of PNG-encoded entries (supported since macOS 10.7).
function icns(entries) {
  const chunks = entries.map(([type, png]) => {
    const head = Buffer.alloc(8);
    head.write(type, 0, "ascii");
    head.writeUInt32BE(png.length + 8, 4);
    return Buffer.concat([head, png]);
  });
  const body = Buffer.concat(chunks);
  const head = Buffer.alloc(8);
  head.write("icns", 0, "ascii");
  head.writeUInt32BE(body.length + 8, 4);
  return Buffer.concat([head, body]);
}

// ICO: little-endian directory of PNG-encoded images (Windows Vista+).
function ico(images) {
  const header = Buffer.alloc(6);
  header.writeUInt16LE(0, 0);
  header.writeUInt16LE(1, 2);
  header.writeUInt16LE(images.length, 4);
  let offset = 6 + 16 * images.length;
  const dir = images.map(([size, png]) => {
    const e = Buffer.alloc(16);
    e.writeUInt8(size >= 256 ? 0 : size, 0);
    e.writeUInt8(size >= 256 ? 0 : size, 1);
    e.writeUInt16LE(1, 4);
    e.writeUInt16LE(32, 6);
    e.writeUInt32LE(png.length, 8);
    e.writeUInt32LE(offset, 12);
    offset += png.length;
    return e;
  });
  return Buffer.concat([header, ...dir, ...images.map(([, png]) => png)]);
}

for (const dir of ["mac", "win", "png"]) fs.mkdirSync(path.join(out, dir), { recursive: true });

for (const size of [16, 24, 32, 48, 64, 128, 256, 512, 1024]) {
  fs.writeFileSync(path.join(out, "png", `${size}x${size}.png`), render(svg, size));
}

const mac = (size) => render(macSvg, size);
fs.writeFileSync(
  path.join(out, "mac", "icon.icns"),
  icns([
    ["icp4", mac(16)],
    ["icp5", mac(32)],
    ["ic11", mac(32)],
    ["icp6", mac(64)],
    ["ic07", mac(128)],
    ["ic12", mac(64)],
    ["ic08", mac(256)],
    ["ic13", mac(256)],
    ["ic09", mac(512)],
    ["ic14", mac(512)],
    ["ic10", mac(1024)],
  ]),
);
fs.writeFileSync(path.join(out, "mac", "dock.png"), mac(512));
fs.writeFileSync(path.join(out, "win", "icon.ico"), ico([16, 24, 32, 48, 64, 128, 256].map((s) => [s, render(svg, s)])));

// Menu bar: macOS "template" images (black + alpha; the system tints them for light/dark menu bars).
// Only the arcs, cropped and with a heavier stroke so they stay legible at 18 pt.
const arcs = [...svg.matchAll(/<path\b[^>]*\/>/g)].map((m) => m[0]).join("");
const traySvg = (recording) => `<svg xmlns="http://www.w3.org/2000/svg" viewBox="96 96 320 320">
  <g fill="none" stroke="#000" stroke-width="26" stroke-linecap="round">${arcs}</g>
  ${recording ? '<circle cx="372" cy="372" r="40" fill="#000"/>' : ""}
</svg>`;
fs.mkdirSync(path.join(out, "tray"), { recursive: true });
for (const [name, recording] of [["trayTemplate", false], ["trayRecordingTemplate", true]]) {
  fs.writeFileSync(path.join(out, "tray", `${name}.png`), render(traySvg(recording), 18));
  fs.writeFileSync(path.join(out, "tray", `${name}@2x.png`), render(traySvg(recording), 36));
}

console.log(`Icons written to ${path.relative(process.cwd(), out) || out}`);
