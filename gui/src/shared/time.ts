/** Meeting-time helpers shared by main and renderer (local wall-clock time, minute precision). */

const pad = (n: number): string => String(n).padStart(2, "0");

/** `YYYY-MM-DDTHH:MM` in local time, the value format of `<input type="datetime-local">`. */
export function toLocalInput(d: Date): string {
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

/** ISO 8601 floored to the minute with the local UTC offset, e.g. `2026-11-09T10:03:00+09:00`. */
export function localIsoMinute(d: Date): string {
  const off = -d.getTimezoneOffset();
  const sign = off >= 0 ? "+" : "-";
  const abs = Math.abs(off);
  return `${toLocalInput(d)}:00${sign}${pad(Math.floor(abs / 60))}:${pad(abs % 60)}`;
}

/** `datetime-local` value (local time) to ISO with offset; null when empty or invalid. */
export function localInputToIso(value: string): string | null {
  if (!/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}/.test(value)) return null;
  const d = new Date(value);
  return Number.isNaN(d.getTime()) ? null : localIsoMinute(d);
}
