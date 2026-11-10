import { describe, expect, it } from "vitest";
import { localInputToIso, localIsoMinute, toLocalInput } from "./time";

describe("meeting time helpers", () => {
  const d = new Date(2026, 10, 9, 10, 3, 47);
  it("floors to the minute in local time with an offset", () => {
    expect(toLocalInput(d)).toBe("2026-11-09T10:03");
    const iso = localIsoMinute(d);
    expect(iso).toMatch(/^2026-11-09T10:03:00[+-]\d{2}:\d{2}$/);
    expect(new Date(iso).getTime()).toBe(new Date(2026, 10, 9, 10, 3).getTime());
  });
  it("converts datetime-local values", () => {
    expect(localInputToIso("2026-11-09T10:03")).toBe(localIsoMinute(d));
    expect(localInputToIso("")).toBeNull();
    expect(localInputToIso("nope")).toBeNull();
  });
});
