import { describe, expect, it } from "vitest";
import { formatDuration, meetingTime } from "./MeetingsView";

describe("meeting list formatting", () => {
  it("formats durations", () => {
    expect(formatDuration(3900)).toBe("1 h 05 min");
    expect(formatDuration(720)).toBe("12 min");
    expect(formatDuration(45)).toBe("45 s");
    expect(formatDuration(null)).toBe("");
  });
  it("prefers the meeting time over the start time", () => {
    expect(meetingTime({ meeting_time: "2026-11-09T09:58:00+09:00", started_at: "2026-11-09T10:03:12+09:00" })).toBe(
      "2026-11-09T09:58:00+09:00",
    );
    expect(meetingTime({ started_at: "2026-11-09T10:03:12+09:00" })).toBe("2026-11-09T10:03:12+09:00");
  });
});
