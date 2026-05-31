import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { useLiveClock } from "@/composables/useLiveClock";

describe("useLiveClock", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-05-24T21:40:30"));
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("formats time as YYYY/MM/DD 星期X HH:MM:SS", () => {
    const { timeString } = useLiveClock();
    expect(timeString.value).toBe("2026/05/24 星期日 21:40:30");
  });

  it("shows correct weekday for Monday", () => {
    vi.setSystemTime(new Date("2026-05-25T08:00:00"));
    const { timeString } = useLiveClock();
    expect(timeString.value).toBe("2026/05/25 星期一 08:00:00");
  });

  it("shows correct weekday for Saturday", () => {
    vi.setSystemTime(new Date("2026-05-30T12:30:45"));
    const { timeString } = useLiveClock();
    expect(timeString.value).toBe("2026/05/30 星期六 12:30:45");
  });

  it("updates every second", () => {
    const { timeString } = useLiveClock();
    expect(timeString.value).toBe("2026/05/24 星期日 21:40:30");

    vi.advanceTimersByTime(1000);
    expect(timeString.value).toBe("2026/05/24 星期日 21:40:31");

    vi.advanceTimersByTime(1000);
    expect(timeString.value).toBe("2026/05/24 星期日 21:40:32");
  });

  it("stop() freezes the time", () => {
    const { timeString, stop } = useLiveClock();
    expect(timeString.value).toBe("2026/05/24 星期日 21:40:30");

    stop();
    vi.advanceTimersByTime(5000);
    expect(timeString.value).toBe("2026/05/24 星期日 21:40:30");
  });
});
