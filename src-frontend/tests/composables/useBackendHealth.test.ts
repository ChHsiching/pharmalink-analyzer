import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { useBackendHealth } from "@/composables/useBackendHealth";
import { useApi } from "@/composables/useApi";

vi.mock("@/composables/useApi", () => ({
  useApi: vi.fn(),
}));

describe("useBackendHealth", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("starts disconnected", () => {
    const mockCheckHealth = vi.fn().mockRejectedValue(new Error("fail"));
    vi.mocked(useApi).mockReturnValue({
      checkHealth: mockCheckHealth,
    } as ReturnType<typeof useApi>);

    const { connected } = useBackendHealth();
    expect(connected.value).toBe(false);
  });

  it("becomes connected after successful health check", async () => {
    const mockCheckHealth = vi.fn().mockResolvedValue({ status: "ok" });
    vi.mocked(useApi).mockReturnValue({
      checkHealth: mockCheckHealth,
    } as ReturnType<typeof useApi>);

    const { connected, start } = useBackendHealth();
    start();

    await vi.advanceTimersByTimeAsync(0);
    expect(connected.value).toBe(true);
    expect(mockCheckHealth).toHaveBeenCalledTimes(1);
  });

  it("polls every 5 seconds", async () => {
    const mockCheckHealth = vi.fn().mockResolvedValue({ status: "ok" });
    vi.mocked(useApi).mockReturnValue({
      checkHealth: mockCheckHealth,
    } as ReturnType<typeof useApi>);

    const { start } = useBackendHealth();
    start();

    await vi.advanceTimersByTimeAsync(0);
    expect(mockCheckHealth).toHaveBeenCalledTimes(1);

    await vi.advanceTimersByTimeAsync(5000);
    expect(mockCheckHealth).toHaveBeenCalledTimes(2);

    await vi.advanceTimersByTimeAsync(5000);
    expect(mockCheckHealth).toHaveBeenCalledTimes(3);
  });

  it("sets disconnected when health check fails", async () => {
    const mockCheckHealth = vi
      .fn()
      .mockResolvedValueOnce({ status: "ok" })
      .mockRejectedValueOnce(new Error("fail"));
    vi.mocked(useApi).mockReturnValue({
      checkHealth: mockCheckHealth,
    } as ReturnType<typeof useApi>);

    const { connected, start } = useBackendHealth();
    start();

    await vi.advanceTimersByTimeAsync(0);
    expect(connected.value).toBe(true);

    await vi.advanceTimersByTimeAsync(5000);
    expect(connected.value).toBe(false);
  });

  it("stop() clears the interval", async () => {
    const mockCheckHealth = vi.fn().mockResolvedValue({ status: "ok" });
    vi.mocked(useApi).mockReturnValue({
      checkHealth: mockCheckHealth,
    } as ReturnType<typeof useApi>);

    const { start, stop } = useBackendHealth();
    start();

    await vi.advanceTimersByTimeAsync(0);
    expect(mockCheckHealth).toHaveBeenCalledTimes(1);

    stop();

    await vi.advanceTimersByTimeAsync(5000);
    expect(mockCheckHealth).toHaveBeenCalledTimes(1);
  });
});
