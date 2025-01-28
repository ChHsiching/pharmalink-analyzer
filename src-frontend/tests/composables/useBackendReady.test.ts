import { describe, it, expect, vi, beforeEach } from "vitest";
import { useBackendReady } from "@/composables/useBackendReady";
import { useApi } from "@/composables/useApi";

vi.mock("@/composables/useApi", () => ({
  useApi: vi.fn(),
}));

describe("useBackendReady", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("starts not ready and becomes ready after successful health check", async () => {
    const mockCheckHealth = vi.fn().mockResolvedValue({ status: "ok" });
    vi.mocked(useApi).mockReturnValue({
      checkHealth: mockCheckHealth,
    } as ReturnType<typeof useApi>);

    const { ready, error, startPolling } = useBackendReady();
    expect(ready.value).toBe(false);

    const promise = startPolling();
    await promise;

    expect(ready.value).toBe(true);
    expect(error.value).toBe("");
    expect(mockCheckHealth).toHaveBeenCalled();
  });

  it("retries on failure and eventually succeeds", async () => {
    const mockCheckHealth = vi
      .fn()
      .mockRejectedValueOnce(new Error("Network error"))
      .mockResolvedValue({ status: "ok" });
    vi.mocked(useApi).mockReturnValue({
      checkHealth: mockCheckHealth,
    } as ReturnType<typeof useApi>);

    const { ready, startPolling } = useBackendReady({ intervalMs: 10, maxAttempts: 5 });
    await startPolling();

    expect(ready.value).toBe(true);
    expect(mockCheckHealth).toHaveBeenCalledTimes(2);
  });

  it("sets error after max attempts exhausted", async () => {
    const mockCheckHealth = vi.fn().mockRejectedValue(new Error("Network error"));
    vi.mocked(useApi).mockReturnValue({
      checkHealth: mockCheckHealth,
    } as ReturnType<typeof useApi>);

    const { ready, error, startPolling } = useBackendReady({ intervalMs: 10, maxAttempts: 2 });
    await startPolling();

    expect(ready.value).toBe(false);
    expect(error.value).toContain("Network error");
    expect(mockCheckHealth).toHaveBeenCalledTimes(2);
  });
});
