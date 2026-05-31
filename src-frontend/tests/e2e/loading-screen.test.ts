/**
 * E2E test: Loading screen → backend ready → main UI transition.
 *
 * Tests App.vue component rendering with mocked API calls.
 * Error-exhaustion behavior is covered by useBackendReady.test.ts at the composable level.
 *
 * Run: cd src-frontend && npx vitest run tests/e2e/loading-screen.test.ts
 */
import { describe, it, expect, vi, beforeEach } from "vitest";
import { mount } from "@vue/test-utils";
import { ref, nextTick, reactive } from "vue";
import App from "@/App.vue";
import { useApi } from "@/composables/useApi";

vi.mock("@/composables/useApi", () => ({
  useApi: vi.fn(),
}));

vi.mock("@/composables/useWorkflow", () => ({
  useWorkflow: () => ({
    init: vi.fn().mockResolvedValue(undefined),
    state: reactive({ datasetsAvailable: false, hasCheckpoint: false, currentExprId: "" }),
    tabEnabled: ref({ "data-import": true }),
    markDatasetsAvailable: vi.fn(),
    markHasCheckpoint: vi.fn(),
    markCurrentExpression: vi.fn(),
  }),
}));

describe("Loading Screen E2E", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("shows loading screen when backend is not ready", async () => {
    const mockCheckHealth = vi.fn().mockRejectedValue(new Error("Network error"));
    vi.mocked(useApi).mockReturnValue({
      checkHealth: mockCheckHealth,
    } as ReturnType<typeof useApi>);

    const wrapper = mount(App, {
      global: {
        stubs: { RouterView: true, TabBar: true },
      },
    });

    await nextTick();
    await nextTick();

    const html = wrapper.html();
    expect(html).toContain("正在启动后端服务");
    expect(html).toContain("PharmaLink Analyzer");
    expect(wrapper.find(".loading-spinner").exists()).toBe(true);
    expect(wrapper.find(".loading-screen").exists()).toBe(true);
  });

  it("transitions to main UI after backend becomes ready", async () => {
    const mockCheckHealth = vi.fn().mockResolvedValue({ status: "ok" });
    vi.mocked(useApi).mockReturnValue({
      checkHealth: mockCheckHealth,
    } as ReturnType<typeof useApi>);

    const wrapper = mount(App, {
      global: {
        stubs: { RouterView: true, TabBar: true },
      },
    });

    await nextTick();
    await vi.waitFor(() => {
      expect(wrapper.find(".loading-screen").exists()).toBe(false);
    }, { timeout: 5000 });

    expect(wrapper.findComponent({ name: "TabBar" }).exists()).toBe(true);
  });
});
