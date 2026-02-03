import { describe, it, expect, vi, beforeEach } from "vitest";
import { mount } from "@vue/test-utils";
import AttentionAnalysisView from "@/views/AttentionAnalysisView.vue";

vi.mock("vue-echarts", () => ({
  default: { name: "VChart", template: "<div />" },
}));

vi.mock("d3", () => ({
  forceSimulation: vi.fn(),
  forceLink: vi.fn(() => ({ id: vi.fn(), distance: vi.fn() })),
  forceManyBody: vi.fn(() => ({ strength: vi.fn() })),
  forceCenter: vi.fn(),
  forceCollide: vi.fn(() => ({ radius: vi.fn() })),
  select: vi.fn(() => ({
    select: vi.fn(() => ({ remove: vi.fn() })),
    append: vi.fn(() => ({ attr: vi.fn(() => ({ attr: vi.fn() })) })),
    selectAll: vi.fn(() => ({
      join: vi.fn(() => ({
        attr: vi.fn(() => ({ attr: vi.fn() })),
        call: vi.fn(),
        append: vi.fn(() => ({ attr: vi.fn() })),
        text: vi.fn(),
      })),
    })),
  })),
}));

vi.mock("vue-router", () => ({
  useRouter: () => ({ push: vi.fn() }),
}));

const mockHeatmap: { value: any } = { value: null };

vi.mock("@/composables/useAnalysis", () => ({
  useAnalysis: () => ({
    heatmap: mockHeatmap,
    network: { value: null },
    loading: { value: false },
    error: { value: "" },
    fetchHeatmap: vi.fn(),
    fetchNetwork: vi.fn(),
  }),
}));

vi.mock("@/composables/useTraining", () => ({
  useTraining: () => ({
    checkpoints: { value: [] },
    fetchCheckpoints: vi.fn(),
  }),
}));

const stubs = {
  PlButton: { template: "<button><slot /></button>" },
  PlSelect: { template: "<select />" },
  PlInput: { template: "<input />" },
  PlCard: { template: "<div><slot /></div>" },
  PlSpinner: { template: "<div />" },
  PlToast: { template: "<div />" },
  PlEmptyState: { template: "<div />" },
  PlPageHeader: { template: "<div />" },
  PlIcon: { template: "<i />" },
  PlProgressBar: { template: "<div />" },
};

function mountView() {
  return mount(AttentionAnalysisView, { global: { stubs } });
}

describe("AttentionAnalysisView", () => {
  beforeEach(() => {
    mockHeatmap.value = {
      model_id: "test-model",
      feature_names: ["A", "B", "C"],
      values: [
        [1.0, 0.3, 0.1],
        [0.8, 0.95, 0.6],
        [0.2, 0.9, 0.85],
      ],
      min_value: 0.1,
      max_value: 1.0,
    };
  });

  it("uses green-to-red 5-stop gradient for heatmap visualMap", () => {
    const wrapper = mountView();
    const option = (wrapper.vm as any).heatmapOption;
    expect(option.visualMap.inRange.color).toEqual([
      "#1aae39",
      "#8bc34a",
      "#f5d75e",
      "#ff9800",
      "#e03131",
    ]);
  });

  it("computes maxAttentionValue as global max of entire heatmap matrix", () => {
    const wrapper = mountView();
    expect((wrapper.vm as any).maxAttentionValue).toBe(1.0);
  });

  it("returns 1 as fallback when heatmap is null", () => {
    mockHeatmap.value = null;
    const wrapper = mountView();
    expect((wrapper.vm as any).maxAttentionValue).toBe(1);
  });
});
