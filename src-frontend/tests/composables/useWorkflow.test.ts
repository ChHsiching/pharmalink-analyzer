import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";

vi.mock("@/composables/useApi", () => ({
  apiClient: {
    get: vi.fn(),
  },
}));

import { useWorkflow, type TabName } from "@/composables/useWorkflow";

describe("useWorkflow tabEnabled", () => {
  const allTabs: TabName[] = [
    "data-import",
    "training",
    "analysis",
    "expression",
    "evaluation",
    "formulation",
  ];

  beforeEach(() => {
    localStorage.clear();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("unlocks formulation when hasCheckpoint is true, even without an expression", () => {
    const { state, tabEnabled } = useWorkflow();
    state.hasCheckpoint = true;
    state.currentExprId = "";

    expect(tabEnabled.value.formulation).toBe(true);
  });

  it("locks formulation when hasCheckpoint is false", () => {
    const { state, tabEnabled } = useWorkflow();
    state.hasCheckpoint = false;
    state.currentExprId = "some-id";

    expect(tabEnabled.value.formulation).toBe(false);
  });

  it("all pages except data-import are locked by default", () => {
    const { state, tabEnabled } = useWorkflow();
    state.datasetsAvailable = false;
    state.targetSelected = false;
    state.hasCheckpoint = false;
    state.currentExprId = "";

    expect(tabEnabled.value["data-import"]).toBe(true);
    for (const tab of allTabs.slice(1)) {
      expect(tabEnabled.value[tab]).toBe(false);
    }
  });

  it("training requires both datasetsAvailable and targetSelected", () => {
    const { state, tabEnabled } = useWorkflow();
    state.datasetsAvailable = false;
    state.targetSelected = false;
    expect(tabEnabled.value.training).toBe(false);

    state.datasetsAvailable = true;
    expect(tabEnabled.value.training).toBe(false);

    state.targetSelected = true;
    expect(tabEnabled.value.training).toBe(true);
  });

  it("analysis, expression, evaluation unlock with hasCheckpoint only", () => {
    const { state, tabEnabled } = useWorkflow();
    state.hasCheckpoint = true;

    expect(tabEnabled.value.analysis).toBe(true);
    expect(tabEnabled.value.expression).toBe(true);
    expect(tabEnabled.value.evaluation).toBe(true);
  });
});
