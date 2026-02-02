import { reactive, computed, watch } from "vue";
import { apiClient } from "./useApi";
import type { DatasetMeta } from "@/types/dataset";
import type { CheckpointInfo } from "@/types/training";

const STORAGE_KEY = "pharmalink-currentExprId";

export type TabName =
  | "data-import"
  | "training"
  | "analysis"
  | "expression"
  | "evaluation"
  | "formulation";

export const TAB_ORDER: TabName[] = [
  "data-import",
  "training",
  "analysis",
  "expression",
  "evaluation",
  "formulation",
];

export const TAB_LABELS: Record<TabName, string> = {
  "data-import": "数据导入",
  training: "模型训练",
  analysis: "注意力分析",
  expression: "表达式推导",
  evaluation: "模型评估",
  formulation: "最优配比",
};

const state = reactive({
  datasetsAvailable: false,
  targetSelected: false,
  hasCheckpoint: false,
  currentExprId: localStorage.getItem(STORAGE_KEY) ?? "",
});

watch(
  () => state.currentExprId,
  (id) => {
    if (id) {
      localStorage.setItem(STORAGE_KEY, id);
    } else {
      localStorage.removeItem(STORAGE_KEY);
    }
  }
);

export function useWorkflow() {
  const tabEnabled = computed<Record<TabName, boolean>>(() => ({
    "data-import": true,
    training: state.datasetsAvailable && state.targetSelected,
    analysis: state.hasCheckpoint,
    expression: state.hasCheckpoint,
    evaluation: state.hasCheckpoint,
    formulation: state.hasCheckpoint,
  }));

  function markDatasetsAvailable() {
    state.datasetsAvailable = true;
  }

  function markHasCheckpoint() {
    state.hasCheckpoint = true;
  }

  function markCurrentExpression(exprId: string) {
    state.currentExprId = exprId;
  }

  function markTargetSelected(selected: boolean) {
    state.targetSelected = selected;
  }

  async function init() {
    const results = await Promise.allSettled([
      apiClient.get<DatasetMeta[]>("/datasets"),
      apiClient.get<CheckpointInfo[]>("/models/checkpoints"),
    ]);
    if (
      results[0].status === "fulfilled" &&
      results[0].value.data.length > 0
    ) {
      state.datasetsAvailable = true;
    }
    if (
      results[1].status === "fulfilled" &&
      results[1].value.data.length > 0
    ) {
      state.hasCheckpoint = true;
    }
    await validateStoredExprId();
  }

  async function validateStoredExprId() {
    if (!state.currentExprId) return;
    try {
      await apiClient.get(`/expressions/${state.currentExprId}`);
    } catch {
      state.currentExprId = "";
    }
  }

  return {
    state,
    tabEnabled,
    markDatasetsAvailable,
    markHasCheckpoint,
    markCurrentExpression,
    markTargetSelected,
    init,
  };
}
