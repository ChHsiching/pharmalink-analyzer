import { reactive, computed } from "vue";
import { apiClient } from "./useApi";
import type { DatasetMeta } from "@/types/dataset";
import type { CheckpointInfo } from "@/types/training";

export type TabName =
  | "data-import"
  | "training"
  | "analysis"
  | "expression"
  | "evaluation";

export const TAB_ORDER: TabName[] = [
  "data-import",
  "training",
  "analysis",
  "expression",
  "evaluation",
];

export const TAB_LABELS: Record<TabName, string> = {
  "data-import": "数据导入",
  training: "模型训练",
  analysis: "注意力分析",
  expression: "表达式推导",
  evaluation: "模型评估",
};

const state = reactive({
  datasetsAvailable: false,
  hasCheckpoint: false,
});

export function useWorkflow() {
  const tabEnabled = computed<Record<TabName, boolean>>(() => ({
    "data-import": true,
    training: state.datasetsAvailable,
    analysis: state.hasCheckpoint,
    expression: state.hasCheckpoint,
    evaluation: state.hasCheckpoint,
  }));

  function markDatasetsAvailable() {
    state.datasetsAvailable = true;
  }

  function markHasCheckpoint() {
    state.hasCheckpoint = true;
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
  }

  return {
    state,
    tabEnabled,
    markDatasetsAvailable,
    markHasCheckpoint,
    init,
  };
}
