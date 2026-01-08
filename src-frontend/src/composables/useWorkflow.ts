import { reactive, computed } from "vue";

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

  function resetDownstream() {
    state.hasCheckpoint = false;
  }

  return {
    state,
    tabEnabled,
    markDatasetsAvailable,
    markHasCheckpoint,
    resetDownstream,
  };
}
