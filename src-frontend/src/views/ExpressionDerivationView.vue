<template>
  <div class="expression-derivation">
    <h2>表达式推导</h2>
    <div class="toolbar">
      <select v-model="selectedCheckpoint" class="checkpoint-select">
        <option value="">选择检查点</option>
        <option v-for="cp in checkpoints" :key="cp.id" :value="cp.id">
          {{ cp.id }} (loss: {{ cp.final_loss.toFixed(4) }})
        </option>
      </select>
      <select v-model="selectedPreset" class="preset-select">
        <option value="quick">快速 (~1 min)</option>
        <option value="standard">标准 (~2-5 min)</option>
        <option value="thorough">深度 (~5-15 min)</option>
      </select>
      <button @click="generate" :disabled="!selectedCheckpoint || loading" class="btn-generate">
        生成表达式
      </button>
      <template v-if="expression">
        <button @click="simplify" :disabled="loading" class="btn-action">精简</button>
        <button @click="optimize" :disabled="loading" class="btn-action">
          优化<template v-if="expression?.pareto_count"> ({{ expression.pareto_index + 1 }}/{{ expression.pareto_count }})</template>
        </button>
        <button @click="undo" :disabled="loading || !canUndo" class="btn-action">撤销</button>
        <button @click="redo" :disabled="loading || !canRedo" class="btn-action">重做</button>
      </template>
    </div>
    <div v-if="loading" class="loading">{{ loadingMessage }}</div>
    <div v-else-if="error" class="error">{{ error }}</div>
    <template v-if="expression">
      <div class="indicators-grid" v-if="expression?.indicators && Object.keys(expression.indicators).length > 0">
        <div class="indicator" v-for="(value, key) in expression.indicators" :key="key">
          <span class="indicator-label">{{ formatIndicatorLabel(key as string) }}</span>
          <span class="indicator-value">{{ formatIndicatorValue(key as string, value) }}</span>
        </div>
      </div>
      <div class="panels">
        <div class="panel latex-panel">
          <h3>符号表达式</h3>
          <div ref="katexRef" class="katex-container"></div>
        </div>
        <div class="panel tree-panel">
          <h3>表达式树</h3>
          <div class="tree-container">
            <TreeNode :node="expression.tree" :depth="0" :variable-impact="expression.variable_impact" />
          </div>
        </div>
      </div>
      <div v-if="impactChartOption" class="panel impact-panel">
        <h3>成分影响力</h3>
        <VChart :option="impactChartOption" class="impact-chart" autoresize />
      </div>
    </template>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, watch, onMounted, onUnmounted, nextTick } from "vue";
import katex from "katex";
import { useTraining } from "@/composables/useTraining";
import { useExpression } from "@/composables/useExpression";
import { useWorkflow } from "@/composables/useWorkflow";
import { useDatasets } from "@/composables/useDatasets";
import TreeNode from "./TreeNode.vue";
import VChart from "vue-echarts";
import { use } from "echarts/core";
import { BarChart } from "echarts/charts";
import {
  GridComponent,
  TooltipComponent,
} from "echarts/components";
import { CanvasRenderer } from "echarts/renderers";

use([BarChart, GridComponent, TooltipComponent, CanvasRenderer]);

const selectedCheckpoint = ref("");
const selectedPreset = ref("standard");
const katexRef = ref<HTMLElement>();

const { checkpoints, fetchCheckpoints } = useTraining();
const { markCurrentExpression } = useWorkflow();
const {
  expression, history, loading, error,
  generateExpression, simplifyExpression, optimizeExpression,
  fetchHistory, undoExpression,
  stopPolling,
} = useExpression();
const { getDataset } = useDatasets();
const targetName = ref("");

const INDICATOR_LABELS: Record<string, string> = {
  train_r2: "R² (训练)",
  test_r2: "R² (测试)",
  train_mae: "MAE (训练)",
  test_mae: "MAE (测试)",
  train_mse: "MSE (训练)",
  test_mse: "MSE (测试)",
  train_nmse: "NMSE (训练)",
  test_nmse: "NMSE (测试)",
  train_rmse: "RMSE (训练)",
  test_rmse: "RMSE (测试)",
  depth: "模型深度",
  length: "模型长度",
};

function formatIndicatorLabel(key: string): string {
  return INDICATOR_LABELS[key] ?? key;
}

function formatIndicatorValue(key: string, value: number): string {
  if (key === "depth" || key === "length") return value.toFixed(0);
  return value.toFixed(4);
}

const canUndo = computed(() => history.value ? history.value.current_index > 0 : false);
const canRedo = computed(() => history.value ? history.value.current_index < history.value.history.length - 1 : false);
const loadingMessage = computed(() => expression.value ? "处理中..." : "符号回归可能需要数分钟，请耐心等待...");

const impactChartOption = computed(() => {
  const impact = expression.value?.variable_impact;
  if (!impact || Object.keys(impact).length === 0) return null;

  const sorted = Object.entries(impact)
    .filter(([, v]) => v > 0)
    .sort(([, a], [, b]) => a - b);

  return {
    tooltip: { trigger: "axis", axisPointer: { type: "shadow" } },
    grid: { left: 100, right: 40, top: 10, bottom: 20 },
    xAxis: { type: "value", max: 1, name: "影响力" },
    yAxis: {
      type: "category",
      data: sorted.map(([name]) => name),
      axisLabel: { fontSize: 12 },
    },
    series: [{
      type: "bar",
      data: sorted.map(([, value]) => value.toFixed(2)),
      itemStyle: { color: "#4caf50" },
      barMaxWidth: 24,
      label: {
        show: true,
        position: "right",
        formatter: "{c}",
        fontSize: 11,
      },
    }],
  };
});

async function generate() {
  if (!selectedCheckpoint.value) return;
  await generateExpression(selectedCheckpoint.value, 10, selectedPreset.value);
  if (expression.value) {
    await fetchHistory(expression.value.expr_id);
    markCurrentExpression(expression.value.expr_id);
  }
}
async function simplify() {
  if (!expression.value) return;
  const ok = await simplifyExpression(expression.value.expr_id);
  if (ok && expression.value) {
    await fetchHistory(expression.value.expr_id);
    markCurrentExpression(expression.value.expr_id);
  }
}
async function optimize() {
  if (!expression.value) return;
  const ok = await optimizeExpression(expression.value.expr_id);
  if (ok && expression.value) {
    await fetchHistory(expression.value.expr_id);
    markCurrentExpression(expression.value.expr_id);
  }
}
async function undo() {
  if (!expression.value) return;
  const ok = await undoExpression(expression.value.expr_id, 1);
  if (ok && expression.value) {
    await fetchHistory(expression.value.expr_id);
    markCurrentExpression(expression.value.expr_id);
  }
}
async function redo() {
  if (!expression.value) return;
  const ok = await undoExpression(expression.value.expr_id, -1);
  if (ok && expression.value) {
    await fetchHistory(expression.value.expr_id);
    markCurrentExpression(expression.value.expr_id);
  }
}

function renderLatex() {
  if (katexRef.value && expression.value?.latex) {
    const prefix = targetName.value ? `${targetName.value} = ` : "";
    katex.render(prefix + expression.value.latex, katexRef.value, {
      displayMode: true,
      throwOnError: false,
    });
  }
}

watch(() => expression.value?.latex, () => nextTick(renderLatex));
watch(targetName, () => nextTick(renderLatex));
watch(selectedCheckpoint, async (cpId) => {
  if (!cpId) {
    targetName.value = "";
    return;
  }
  const cp = checkpoints.value.find((c) => c.id === cpId);
  if (cp?.dataset_id) {
    try {
      const ds = await getDataset(cp.dataset_id);
      targetName.value = ds.target;
    } catch {
      targetName.value = "";
    }
  }
});
onMounted(() => { fetchCheckpoints(); });
onUnmounted(() => { stopPolling(); });
</script>

<style scoped>
.expression-derivation {
  max-width: 1200px;
  margin: 0 auto;
  padding: 20px;
  font-family: system-ui, sans-serif;
}
h2 { color: #2e7d32; }
.toolbar {
  display: flex;
  gap: 8px;
  align-items: center;
  margin-bottom: 16px;
  flex-wrap: wrap;
}
.checkpoint-select {
  padding: 6px 12px;
  border: 1px solid #ccc;
  border-radius: 4px;
}
.preset-select {
  padding: 6px 12px;
  border: 1px solid #ccc;
  border-radius: 4px;
}
.btn-generate, .btn-action {
  padding: 6px 16px;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
}
.btn-generate { background: #4caf50; }
.btn-action { background: #2196f3; }
.btn-generate:disabled, .btn-action:disabled {
  background: #ccc;
  cursor: not-allowed;
}
.indicators-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
  gap: 8px;
  margin-bottom: 16px;
}
.indicator {
  background: #e8f5e9;
  padding: 6px 12px;
  border-radius: 4px;
  font-size: 13px;
  display: flex;
  flex-direction: column;
  gap: 2px;
}
.indicator-label {
  color: #666;
  font-size: 11px;
}
.indicator-value {
  font-weight: 600;
  color: #2e7d32;
}
.panels {
  display: flex;
  gap: 20px;
}
.panel {
  flex: 1;
  border: 1px solid #e0e0e0;
  border-radius: 8px;
  padding: 16px;
}
.panel h3 { margin-top: 0; color: #333; }
.katex-container {
  min-height: 80px;
  display: flex;
  align-items: center;
  justify-content: center;
  overflow-x: auto;
}
.tree-container {
  max-height: 500px;
  overflow-y: auto;
  font-family: monospace;
  font-size: 14px;
}
.loading { color: #666; }
.error { color: #f44336; }
.impact-panel { margin-top: 20px; }
.impact-chart { height: 240px; width: 100%; }
</style>
