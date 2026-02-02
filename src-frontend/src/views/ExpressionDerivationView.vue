<template>
  <div class="expression-derivation">
    <PlPageHeader
      :step="4"
      title="表达式推导"
      subtitle="基于符号回归推导数学表达式"
    />

    <div class="expression-derivation__grid">
      <!-- Left panel (1fr): Regression config -->
      <PlCard variant="base" padding="md" class="expression-derivation__left">
        <h3 class="card-title">回归配置</h3>

        <PlSelect
          :model-value="selectedCheckpoint"
          :options="checkpointOptions"
          label="检查点"
          @update:model-value="selectedCheckpoint = $event"
        />

        <PlSelect
          :model-value="selectedPreset"
          :options="presetOptions"
          label="预设"
          @update:model-value="selectedPreset = $event"
        />

        <PlButton
          variant="primary"
          :disabled="!selectedCheckpoint || loading"
          @click="generate"
        >
          生成表达式
        </PlButton>

        <template v-if="expression">
          <div class="action-group">
            <PlButton variant="secondary" :disabled="loading" @click="simplify">
              精简
            </PlButton>
            <PlButton variant="secondary" :disabled="loading" @click="optimize">
              优化<template v-if="expression?.pareto_count"> ({{ expression.pareto_index + 1 }}/{{ expression.pareto_count }})</template>
            </PlButton>
          </div>
          <div class="action-group">
            <PlButton variant="ghost" :disabled="loading || !canUndo" @click="undo">
              撤销
            </PlButton>
            <PlButton variant="ghost" :disabled="loading || !canRedo" @click="redo">
              重做
            </PlButton>
          </div>
        </template>
      </PlCard>

      <!-- Right panel (2fr): Results -->
      <div class="expression-derivation__right">
        <!-- Loading / Error -->
        <div v-if="loading" class="status-area">
          <PlSpinner size="sm" />
          <span class="status-text">{{ loadingMessage }}</span>
        </div>
        <PlToast v-if="error" :message="error" variant="error" />

        <template v-if="expression">
          <!-- Indicators grid -->
          <div
            v-if="expression?.indicators && Object.keys(expression.indicators).length > 0"
            class="indicators-grid"
          >
            <PlStatCard
              v-for="(value, key) in expression.indicators"
              :key="key"
              :label="formatIndicatorLabel(key as string)"
              :value="formatIndicatorValue(key as string, value)"
              :variant="indicatorVariant(key as string)"
            />
          </div>

          <!-- KaTeX expression card -->
          <PlCard variant="base" padding="md" class="latex-card">
            <h3 class="card-title">数学表达式</h3>
            <div ref="katexRef" class="katex-container"></div>
          </PlCard>

          <!-- Bottom row: expression tree + variable impact chart -->
          <div class="bottom-row">
            <PlCard variant="base" padding="md" class="tree-card">
              <h3 class="card-title">表达式树</h3>
              <div class="tree-container">
                <TreeNode :node="expression.tree" :depth="0" :variable-impact="expression.variable_impact" />
              </div>
            </PlCard>

            <PlCard v-if="impactChartOption" variant="base" padding="md" class="impact-card">
              <h3 class="card-title">影响力</h3>
              <VChart :option="impactChartOption" theme="pharmalink" class="impact-chart" autoresize />
            </PlCard>
          </div>
        </template>
      </div>
    </div>

    <!-- Footer: prev/next navigation -->
    <div class="expression-derivation__footer">
      <PlButton variant="secondary" @click="goPrev">
        <PlIcon name="arrow-left" size="sm" />
        上一步：注意力分析
      </PlButton>
      <PlButton variant="dark" @click="goNext">
        下一步：模型评估
        <PlIcon name="arrow-right" size="sm" />
      </PlButton>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, watch, onMounted, onUnmounted, nextTick } from "vue";
import { useRouter } from "vue-router";
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
import PlButton from "@/components/PlButton.vue";
import PlSelect from "@/components/PlSelect.vue";
import PlCard from "@/components/PlCard.vue";
import PlStatCard from "@/components/PlStatCard.vue";
import PlSpinner from "@/components/PlSpinner.vue";
import PlToast from "@/components/PlToast.vue";
import PlPageHeader from "@/components/PlPageHeader.vue";
import PlIcon from "@/components/PlIcon.vue";
import { CHART_PALETTE } from "@/utils/chart-palette";

use([BarChart, GridComponent, TooltipComponent, CanvasRenderer]);

const router = useRouter();

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
const featureNames = ref<string[]>([]);

const checkpointOptions = computed(() => [
  { value: "", label: "选择检查点" },
  ...checkpoints.value.map((cp) => ({
    value: cp.id,
    label: `${cp.id} (loss: ${cp.final_loss.toFixed(4)})`,
  })),
]);

const presetOptions = [
  { value: "quick", label: "快速 (~1 min)" },
  { value: "standard", label: "标准 (~2-5 min)" },
  { value: "thorough", label: "深度 (~5-15 min)" },
];

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

function indicatorVariant(key: string): "mint" | "lavender" | "sky" {
  if (key.startsWith("train_")) return "mint";
  if (key.startsWith("test_")) return "lavender";
  return "sky";
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
      itemStyle: { color: CHART_PALETTE[0] },
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

function replaceFeatureVariables(latex: string): string {
  if (featureNames.value.length === 0) return latex;
  return latex.replace(/X(\d+)/g, (_, i) => {
    const idx = parseInt(i, 10);
    return featureNames.value[idx] ?? ("X" + i);
  });
}

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
    const substituted = replaceFeatureVariables(expression.value.latex);
    const prefix = targetName.value ? `${targetName.value} = ` : "";
    katex.render(prefix + substituted, katexRef.value, {
      displayMode: true,
      throwOnError: false,
    });
  }
}

function goPrev() {
  router.push({ name: "analysis" });
}

function goNext() {
  router.push({ name: "evaluation" });
}

watch(() => expression.value?.latex, () => nextTick(renderLatex));
watch(targetName, () => nextTick(renderLatex));
watch(featureNames, () => nextTick(renderLatex));
watch(selectedCheckpoint, async (cpId) => {
  if (!cpId) {
    targetName.value = "";
    featureNames.value = [];
    return;
  }
  const cp = checkpoints.value.find((c) => c.id === cpId);
  if (cp?.dataset_id) {
    try {
      const ds = await getDataset(cp.dataset_id);
      targetName.value = ds.target;
      featureNames.value = ds.feature_names ?? [];
    } catch {
      targetName.value = "";
      featureNames.value = [];
    }
  }
});
onMounted(() => { fetchCheckpoints(); });
onUnmounted(() => { stopPolling(); });
</script>

<style scoped>
.expression-derivation {
  max-width: var(--content-max-width);
  margin: 0 auto;
  padding: var(--space-10);
  display: flex;
  flex-direction: column;
  gap: var(--space-6);
  font-family: var(--font-family);
}

/* -- Grid: 1fr config + 2fr results -- */
.expression-derivation__grid {
  display: grid;
  grid-template-columns: 1fr 2fr;
  gap: var(--space-6);
}

@media (max-width: 960px) {
  .expression-derivation__grid {
    grid-template-columns: 1fr;
  }
}

/* -- Left panel: Config card -- */
.expression-derivation__left {
  display: flex;
  flex-direction: column;
  gap: var(--space-4);
}

.card-title {
  font-size: var(--text-h3);
  font-weight: var(--text-h3-weight);
  color: var(--color-charcoal);
  margin: 0 0 var(--space-2);
}

.action-group {
  display: flex;
  gap: var(--space-2);
}

/* -- Right panel: Results -- */
.expression-derivation__right {
  display: flex;
  flex-direction: column;
  gap: var(--space-4);
}

.status-area {
  display: flex;
  align-items: center;
  gap: var(--space-2);
}

.status-text {
  font-size: var(--text-sm);
  color: var(--color-steel);
}

/* -- Indicators grid -- */
.indicators-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
  gap: var(--space-3);
}

/* -- KaTeX expression card -- */
.latex-card {
  min-height: 0;
}

.katex-container {
  min-height: 80px;
  display: flex;
  align-items: center;
  justify-content: center;
  overflow-x: auto;
  padding: var(--space-3) 0;
}

/* -- Bottom row: tree + impact chart (2-column) -- */
.bottom-row {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: var(--space-4);
}

@media (max-width: 1200px) {
  .bottom-row {
    grid-template-columns: 1fr;
  }
}

.tree-card {
  min-height: 0;
}

.tree-container {
  max-height: 400px;
  overflow-y: auto;
  font-family: monospace;
  font-size: var(--text-sm);
}

.impact-card {
  min-height: 0;
}

.impact-chart {
  height: 260px;
  width: 100%;
}

/* -- Footer: prev/next navigation -- */
.expression-derivation__footer {
  display: flex;
  justify-content: space-between;
  padding-top: var(--space-4);
  border-top: 1px solid var(--color-hairline);
}
</style>
