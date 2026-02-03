<template>
  <div class="formulation">
    <PlPageHeader
      :step="6"
      title="最优配比"
      subtitle="基于注意力权重与表达式寻找最优成分组合"
    />

    <!-- Config toolbar -->
    <PlCard variant="base" padding="md" class="formulation__toolbar-card">
      <div class="toolbar">
        <PlSelect
          :model-value="selectedCheckpoint"
          :options="checkpointOptions"
          label="检查点"
          @update:model-value="handleCheckpointChange"
        />
        <PlSelect
          v-if="expressionOptions.length"
          :model-value="selectedExprId"
          :options="expressionOptions"
          label="表达式"
          @update:model-value="handleExpressionChange"
        />
        <PlInput
          :model-value="topK"
          type="number"
          label="Top K"
          @update:model-value="topK = Number($event)"
        />
        <PlInput
          :model-value="nSamples"
          type="number"
          label="采样数"
          @update:model-value="nSamples = Number($event)"
        />
        <PlButton
          variant="primary"
          :disabled="!selectedExprId || loading"
          @click="runFormulation"
        >
          开始寻优
        </PlButton>
      </div>
    </PlCard>

    <template v-if="selectedExprId">
      <PlSpinner v-if="loading" size="md" />

      <PlToast v-if="error" :message="error" variant="error" />

      <div v-if="result?.attention_weak" class="warning-banner">
        注意力权重过于均匀，结果已通过 softmax 增强。建议增加训练层数或训练轮次以获得更显著的注意力分布。
      </div>

      <template v-if="result">
        <!-- 2-col grid: Radar chart + Candidates table -->
        <div class="formulation__grid">
          <PlCard variant="base" padding="md" class="formulation__radar-card">
            <h3 class="card-title">候选配比对比</h3>
            <v-chart
              class="chart"
              :option="radarOption"
              theme="pharmalink"
              autoresize
            />
          </PlCard>

          <PlCard variant="base" padding="md" class="formulation__table-card">
            <div class="card-header">
              <h3 class="card-title">候选配比</h3>
              <PlButton variant="secondary" @click="exportCsv">
                <PlIcon name="download" size="sm" />
                导出 CSV
              </PlButton>
            </div>
            <div class="table-wrapper">
              <table>
                <thead>
                  <tr>
                    <th>排名</th>
                    <th v-for="f in result.feature_names" :key="f">{{ f }}</th>
                    <th>预测药效指标值</th>
                  </tr>
                </thead>
                <tbody>
                  <tr
                    v-for="c in result.candidates"
                    :key="c.rank"
                    :class="{ highlight: c.rank === 1 }"
                  >
                    <td>
                      <PlBadge
                        v-if="c.rank === 1"
                        text="Best"
                        variant="teal"
                      />
                      <template v-else>{{ c.rank }}</template>
                    </td>
                    <td v-for="f in result.feature_names" :key="f">
                      {{ c.components[f].toFixed(4) }}
                    </td>
                    <td class="td--response">
                      {{ c.predicted_response.toFixed(4) }}
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </PlCard>
        </div>

        <!-- Full-width: Attention weights bar chart -->
        <PlCard variant="base" padding="md" class="formulation__attention-card">
          <h3 class="card-title">注意力权重</h3>
          <v-chart
            class="chart chart--attention"
            :option="attentionOption"
            theme="pharmalink"
            autoresize
          />
        </PlCard>
      </template>
    </template>

    <!-- Footer: prev navigation only (last step) -->
    <div class="formulation__footer">
      <PlButton variant="secondary" @click="goPrev">
        <PlIcon name="arrow-left" size="sm" />
        上一步：模型评估
      </PlButton>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from "vue";
import { useRouter } from "vue-router";
import VChart from "vue-echarts";
import { use } from "echarts/core";
import { BarChart, RadarChart } from "echarts/charts";
import {
  GridComponent,
  TooltipComponent,
  LegendComponent,
} from "echarts/components";
import { CanvasRenderer } from "echarts/renderers";

import PlButton from "@/components/PlButton.vue";
import PlInput from "@/components/PlInput.vue";
import PlSelect from "@/components/PlSelect.vue";
import PlCard from "@/components/PlCard.vue";
import PlSpinner from "@/components/PlSpinner.vue";
import PlToast from "@/components/PlToast.vue";
import PlPageHeader from "@/components/PlPageHeader.vue";
import PlIcon from "@/components/PlIcon.vue";
import PlBadge from "@/components/PlBadge.vue";

import { CHART_PALETTE } from "@/utils/chart-palette";
import { useWorkflow } from "@/composables/useWorkflow";
import { useFormulation } from "@/composables/useFormulation";
import { useTraining } from "@/composables/useTraining";
import { apiClient } from "@/composables/useApi";

use([
  BarChart,
  RadarChart,
  GridComponent,
  TooltipComponent,
  LegendComponent,
  CanvasRenderer,
]);

const router = useRouter();
const { state, markCurrentExpression } = useWorkflow();
const { result, loading, error, formulate, exportCsv } = useFormulation();
const { checkpoints, fetchCheckpoints } = useTraining();

const selectedCheckpoint = ref("");
const selectedExprId = ref(state.currentExprId);
const expressionList = ref<{ expr_id: string; latex: string }[]>([]);

const topK = ref(5);
const nSamples = ref(1000);

const checkpointOptions = computed(() =>
  checkpoints.value.map((cp) => ({
    value: cp.id,
    label: `${cp.id} (loss: ${cp.final_loss.toFixed(4)})`,
  })),
);

const expressionOptions = computed(() =>
  expressionList.value.map((e) => ({ value: e.expr_id, label: e.expr_id })),
);

async function handleCheckpointChange(cpId: string) {
  selectedCheckpoint.value = cpId;
  selectedExprId.value = "";
  expressionList.value = [];
  try {
    const { data } = await apiClient.get<
      { expr_id: string; latex: string }[]
    >(`/expressions?checkpoint_id=${cpId}`);
    expressionList.value = data;
    if (data.length === 1) {
      selectedExprId.value = data[0].expr_id;
      markCurrentExpression(data[0].expr_id);
    }
  } catch {
    /* no expressions */
  }
}

function handleExpressionChange(exprId: string) {
  selectedExprId.value = exprId;
  markCurrentExpression(exprId);
}

async function runFormulation() {
  if (!selectedExprId.value) return;
  await formulate(selectedExprId.value, topK.value, nSamples.value);
}

function goPrev() {
  router.push({ name: "evaluation" });
}

const attentionOption = computed(() => {
  if (!result.value) return {};
  const entries = Object.entries(result.value.attention_weights).sort(
    (a, b) => b[1] - a[1],
  );
  return {
    tooltip: { trigger: "axis" },
    grid: { left: 100, right: 30, top: 10, bottom: 30 },
    xAxis: { type: "value", name: "注意力权重" },
    yAxis: {
      type: "category",
      data: entries.map((e) => e[0]),
    },
    series: [
      {
        type: "bar",
        data: entries.map((e) => e[1]),
        itemStyle: { color: CHART_PALETTE[0] },
      },
    ],
  };
});

const radarOption = computed(() => {
  if (!result.value) return {};
  const { candidates, feature_names } = result.value;
  const top = candidates.slice(0, Math.min(3, candidates.length));

  const maxValues: Record<string, number> = {};
  for (const f of feature_names) {
    maxValues[f] = Math.max(...candidates.map((c) => c.components[f]));
  }

  const indicator = feature_names.map((f) => ({
    name: f,
    max: maxValues[f] > 0 ? maxValues[f] * 1.1 : 1,
  }));

  const series = top.map((c, i) => ({
    value: feature_names.map((f) => c.components[f]),
    name: `#${c.rank}`,
    areaStyle: { opacity: 0.15 },
    lineStyle: { color: CHART_PALETTE[i] },
    itemStyle: { color: CHART_PALETTE[i] },
  }));

  return {
    tooltip: {},
    legend: { data: top.map((c) => `#${c.rank}`), top: 0 },
    radar: { indicator },
    series: [{ type: "radar", data: series }],
  };
});

onMounted(async () => {
  await fetchCheckpoints();
  if (state.currentExprId && !selectedCheckpoint.value) {
    const cp = checkpoints.value[0];
    if (cp) {
      selectedCheckpoint.value = cp.id;
      await handleCheckpointChange(cp.id);
      selectedExprId.value = state.currentExprId;
    }
  }
});
</script>

<style scoped>
.formulation {
  max-width: var(--content-max-width);
  margin: 0 auto;
  padding: var(--space-10);
  display: flex;
  flex-direction: column;
  gap: var(--space-6);
  font-family: var(--font-family);
}

/* -- Toolbar card -- */
.formulation__toolbar-card {
  display: flex;
  flex-direction: column;
}

.toolbar {
  display: flex;
  gap: var(--space-3);
  align-items: flex-end;
  flex-wrap: wrap;
}

/* -- Warning banner -- */
.warning-banner {
  padding: var(--space-3) var(--space-4);
  background: var(--color-tint-sky);
  border-left: 3px solid var(--color-link-blue);
  border-radius: var(--radius-sm);
  font-size: var(--text-sm);
  line-height: var(--text-sm-line);
  color: var(--color-charcoal);
}

/* -- 2-col layout: Radar + Candidates table -- */
.formulation__grid {
  display: flex;
  gap: var(--space-6);
}

@media (max-width: 960px) {
  .formulation__grid {
    flex-direction: column;
  }
}

.formulation__radar-card {
  min-width: 280px;
  max-width: 320px;
  flex-shrink: 0;
}

.formulation__table-card {
  flex: 1;
  min-width: 0;
}

/* -- Card title -- */
.card-title {
  font-size: var(--text-h3);
  font-weight: var(--text-h3-weight);
  color: var(--color-charcoal);
  margin: 0 0 var(--space-3);
}

/* -- Card header with title + action -- */
.card-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: var(--space-3);
}

.card-header .card-title {
  margin-bottom: 0;
}

/* -- Chart sizing -- */
.chart {
  height: 360px;
  width: 100%;
}

.chart--attention {
  height: 300px;
}

/* -- Table (styled to match design system) -- */
.table-wrapper {
  overflow-x: auto;
}

table {
  border-collapse: collapse;
  width: 100%;
  font-size: var(--text-xs);
}

th,
td {
  padding: var(--space-2) var(--space-3);
  border: 1px solid var(--color-hairline);
  text-align: center;
  white-space: nowrap;
}

th {
  background: var(--color-surface);
  font-weight: var(--text-sm-weight);
  font-size: var(--text-sm);
  color: var(--color-charcoal);
}

tr.highlight {
  background: var(--color-tint-mint);
  font-weight: 600;
}

.td--response {
  font-variant-numeric: tabular-nums;
  color: var(--color-primary);
  font-weight: 600;
}

/* -- Attention weights full-width card -- */
.formulation__attention-card {
  /* full span */
}

/* -- Footer: prev navigation -- */
.formulation__footer {
  display: flex;
  justify-content: flex-start;
  padding-top: var(--space-4);
  border-top: 1px solid var(--color-hairline);
}
</style>
