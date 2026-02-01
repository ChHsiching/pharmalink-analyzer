<template>
  <div class="formulation">
    <h2>最优配比</h2>

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
      <PlButton v-if="result" variant="secondary" @click="exportCsv">
        导出 CSV
      </PlButton>
    </div>

    <PlEmptyState
      v-if="!selectedExprId && !expressionOptions.length"
      title="未生成表达式"
      description="请先在「表达式推导」页面生成表达式后再进行配比寻优"
      action="前往表达式推导"
      @action="goToExpression"
    />

    <template v-else-if="selectedExprId">
      <PlSpinner v-if="loading" size="md" />

      <PlToast v-if="error" :message="error" variant="error" />

      <div v-if="result?.attention_weak" class="warning-banner">
        注意力权重过于均匀，结果已通过 softmax 增强。建议增加训练层数或训练轮次以获得更显著的注意力分布。
      </div>

      <template v-if="result">
        <PlCard variant="base" padding="md">
          <h3>注意力权重</h3>
          <v-chart
            :option="attentionOption"
            autoresize
            style="height: 260px"
          />
        </PlCard>

        <PlCard variant="base" padding="md">
          <h3>候选配比</h3>
          <table class="result-table">
            <thead>
              <tr>
                <th>排名</th>
                <th v-for="f in result.feature_names" :key="f">{{ f }}</th>
                <th>预测药效</th>
              </tr>
            </thead>
            <tbody>
              <tr
                v-for="c in result.candidates"
                :key="c.rank"
                :class="{ highlight: c.rank === 1 }"
              >
                <td>{{ c.rank }}</td>
                <td v-for="f in result.feature_names" :key="f">
                  {{ c.components[f].toFixed(4) }}
                </td>
                <td>{{ c.predicted_response.toFixed(4) }}</td>
              </tr>
            </tbody>
          </table>
        </PlCard>

        <PlCard variant="base" padding="md">
          <h3>候选配比对比</h3>
          <v-chart
            :option="radarOption"
            autoresize
            style="height: 360px"
          />
        </PlCard>
      </template>
    </template>
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
import PlEmptyState from "@/components/PlEmptyState.vue";
import PlSpinner from "@/components/PlSpinner.vue";
import PlToast from "@/components/PlToast.vue";

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

function goToExpression() {
  router.push("/expression");
}

async function runFormulation() {
  if (!selectedExprId.value) return;
  await formulate(selectedExprId.value, topK.value, nSamples.value);
}

const attentionOption = computed(() => {
  if (!result.value) return {};
  const entries = Object.entries(result.value.attention_weights).sort(
    (a, b) => b[1] - a[1],
  );
  return {
    tooltip: { trigger: "axis" },
    grid: { left: 100, right: 30, top: 10, bottom: 30 },
    xAxis: { type: "value", name: "权重" },
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
  max-width: 1200px;
  margin: 0 auto;
  padding: 20px;
  font-family: system-ui, sans-serif;
}

h2 {
  color: var(--color-ink);
}

h3 {
  margin: 0 0 8px 0;
  font-size: 14px;
  color: var(--color-charcoal);
}

.toolbar {
  display: flex;
  gap: 12px;
  align-items: flex-end;
  margin-bottom: 16px;
  flex-wrap: wrap;
}

.warning-banner {
  padding: 10px 14px;
  background: var(--color-tint-sky);
  border-left: 3px solid var(--color-link-blue);
  border-radius: var(--radius-sm);
  margin-bottom: 16px;
  font-size: 13px;
  color: var(--color-charcoal);
}

.result-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
}

.result-table th,
.result-table td {
  border: 1px solid var(--color-hairline);
  padding: 6px 10px;
  text-align: center;
}

.result-table th {
  background: var(--color-surface);
}

.result-table tr.highlight {
  background: var(--color-tint-mint);
  font-weight: bold;
}
</style>
