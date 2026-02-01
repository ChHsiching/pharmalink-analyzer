<template>
  <div class="evaluation">
    <h2>模型评估</h2>

    <div class="toolbar">
      <PlSelect
        :model-value="selectedCheckpoint"
        :options="checkpointOptions"
        label="检查点"
        @update:model-value="selectedCheckpoint = $event"
      />
      <PlButton variant="primary" :disabled="!selectedCheckpoint" :loading="loading" @click="evaluate">
        评估
      </PlButton>
      <PlSpinner v-if="loading" size="sm" />
      <PlToast v-if="error" :message="error" variant="error" />
    </div>

    <div v-if="metrics" class="metrics-bar">
      <PlStatCard
        label="R²"
        :value="`${metrics.aggregate.r2_mean.toFixed(4)} ± ${metrics.aggregate.r2_std.toFixed(4)}`"
        variant="mint"
      />
      <PlStatCard
        label="MSE"
        :value="`${metrics.aggregate.mse_mean.toFixed(4)} ± ${metrics.aggregate.mse_std.toFixed(4)}`"
        variant="lavender"
      />
      <PlStatCard
        label="MAE"
        :value="`${metrics.aggregate.mae_mean.toFixed(4)} ± ${metrics.aggregate.mae_std.toFixed(4)}`"
        variant="sky"
      />
    </div>

    <div v-if="metrics" class="charts-grid">
      <PlCard variant="base" padding="md">
        <h3>Predicted vs Actual</h3>
        <v-chart :option="scatterOption" autoresize style="height: 300px" />
      </PlCard>

      <PlCard variant="base" padding="md">
        <h3>残差分布</h3>
        <v-chart :option="histogramOption" autoresize style="height: 300px" />
      </PlCard>

      <PlCard variant="base" padding="md">
        <h3>训练损失曲线</h3>
        <v-chart :option="lossCurveOption" autoresize style="height: 300px" />
      </PlCard>

      <PlCard variant="base" padding="md">
        <h3>K-Fold 指标</h3>
        <table class="metrics-table">
          <thead>
            <tr>
              <th>Fold</th>
              <th>R2</th>
              <th>MSE</th>
              <th>MAE</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="f in metrics.folds" :key="f.fold">
              <td>{{ f.fold }}</td>
              <td>{{ f.r2.toFixed(4) }}</td>
              <td>{{ f.mse.toFixed(4) }}</td>
              <td>{{ f.mae.toFixed(4) }}</td>
            </tr>
          </tbody>
          <tfoot>
            <tr>
              <td>Mean ± Std</td>
              <td>{{ metrics.aggregate.r2_mean.toFixed(4) }} ± {{ metrics.aggregate.r2_std.toFixed(4) }}</td>
              <td>{{ metrics.aggregate.mse_mean.toFixed(4) }} ± {{ metrics.aggregate.mse_std.toFixed(4) }}</td>
              <td>{{ metrics.aggregate.mae_mean.toFixed(4) }} ± {{ metrics.aggregate.mae_std.toFixed(4) }}</td>
            </tr>
          </tfoot>
        </table>
      </PlCard>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from "vue";
import VChart from "vue-echarts";
import { use } from "echarts/core";
import { ScatterChart, LineChart, BarChart } from "echarts/charts";
import { GridComponent, TooltipComponent, LegendComponent } from "echarts/components";
import { CanvasRenderer } from "echarts/renderers";
import { useEvaluation } from "@/composables/useEvaluation";
import { useTraining } from "@/composables/useTraining";
import { CHART_PALETTE, CHART_COLORS } from "@/utils/chart-palette";
import PlSelect from "@/components/PlSelect.vue";
import PlButton from "@/components/PlButton.vue";
import PlCard from "@/components/PlCard.vue";
import PlStatCard from "@/components/PlStatCard.vue";
import PlSpinner from "@/components/PlSpinner.vue";
import PlToast from "@/components/PlToast.vue";

use([ScatterChart, LineChart, BarChart, GridComponent, TooltipComponent, LegendComponent, CanvasRenderer]);

const {
  metrics, predictions, residuals, lossCurve,
  loading, error, fetchAll,
} = useEvaluation();
const { checkpoints, fetchCheckpoints } = useTraining();

const selectedCheckpoint = ref("");

const checkpointOptions = computed(() =>
  checkpoints.value.map((cp) => ({ value: cp.id, label: `${cp.id} (loss: ${cp.final_loss.toFixed(4)})` }))
);

onMounted(() => {
  fetchCheckpoints();
});

async function evaluate() {
  if (!selectedCheckpoint.value) return;
  await fetchAll(selectedCheckpoint.value);
}

const scatterOption = computed(() => {
  if (!predictions.value) return {};
  const data = predictions.value.predictions.map((p) => [p.actual, p.predicted]);
  const allVals = data.flat();
  const min = Math.min(...allVals);
  const max = Math.max(...allVals);
  return {
    tooltip: { trigger: "item" },
    xAxis: { name: "Actual", type: "value", min, max },
    yAxis: { name: "Predicted", type: "value", min, max },
    series: [
      {
        type: "scatter",
        data,
        symbolSize: 6,
        color: CHART_PALETTE[0],
      },
      {
        type: "line",
        data: [
          [min, min],
          [max, max],
        ],
        lineStyle: { type: "dashed", color: CHART_COLORS.muted },
        symbol: "none",
      },
    ],
  };
});

const histogramOption = computed(() => {
  if (!residuals.value) return {};
  const bins = residuals.value.bins;
  const categories = bins.map(
    (b) => `${b.range_start.toFixed(2)}~${b.range_end.toFixed(2)}`,
  );
  return {
    tooltip: { trigger: "axis" },
    xAxis: { type: "category", data: categories, axisLabel: { rotate: 45 } },
    yAxis: { name: "Count", type: "value" },
    series: [
      {
        type: "bar",
        data: bins.map((b) => b.count),
        color: CHART_PALETTE[1],
      },
    ],
  };
});

const lossCurveOption = computed(() => {
  if (!lossCurve.value || lossCurve.value.folds.length === 0) return {};
  const series: Array<Record<string, unknown>> = [];
  for (const [idx, fold] of lossCurve.value.folds.entries()) {
    const foldColor = CHART_PALETTE[idx % CHART_PALETTE.length];
    series.push({
      name: `Fold ${fold.fold} - Train`,
      type: "line",
      data: fold.points.map((p) => [p.epoch, p.train_loss]),
      smooth: true,
      color: foldColor,
    });
    series.push({
      name: `Fold ${fold.fold} - Val`,
      type: "line",
      data: fold.points.map((p) => [p.epoch, p.val_loss]),
      smooth: true,
      lineStyle: { type: "dashed" },
      color: foldColor,
    });
  }
  return {
    tooltip: { trigger: "axis" },
    legend: { top: 0 },
    xAxis: { name: "Epoch", type: "value" },
    yAxis: { name: "Loss", type: "value" },
    series,
  };
});
</script>

<style scoped>
.evaluation {
  max-width: 1200px;
  margin: 0 auto;
  padding: 20px;
  font-family: system-ui, sans-serif;
}

.toolbar {
  display: flex;
  gap: 12px;
  align-items: center;
  margin-bottom: 16px;
}

.metrics-bar {
  display: flex;
  gap: 16px;
  margin-bottom: 16px;
}

.charts-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
}

.charts-grid h3 {
  margin: 0 0 8px 0;
  font-size: 14px;
  color: var(--color-charcoal);
}

.metrics-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
}

.metrics-table th,
.metrics-table td {
  border: 1px solid var(--color-hairline);
  padding: 6px 10px;
  text-align: center;
}

.metrics-table th {
  background: var(--color-surface);
}

.metrics-table tfoot td {
  font-weight: bold;
  background: var(--color-surface-soft);
}

@media (max-width: 1024px) {
  .charts-grid {
    grid-template-columns: 1fr;
  }
}
</style>
