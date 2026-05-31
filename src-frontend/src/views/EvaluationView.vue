<template>
  <div class="evaluation">
    <PlPageHeader
      :step="5"
      title="模型评估"
      subtitle="评估模型预测性能与泛化能力"
    />

    <!-- Toolbar: checkpoint selection -->
    <div class="evaluation__toolbar">
      <PlSelect
        :model-value="selectedCheckpoint"
        :options="checkpointOptions"
        label="检查点"
        @update:model-value="selectedCheckpoint = $event"
      />
      <PlButton variant="primary" :disabled="!selectedCheckpoint" :loading="loading" @click="evaluate">
        评估
      </PlButton>
    </div>

    <!-- Metrics row: 3-col grid -->
    <div v-if="metrics" class="evaluation__metrics">
      <PlStatCard
        label="R²"
        :value="`${metrics.aggregate.r2_mean.toFixed(4)} ± ${metrics.aggregate.r2_std.toFixed(4)}`"
        variant="mint"
      />
      <PlStatCard
        label="RMSE"
        :value="rmseDisplay"
        variant="sky"
      />
      <PlStatCard
        label="MAE"
        :value="`${metrics.aggregate.mae_mean.toFixed(4)} ± ${metrics.aggregate.mae_std.toFixed(4)}`"
        variant="peach"
      />
    </div>

    <!-- Charts row: 2-col grid (scatter + residuals) -->
    <div v-if="metrics" class="evaluation__charts-row">
      <PlCard variant="base" padding="md">
        <h3 class="card-title">Predicted vs Actual</h3>
        <v-chart
          class="chart"
          :option="scatterOption"
          theme="pharmalink"
          autoresize
        />
      </PlCard>
      <PlCard variant="base" padding="md">
        <h3 class="card-title">残差分布</h3>
        <v-chart
          class="chart"
          :option="histogramOption"
          theme="pharmalink"
          autoresize
        />
      </PlCard>
    </div>

    <!-- Full-width: Loss curves -->
    <PlCard v-if="metrics && lossCurve?.folds.length" variant="base" padding="md" class="evaluation__full-card">
      <h3 class="card-title">训练损失曲线</h3>
      <v-chart
        class="chart chart--lg"
        :option="lossCurveOption"
        theme="pharmalink"
        autoresize
      />
    </PlCard>

    <!-- Full-width: K-Fold table -->
    <PlCard v-if="metrics" variant="base" padding="md" class="evaluation__full-card">
      <h3 class="card-title">K-Fold 评估指标</h3>
      <div class="table-wrapper">
        <table class="kfold-table">
          <thead>
            <tr>
              <th>Fold</th>
              <th>R²</th>
              <th>RMSE</th>
              <th>MAE</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="f in metrics.folds" :key="f.fold">
              <td class="td--fold">{{ f.fold }}</td>
              <td class="td--num">{{ f.r2.toFixed(4) }}</td>
              <td class="td--num">{{ Math.sqrt(f.mse).toFixed(4) }}</td>
              <td class="td--num">{{ f.mae.toFixed(4) }}</td>
            </tr>
          </tbody>
          <tfoot>
            <tr>
              <td class="td--summary">Mean ± Std</td>
              <td class="td--num">
                {{ metrics.aggregate.r2_mean.toFixed(4) }} ± {{ metrics.aggregate.r2_std.toFixed(4) }}
              </td>
              <td class="td--num">{{ rmseDisplay }}</td>
              <td class="td--num">
                {{ metrics.aggregate.mae_mean.toFixed(4) }} ± {{ metrics.aggregate.mae_std.toFixed(4) }}
              </td>
            </tr>
          </tfoot>
        </table>
      </div>
    </PlCard>

    <!-- Footer: prev/next navigation -->
    <div class="evaluation__footer">
      <PlButton variant="secondary" @click="goPrev">
        <PlIcon name="arrow-left" size="sm" />
        上一步：表达推导
      </PlButton>
      <PlButton variant="dark" @click="goNext">
        下一步：处方优化
        <PlIcon name="arrow-right" size="sm" />
      </PlButton>
    </div>

    <PlToast v-if="error" :message="error" variant="error" />
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from "vue";
import { useRouter } from "vue-router";
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
import PlToast from "@/components/PlToast.vue";
import PlPageHeader from "@/components/PlPageHeader.vue";
import PlIcon from "@/components/PlIcon.vue";

use([ScatterChart, LineChart, BarChart, GridComponent, TooltipComponent, LegendComponent, CanvasRenderer]);

const router = useRouter();

const {
  metrics, predictions, residuals, lossCurve,
  loading, error, fetchAll,
} = useEvaluation();
const { checkpoints, fetchCheckpoints } = useTraining();

const selectedCheckpoint = ref("");

const checkpointOptions = computed(() =>
  checkpoints.value.map((cp) => ({ value: cp.id, label: `${cp.id} (loss: ${cp.final_loss.toFixed(4)})` }))
);

const rmseDisplay = computed(() => {
  if (!metrics.value) return "--";
  const rmseMean = Math.sqrt(metrics.value.aggregate.mse_mean);
  const rmseStd = Math.sqrt(metrics.value.aggregate.mse_std);
  return `${rmseMean.toFixed(4)} ± ${rmseStd.toFixed(4)}`;
});

onMounted(() => {
  fetchCheckpoints();
});

async function evaluate() {
  if (!selectedCheckpoint.value) return;
  await fetchAll(selectedCheckpoint.value);
}

function goPrev() {
  router.push({ name: "expression" });
}

function goNext() {
  router.push({ name: "formulation" });
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
  max-width: var(--content-max-width);
  margin: 0 auto;
  padding: var(--space-10);
  display: flex;
  flex-direction: column;
  gap: var(--space-6);
  font-family: var(--font-family);
}

/* -- Toolbar -- */
.evaluation__toolbar {
  display: flex;
  gap: var(--space-3);
  align-items: flex-end;
}

/* -- Metrics row: 3-col grid -- */
.evaluation__metrics {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: var(--space-4);
}

@media (max-width: 640px) {
  .evaluation__metrics {
    grid-template-columns: 1fr;
  }
}

/* -- Charts row: 2-col grid -- */
.evaluation__charts-row {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: var(--space-4);
}

@media (max-width: 960px) {
  .evaluation__charts-row {
    grid-template-columns: 1fr;
  }
}

/* -- Card title -- */
.card-title {
  font-size: var(--text-h3);
  font-weight: var(--text-h3-weight);
  color: var(--color-charcoal);
  margin: 0 0 var(--space-3);
}

/* -- Chart sizing -- */
.chart {
  height: 300px;
  width: 100%;
}

.chart--lg {
  height: 350px;
}

/* -- Full-width card -- */
.evaluation__full-card {
  width: 100%;
}

/* -- K-Fold table -- */
.table-wrapper {
  overflow-x: auto;
}

.kfold-table {
  width: 100%;
  border-collapse: collapse;
  font-size: var(--text-xs);
}

.kfold-table th,
.kfold-table td {
  padding: var(--space-2) var(--space-3);
  border: 1px solid var(--color-hairline);
  text-align: center;
  white-space: nowrap;
}

.kfold-table th {
  background: var(--color-surface);
  font-weight: var(--text-sm-weight);
  font-size: var(--text-sm);
  color: var(--color-charcoal);
}

.kfold-table tfoot td {
  font-weight: 600;
  background: var(--color-surface-soft);
}

.td--fold {
  font-weight: 600;
  color: var(--color-ink);
}

.td--num {
  font-variant-numeric: tabular-nums;
  color: var(--color-primary);
}

.td--summary {
  font-weight: 600;
  color: var(--color-charcoal);
}

/* -- Footer: prev/next navigation -- */
.evaluation__footer {
  display: flex;
  justify-content: space-between;
  padding-top: var(--space-4);
  border-top: 1px solid var(--color-hairline);
}
</style>
