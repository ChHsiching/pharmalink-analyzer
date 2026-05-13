<template>
  <div class="evaluation">
    <h2>模型评估</h2>

    <div class="toolbar">
      <select v-model="selectedCheckpoint" class="select">
        <option value="">选择检查点</option>
        <option v-for="cp in checkpoints" :key="cp.id" :value="cp.id">
          {{ cp.id }} (loss: {{ cp.final_loss.toFixed(4) }})
        </option>
      </select>
      <button :disabled="!selectedCheckpoint || loading" @click="evaluate">
        评估
      </button>
      <span v-if="loading">加载中...</span>
      <span v-if="error" class="error">{{ error }}</span>
    </div>

    <div v-if="metrics" class="metrics-bar">
      <span>R2 = {{ metrics.aggregate.r2_mean.toFixed(4) }} +/- {{ metrics.aggregate.r2_std.toFixed(4) }}</span>
      <span>MSE = {{ metrics.aggregate.mse_mean.toFixed(4) }} +/- {{ metrics.aggregate.mse_std.toFixed(4) }}</span>
      <span>MAE = {{ metrics.aggregate.mae_mean.toFixed(4) }} +/- {{ metrics.aggregate.mae_std.toFixed(4) }}</span>
    </div>

    <div v-if="metrics" class="charts-grid">
      <div class="chart-panel">
        <h3>Predicted vs Actual</h3>
        <v-chart :option="scatterOption" autoresize style="height: 300px" />
      </div>

      <div class="chart-panel">
        <h3>残差分布</h3>
        <v-chart :option="histogramOption" autoresize style="height: 300px" />
      </div>

      <div class="chart-panel">
        <h3>训练损失曲线</h3>
        <v-chart :option="lossCurveOption" autoresize style="height: 300px" />
      </div>

      <div class="chart-panel">
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
              <td>Mean +/- Std</td>
              <td>{{ metrics.aggregate.r2_mean.toFixed(4) }} +/- {{ metrics.aggregate.r2_std.toFixed(4) }}</td>
              <td>{{ metrics.aggregate.mse_mean.toFixed(4) }} +/- {{ metrics.aggregate.mse_std.toFixed(4) }}</td>
              <td>{{ metrics.aggregate.mae_mean.toFixed(4) }} +/- {{ metrics.aggregate.mae_std.toFixed(4) }}</td>
            </tr>
          </tfoot>
        </table>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from "vue";
import VChart from "vue-echarts";
import { useEvaluation } from "@/composables/useEvaluation";
import { useTraining } from "@/composables/useTraining";
import "echarts";

const {
  metrics, predictions, residuals, lossCurve,
  loading, error, fetchAll,
} = useEvaluation();
const { checkpoints, fetchCheckpoints } = useTraining();

const selectedCheckpoint = ref("");

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
        itemStyle: { color: "#4caf50" },
      },
      {
        type: "line",
        data: [
          [min, min],
          [max, max],
        ],
        lineStyle: { type: "dashed", color: "#999" },
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
        itemStyle: { color: "#2196f3" },
      },
    ],
  };
});

const lossCurveOption = computed(() => {
  if (!lossCurve.value || lossCurve.value.folds.length === 0) return {};
  const series: Array<Record<string, unknown>> = [];
  for (const fold of lossCurve.value.folds) {
    series.push({
      name: `Fold ${fold.fold} - Train`,
      type: "line",
      data: fold.points.map((p) => [p.epoch, p.train_loss]),
      smooth: true,
    });
    series.push({
      name: `Fold ${fold.fold} - Val`,
      type: "line",
      data: fold.points.map((p) => [p.epoch, p.val_loss]),
      smooth: true,
      lineStyle: { type: "dashed" },
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

.select {
  padding: 6px 12px;
  min-width: 300px;
}

.toolbar button {
  padding: 6px 16px;
  background: var(--color-primary);
  color: white;
  border: none;
  border-radius: var(--radius-sm);
  cursor: pointer;
}

.toolbar button:disabled {
  background: var(--color-muted);
  cursor: not-allowed;
}

.error { color: var(--color-error); }

.metrics-bar {
  display: flex;
  gap: 24px;
  padding: 12px;
  background: var(--color-surface);
  border-radius: var(--radius-sm);
  margin-bottom: 16px;
  font-size: 14px;
}

.charts-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
}

.chart-panel {
  border: 1px solid var(--color-hairline);
  border-radius: var(--radius-sm);
  padding: 12px;
}

.chart-panel h3 {
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
</style>
