<template>
  <div class="training-config">
    <section class="config-panel">
      <h2>训练配置</h2>

      <PlSelect
        :model-value="config.dataset_id"
        :options="datasetOptions"
        label="数据集"
        @update:model-value="config.dataset_id = $event"
      />

      <div class="form-row">
        <PlInput
          :model-value="String(config.d_model)"
          type="number"
          label="d_model"
          @update:model-value="config.d_model = Number($event)"
        />
        <PlInput
          :model-value="String(config.n_heads)"
          type="number"
          label="Heads"
          @update:model-value="config.n_heads = Number($event)"
        />
        <PlInput
          :model-value="String(config.n_layers)"
          type="number"
          label="Layers"
          @update:model-value="config.n_layers = Number($event)"
        />
      </div>

      <div class="form-row">
        <PlInput
          :model-value="String(config.learning_rate)"
          type="number"
          label="Learning Rate"
          @update:model-value="config.learning_rate = Number($event)"
        />
        <PlInput
          :model-value="String(config.epochs)"
          type="number"
          label="Epochs"
          @update:model-value="config.epochs = Number($event)"
        />
        <PlInput
          :model-value="String(config.k_folds)"
          type="number"
          label="K-Fold"
          @update:model-value="config.k_folds = Number($event)"
        />
      </div>

      <div class="form-row">
        <PlInput
          :model-value="String(config.dropout)"
          type="number"
          label="Dropout"
          @update:model-value="config.dropout = Number($event)"
        />
        <PlInput
          :model-value="String(config.weight_decay)"
          type="number"
          label="Weight Decay"
          @update:model-value="config.weight_decay = Number($event)"
        />
        <PlInput
          :model-value="String(config.early_stopping_patience)"
          type="number"
          label="Early Stop"
          @update:model-value="config.early_stopping_patience = Number($event)"
        />
      </div>

      <div class="section-title">数据增强</div>
      <div class="form-row">
        <div class="form-group">
          <label class="toggle-label">启用</label>
          <PlButton
            :variant="config.augmentation.enabled ? 'primary' : 'secondary'"
            @click="config.augmentation.enabled = !config.augmentation.enabled"
          >
            {{ config.augmentation.enabled ? '已启用' : '未启用' }}
          </PlButton>
        </div>
        <PlInput
          v-if="config.augmentation.enabled"
          :model-value="String(config.augmentation.gaussian_noise_sigma)"
          type="number"
          label="噪声 sigma"
          @update:model-value="config.augmentation.gaussian_noise_sigma = Number($event)"
        />
        <div class="form-group" v-if="config.augmentation.enabled">
          <label class="toggle-label">Bootstrap</label>
          <PlButton
            :variant="config.augmentation.bootstrap_enabled ? 'primary' : 'secondary'"
            @click="config.augmentation.bootstrap_enabled = !config.augmentation.bootstrap_enabled"
          >
            {{ config.augmentation.bootstrap_enabled ? '已启用' : '未启用' }}
          </PlButton>
        </div>
      </div>

      <div class="actions">
        <PlButton
          variant="primary"
          :disabled="!config.dataset_id || isRunning"
          @click="handleStart"
        >
          开始训练
        </PlButton>
        <PlButton
          variant="ghost"
          :disabled="!isRunning"
          @click="handleStop"
        >
          停止训练
        </PlButton>
      </div>

      <PlToast v-if="error" :message="error" variant="error" />
    </section>

    <section class="progress-panel">
      <h2>训练进度</h2>

      <div v-if="statusData" class="status-bar">
        <PlBadge :text="statusText" :variant="statusVariant" />
        <span v-if="isRunning">
          Fold {{ statusData.current_fold }}/{{ config.k_folds }} -
          Epoch {{ statusData.current_epoch }}
        </span>
      </div>

      <div v-if="isRunning && latestMetrics" class="stat-row">
        <PlStatCard label="Train Loss" :value="latestMetrics.trainLoss" variant="mint" />
        <PlStatCard label="Val Loss" :value="latestMetrics.valLoss" variant="lavender" />
        <PlStatCard label="Epoch" :value="latestMetrics.epoch" variant="sky" />
      </div>

      <div v-if="isRunning" class="progress-bar-row">
        <PlProgressBar :value="statusData?.current_epoch || 0" :max="config.epochs" label="Epoch 进度" />
      </div>

      <div v-if="progress.length" class="chart-container">
        <v-chart class="chart" :option="chartOption" autoresize />
      </div>

      <PlEmptyState v-else title="等待训练开始" description="配置训练参数并点击「开始训练」" />

      <div v-if="checkpoints.length" class="checkpoints">
        <h3>已保存检查点</h3>
        <div class="checkpoint-list">
          <PlCard v-for="cp in checkpoints" :key="cp.id" variant="base" padding="sm">
            <div class="checkpoint-item">
              <span>{{ cp.dataset_id }}</span>
              <span class="loss">Loss: {{ cp.final_loss.toFixed(4) }}</span>
              <PlButton variant="primary" @click="handleLoadCheckpoint(cp.id)">
                加载
              </PlButton>
            </div>
          </PlCard>
        </div>
      </div>
    </section>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted } from "vue";
import VChart from "vue-echarts";
import { use } from "echarts/core";
import { LineChart } from "echarts/charts";
import {
  GridComponent,
  TooltipComponent,
  LegendComponent,
} from "echarts/components";
import { CanvasRenderer } from "echarts/renderers";

import { useDatasets } from "@/composables/useDatasets";
import { useTraining } from "@/composables/useTraining";
import type { DatasetMeta } from "@/types/dataset";
import type { TrainingConfig } from "@/types/training";
import { useWorkflow } from "@/composables/useWorkflow";
import { CHART_PALETTE } from "@/utils/chart-palette";

import PlButton from "@/components/PlButton.vue";
import PlInput from "@/components/PlInput.vue";
import PlSelect from "@/components/PlSelect.vue";
import PlCard from "@/components/PlCard.vue";
import PlStatCard from "@/components/PlStatCard.vue";
import PlBadge from "@/components/PlBadge.vue";
import PlProgressBar from "@/components/PlProgressBar.vue";
import PlEmptyState from "@/components/PlEmptyState.vue";
import PlToast from "@/components/PlToast.vue";

use([LineChart, GridComponent, TooltipComponent, LegendComponent, CanvasRenderer]);

const { listDatasets } = useDatasets();
const {
  status: statusData,
  progress,
  checkpoints,
  error,
  fetchStatus,
  fetchCheckpoints,
  startTraining,
  stopTraining,
  loadCheckpoint,
  disconnectProgress,
} = useTraining();

const { markHasCheckpoint } = useWorkflow();

const datasets = ref<DatasetMeta[]>([]);

const config = ref<TrainingConfig>({
  dataset_id: "",
  d_model: 64,
  n_heads: 4,
  n_layers: 2,
  dropout: 0.1,
  learning_rate: 0.001,
  weight_decay: 0.0001,
  epochs: 100,
  k_folds: 5,
  early_stopping_patience: 20,
  augmentation: {
    enabled: true,
    gaussian_noise_sigma: 0.05,
    bootstrap_enabled: true,
  },
});

const isRunning = computed(() => statusData.value?.status === "running");

const statusText = computed(() => {
  const s = statusData.value?.status;
  const map: Record<string, string> = {
    idle: "空闲",
    running: "训练中",
    completed: "已完成",
    stopped: "已停止",
    error: "错误",
  };
  return map[s || "idle"] || s;
});

const statusVariant = computed(() => {
  const map: Record<string, "teal" | "orange" | "pink"> = {
    idle: "teal",
    running: "orange",
    completed: "teal",
    stopped: "pink",
    error: "pink",
  };
  return map[statusData.value?.status || "idle"] || "teal";
});

const datasetOptions = computed(() => [
  { value: "", label: "-- 选择数据集 --" },
  ...datasets.value.map((ds) => ({ value: ds.id, label: `${ds.name} (${ds.target})` })),
]);

const latestMetrics = computed(() => {
  if (progress.value.length === 0) return null;
  const last = progress.value[progress.value.length - 1];
  return {
    trainLoss: last.train_loss.toFixed(4),
    valLoss: last.val_loss.toFixed(4),
    epoch: last.epoch,
  };
});

const chartOption = computed(() => {
  const trainData = progress.value.map((p) => [p.epoch, p.train_loss]);
  const valData = progress.value.map((p) => [p.epoch, p.val_loss]);
  return {
    color: [CHART_PALETTE[0], CHART_PALETTE[1]],
    tooltip: { trigger: "axis" },
    legend: { data: ["Train Loss", "Val Loss"] },
    grid: { left: 60, right: 20, top: 40, bottom: 30 },
    xAxis: { type: "value", name: "Epoch" },
    yAxis: { type: "value", name: "Loss" },
    series: [
      { name: "Train Loss", type: "line", data: trainData, smooth: true },
      { name: "Val Loss", type: "line", data: valData, smooth: true },
    ],
  };
});

async function handleStart() {
  await startTraining(config.value);
  await fetchCheckpoints();
  if (checkpoints.value.length > 0) {
    markHasCheckpoint();
  }
}

async function handleStop() {
  await stopTraining();
}

async function handleLoadCheckpoint(id: string) {
  await loadCheckpoint(id);
  markHasCheckpoint();
}

onMounted(async () => {
  try {
    datasets.value = await listDatasets();
  } catch {
    // ignore
  }
  await fetchStatus();
  await fetchCheckpoints();
  if (checkpoints.value.length > 0) {
    markHasCheckpoint();
  }
});

onUnmounted(() => {
  disconnectProgress();
});
</script>

<style scoped>
.training-config {
  display: grid;
  grid-template-columns: minmax(280px, 360px) 1fr;
  height: calc(100vh - 48px);
  font-family: var(--font-family);
}

@media (max-width: 768px) {
  .training-config {
    grid-template-columns: 1fr;
  }

  .config-panel {
    border-right: none;
    border-bottom: 1px solid var(--color-hairline);
  }
}

.config-panel {
  border-right: 1px solid var(--color-hairline);
  padding: 16px;
  overflow-y: auto;
  background: var(--color-surface-soft);
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.config-panel h2 {
  margin: 0 0 8px;
  font-size: var(--font-size-heading);
  font-weight: var(--font-weight-heading);
  color: var(--color-ink);
}

.section-title {
  font-weight: var(--font-weight-medium);
  margin: 8px 0 4px;
  color: var(--color-charcoal);
  font-size: var(--font-size-body);
}

.form-row {
  display: flex;
  gap: 8px;
}

.form-group {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.toggle-label {
  font-size: var(--font-size-caption);
  font-weight: var(--font-weight-medium);
  color: var(--color-charcoal);
}

.actions {
  display: flex;
  gap: 8px;
  margin-top: 12px;
}

.progress-panel {
  padding: 16px;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.progress-panel h2 {
  margin: 0;
  font-size: var(--font-size-heading);
  font-weight: var(--font-weight-heading);
  color: var(--color-success);
}

.status-bar {
  display: flex;
  gap: 12px;
  align-items: center;
  font-size: var(--font-size-body);
}

.stat-row {
  display: flex;
  gap: 12px;
}

.progress-bar-row {
  margin: 4px 0;
}

.chart-container {
  margin-bottom: 8px;
}

.chart {
  height: 300px;
  width: 100%;
}

.checkpoints h3 {
  font-size: var(--font-size-body);
  font-weight: var(--font-weight-medium);
  color: var(--color-ink);
  margin: 8px 0;
}

.checkpoint-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.checkpoint-item {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: var(--font-size-body);
}

.checkpoint-item .loss {
  color: var(--color-slate);
  flex: 1;
}
</style>
