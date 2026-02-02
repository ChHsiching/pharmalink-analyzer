<template>
  <div class="training-config">
    <PlPageHeader
      :step="2"
      title="模型训练"
      subtitle="配置并训练 Transformer 模型"
    />

    <div class="training-config__grid">
      <!-- Left panel (1fr): Config -->
      <PlCard variant="base" padding="md" class="training-config__left">
        <h3 class="card-title">训练配置</h3>

        <PlSelect
          :model-value="config.dataset_id"
          :options="datasetOptions"
          label="数据集"
          :disabled="isRunning"
          @update:model-value="config.dataset_id = $event"
        />

        <div class="config-section">
          <span class="config-section__label">模型结构</span>
          <div class="config-fields">
            <PlInput
              :model-value="String(config.d_model)"
              type="number"
              label="d_model"
              :disabled="isRunning"
              @update:model-value="config.d_model = Number($event)"
            />
            <PlInput
              :model-value="String(config.n_heads)"
              type="number"
              label="Heads"
              :disabled="isRunning"
              @update:model-value="config.n_heads = Number($event)"
            />
            <PlInput
              :model-value="String(config.n_layers)"
              type="number"
              label="Layers"
              :disabled="isRunning"
              @update:model-value="config.n_layers = Number($event)"
            />
          </div>
        </div>

        <div class="config-section">
          <span class="config-section__label">训练参数</span>
          <div class="config-fields">
            <PlInput
              :model-value="String(config.learning_rate)"
              type="number"
              label="Learning Rate"
              :disabled="isRunning"
              @update:model-value="config.learning_rate = Number($event)"
            />
            <PlInput
              :model-value="String(config.epochs)"
              type="number"
              label="Epochs"
              :disabled="isRunning"
              @update:model-value="config.epochs = Number($event)"
            />
            <PlInput
              :model-value="String(config.k_folds)"
              type="number"
              label="K-Fold"
              :disabled="isRunning"
              @update:model-value="config.k_folds = Number($event)"
            />
          </div>
        </div>

        <div class="config-section">
          <span class="config-section__label">正则化</span>
          <div class="config-fields">
            <PlInput
              :model-value="String(config.dropout)"
              type="number"
              label="Dropout"
              :disabled="isRunning"
              @update:model-value="config.dropout = Number($event)"
            />
            <PlInput
              :model-value="String(config.weight_decay)"
              type="number"
              label="Weight Decay"
              :disabled="isRunning"
              @update:model-value="config.weight_decay = Number($event)"
            />
            <PlInput
              :model-value="String(config.early_stopping_patience)"
              type="number"
              label="Early Stop"
              :disabled="isRunning"
              @update:model-value="config.early_stopping_patience = Number($event)"
            />
          </div>
        </div>

        <div class="config-section">
          <span class="config-section__label">数据增强</span>
          <div class="augment-row">
            <PlButton
              :variant="config.augmentation.enabled ? 'primary' : 'secondary'"
              :disabled="isRunning"
              @click="config.augmentation.enabled = !config.augmentation.enabled"
            >
              {{ config.augmentation.enabled ? '已启用' : '未启用' }}
            </PlButton>
            <PlInput
              v-if="config.augmentation.enabled"
              :model-value="String(config.augmentation.gaussian_noise_sigma)"
              type="number"
              label="噪声 sigma"
              :disabled="isRunning"
              @update:model-value="config.augmentation.gaussian_noise_sigma = Number($event)"
            />
            <PlButton
              v-if="config.augmentation.enabled"
              :variant="config.augmentation.bootstrap_enabled ? 'primary' : 'secondary'"
              :disabled="isRunning"
              @click="config.augmentation.bootstrap_enabled = !config.augmentation.bootstrap_enabled"
            >
              Bootstrap {{ config.augmentation.bootstrap_enabled ? '已启用' : '未启用' }}
            </PlButton>
          </div>
        </div>

        <div class="config-actions">
          <PlButton
            variant="primary"
            :disabled="!config.dataset_id || isRunning"
            @click="handleStart"
          >
            <PlIcon name="loader" size="sm" />
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
      </PlCard>

      <!-- Right panel (2fr): Results -->
      <div class="training-config__right">
        <!-- Stats row -->
        <div v-if="isRunning && latestMetrics" class="stat-row">
          <PlStatCard label="当前 Loss" :value="latestMetrics.trainLoss" variant="mint" />
          <PlStatCard label="Epoch" :value="String(latestMetrics.epoch)" variant="sky" />
          <PlStatCard label="最佳 Loss" :value="bestLoss" variant="peach" />
        </div>

        <!-- Progress bar -->
        <div v-if="isRunning" class="progress-bar-row">
          <PlProgressBar
            :value="statusData?.current_epoch || 0"
            :max="config.epochs"
            label="Epoch 进度"
          />
        </div>

        <!-- Loss chart -->
        <PlCard v-if="progress.length" variant="base" padding="md" class="chart-card">
          <h3 class="card-title">Loss 曲线</h3>
          <v-chart
            class="chart"
            :option="chartOption"
            theme="pharmalink"
            autoresize
          />
        </PlCard>

        <PlEmptyState
          v-else
          title="等待训练开始"
          description="配置训练参数并点击「开始训练」"
        />

        <!-- Checkpoints -->
        <PlCard v-if="checkpoints.length" variant="base" padding="md" class="checkpoints-card">
          <h3 class="card-title">已保存检查点</h3>
          <div class="table-wrapper">
            <table>
              <thead>
                <tr>
                  <th>检查点 ID</th>
                  <th>数据集</th>
                  <th>最终 Loss</th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="cp in checkpoints" :key="cp.id">
                  <td class="td--id">{{ cp.id }}</td>
                  <td>{{ cp.dataset_id }}</td>
                  <td class="td--loss">{{ cp.final_loss.toFixed(4) }}</td>
                  <td>
                    <PlButton variant="primary" @click="handleLoadCheckpoint(cp.id)">
                      加载
                    </PlButton>
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </PlCard>
      </div>
    </div>

    <!-- Footer: prev/next navigation -->
    <div class="training-config__footer">
      <PlButton variant="secondary" @click="goPrev">
        <PlIcon name="arrow-left" size="sm" />
        上一步：数据导入
      </PlButton>
      <PlButton variant="dark" @click="goNext">
        下一步：注意力分析
        <PlIcon name="arrow-right" size="sm" />
      </PlButton>
    </div>

    <PlToast v-if="error" :message="error" variant="error" />
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted } from "vue";
import { useRouter } from "vue-router";
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
import PlProgressBar from "@/components/PlProgressBar.vue";
import PlEmptyState from "@/components/PlEmptyState.vue";
import PlToast from "@/components/PlToast.vue";
import PlPageHeader from "@/components/PlPageHeader.vue";
import PlIcon from "@/components/PlIcon.vue";

use([LineChart, GridComponent, TooltipComponent, LegendComponent, CanvasRenderer]);

const router = useRouter();

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

const bestLoss = computed(() => {
  if (progress.value.length === 0) return "--";
  const minVal = Math.min(...progress.value.map((p) => p.val_loss));
  return minVal.toFixed(4);
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

function goPrev() {
  router.push({ name: "data-import" });
}

function goNext() {
  router.push({ name: "analysis" });
}

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
  max-width: var(--content-max-width);
  margin: 0 auto;
  padding: var(--space-10);
  display: flex;
  flex-direction: column;
  gap: var(--space-6);
  font-family: var(--font-family);
}

/* -- Grid: 1fr config + 2fr results -- */
.training-config__grid {
  display: grid;
  grid-template-columns: 1fr 2fr;
  gap: var(--space-6);
}

@media (max-width: 960px) {
  .training-config__grid {
    grid-template-columns: 1fr;
  }
}

/* -- Left panel: Config card -- */
.training-config__left {
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

.config-section {
  display: flex;
  flex-direction: column;
  gap: var(--space-2);
}

.config-section__label {
  font-size: var(--text-xs);
  font-weight: var(--text-xs-weight);
  text-transform: uppercase;
  letter-spacing: 0.5px;
  color: var(--color-slate);
}

.config-fields {
  display: flex;
  flex-direction: column;
  gap: var(--space-2);
}

.augment-row {
  display: flex;
  align-items: center;
  gap: var(--space-3);
  flex-wrap: wrap;
}

.config-actions {
  display: flex;
  gap: var(--space-2);
  padding-top: var(--space-4);
  border-top: 1px solid var(--color-hairline);
}

/* -- Right panel: Results -- */
.training-config__right {
  display: flex;
  flex-direction: column;
  gap: var(--space-4);
}

.stat-row {
  display: flex;
  gap: var(--space-3);
}

.progress-bar-row {
  padding: 0;
}

/* -- Chart card -- */
.chart-card {
  flex: 1;
  min-height: 0;
}

.chart {
  height: 300px;
  width: 100%;
}

/* -- Checkpoints card -- */
.checkpoints-card {
  min-height: 0;
}

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
  text-align: left;
  white-space: nowrap;
}

th {
  background: var(--color-surface);
  font-weight: var(--text-sm-weight);
  font-size: var(--text-sm);
  color: var(--color-charcoal);
}

.td--id {
  font-weight: 600;
  color: var(--color-ink);
}

.td--loss {
  font-variant-numeric: tabular-nums;
  color: var(--color-primary);
}

/* -- Footer: prev/next navigation -- */
.training-config__footer {
  display: flex;
  justify-content: space-between;
  padding-top: var(--space-4);
  border-top: 1px solid var(--color-hairline);
}
</style>
