<template>
  <div class="training-config">
    <section class="config-panel">
      <h2>训练配置</h2>

      <div class="form-group">
        <label>数据集</label>
        <select v-model="config.dataset_id">
          <option value="">-- 选择数据集 --</option>
          <option v-for="ds in datasets" :key="ds.id" :value="ds.id">
            {{ ds.name }} ({{ ds.target }})
          </option>
        </select>
      </div>

      <div class="form-row">
        <div class="form-group">
          <label>d_model</label>
          <input v-model.number="config.d_model" type="number" min="8" max="256" />
        </div>
        <div class="form-group">
          <label>Heads</label>
          <input v-model.number="config.n_heads" type="number" min="1" max="16" />
        </div>
        <div class="form-group">
          <label>Layers</label>
          <input v-model.number="config.n_layers" type="number" min="1" max="8" />
        </div>
      </div>

      <div class="form-row">
        <div class="form-group">
          <label>Learning Rate</label>
          <input v-model.number="config.learning_rate" type="number" step="0.0001" />
        </div>
        <div class="form-group">
          <label>Epochs</label>
          <input v-model.number="config.epochs" type="number" min="1" max="1000" />
        </div>
        <div class="form-group">
          <label>K-Fold</label>
          <input v-model.number="config.k_folds" type="number" min="2" max="10" />
        </div>
      </div>

      <div class="form-row">
        <div class="form-group">
          <label>Dropout</label>
          <input v-model.number="config.dropout" type="number" step="0.05" min="0" max="0.5" />
        </div>
        <div class="form-group">
          <label>Weight Decay</label>
          <input v-model.number="config.weight_decay" type="number" step="0.0001" />
        </div>
        <div class="form-group">
          <label>Early Stop</label>
          <input v-model.number="config.early_stopping_patience" type="number" min="1" max="100" />
        </div>
      </div>

      <div class="section-title">数据增强</div>
      <div class="form-row">
        <div class="form-group">
          <label>
            <input v-model="config.augmentation.enabled" type="checkbox" /> 启用
          </label>
        </div>
        <div class="form-group" v-if="config.augmentation.enabled">
          <label>噪声 sigma</label>
          <input v-model.number="config.augmentation.gaussian_noise_sigma" type="number" step="0.01" />
        </div>
        <div class="form-group" v-if="config.augmentation.enabled">
          <label>
            <input v-model="config.augmentation.bootstrap_enabled" type="checkbox" /> Bootstrap
          </label>
        </div>
      </div>

      <div class="actions">
        <button class="btn start" @click="handleStart" :disabled="!config.dataset_id || isRunning">
          开始训练
        </button>
        <button class="btn stop" @click="handleStop" :disabled="!isRunning">
          停止训练
        </button>
      </div>

      <p v-if="error" class="error">{{ error }}</p>
    </section>

    <section class="progress-panel">
      <h2>训练进度</h2>

      <div v-if="statusData" class="status-bar">
        <span class="status-badge" :class="statusData.status">
          {{ statusText }}
        </span>
        <span v-if="isRunning">
          Fold {{ statusData.current_fold }}/{{ config.k_folds }} -
          Epoch {{ statusData.current_epoch }}
        </span>
      </div>

      <div v-if="progress.length" class="chart-container">
        <v-chart class="chart" :option="chartOption" autoresize />
      </div>

      <div v-else class="empty">等待训练开始...</div>

      <div v-if="checkpoints.length" class="checkpoints">
        <h3>已保存检查点</h3>
        <ul>
          <li v-for="cp in checkpoints" :key="cp.id">
            <span>{{ cp.dataset_id }}</span>
            <span class="loss">Loss: {{ cp.final_loss.toFixed(4) }}</span>
            <button class="btn small" @click="handleLoadCheckpoint(cp.id)">
              加载
            </button>
          </li>
        </ul>
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

const chartOption = computed(() => {
  const trainData = progress.value.map((p) => [p.epoch, p.train_loss]);
  const valData = progress.value.map((p) => [p.epoch, p.val_loss]);
  return {
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
  grid-template-columns: 360px 1fr;
  height: calc(100vh - 48px);
  font-family: system-ui, sans-serif;
}

.config-panel {
  border-right: 1px solid #e0e0e0;
  padding: 16px;
  overflow-y: auto;
  background: #fafafa;
}

.config-panel h2 {
  margin: 0 0 16px;
  font-size: 16px;
  color: #2e7d32;
}

.section-title {
  font-weight: 500;
  margin: 12px 0 8px;
  color: #333;
}

.form-row {
  display: flex;
  gap: 8px;
  margin-bottom: 8px;
}

.form-group {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.form-group label {
  font-size: 12px;
  color: #666;
}

.form-group input[type="number"],
.form-group select {
  padding: 4px 8px;
  border: 1px solid #ddd;
  border-radius: 4px;
  font-size: 13px;
}

.actions {
  display: flex;
  gap: 8px;
  margin-top: 16px;
}

.btn {
  padding: 8px 16px;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 14px;
}

.btn.start {
  background: #4caf50;
  color: white;
  flex: 1;
}

.btn.start:disabled {
  background: #a5d6a7;
  cursor: not-allowed;
}

.btn.stop {
  background: #f44336;
  color: white;
}

.btn.stop:disabled {
  background: #ef9a9a;
  cursor: not-allowed;
}

.btn.small {
  padding: 2px 8px;
  font-size: 12px;
  background: #4caf50;
  color: white;
}

.error {
  color: #f44336;
  font-size: 13px;
  margin-top: 8px;
}

.progress-panel {
  padding: 16px;
  overflow-y: auto;
}

.progress-panel h2 {
  margin: 0 0 12px;
  font-size: 16px;
  color: #2e7d32;
}

.status-bar {
  display: flex;
  gap: 12px;
  align-items: center;
  margin-bottom: 12px;
  font-size: 13px;
}

.status-badge {
  padding: 2px 10px;
  border-radius: 10px;
  font-weight: 500;
}

.status-badge.idle { background: #e0e0e0; }
.status-badge.running { background: #fff3e0; color: #e65100; }
.status-badge.completed { background: #e8f5e9; color: #2e7d32; }
.status-badge.stopped { background: #fce4ec; color: #c62828; }
.status-badge.error { background: #ffebee; color: #b71c1c; }

.chart-container {
  margin-bottom: 16px;
}

.chart {
  height: 300px;
  width: 100%;
}

.empty {
  color: #999;
  text-align: center;
  padding: 40px;
}

.checkpoints h3 {
  font-size: 14px;
  color: #2e7d32;
  margin: 12px 0 8px;
}

.checkpoints ul {
  list-style: none;
  padding: 0;
}

.checkpoints li {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 6px 0;
  border-bottom: 1px solid #eee;
  font-size: 13px;
}

.checkpoints .loss {
  color: #666;
}
</style>
