<template>
  <div class="data-import">
    <aside class="sidebar">
      <h2>数据集</h2>
      <button class="upload-btn" @click="triggerUpload">+ 导入 CSV</button>
      <input
        ref="fileInput"
        type="file"
        accept=".csv"
        hidden
        @change="handleUpload"
      />
      <ul class="dataset-list">
        <li
          v-for="ds in datasets"
          :key="ds.id"
          :class="{ active: selectedId === ds.id }"
          @click="selectDataset(ds.id)"
        >
          <span class="ds-name">{{ ds.name }}</span>
          <span class="ds-meta">{{ ds.target }} · {{ ds.plant_part }}</span>
          <button
            v-if="!ds.is_preset"
            class="delete-btn"
            title="删除"
            @click.stop="handleDelete(ds.id)"
          >
            ×
          </button>
        </li>
      </ul>
      <p v-if="loading" class="status">加载中...</p>
      <p v-if="error" class="status error">{{ error }}</p>
    </aside>

    <main class="content">
      <template v-if="selected">
        <div class="meta-tags">
          <span class="tag">样本数: {{ selected.n_samples }}</span>
          <span class="tag">特征数: {{ selected.n_features }}</span>
          <span class="tag highlight">Target: {{ selected.target }}</span>
          <span class="tag">部位: {{ selected.plant_part }}</span>
          <span v-if="!selected.is_preset" class="tag custom">自定义</span>
          <span v-else class="tag">预置</span>
        </div>

        <h3>数据预览</h3>
        <div class="table-wrapper">
          <table v-if="selected.preview?.length">
            <thead>
              <tr>
                <th
                  v-for="col in columns"
                  :key="col"
                  :class="{ target: col === selected.target }"
                >
                  {{ col }}
                </th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="(row, i) in selected.preview" :key="i">
                <td v-for="col in columns" :key="col">{{ row[col] }}</td>
              </tr>
            </tbody>
          </table>
        </div>

        <h3>特征统计</h3>
        <div v-if="stats" class="table-wrapper">
          <table class="stats-table">
            <thead>
              <tr>
                <th>特征</th>
                <th>均值</th>
                <th>标准差</th>
                <th>最小值</th>
                <th>最大值</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="s in stats.stats" :key="s.column">
                <td :class="{ target: s.column === selected.target }">
                  {{ s.column }}
                </td>
                <td>{{ s.mean.toFixed(2) }}</td>
                <td>{{ s.std.toFixed(2) }}</td>
                <td>{{ s.min.toFixed(2) }}</td>
                <td>{{ s.max.toFixed(2) }}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </template>

      <div v-else class="empty">
        <p>请从左侧选择一个数据集，或导入自定义 CSV 文件</p>
      </div>
    </main>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from "vue";
import { useDatasets } from "@/composables/useDatasets";
import { useWorkflow } from "@/composables/useWorkflow";
import type {
  DatasetMeta,
  DatasetDetail,
  DatasetStatsResponse,
} from "@/types/dataset";

const { listDatasets, getDataset, getDatasetStats, uploadDataset, deleteDataset } =
  useDatasets();

const { markDatasetsAvailable } = useWorkflow();
const datasets = ref<DatasetMeta[]>([]);
const selected = ref<DatasetDetail | null>(null);
const stats = ref<DatasetStatsResponse | null>(null);
const selectedId = ref("");
const loading = ref(false);
const error = ref("");
const fileInput = ref<HTMLInputElement | null>(null);

const columns = computed(() => {
  if (!selected.value) return [];
  return [...selected.value.feature_names, selected.value.target];
});

async function fetchData() {
  loading.value = true;
  error.value = "";
  try {
    datasets.value = await listDatasets();
    if (datasets.value.length > 0) {
      markDatasetsAvailable();
    }
  } catch {
    error.value = "无法连接后端";
  } finally {
    loading.value = false;
  }
}

async function selectDataset(id: string) {
  selectedId.value = id;
  error.value = "";
  try {
    selected.value = await getDataset(id);
    stats.value = await getDatasetStats(id);
  } catch {
    error.value = "加载数据集失败";
  }
}

function triggerUpload() {
  fileInput.value?.click();
}

async function handleUpload(event: Event) {
  const input = event.target as HTMLInputElement;
  const file = input.files?.[0];
  if (!file) return;
  try {
    const meta = await uploadDataset(file);
    await fetchData();
    await selectDataset(meta.id);
  } catch {
    error.value = "导入失败，请检查 CSV 格式";
  }
  input.value = "";
}

async function handleDelete(id: string) {
  try {
    await deleteDataset(id);
    if (selectedId.value === id) {
      selected.value = null;
      stats.value = null;
      selectedId.value = "";
    }
    await fetchData();
  } catch {
    error.value = "删除失败";
  }
}

onMounted(fetchData);
</script>

<style scoped>
.data-import {
  display: grid;
  grid-template-columns: 260px 1fr;
  height: calc(100vh - 48px);
  font-family: system-ui, sans-serif;
}

.sidebar {
  border-right: 1px solid #e0e0e0;
  padding: 16px;
  overflow-y: auto;
  background: #fafafa;
}

.sidebar h2 {
  margin: 0 0 12px;
  font-size: 16px;
  color: #2e7d32;
}

.upload-btn {
  width: 100%;
  padding: 8px;
  background: #4caf50;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  margin-bottom: 12px;
  font-size: 14px;
}

.upload-btn:hover {
  background: #388e3c;
}

.dataset-list {
  list-style: none;
  padding: 0;
  margin: 0;
}

.dataset-list li {
  padding: 8px;
  border-radius: 4px;
  cursor: pointer;
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 4px;
  margin-bottom: 4px;
}

.dataset-list li:hover {
  background: #e8f5e9;
}

.dataset-list li.active {
  background: #c8e6c9;
}

.ds-name {
  font-weight: 500;
  flex: 1 1 100%;
}

.ds-meta {
  font-size: 12px;
  color: #666;
}

.delete-btn {
  background: none;
  border: none;
  color: #f44336;
  cursor: pointer;
  font-size: 16px;
  padding: 0 4px;
  margin-left: auto;
}

.content {
  padding: 20px;
  overflow-y: auto;
}

.meta-tags {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
  margin-bottom: 16px;
}

.tag {
  padding: 4px 10px;
  background: #e8f5e9;
  border-radius: 12px;
  font-size: 13px;
  color: #333;
}

.tag.highlight {
  background: #fff3e0;
  color: #e65100;
}

.tag.custom {
  background: #e3f2fd;
  color: #1565c0;
}

h3 {
  color: #2e7d32;
  margin: 16px 0 8px;
  font-size: 15px;
}

.table-wrapper {
  overflow-x: auto;
  margin-bottom: 16px;
}

table {
  border-collapse: collapse;
  width: 100%;
  font-size: 13px;
}

th,
td {
  padding: 6px 10px;
  border: 1px solid #e0e0e0;
  text-align: right;
  white-space: nowrap;
}

th {
  background: #f5f5f5;
  font-weight: 500;
  text-align: center;
}

th.target,
td.target {
  background: #fff3e0;
  font-weight: 600;
}

.status {
  color: #666;
  font-size: 13px;
  margin-top: 8px;
}

.status.error {
  color: #f44336;
}

.empty {
  display: flex;
  align-items: center;
  justify-content: center;
  height: 100%;
  color: #999;
}
</style>
