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
          <span class="tag highlight">
            Target:
            <select
              class="target-select"
              :value="selected.target"
              @change="handleTargetChange"
            >
              <option v-for="col in selected.columns" :key="col" :value="col">
                {{ col }}
              </option>
            </select>
          </span>
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

const {
  listDatasets,
  getDataset,
  getDatasetStats,
  uploadDataset,
  deleteDataset,
  updateTarget,
} = useDatasets();

const { markDatasetsAvailable, markTargetSelected } = useWorkflow();
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
    markTargetSelected(!!selected.value?.target);
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

async function handleTargetChange(event: Event) {
  const select = event.target as HTMLSelectElement;
  const newTarget = select.value;
  if (!selected.value || newTarget === selected.value.target) return;
  try {
    const updated = await updateTarget(selected.value.id, newTarget);
    await selectDataset(updated.id);
    markTargetSelected(true);
  } catch {
    error.value = "切换目标变量失败";
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
  border-right: 1px solid var(--color-hairline);
  padding: 16px;
  overflow-y: auto;
  background: var(--color-surface-soft);
}

.sidebar h2 {
  margin: 0 0 12px;
  font-size: 16px;
  color: var(--color-ink);
}

.upload-btn {
  width: 100%;
  padding: 8px;
  background: var(--color-primary);
  color: white;
  border: none;
  border-radius: var(--radius-sm);
  cursor: pointer;
  margin-bottom: 12px;
  font-size: 14px;
}

.upload-btn:hover {
  background: var(--color-primary-pressed);
}

.dataset-list {
  list-style: none;
  padding: 0;
  margin: 0;
}

.dataset-list li {
  padding: 8px;
  border-radius: var(--radius-sm);
  cursor: pointer;
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 4px;
  margin-bottom: 4px;
}

.dataset-list li:hover {
  background: var(--color-tint-mint);
}

.dataset-list li.active {
  background: var(--color-tint-mint);
}

.ds-name {
  font-weight: 500;
  flex: 1 1 100%;
}

.ds-meta {
  font-size: 12px;
  color: var(--color-steel);
}

.delete-btn {
  background: none;
  border: none;
  color: var(--color-error);
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
  background: var(--color-tint-mint);
  border-radius: var(--radius-lg);
  font-size: 13px;
  color: var(--color-charcoal);
}

.tag.highlight {
  background: var(--color-tint-peach);
  color: var(--color-warning);
}

.tag.custom {
  background: var(--color-tint-sky);
  color: var(--color-link-blue);
}

h3 {
  color: var(--color-success);
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
  border: 1px solid var(--color-hairline);
  text-align: right;
  white-space: nowrap;
}

th {
  background: var(--color-surface);
  font-weight: 500;
  text-align: center;
}

th.target,
td.target {
  background: var(--color-tint-peach);
  font-weight: 600;
}

.status {
  color: var(--color-steel);
  font-size: 13px;
  margin-top: 8px;
}

.status.error {
  color: var(--color-error);
}

.empty {
  display: flex;
  align-items: center;
  justify-content: center;
  height: 100%;
  color: var(--color-stone);
}

.target-select {
  background: transparent;
  border: 1px solid var(--color-warning);
  border-radius: var(--radius-sm);
  color: var(--color-warning);
  font-size: 13px;
  padding: 0 4px;
  margin-left: 4px;
  cursor: pointer;
}

.target-select:focus {
  outline: 2px solid var(--color-warning);
  outline-offset: 1px;
}
</style>
