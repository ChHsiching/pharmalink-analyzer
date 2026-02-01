<template>
  <div class="data-import">
    <aside class="sidebar">
      <h2 class="sidebar__title">数据集</h2>

      <div
        :class="['drop-zone', { 'drop-zone--active': isDragging }]"
        @dragover.prevent="isDragging = true"
        @dragleave="isDragging = false"
        @drop.prevent="handleDrop"
      >
        <PlButton variant="primary" icon="+" @click="triggerUpload">
          导入 CSV
        </PlButton>
        <span v-if="isDragging" class="drop-zone__hint">释放以导入</span>
        <span v-else class="drop-zone__hint">或拖放 CSV 文件到此处</span>
      </div>

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
          :class="['dataset-list__item', { 'dataset-list__item--active': selectedId === ds.id }]"
          @click="selectDataset(ds.id)"
        >
          <span class="dataset-list__name">{{ ds.name }}</span>
          <span class="dataset-list__meta">{{ ds.target }} · {{ ds.plant_part }}</span>
          <PlButton
            v-if="!ds.is_preset"
            variant="ghost"
            icon="×"
            @click.stop="handleDelete(ds.id)"
          />
        </li>
      </ul>

      <div v-if="loading" class="sidebar__status">
        <PlSpinner size="sm" />
      </div>
      <PlToast :message="error" variant="error" />
    </aside>

    <main class="content">
      <template v-if="selected">
        <div class="meta-tags">
          <PlBadge :text="`样本数: ${selected.n_samples}`" variant="teal" />
          <PlBadge :text="`特征数: ${selected.n_features}`" variant="teal" />

          <PlCard variant="feature" padding="sm" class="target-card">
            <PlSelect
              :model-value="selected.target"
              :options="targetOptions"
              label="Target"
              @update:model-value="handleTargetChange"
            />
          </PlCard>

          <PlBadge :text="`部位: ${selected.plant_part}`" variant="teal" />
          <PlBadge
            v-if="!selected.is_preset"
            text="自定义"
            variant="tag-green"
          />
          <PlBadge v-else text="预置" variant="tag-orange" />
        </div>

        <h3 class="section-heading">数据预览</h3>
        <div class="table-wrapper">
          <table v-if="selected.preview?.length">
            <thead>
              <tr>
                <th
                  v-for="col in columns"
                  :key="col"
                  :class="{ 'th--target': col === selected.target }"
                >
                  {{ col }}
                </th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="(row, i) in selected.preview" :key="i">
                <td
                  v-for="col in columns"
                  :key="col"
                  :class="{ 'td--target': col === selected.target }"
                >
                  {{ row[col] }}
                </td>
              </tr>
            </tbody>
          </table>
        </div>

        <h3 class="section-heading">特征统计</h3>
        <div v-if="stats" class="table-wrapper">
          <table>
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
                <td :class="{ 'td--target': s.column === selected.target }">
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

      <PlEmptyState
        v-else
        title="暂无选择的数据集"
        description="请从左侧选择一个数据集，或导入自定义 CSV 文件"
        action="导入 CSV"
        @action="triggerUpload"
      />
    </main>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from "vue";
import { useDatasets } from "@/composables/useDatasets";
import { useWorkflow } from "@/composables/useWorkflow";
import PlButton from "@/components/PlButton.vue";
import PlSelect from "@/components/PlSelect.vue";
import PlCard from "@/components/PlCard.vue";
import PlBadge from "@/components/PlBadge.vue";
import PlSpinner from "@/components/PlSpinner.vue";
import PlToast from "@/components/PlToast.vue";
import PlEmptyState from "@/components/PlEmptyState.vue";
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
const isDragging = ref(false);

const columns = computed(() => {
  if (!selected.value) return [];
  return [...selected.value.feature_names, selected.value.target];
});

const targetOptions = computed(() =>
  selected.value?.columns.map((c) => ({ value: c, label: c })) ?? [],
);

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

async function processFile(file: File) {
  try {
    const meta = await uploadDataset(file);
    await fetchData();
    await selectDataset(meta.id);
  } catch {
    error.value = "导入失败，请检查 CSV 格式";
  }
}

async function handleUpload(event: Event) {
  const input = event.target as HTMLInputElement;
  const file = input.files?.[0];
  if (!file) return;
  await processFile(file);
  input.value = "";
}

async function handleDrop(event: DragEvent) {
  isDragging.value = false;
  const file = event.dataTransfer?.files[0];
  if (!file || !file.name.endsWith(".csv")) {
    error.value = "请拖入 CSV 文件";
    return;
  }
  await processFile(file);
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

async function handleTargetChange(newTarget: string) {
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
  grid-template-columns: 280px 1fr;
  height: calc(100vh - 48px);
  font-family: var(--font-family);
}

/* ── Sidebar ── */
.sidebar {
  border-right: 1px solid var(--color-hairline);
  padding: var(--radius-md);
  overflow-y: auto;
  background: var(--color-surface-soft);
}

.sidebar__title {
  margin: 0 0 12px;
  font-size: var(--font-size-heading);
  font-weight: var(--font-weight-heading);
  color: var(--color-ink);
}

.sidebar__status {
  display: flex;
  justify-content: center;
  margin-top: 12px;
}

/* ── Drop zone ── */
.drop-zone {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;
  padding: 16px 12px;
  margin-bottom: 12px;
  border: 2px dashed var(--color-hairline-strong);
  border-radius: var(--radius-lg);
  background: var(--color-canvas);
  transition: border-color 0.2s, background 0.2s;
}

.drop-zone--active {
  border-color: var(--color-primary);
  background: var(--color-primary-tint);
}

.drop-zone__hint {
  font-size: 12px;
  color: var(--color-stone);
}

/* ── Dataset list ── */
.dataset-list {
  list-style: none;
  padding: 0;
  margin: 0;
}

.dataset-list__item {
  padding: 8px;
  border-radius: var(--radius-sm);
  cursor: pointer;
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 4px;
  margin-bottom: 4px;
}

.dataset-list__item:hover {
  background: var(--color-tint-mint);
}

.dataset-list__item--active {
  background: var(--color-tint-mint);
}

.dataset-list__name {
  font-weight: var(--font-weight-medium);
  flex: 1 1 100%;
  font-size: var(--font-size-body);
  color: var(--color-ink);
}

.dataset-list__meta {
  font-size: 12px;
  color: var(--color-steel);
}

/* ── Content ── */
.content {
  padding: 20px;
  overflow-y: auto;
}

.meta-tags {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
  align-items: center;
  margin-bottom: 16px;
}

.target-card {
  display: inline-block;
}

.section-heading {
  color: var(--color-success);
  margin: 16px 0 8px;
  font-size: var(--font-size-heading);
  font-weight: var(--font-weight-heading);
}

/* ── Tables ── */
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
  font-weight: var(--font-weight-medium);
  text-align: center;
}

.th--target,
.td--target {
  background: var(--color-tint-peach);
  font-weight: var(--font-weight-heading);
}

/* ── Responsive ── */
@media (max-width: 1280px) {
  .data-import {
    grid-template-columns: 240px 1fr;
  }
}

@media (max-width: 1024px) {
  .data-import {
    grid-template-columns: 1fr;
    height: auto;
  }

  .sidebar {
    border-right: none;
    border-bottom: 1px solid var(--color-hairline);
    max-height: 320px;
  }
}
</style>
