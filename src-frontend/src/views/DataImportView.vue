<template>
  <div class="data-import">
    <PlPageHeader
      :step="1"
      title="数据导入"
      subtitle="导入中药成分数据集，选择目标变量进行后续分析"
    />

    <div class="data-import__grid-top">
      <!-- Card 1: Upload area -->
      <PlCard variant="base" padding="lg">
        <div
          :class="['upload-zone', { 'upload-zone--active': isDragging }]"
          @dragover.prevent="isDragging = true"
          @dragleave="isDragging = false"
          @drop.prevent="handleDrop"
        >
          <PlIcon name="upload" size="lg" />
          <p class="upload-zone__text">将 CSV 文件拖放到此处</p>
          <span class="upload-zone__or">或</span>
          <PlButton variant="primary" @click="triggerUpload">选择文件</PlButton>
          <span v-if="isDragging" class="upload-zone__hint">释放以导入</span>
        </div>

        <input
          ref="fileInput"
          type="file"
          accept=".csv"
          hidden
          @change="handleUpload"
        />

        <div v-if="loading" class="upload-zone__spinner">
          <PlSpinner size="sm" />
        </div>
      </PlCard>

      <!-- Card 2: Dataset list -->
      <PlCard variant="base" padding="md">
        <h3 class="card-title">数据集列表</h3>

        <ul v-if="datasets.length" class="dataset-list">
          <li
            v-for="ds in datasets"
            :key="ds.id"
            :class="['dataset-list__item', { 'dataset-list__item--active': selectedId === ds.id }]"
            @click="selectDataset(ds.id)"
          >
            <span class="dataset-list__name">{{ ds.name }}</span>
            <div class="dataset-list__meta">
              <PlBadge :text="ds.plant_part" variant="neutral" />
              <PlBadge v-if="ds.is_preset" text="预置" variant="tag-orange" />
              <PlBadge v-else text="自定义" variant="tag-green" />
              <PlButton
                v-if="!ds.is_preset"
                variant="ghost"
                icon="×"
                @click.stop="handleDelete(ds.id)"
              />
            </div>
          </li>
        </ul>

        <PlEmptyState
          v-else
          title="暂无数据集"
          description="导入 CSV 文件以开始分析"
          action="导入 CSV"
          @action="triggerUpload"
        />
      </PlCard>
    </div>

    <!-- Card 3: Data preview (full-width) -->
    <PlCard v-if="selected" variant="base" padding="md">
      <h3 class="card-title">数据预览</h3>

      <div class="preview-meta">
        <PlBadge :text="`样本数: ${selected.n_samples}`" variant="teal" />
        <PlBadge :text="`成分数: ${selected.n_features}`" variant="teal" />
        <PlBadge :text="`部位: ${selected.plant_part}`" variant="teal" />
        <PlBadge
          v-if="!selected.is_preset"
          text="自定义"
          variant="tag-green"
        />
        <PlBadge v-else text="预置" variant="tag-orange" />

        <div class="target-select">
          <PlSelect
            :model-value="selected.target"
            :options="targetOptions"
            label="药效指标"
            @update:model-value="handleTargetChange"
          />
        </div>
      </div>

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
    </PlCard>

    <!-- Card 4: Feature statistics (full-width) -->
    <PlCard v-if="stats" variant="stat" padding="md">
      <h3 class="card-title">成分统计</h3>

      <div class="table-wrapper">
        <table>
          <thead>
            <tr>
              <th>成分</th>
              <th>均值</th>
              <th>标准差</th>
              <th>最小值</th>
              <th>最大值</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="s in stats.stats" :key="s.column">
              <td :class="{ 'td--target': selected && s.column === selected.target }">
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
    </PlCard>

    <!-- Empty state when no dataset selected (shown below the top grid) -->
    <PlCard v-if="!selected && datasets.length > 0" variant="base" padding="lg">
      <PlEmptyState
        title="暂未选择数据集"
        description="从上方列表中选择一个数据集，或导入自定义 CSV 文件"
        action="导入 CSV"
        @action="triggerUpload"
      />
    </PlCard>

    <!-- Footer: Next step button -->
    <div class="data-import__footer">
      <PlButton variant="dark" :disabled="!selected" @click="goNext">
        下一步：模型训练
      </PlButton>
    </div>

    <PlToast :message="error" variant="error" />
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from "vue";
import { useRouter } from "vue-router";
import { useDatasets } from "@/composables/useDatasets";
import { useWorkflow } from "@/composables/useWorkflow";
import PlPageHeader from "@/components/PlPageHeader.vue";
import PlButton from "@/components/PlButton.vue";
import PlSelect from "@/components/PlSelect.vue";
import PlCard from "@/components/PlCard.vue";
import PlBadge from "@/components/PlBadge.vue";
import PlIcon from "@/components/PlIcon.vue";
import PlSpinner from "@/components/PlSpinner.vue";
import PlToast from "@/components/PlToast.vue";
import PlEmptyState from "@/components/PlEmptyState.vue";
import type {
  DatasetMeta,
  DatasetDetail,
  DatasetStatsResponse,
} from "@/types/dataset";

const router = useRouter();

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

function goNext() {
  router.push({ name: "training" });
}

onMounted(fetchData);
</script>

<style scoped>
.data-import {
  max-width: var(--content-max-width);
  margin: 0 auto;
  padding: var(--space-10);
  display: flex;
  flex-direction: column;
  gap: var(--space-6);
  font-family: var(--font-family);
}

/* ── Top grid: upload + dataset list ── */
.data-import__grid-top {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: var(--space-6);
}

@media (max-width: 960px) {
  .data-import__grid-top {
    grid-template-columns: 1fr;
  }
}

/* ── Upload zone ── */
.upload-zone {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: var(--space-3);
  padding: var(--space-8) var(--space-4);
  border: 2px dashed var(--color-hairline-strong);
  border-radius: var(--radius-lg);
  background: var(--color-surface);
  transition: border-color 0.2s, background 0.2s;
}

.upload-zone--active {
  border-color: var(--color-primary);
  background: var(--color-primary-tint);
}

.upload-zone__text {
  font-size: var(--text-body);
  color: var(--color-charcoal);
  margin: 0;
}

.upload-zone__or {
  font-size: var(--text-xs);
  color: var(--color-stone);
}

.upload-zone__hint {
  font-size: var(--text-xs);
  color: var(--color-primary);
}

.upload-zone__spinner {
  display: flex;
  justify-content: center;
  margin-top: var(--space-3);
}

/* ── Card title ── */
.card-title {
  font-size: var(--text-h3);
  font-weight: var(--text-h3-weight);
  color: var(--color-charcoal);
  margin: 0 0 var(--space-4);
}

/* ── Dataset list ── */
.dataset-list {
  list-style: none;
  padding: 0;
  margin: 0;
  display: flex;
  flex-direction: column;
  gap: var(--space-1);
}

.dataset-list__item {
  padding: var(--space-3) var(--space-3);
  border-radius: var(--radius-md);
  cursor: pointer;
  display: flex;
  flex-direction: column;
  gap: var(--space-1);
  transition: background 0.15s;
}

.dataset-list__item:hover {
  background: var(--color-surface);
}

.dataset-list__item--active {
  background: var(--color-primary-tint);
  outline: 2px solid var(--color-primary);
}

.dataset-list__name {
  font-size: var(--text-body);
  font-weight: var(--text-body-weight);
  color: var(--color-ink);
}

.dataset-list__meta {
  display: flex;
  align-items: center;
  gap: var(--space-2);
  flex-wrap: wrap;
}

/* ── Preview meta badges ── */
.preview-meta {
  display: flex;
  gap: var(--space-2);
  flex-wrap: wrap;
  align-items: center;
  margin-bottom: var(--space-4);
}

.target-select {
  margin-left: auto;
}

/* ── Tables ── */
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
  text-align: right;
  white-space: nowrap;
}

th {
  background: var(--color-surface);
  font-weight: var(--text-sm-weight);
  font-size: var(--text-sm);
  text-align: center;
  color: var(--color-charcoal);
}

.th--target,
.td--target {
  background: var(--color-tint-peach);
  font-weight: 600;
}

/* ── Footer ── */
.data-import__footer {
  display: flex;
  justify-content: flex-end;
  padding-top: var(--space-4);
}
</style>
