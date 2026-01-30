<template>
  <div class="formulation">
    <h2>最优配比</h2>

    <div v-if="exprId" class="expr-info">
      当前表达式: <code>{{ exprId }}</code>
    </div>

    <div v-if="!exprId" class="empty-hint">
      请先在「表达式推导」页面生成表达式
    </div>

    <template v-else>
      <div class="toolbar">
        <label class="param-label">
          Top K
          <input
            type="number"
            v-model.number="topK"
            min="1"
            max="50"
            class="input"
          />
        </label>
        <label class="param-label">
          采样数
          <input
            type="number"
            v-model.number="nSamples"
            min="100"
            max="10000"
            step="100"
            class="input"
          />
        </label>
        <button
          :disabled="loading"
          @click="runFormulation"
          class="btn-primary"
        >
          开始寻优
        </button>
        <button v-if="result" @click="exportCsv" class="btn-secondary">
          导出 CSV
        </button>
      </div>

      <div v-if="loading" class="status">寻优计算中...</div>
      <div v-else-if="error" class="error">{{ error }}</div>

      <template v-if="result">
        <div class="section">
          <h3>注意力权重</h3>
          <v-chart
            :option="attentionOption"
            autoresize
            style="height: 260px"
          />
        </div>

        <div class="section">
          <h3>候选配比</h3>
          <table class="result-table">
            <thead>
              <tr>
                <th>排名</th>
                <th v-for="f in result.feature_names" :key="f">{{ f }}</th>
                <th>预测药效</th>
              </tr>
            </thead>
            <tbody>
              <tr
                v-for="c in result.candidates"
                :key="c.rank"
                :class="{ highlight: c.rank === 1 }"
              >
                <td>{{ c.rank }}</td>
                <td v-for="f in result.feature_names" :key="f">
                  {{ c.components[f].toFixed(4) }}
                </td>
                <td>{{ c.predicted_response.toFixed(4) }}</td>
              </tr>
            </tbody>
          </table>
        </div>

        <div class="section">
          <h3>候选配比对比</h3>
          <v-chart
            :option="radarOption"
            autoresize
            style="height: 360px"
          />
        </div>
      </template>
    </template>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from "vue";
import VChart from "vue-echarts";
import "echarts";
import { useWorkflow } from "@/composables/useWorkflow";
import { useFormulation } from "@/composables/useFormulation";

const { state } = useWorkflow();
const { result, loading, error, formulate, exportCsv } = useFormulation();

const exprId = computed(() => state.currentExprId);

const topK = ref(5);
const nSamples = ref(1000);

async function runFormulation() {
  if (!exprId.value) return;
  await formulate(exprId.value, topK.value, nSamples.value);
}

const attentionOption = computed(() => {
  if (!result.value) return {};
  const entries = Object.entries(result.value.attention_weights).sort(
    (a, b) => b[1] - a[1],
  );
  return {
    tooltip: { trigger: "axis" },
    grid: { left: 100, right: 30, top: 10, bottom: 30 },
    xAxis: { type: "value", name: "权重" },
    yAxis: {
      type: "category",
      data: entries.map((e) => e[0]),
    },
    series: [
      {
        type: "bar",
        data: entries.map((e) => e[1]),
        itemStyle: { color: "#4caf50" },
      },
    ],
  };
});

const radarOption = computed(() => {
  if (!result.value) return {};
  const { candidates, feature_names } = result.value;
  const top = candidates.slice(0, Math.min(3, candidates.length));

  const maxValues: Record<string, number> = {};
  for (const f of feature_names) {
    maxValues[f] = Math.max(...candidates.map((c) => c.components[f]));
  }

  const indicator = feature_names.map((f) => ({
    name: f,
    max: maxValues[f] > 0 ? maxValues[f] * 1.1 : 1,
  }));

  const colors = ["#4caf50", "#2196f3", "#ff9800"];
  const series = top.map((c, i) => ({
    value: feature_names.map((f) => c.components[f]),
    name: `#${c.rank}`,
    areaStyle: { opacity: 0.15 },
    lineStyle: { color: colors[i] },
    itemStyle: { color: colors[i] },
  }));

  return {
    tooltip: {},
    legend: { data: top.map((c) => `#${c.rank}`), top: 0 },
    radar: { indicator },
    series: [{ type: "radar", data: series }],
  };
});
</script>

<style scoped>
.formulation {
  max-width: 1200px;
  margin: 0 auto;
  padding: 20px;
  font-family: system-ui, sans-serif;
}

h2 {
  color: #2e7d32;
}

.expr-info {
  padding: 8px 12px;
  background: #e3f2fd;
  border-radius: 4px;
  margin-bottom: 12px;
  font-size: 14px;
  color: #1565c0;
}
.expr-info code {
  background: #bbdefb;
  padding: 2px 6px;
  border-radius: 3px;
  font-family: monospace;
}

h3 {
  margin: 0 0 8px 0;
  font-size: 14px;
  color: #333;
}

.empty-hint {
  padding: 40px;
  text-align: center;
  color: #666;
  background: #f5f5f5;
  border-radius: 4px;
}

.toolbar {
  display: flex;
  gap: 12px;
  align-items: center;
  margin-bottom: 16px;
  flex-wrap: wrap;
}

.param-label {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 14px;
}

.input {
  width: 90px;
  padding: 6px 8px;
  border: 1px solid #ccc;
  border-radius: 4px;
}

.btn-primary,
.btn-secondary {
  padding: 6px 16px;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
}

.btn-primary {
  background: #4caf50;
}

.btn-secondary {
  background: #2196f3;
}

.btn-primary:disabled {
  background: #ccc;
  cursor: not-allowed;
}

.status {
  color: #666;
}

.error {
  color: #f44336;
}

.section {
  margin-bottom: 20px;
  border: 1px solid #e0e0e0;
  border-radius: 4px;
  padding: 12px;
}

.result-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
}

.result-table th,
.result-table td {
  border: 1px solid #e0e0e0;
  padding: 6px 10px;
  text-align: center;
}

.result-table th {
  background: #f5f5f5;
}

.result-table tr.highlight {
  background: #e8f5e9;
  font-weight: bold;
}
</style>
