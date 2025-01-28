<template>
  <div class="expression-derivation">
    <h2>表达式推导</h2>
    <div class="toolbar">
      <select v-model="selectedCheckpoint" class="checkpoint-select">
        <option value="">选择检查点</option>
        <option v-for="cp in checkpoints" :key="cp.id" :value="cp.id">
          {{ cp.id }} (loss: {{ cp.final_loss.toFixed(4) }})
        </option>
      </select>
      <select v-model="selectedPreset" class="preset-select">
        <option value="quick">快速 (~1 min)</option>
        <option value="standard">标准 (~2-5 min)</option>
        <option value="thorough">深度 (~5-15 min)</option>
      </select>
      <button @click="generate" :disabled="!selectedCheckpoint || loading" class="btn-generate">
        生成表达式
      </button>
      <template v-if="expression">
        <button @click="simplify" :disabled="loading" class="btn-action">精简</button>
        <button @click="optimize" :disabled="loading" class="btn-action">
          优化<template v-if="expression?.pareto_count"> ({{ expression.pareto_index + 1 }}/{{ expression.pareto_count }})</template>
        </button>
        <button @click="undo" :disabled="loading || !canUndo" class="btn-action">撤销</button>
        <button @click="redo" :disabled="loading || !canRedo" class="btn-action">重做</button>
      </template>
    </div>
    <div v-if="loading" class="loading">{{ loadingMessage }}</div>
    <div v-else-if="error" class="error">{{ error }}</div>
    <template v-if="expression">
      <div class="metrics">
        <span class="metric">复杂度: {{ expression.complexity }}</span>
        <span class="metric">R²: {{ expression.r2_score.toFixed(4) }}</span>
      </div>
      <div class="panels">
        <div class="panel latex-panel">
          <h3>符号表达式</h3>
          <div ref="katexRef" class="katex-container"></div>
        </div>
        <div class="panel tree-panel">
          <h3>表达式树</h3>
          <div class="tree-container">
            <TreeNode :node="expression.tree" :depth="0" />
          </div>
        </div>
      </div>
    </template>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, watch, onMounted, onUnmounted, nextTick } from "vue";
import katex from "katex";
import { useTraining } from "@/composables/useTraining";
import { useExpression } from "@/composables/useExpression";
import { useWorkflow } from "@/composables/useWorkflow";
import TreeNode from "./TreeNode.vue";

const selectedCheckpoint = ref("");
const selectedPreset = ref("standard");
const katexRef = ref<HTMLElement>();

const { checkpoints, fetchCheckpoints } = useTraining();
const { markCurrentExpression } = useWorkflow();
const {
  expression, history, loading, error,
  generateExpression, simplifyExpression, optimizeExpression,
  fetchHistory, undoExpression,
  stopPolling,
} = useExpression();

const canUndo = computed(() => history.value ? history.value.current_index > 0 : false);
const canRedo = computed(() => history.value ? history.value.current_index < history.value.history.length - 1 : false);
const loadingMessage = computed(() => expression.value ? "处理中..." : "符号回归可能需要数分钟，请耐心等待...");

async function generate() {
  if (!selectedCheckpoint.value) return;
  await generateExpression(selectedCheckpoint.value, 10, selectedPreset.value);
  if (expression.value) {
    await fetchHistory(expression.value.expr_id);
    markCurrentExpression(expression.value.expr_id);
  }
}
async function simplify() {
  if (!expression.value) return;
  const ok = await simplifyExpression(expression.value.expr_id);
  if (ok && expression.value) {
    await fetchHistory(expression.value.expr_id);
    markCurrentExpression(expression.value.expr_id);
  }
}
async function optimize() {
  if (!expression.value) return;
  const ok = await optimizeExpression(expression.value.expr_id);
  if (ok && expression.value) {
    await fetchHistory(expression.value.expr_id);
    markCurrentExpression(expression.value.expr_id);
  }
}
async function undo() {
  if (!expression.value) return;
  const ok = await undoExpression(expression.value.expr_id, 1);
  if (ok && expression.value) {
    await fetchHistory(expression.value.expr_id);
    markCurrentExpression(expression.value.expr_id);
  }
}
async function redo() {
  if (!expression.value) return;
  const ok = await undoExpression(expression.value.expr_id, -1);
  if (ok && expression.value) {
    await fetchHistory(expression.value.expr_id);
    markCurrentExpression(expression.value.expr_id);
  }
}

function renderLatex() {
  if (katexRef.value && expression.value?.latex) {
    katex.render(expression.value.latex, katexRef.value, {
      displayMode: true,
      throwOnError: false,
    });
  }
}

watch(() => expression.value?.latex, () => nextTick(renderLatex));
onMounted(() => { fetchCheckpoints(); });
onUnmounted(() => { stopPolling(); });
</script>

<style scoped>
.expression-derivation {
  max-width: 1200px;
  margin: 0 auto;
  padding: 20px;
  font-family: system-ui, sans-serif;
}
h2 { color: #2e7d32; }
.toolbar {
  display: flex;
  gap: 8px;
  align-items: center;
  margin-bottom: 16px;
  flex-wrap: wrap;
}
.checkpoint-select {
  padding: 6px 12px;
  border: 1px solid #ccc;
  border-radius: 4px;
}
.preset-select {
  padding: 6px 12px;
  border: 1px solid #ccc;
  border-radius: 4px;
}
.btn-generate, .btn-action {
  padding: 6px 16px;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
}
.btn-generate { background: #4caf50; }
.btn-action { background: #2196f3; }
.btn-generate:disabled, .btn-action:disabled {
  background: #ccc;
  cursor: not-allowed;
}
.metrics {
  display: flex;
  gap: 20px;
  margin-bottom: 16px;
}
.metric {
  background: #e8f5e9;
  padding: 6px 14px;
  border-radius: 4px;
  font-size: 14px;
}
.panels {
  display: flex;
  gap: 20px;
}
.panel {
  flex: 1;
  border: 1px solid #e0e0e0;
  border-radius: 8px;
  padding: 16px;
}
.panel h3 { margin-top: 0; color: #333; }
.katex-container {
  min-height: 80px;
  display: flex;
  align-items: center;
  justify-content: center;
  overflow-x: auto;
}
.tree-container {
  max-height: 500px;
  overflow-y: auto;
  font-family: monospace;
  font-size: 14px;
}
.loading { color: #666; }
.error { color: #f44336; }
</style>
