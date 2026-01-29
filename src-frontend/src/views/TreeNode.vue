<template>
  <div class="tree-node" :style="{ paddingLeft: depth * 20 + 'px' }">
    <div class="node-label" @click="toggle" :class="{ 'zero-impact': isZeroImpact }">
      <span class="toggle">{{ expanded ? "▾" : "▸" }}</span>
      <span :class="['node-badge', node.type]">{{ node.type[0].toUpperCase() }}</span>
      <span class="node-value">{{ node.value }}</span>
      <span v-if="impactValue !== null" class="impact-badge">
        {{ (impactValue * 100).toFixed(0) }}%
      </span>
    </div>
    <div v-if="expanded && node.children.length > 0" class="node-children">
      <TreeNode
        v-for="(child, i) in node.children"
        :key="i"
        :node="child"
        :depth="depth + 1"
        :variable-impact="variableImpact"
      />
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from "vue";
import type { ExpressionNode } from "@/types/expression";

const props = defineProps<{
  node: ExpressionNode;
  depth: number;
  variableImpact?: Record<string, number>;
}>();

const expanded = ref(props.depth < 3);

const impactValue = computed(() => {
  if (props.node.type !== "variable" || !props.variableImpact) return null;
  return props.variableImpact[props.node.value] ?? null;
});

const isZeroImpact = computed(() => impactValue.value !== null && impactValue.value === 0);

function toggle() {
  expanded.value = !expanded.value;
}
</script>

<style scoped>
.tree-node { margin: 2px 0; }
.node-label {
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 2px 4px;
  border-radius: 3px;
}
.node-label:hover { background: #f5f5f5; }
.toggle { color: #999; width: 12px; }
.node-badge {
  font-size: 11px;
  padding: 1px 6px;
  border-radius: 3px;
  color: white;
}
.node-badge.operator { background: #2196f3; }
.node-badge.function { background: #ff9800; }
.node-badge.variable { background: #4caf50; }
.node-badge.constant { background: #9e9e9e; }
.node-value { font-family: monospace; }
.impact-badge {
  font-size: 10px;
  padding: 1px 5px;
  border-radius: 3px;
  background: #e8f5e9;
  color: #2e7d32;
  margin-left: 4px;
}
.zero-impact { opacity: 0.4; }
</style>
