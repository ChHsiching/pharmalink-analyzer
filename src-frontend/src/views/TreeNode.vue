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
  border-radius: var(--radius-xs);
}
.node-label:hover { background: var(--color-surface); }
.toggle { color: var(--color-stone); width: 12px; }
.node-badge {
  font-size: 11px;
  padding: 1px 6px;
  border-radius: var(--radius-xs);
  color: white;
}
.node-badge.operator { background: var(--color-link-blue); }
.node-badge.function { background: var(--color-warning); }
.node-badge.variable { background: var(--color-success); }
.node-badge.constant { background: var(--color-muted); }
.node-value { font-family: monospace; }
.impact-badge {
  font-size: 10px;
  padding: 1px 5px;
  border-radius: var(--radius-xs);
  background: var(--color-tint-mint);
  color: var(--color-success);
  margin-left: 4px;
}
.zero-impact { opacity: 0.4; }
</style>
