<template>
  <nav class="tab-bar">
    <router-link
      v-for="(tab, index) in TAB_ORDER"
      :key="tab"
      :to="enabled[tab] ? { name: tab } : {}"
      :class="['tab', { active: isActive(tab), disabled: !enabled[tab] }]"
      :tabindex="enabled[tab] ? 0 : -1"
      @click.prevent="enabled[tab] && $router.push({ name: tab })"
    >
      <span class="step">{{ index + 1 }}</span>
      <span class="label">{{ TAB_LABELS[tab] }}</span>
    </router-link>
  </nav>
</template>

<script setup lang="ts">
import { useRoute } from "vue-router";
import {
  TAB_ORDER,
  TAB_LABELS,
  useWorkflow,
  type TabName,
} from "@/composables/useWorkflow";

const route = useRoute();
const { tabEnabled: enabled } = useWorkflow();

function isActive(tab: TabName): boolean {
  return route.name === tab;
}
</script>

<style scoped>
.tab-bar {
  display: flex;
  height: 48px;
  background: var(--color-canvas);
  border-bottom: 2px solid var(--color-hairline);
  font-family: system-ui, sans-serif;
}

.tab {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 0 20px;
  text-decoration: none;
  color: var(--color-steel);
  font-size: 14px;
  cursor: pointer;
  border-bottom: 3px solid transparent;
  transition: color 0.2s, border-color 0.2s;
  user-select: none;
}

.tab:hover:not(.disabled) {
  color: var(--color-ink);
  background: var(--color-tint-mint);
}

.tab.active {
  color: var(--color-ink);
  font-weight: 600;
  border-bottom-color: var(--color-primary);
}

.tab.disabled {
  color: var(--color-muted);
  cursor: not-allowed;
}

.step {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 22px;
  height: 22px;
  border-radius: 50%;
  background: var(--color-hairline);
  color: var(--color-steel);
  font-size: 12px;
  font-weight: 600;
}

.tab.active .step {
  background: var(--color-primary);
  color: var(--color-on-primary);
}

.tab.disabled .step {
  background: var(--color-hairline-soft);
  color: var(--color-muted);
}
</style>
