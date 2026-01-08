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
  background: #fff;
  border-bottom: 2px solid #e0e0e0;
  font-family: system-ui, sans-serif;
}

.tab {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 0 20px;
  text-decoration: none;
  color: #666;
  font-size: 14px;
  cursor: pointer;
  border-bottom: 3px solid transparent;
  transition: color 0.2s, border-color 0.2s;
  user-select: none;
}

.tab:hover:not(.disabled) {
  color: #2e7d32;
  background: #f1f8e9;
}

.tab.active {
  color: #2e7d32;
  font-weight: 600;
  border-bottom-color: #4caf50;
}

.tab.disabled {
  color: #bbb;
  cursor: not-allowed;
}

.step {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 22px;
  height: 22px;
  border-radius: 50%;
  background: #e0e0e0;
  color: #666;
  font-size: 12px;
  font-weight: 600;
}

.tab.active .step {
  background: #4caf50;
  color: #fff;
}

.tab.disabled .step {
  background: #eee;
  color: #ccc;
}
</style>
