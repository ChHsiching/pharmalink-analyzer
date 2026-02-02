<template>
  <nav class="pl-step-nav">
    <div class="pl-step-nav__inner">
      <template v-for="(tab, index) in TAB_ORDER" :key="tab">
        <div v-if="index > 0" class="pl-step-nav__connector" />
        <router-link
          :to="enabled[tab] ? { name: tab } : {}"
          :class="[
            'pl-step-nav__step',
            {
              'pl-step-nav__step--active': isActive(tab),
              'pl-step-nav__step--completed': isCompleted(tab),
              'pl-step-nav__step--locked': !enabled[tab],
            },
          ]"
          :tabindex="enabled[tab] ? 0 : -1"
          @click.prevent="enabled[tab] && $router.push({ name: tab })"
        >
          <span class="pl-step-nav__circle">
            <PlIcon v-if="isCompleted(tab)" name="check" size="sm" />
            <span v-else>{{ index + 1 }}</span>
          </span>
          <span class="pl-step-nav__label">{{ TAB_LABELS[tab] }}</span>
        </router-link>
      </template>
    </div>
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
import PlIcon from "./PlIcon.vue";

const route = useRoute();
const { tabEnabled: enabled } = useWorkflow();

function isActive(tab: TabName): boolean {
  return route.name === tab;
}

function isCompleted(tab: TabName): boolean {
  if (!enabled[tab]) return false;
  const idx = TAB_ORDER.indexOf(tab);
  const activeIdx = TAB_ORDER.indexOf(route.name as TabName);
  return idx < activeIdx;
}
</script>

<style scoped>
.pl-step-nav {
  height: var(--stepnav-height);
  background: var(--color-surface-soft);
  border-bottom: 1px solid var(--color-hairline);
  padding: 0 var(--space-6);
}
.pl-step-nav__inner {
  display: flex;
  align-items: center;
  height: 100%;
  max-width: var(--content-max-width);
  margin: 0 auto;
}
.pl-step-nav__connector {
  width: var(--space-6);
  height: 1px;
  background: var(--color-hairline-strong);
  flex-shrink: 0;
}
.pl-step-nav__step {
  display: flex;
  align-items: center;
  gap: var(--space-1);
  padding: var(--space-1) var(--space-3);
  border-radius: var(--radius-md);
  text-decoration: none;
  color: var(--color-steel);
  cursor: pointer;
  transition: background 0.15s, color 0.15s;
  user-select: none;
  white-space: nowrap;
}
.pl-step-nav__step:hover:not(.pl-step-nav__step--locked) {
  background: var(--color-surface);
}
.pl-step-nav__step--locked {
  color: var(--color-muted);
  cursor: not-allowed;
}
.pl-step-nav__step--active {
  background: var(--color-primary-tint);
  color: var(--color-primary-deep);
}
.pl-step-nav__step--completed {
  color: var(--color-slate);
}
.pl-step-nav__circle {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 1.375rem;
  height: 1.375rem;
  border-radius: 50%;
  font-size: var(--text-xs);
  font-weight: 600;
  flex-shrink: 0;
  background: var(--color-hairline);
  color: var(--color-muted);
}
.pl-step-nav__step--active .pl-step-nav__circle {
  background: var(--color-primary);
  color: var(--color-on-primary);
}
.pl-step-nav__step--completed .pl-step-nav__circle {
  background: var(--color-success);
  color: var(--color-on-primary);
}
.pl-step-nav__label {
  font-size: var(--text-sm);
  font-weight: 500;
}
.pl-step-nav__step--active .pl-step-nav__label {
  font-weight: 600;
  color: var(--color-primary-deep);
}
</style>
