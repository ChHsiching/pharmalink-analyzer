<template>
  <div :class="['pl-card', `pl-card--${variant}`, `pl-card--pad-${padding}`, { 'pl-card--selectable': selectable }]">
    <slot name="header" />
    <slot />
    <slot name="footer" />
  </div>
</template>

<script setup lang="ts">
withDefaults(defineProps<{
  variant?: "base" | "feature" | "stat";
  padding?: "sm" | "md" | "lg";
  selectable?: boolean;
}>(), {
  variant: "base",
  padding: "md",
  selectable: false,
});
</script>

<style scoped>
.pl-card {
  border-radius: var(--radius-lg);
  background: var(--color-canvas);
  border: 1px solid var(--color-hairline);
  box-shadow: var(--shadow-card);
}
.pl-card--feature {
  box-shadow: var(--shadow-card);
}
.pl-card--stat {
  background: var(--color-surface);
}
.pl-card--selectable {
  cursor: pointer;
  transition: box-shadow 0.15s;
}
.pl-card--selectable:hover {
  box-shadow: var(--shadow-raised);
}
.pl-card--pad-sm { padding: var(--space-3); }
.pl-card--pad-md { padding: var(--space-4); }
.pl-card--pad-lg { padding: var(--space-6); }
</style>
