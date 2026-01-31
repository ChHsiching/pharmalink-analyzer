<template>
  <div :class="['pl-card', `pl-card--${variant}`, `pl-card--pad-${padding}`]">
    <slot />
  </div>
</template>

<script setup lang="ts">
withDefaults(defineProps<{
  variant?: "base" | "feature" | "stat";
  padding?: "sm" | "md" | "lg";
}>(), {
  variant: "base",
  padding: "md",
});
</script>

<style scoped>
.pl-card {
  border-radius: var(--radius-lg);
  background: var(--color-canvas);
  border: 1px solid var(--color-hairline);
}
.pl-card--feature {
  box-shadow: var(--shadow-card);
}
.pl-card--stat {
  background: var(--color-surface);
}
.pl-card--pad-sm { padding: 12px; }
.pl-card--pad-md { padding: 20px; }
.pl-card--pad-lg { padding: 28px; }
</style>
