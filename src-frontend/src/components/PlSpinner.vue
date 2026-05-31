<template>
  <div :class="['pl-spinner', `pl-spinner--${size}`]" />
</template>

<script setup lang="ts">
withDefaults(defineProps<{
  size?: "sm" | "md";
}>(), { size: "md" });
</script>

<style scoped>
.pl-spinner {
  border: 2px solid var(--color-hairline-strong);
  border-top-color: var(--color-primary);
  border-radius: 50%;
  animation: pl-spin 0.6s linear infinite;
}
.pl-spinner--sm { width: calc(var(--space-2) + var(--space-1)); height: calc(var(--space-2) + var(--space-1)); }
.pl-spinner--md { width: var(--space-4); height: var(--space-4); }
@keyframes pl-spin {
  to { transform: rotate(360deg); }
}
</style>
