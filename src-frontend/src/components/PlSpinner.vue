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
  border: 2px solid var(--color-hairline);
  border-top-color: var(--color-primary);
  border-radius: 50%;
  animation: pl-spin 0.6s linear infinite;
}
.pl-spinner--sm { width: 16px; height: 16px; }
.pl-spinner--md { width: 24px; height: 24px; }
@keyframes pl-spin {
  to { transform: rotate(360deg); }
}
</style>
