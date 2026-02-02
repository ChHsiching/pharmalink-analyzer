<template>
  <div class="pl-progress">
    <label v-if="label" class="pl-progress__label">{{ label }}</label>
    <div class="pl-progress__track">
      <div class="pl-progress__fill" :style="{ width: `${(value / max) * 100}%` }" />
    </div>
  </div>
</template>

<script setup lang="ts">
withDefaults(defineProps<{
  value: number;
  max?: number;
  label?: string;
}>(), { max: 100 });
</script>

<style scoped>
.pl-progress {
  display: flex;
  flex-direction: column;
  gap: var(--space-1);
}
.pl-progress__label {
  font-size: var(--text-sm);
  color: var(--color-slate);
}
.pl-progress__track {
  height: 6px;
  background: var(--color-surface);
  border-radius: var(--radius-full);
  overflow: hidden;
}
.pl-progress__fill {
  height: 100%;
  background: var(--color-primary);
  border-radius: var(--radius-full);
  transition: width 0.3s ease;
}
</style>
