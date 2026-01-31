<template>
  <div :class="['pl-stat-card', `pl-stat-card--${variant}`]">
    <span class="pl-stat-card__label">{{ label }}</span>
    <span class="pl-stat-card__value">{{ value }}</span>
  </div>
</template>

<script setup lang="ts">
defineProps<{
  label: string;
  value: string | number;
  variant?: "mint" | "lavender" | "sky" | "neutral";
}>();
</script>

<style scoped>
.pl-stat-card {
  display: flex;
  flex-direction: column;
  gap: 4px;
  padding: 16px;
  border-radius: var(--radius-lg);
  min-width: 120px;
}
.pl-stat-card--mint { background: var(--color-tint-mint); }
.pl-stat-card--lavender { background: var(--color-tint-rose); }
.pl-stat-card--sky { background: var(--color-tint-sky); }
.pl-stat-card--neutral { background: var(--color-surface); }
.pl-stat-card__label {
  font-size: 12px;
  font-weight: var(--font-weight-medium);
  color: var(--color-slate);
  text-transform: uppercase;
  letter-spacing: 0.5px;
}
.pl-stat-card__value {
  font-size: 24px;
  font-weight: var(--font-weight-heading);
  color: var(--color-ink);
}
</style>
