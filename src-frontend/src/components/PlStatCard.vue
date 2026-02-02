<template>
  <div :class="['pl-stat-card', `pl-stat-card--${variant}`]">
    <span class="pl-stat-card__label">{{ label }}</span>
    <span class="pl-stat-card__value">{{ value }}</span>
    <span v-if="trend" class="pl-stat-card__trend">{{ trend }}</span>
  </div>
</template>

<script setup lang="ts">
defineProps<{
  label: string;
  value: string | number;
  variant?: "mint" | "sky" | "peach" | "rose" | "neutral" | "lavender";
  trend?: string;
}>();
</script>

<style scoped>
.pl-stat-card {
  display: flex;
  flex-direction: column;
  gap: var(--space-1);
  padding: var(--space-4);
  border-radius: var(--radius-lg);
  min-width: 120px;
}
.pl-stat-card--mint { background: var(--color-tint-mint); }
.pl-stat-card--sky { background: var(--color-tint-sky); }
.pl-stat-card--peach { background: var(--color-tint-peach); }
.pl-stat-card--rose { background: var(--color-tint-rose); }
.pl-stat-card--neutral { background: var(--color-surface); }
.pl-stat-card--lavender { background: var(--color-tint-rose); }
.pl-stat-card__label {
  font-size: var(--text-xs);
  font-weight: var(--text-xs-weight);
  color: var(--color-slate);
  text-transform: uppercase;
  letter-spacing: 0.5px;
}
.pl-stat-card__value {
  font-size: var(--text-h2);
  font-weight: var(--text-h2-weight);
  color: var(--color-ink);
}
.pl-stat-card__trend {
  font-size: var(--text-xs);
  color: var(--color-steel);
}
</style>
