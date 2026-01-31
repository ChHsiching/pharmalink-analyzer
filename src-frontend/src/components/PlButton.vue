<template>
  <button
    :class="['pl-button', `pl-button--${variant}`, { 'pl-button--disabled': disabled, 'pl-button--loading': loading }]"
    :disabled="disabled || loading"
    @click="$emit('click', $event)"
  >
    <span v-if="loading" class="pl-button__spinner" />
    <span v-if="icon && !loading" class="pl-button__icon">{{ icon }}</span>
    <span class="pl-button__label"><slot /></span>
  </button>
</template>

<script setup lang="ts">
defineProps<{
  variant?: "primary" | "secondary" | "ghost" | "dark";
  disabled?: boolean;
  loading?: boolean;
  icon?: string;
}>();

defineEmits<{ click: [e: MouseEvent] }>();
</script>

<style scoped>
.pl-button {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 6px;
  padding: 8px 16px;
  font-family: var(--font-family);
  font-size: var(--font-size-body);
  font-weight: var(--font-weight-medium);
  line-height: 1;
  border: 1px solid transparent;
  border-radius: var(--radius-md);
  cursor: pointer;
  transition: background 0.15s, border-color 0.15s, opacity 0.15s;
}
.pl-button--primary {
  background: var(--color-primary);
  color: var(--color-on-primary);
}
.pl-button--primary:hover:not(:disabled) {
  background: var(--color-primary-pressed);
}
.pl-button--secondary {
  background: var(--color-surface);
  color: var(--color-ink);
  border-color: var(--color-hairline);
}
.pl-button--secondary:hover:not(:disabled) {
  background: var(--color-surface-soft);
}
.pl-button--ghost {
  background: transparent;
  color: var(--color-slate);
}
.pl-button--ghost:hover:not(:disabled) {
  background: var(--color-surface);
}
.pl-button--dark {
  background: var(--color-ink);
  color: var(--color-on-dark);
}
.pl-button--disabled {
  opacity: 0.5;
  cursor: not-allowed;
}
.pl-button--loading {
  cursor: wait;
}
.pl-button__spinner {
  width: 14px;
  height: 14px;
  border: 2px solid currentColor;
  border-top-color: transparent;
  border-radius: 50%;
  animation: pl-spin 0.6s linear infinite;
}
@keyframes pl-spin {
  to { transform: rotate(360deg); }
}
</style>
