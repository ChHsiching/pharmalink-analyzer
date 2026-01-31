<template>
  <div v-if="message" :class="['pl-toast', `pl-toast--${variant}`]">
    <span class="pl-toast__message">{{ message }}</span>
  </div>
</template>

<script setup lang="ts">
withDefaults(defineProps<{
  message?: string;
  variant?: "success" | "warning" | "error";
}>(), { variant: "success" });
</script>

<style scoped>
.pl-toast {
  display: flex;
  align-items: center;
  padding: 10px 16px;
  border-radius: var(--radius-md);
  font-size: var(--font-size-body);
  font-weight: var(--font-weight-medium);
  box-shadow: var(--shadow-card);
}
.pl-toast--success { background: var(--color-tint-mint); color: var(--color-success); }
.pl-toast--warning { background: var(--color-tint-peach); color: var(--color-warning); }
.pl-toast--error { background: var(--color-tint-rose); color: var(--color-error); }
</style>
