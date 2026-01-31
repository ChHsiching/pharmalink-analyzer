<template>
  <div class="pl-empty">
    <h3 class="pl-empty__title">{{ title }}</h3>
    <p class="pl-empty__desc">{{ description }}</p>
    <button v-if="action" class="pl-empty__action" @click="$emit('action')">
      {{ action }}
    </button>
  </div>
</template>

<script setup lang="ts">
defineProps<{
  title: string;
  description?: string;
  action?: string;
}>();

defineEmits<{ action: [] }>();
</script>

<style scoped>
.pl-empty {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 48px 24px;
  text-align: center;
}
.pl-empty__title {
  font-size: var(--font-size-heading);
  font-weight: var(--font-weight-heading);
  color: var(--color-charcoal);
  margin: 0 0 8px;
}
.pl-empty__desc {
  font-size: var(--font-size-body);
  color: var(--color-stone);
  margin: 0 0 16px;
  max-width: 320px;
}
.pl-empty__action {
  padding: 8px 20px;
  background: var(--color-primary);
  color: var(--color-on-primary);
  border: none;
  border-radius: var(--radius-md);
  font-size: var(--font-size-body);
  font-weight: var(--font-weight-medium);
  cursor: pointer;
}
.pl-empty__action:hover {
  background: var(--color-primary-pressed);
}
</style>
