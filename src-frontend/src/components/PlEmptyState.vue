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
  padding: var(--space-16) var(--space-6);
  text-align: center;
}
.pl-empty__title {
  font-size: var(--text-h3);
  font-weight: var(--text-h3-weight);
  color: var(--color-charcoal);
  margin: 0 0 var(--space-2);
}
.pl-empty__desc {
  font-size: var(--text-sm);
  color: var(--color-stone);
  margin: 0 0 var(--space-4);
  max-width: 320px;
}
.pl-empty__action {
  padding: var(--space-2) var(--space-5);
  background: var(--color-primary);
  color: var(--color-on-primary);
  border: none;
  border-radius: var(--radius-md);
  font-family: var(--font-family);
  font-size: var(--text-body);
  font-weight: var(--text-body-weight);
  cursor: pointer;
}
.pl-empty__action:hover {
  background: var(--color-primary-pressed);
}
</style>
