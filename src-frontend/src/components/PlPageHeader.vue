<template>
  <div class="pl-page-header">
    <span class="pl-page-header__step">Step {{ step }}</span>
    <h1 class="pl-page-header__title">{{ title }}</h1>
    <p v-if="subtitle" class="pl-page-header__subtitle">{{ subtitle }}</p>
  </div>
</template>

<script setup lang="ts">
defineProps<{
  step: number;
  title: string;
  subtitle?: string;
}>();
</script>

<style scoped>
.pl-page-header {
  margin-bottom: var(--space-6);
}
.pl-page-header__step {
  display: block;
  font-size: var(--text-micro);
  font-weight: var(--text-micro-weight);
  text-transform: uppercase;
  letter-spacing: 0.5px;
  color: var(--color-primary);
  margin-bottom: var(--space-1);
}
.pl-page-header__title {
  font-size: var(--text-h2);
  font-weight: var(--text-h2-weight);
  line-height: var(--text-h2-line);
  color: var(--color-charcoal);
  margin: 0 0 var(--space-1);
}
.pl-page-header__subtitle {
  font-size: var(--text-sm);
  line-height: var(--text-sm-line);
  color: var(--color-steel);
  margin: 0;
}
</style>
