<template>
  <span :class="['pl-badge', `pl-badge--${variant}`]">{{ text }}</span>
</template>

<script setup lang="ts">
defineProps<{
  text: string;
  variant?: "teal" | "pink" | "orange" | "yellow" | "neutral" | "tag-green" | "tag-orange";
}>();
</script>

<style scoped>
.pl-badge {
  display: inline-flex;
  align-items: center;
  padding: 2px 10px;
  font-size: var(--text-xs);
  font-weight: var(--text-xs-weight);
  border-radius: var(--radius-full);
  white-space: nowrap;
}
.pl-badge--teal { background: var(--color-tint-mint); color: var(--color-success); }
.pl-badge--pink { background: var(--color-tint-rose); color: var(--color-error); }
.pl-badge--orange { background: var(--color-tint-peach); color: var(--color-warning); }
.pl-badge--yellow { background: var(--color-tint-yellow); color: #523410; }
.pl-badge--neutral { background: var(--color-tint-gray); color: var(--color-charcoal); }
.pl-badge--tag-green { background: var(--color-tint-mint); color: var(--color-charcoal); }
.pl-badge--tag-orange { background: var(--color-tint-peach); color: var(--color-charcoal); }
</style>
