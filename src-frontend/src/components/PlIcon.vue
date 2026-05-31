<template>
  <span :class="['pl-icon', `pl-icon--${size}`]" v-html="svgHtml" />
</template>

<script setup lang="ts">
import { ref, watch } from "vue";

const props = withDefaults(defineProps<{
  name: string;
  size?: "sm" | "md" | "lg";
}>(), { size: "md" });

const svgHtml = ref("");

const icons = import.meta.glob("@/assets/icons/*.svg", { query: "?raw", import: "default" }) as Record<string, () => Promise<string>>;

async function loadIcon() {
  const key = `/src/assets/icons/${props.name}.svg`;
  const loader = icons[key];
  if (!loader) {
    svgHtml.value = "?";
    return;
  }
  svgHtml.value = await loader();
}

watch(() => props.name, loadIcon, { immediate: true });
</script>

<style scoped>
.pl-icon {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  line-height: 1;
}
.pl-icon :deep(svg) {
  width: 100%;
  height: 100%;
}
.pl-icon--sm { width: 1rem; height: 1rem; }
.pl-icon--md { width: 1.5rem; height: 1.5rem; }
.pl-icon--lg { width: 2rem; height: 2rem; }
</style>
