<template>
  <div class="pl-app-shell">
    <header class="pl-app-shell__header">
      <div class="pl-app-shell__brand">
        <img src="@/assets/logo.png" alt="PharmaLink" class="pl-app-shell__logo" />
        <span class="pl-app-shell__name">PharmaLink Analyzer</span>
      </div>
      <span class="pl-app-shell__version">v1.0</span>
    </header>
    <PlStepNav />
    <main class="pl-app-shell__content">
      <slot />
    </main>
  </div>
</template>

<script setup lang="ts">
import PlStepNav from "./PlStepNav.vue";
</script>

<style scoped>
.pl-app-shell {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  background: var(--color-canvas);
  font-family: var(--font-family);
}
.pl-app-shell__header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  height: var(--header-height);
  padding: 0 var(--space-6);
  background: var(--color-canvas);
  border-bottom: 1px solid var(--color-hairline);
}
.pl-app-shell__brand {
  display: flex;
  align-items: center;
  gap: var(--space-2);
}
.pl-app-shell__logo {
  width: 2rem;
  height: 2rem;
  border-radius: var(--radius-md);
  object-fit: contain;
}
.pl-app-shell__name {
  font-size: var(--text-h3);
  font-weight: var(--text-h3-weight);
  color: var(--color-charcoal);
}
.pl-app-shell__version {
  font-size: var(--text-xs);
  color: var(--color-steel);
}
.pl-app-shell__content {
  flex: 1;
  overflow-y: auto;
}
</style>
