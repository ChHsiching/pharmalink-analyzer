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
    <footer class="pl-app-shell__footer">
      <div class="pl-app-shell__footer-left">
        <span
          class="pl-app-shell__health-dot"
          :class="{ 'pl-app-shell__health-dot--connected': connected }"
        />
        <span class="pl-app-shell__health-text">{{ connected ? '已连接' : '已断开' }}</span>
      </div>
      <div class="pl-app-shell__footer-center">
        <span class="pl-app-shell__author">ChHsiching</span>
        <a href="https://github.com/ChHsiching" target="_blank" rel="noopener noreferrer" class="pl-app-shell__github">
          <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
            <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0024 12c0-6.63-5.37-12-12-12z"/>
          </svg>
        </a>
      </div>
      <div class="pl-app-shell__footer-right">
        <span class="pl-app-shell__clock">{{ timeString }}</span>
      </div>
    </footer>
  </div>
</template>

<script setup lang="ts">
import PlStepNav from "./PlStepNav.vue";
import { useBackendHealth } from "@/composables/useBackendHealth";
import { useLiveClock } from "@/composables/useLiveClock";

const { connected, start: startHealth } = useBackendHealth();
const { timeString } = useLiveClock();

startHealth();
</script>

<style scoped>
.pl-app-shell {
  height: 100vh;
  display: flex;
  flex-direction: column;
  background: var(--color-canvas);
  font-family: var(--font-family);
  overflow: hidden;
}

.pl-app-shell__header {
  position: sticky;
  top: 0;
  z-index: 10;
  display: flex;
  align-items: center;
  justify-content: space-between;
  height: var(--header-height);
  padding: 0 var(--space-6);
  background: var(--color-canvas);
  border-bottom: 1px solid var(--color-hairline);
  flex-shrink: 0;
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

.pl-app-shell__footer {
  position: sticky;
  bottom: 0;
  z-index: 10;
  display: flex;
  align-items: center;
  justify-content: space-between;
  height: 2rem;
  padding: 0 var(--space-6);
  background: var(--color-canvas);
  border-top: 1px solid var(--color-hairline);
  font-size: var(--text-sm);
  color: var(--color-steel);
  flex-shrink: 0;
}

.pl-app-shell__footer-left {
  display: flex;
  align-items: center;
  gap: var(--space-2);
}

.pl-app-shell__health-dot {
  width: 6px;
  height: 6px;
  border-radius: var(--radius-full);
  background: var(--color-error);
}

.pl-app-shell__health-dot--connected {
  background: var(--color-success);
  animation: pulse 2s ease-in-out infinite;
}

.pl-app-shell__health-text {
  font-size: var(--text-xs);
}

.pl-app-shell__footer-center {
  display: flex;
  align-items: center;
  gap: var(--space-2);
}

.pl-app-shell__author {
  font-size: var(--text-xs);
  color: var(--color-slate);
}

.pl-app-shell__github {
  color: var(--color-steel);
  display: flex;
  align-items: center;
}

.pl-app-shell__github:hover {
  color: var(--color-charcoal);
}

.pl-app-shell__footer-right {
  display: flex;
  align-items: center;
}

.pl-app-shell__clock {
  font-size: var(--text-xs);
  font-variant-numeric: tabular-nums;
}

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.4; }
}
</style>
