<template>
  <div v-if="!ready && !loadingError" class="loading-screen">
    <div class="loading-screen__inner">
      <img src="@/assets/logo.png" alt="PharmaLink" class="loading-screen__logo" />
      <h1 class="loading-screen__title">PharmaLink Analyzer</h1>
      <p class="loading-screen__subtitle">基于注意力机制的中药成分关联性分析</p>
      <div class="loading-screen__bar">
        <div class="loading-screen__bar-fill" />
      </div>
      <p class="loading-screen__status">正在启动后端服务...</p>
    </div>
  </div>
  <div v-else-if="loadingError" class="loading-screen">
    <div class="loading-screen__inner">
      <PlIcon name="error" size="lg" class="loading-screen__error-icon" />
      <h1 class="loading-screen__title loading-screen__title--error">后端启动失败</h1>
      <p class="loading-screen__subtitle">{{ loadingError }}</p>
      <PlButton variant="primary" @click="retry">重试</PlButton>
    </div>
  </div>
  <PlAppShell v-else>
    <RouterView />
  </PlAppShell>
</template>

<script setup lang="ts">
import { onMounted } from "vue";
import { RouterView } from "vue-router";
import PlAppShell from "@/components/PlAppShell.vue";
import PlButton from "@/components/PlButton.vue";
import PlIcon from "@/components/PlIcon.vue";
import { useWorkflow } from "@/composables/useWorkflow";
import { useBackendReady } from "@/composables/useBackendReady";

const { init } = useWorkflow();
const { ready, error: loadingError, startPolling } = useBackendReady({
  intervalMs: 500,
  maxAttempts: 60,
});

function retry() {
  window.location.reload();
}

onMounted(async () => {
  await startPolling();
  if (ready.value) {
    await init();
  }
});
</script>

<style>
*,
*::before,
*::after {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}
html {
  font-size: 14px;
}
body {
  font-family: var(--font-family);
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}
</style>

<style scoped>
.loading-screen {
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 100vh;
  background: var(--color-canvas);
}
.loading-screen__inner {
  display: flex;
  flex-direction: column;
  align-items: center;
  text-align: center;
  padding: var(--space-8);
}
.loading-screen__logo {
  width: 4rem;
  height: 4rem;
  border-radius: var(--radius-lg);
  object-fit: contain;
  margin-bottom: var(--space-4);
}
.loading-screen__title {
  font-size: var(--text-hero);
  font-weight: var(--text-hero-weight);
  color: var(--color-charcoal);
  margin-bottom: var(--space-2);
}
.loading-screen__title--error {
  color: var(--color-error);
}
.loading-screen__subtitle {
  font-size: var(--text-sm);
  color: var(--color-steel);
  margin-bottom: var(--space-6);
  max-width: 24rem;
}
.loading-screen__bar {
  width: 12.5rem;
  height: 4px;
  background: var(--color-surface);
  border-radius: var(--radius-full);
  overflow: hidden;
  margin-bottom: var(--space-4);
}
.loading-screen__bar-fill {
  height: 100%;
  width: 40%;
  background: var(--color-primary);
  border-radius: var(--radius-full);
  animation: loading-slide 1.5s ease-in-out infinite;
}
@keyframes loading-slide {
  0% { transform: translateX(-100%); }
  100% { transform: translateX(350%); }
}
.loading-screen__status {
  font-size: var(--text-xs);
  color: var(--color-stone);
}
.loading-screen__error-icon {
  color: var(--color-error);
  margin-bottom: var(--space-3);
}
</style>
