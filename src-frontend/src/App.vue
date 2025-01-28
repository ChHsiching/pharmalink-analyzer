<template>
  <div v-if="!ready && !loadingError" class="loading-screen">
    <div class="loading-spinner"></div>
    <p class="loading-text">正在启动后端服务...</p>
    <p class="loading-subtext">PharmaLink Analyzer</p>
  </div>
  <div v-else-if="loadingError" class="loading-screen loading-error">
    <p class="loading-text">后端启动失败</p>
    <p class="loading-subtext">{{ loadingError }}</p>
  </div>
  <template v-else>
    <TabBar />
    <main class="app-content">
      <RouterView />
    </main>
  </template>
</template>

<script setup lang="ts">
import { onMounted } from "vue";
import { RouterView } from "vue-router";
import TabBar from "@/components/TabBar.vue";
import { useWorkflow } from "@/composables/useWorkflow";
import { useBackendReady } from "@/composables/useBackendReady";

const { init } = useWorkflow();
const { ready, error: loadingError, startPolling } = useBackendReady({
  intervalMs: 500,
  maxAttempts: 60,
});

onMounted(async () => {
  await startPolling();
  if (ready.value) {
    await init();
  }
});
</script>

<style>
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

.app-content {
  height: calc(100vh - 48px);
  overflow: auto;
}

.loading-screen {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100vh;
  background: #1a1a2e;
  color: #e0e0e0;
}

.loading-spinner {
  width: 40px;
  height: 40px;
  border: 3px solid #333;
  border-top: 3px solid #4fc3f7;
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
  margin-bottom: 16px;
}

@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}

.loading-text {
  font-size: 18px;
  font-weight: 500;
  margin-bottom: 8px;
}

.loading-subtext {
  font-size: 14px;
  color: #888;
}

.loading-error .loading-text {
  color: #ef5350;
}
</style>
