<template>
  <div class="home">
    <h1>PharmaLink Analyzer</h1>
    <p>基于注意力机制的中药成分关联性分析</p>
    <p v-if="backendStatus" class="status">
      Backend: {{ backendStatus }}
    </p>
    <p v-else class="status error">Backend: connecting...</p>
  </div>
</template>

<script setup lang="ts">
import { onMounted, ref } from "vue";
import { useApi } from "@/composables/useApi";

const { checkHealth } = useApi();
const backendStatus = ref<string>("");

onMounted(async () => {
  try {
    const health = await checkHealth();
    backendStatus.value = health.status;
  } catch {
    backendStatus.value = "unavailable";
  }
});
</script>

<style scoped>
.home {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  min-height: 100vh;
  font-family: system-ui, sans-serif;
}
h1 { color: #2e7d32; }
.status { color: #4caf50; }
.status.error { color: #f44336; }
</style>
