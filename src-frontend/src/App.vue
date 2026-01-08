<template>
  <TabBar />
  <main class="app-content">
    <RouterView />
  </main>
</template>

<script setup lang="ts">
import { RouterView } from "vue-router";
import TabBar from "@/components/TabBar.vue";
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
</style>
