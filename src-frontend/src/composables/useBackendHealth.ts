import { ref, onUnmounted } from "vue";
import { useApi } from "./useApi";

const POLL_INTERVAL_MS = 5000;

export function useBackendHealth() {
  const { checkHealth } = useApi();
  const connected = ref(false);
  let timer: ReturnType<typeof setInterval> | null = null;

  async function poll(): Promise<void> {
    try {
      await checkHealth();
      connected.value = true;
    } catch {
      connected.value = false;
    }
  }

  function start(): void {
    poll();
    timer = setInterval(poll, POLL_INTERVAL_MS);
  }

  function stop(): void {
    if (timer !== null) {
      clearInterval(timer);
      timer = null;
    }
  }

  onUnmounted(stop);

  return { connected, start, stop };
}
