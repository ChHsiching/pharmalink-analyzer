import { ref } from "vue";
import { useApi } from "./useApi";

interface PollingOptions {
  intervalMs: number;
  maxAttempts: number;
}

const DEFAULT_OPTIONS: PollingOptions = {
  intervalMs: 1000,
  maxAttempts: 30,
};

export function useBackendReady(options?: Partial<PollingOptions>) {
  const { intervalMs, maxAttempts } = { ...DEFAULT_OPTIONS, ...options };
  const { checkHealth } = useApi();

  const ready = ref(false);
  const error = ref("");

  async function startPolling(): Promise<void> {
    let lastError: unknown;
    for (let attempt = 0; attempt < maxAttempts; attempt++) {
      try {
        await checkHealth();
        ready.value = true;
        return;
      } catch (e) {
        lastError = e;
        if (attempt < maxAttempts - 1) {
          await new Promise((r) => setTimeout(r, intervalMs));
        }
      }
    }
    error.value = `Backend not ready after ${maxAttempts} attempts: ${lastError instanceof Error ? lastError.message : String(lastError)}`;
  }

  return { ready, error, startPolling };
}
