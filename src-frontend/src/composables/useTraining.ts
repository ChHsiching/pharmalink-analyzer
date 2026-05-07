import { ref } from "vue";
import { apiClient } from "./useApi";
import { WS_BASE_URL } from "@/config";
import type {
  TrainingConfig,
  TrainingStatusResponse,
  TrainingProgress,
  CheckpointInfo,
} from "@/types/training";

export function useTraining() {
  const status = ref<TrainingStatusResponse | null>(null);
  const progress = ref<TrainingProgress[]>([]);
  const checkpoints = ref<CheckpointInfo[]>([]);
  const loading = ref(false);
  const error = ref("");
  let ws: WebSocket | null = null;

  async function fetchStatus() {
    const res = await apiClient.get<TrainingStatusResponse>(
      "/models/train/status",
    );
    status.value = res.data;
  }

  async function fetchCheckpoints() {
    const res = await apiClient.get<CheckpointInfo[]>("/models/checkpoints");
    checkpoints.value = res.data;
  }

  async function startTraining(config: TrainingConfig) {
    loading.value = true;
    error.value = "";
    progress.value = [];
    try {
      await apiClient.post("/models/train", config);
      connectProgress();
      await fetchStatus();
    } catch (e: any) {
      error.value = e.response?.data?.detail || "启动训练失败";
    } finally {
      loading.value = false;
    }
  }

  async function stopTraining() {
    try {
      await apiClient.post("/models/train/stop");
    } catch {
      error.value = "停止训练失败";
    }
  }

  async function loadCheckpoint(checkpointId: string) {
    try {
      await apiClient.post(
        `/models/checkpoints/load?checkpoint_id=${checkpointId}`,
      );
    } catch {
      error.value = "加载检查点失败";
    }
  }

  function connectProgress() {
    if (ws) ws.close();
    const url = WS_BASE_URL + "/ws/train";
    ws = new WebSocket(url);
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.status === "heartbeat") return;
      if (data.epoch !== undefined) {
        progress.value = [...progress.value, data as TrainingProgress];
      }
      if (["completed", "stopped", "error"].includes(data.status)) {
        fetchStatus();
        fetchCheckpoints();
        ws?.close();
        ws = null;
      }
    };
    ws.onerror = () => {
      error.value = "WebSocket 连接失败";
    };
  }

  function disconnectProgress() {
    if (ws) {
      ws.close();
      ws = null;
    }
  }

  return {
    status,
    progress,
    checkpoints,
    loading,
    error,
    fetchStatus,
    fetchCheckpoints,
    startTraining,
    stopTraining,
    loadCheckpoint,
    connectProgress,
    disconnectProgress,
  };
}
