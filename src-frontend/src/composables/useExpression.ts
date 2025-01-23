import { ref } from "vue";
import { apiClient, safeRequest } from "./useApi";
import type {
  ExpressionResponse,
  ExpressionHistoryResponse,
  TaskStatusResponse,
} from "@/types/expression";

const POLL_INTERVAL_MS = 2000;

export function useExpression() {
  const expression = ref<ExpressionResponse | null>(null);
  const history = ref<ExpressionHistoryResponse | null>(null);
  const loading = ref(false);
  const error = ref("");
  let pollTimer: ReturnType<typeof setInterval> | null = null;

  function stopPolling() {
    if (pollTimer !== null) {
      clearInterval(pollTimer);
      pollTimer = null;
    }
  }

  async function generateExpression(checkpointId: string, topK = 10, preset = "standard") {
    stopPolling();
    loading.value = true;
    error.value = "";

    try {
      const startResp = await apiClient.post<TaskStatusResponse>(
        `/expressions/generate/${checkpointId}`,
        null,
        { params: { top_k: topK, preset }, timeout: 30000 },
      );
      const taskId = startResp.data.task_id;

      pollTimer = setInterval(async () => {
        try {
          const statusResp = await apiClient.get<TaskStatusResponse>(
            `/expressions/result/${taskId}`,
            { timeout: 10000 },
          );
          const task = statusResp.data;

          if (task.status === "completed" && task.result) {
            stopPolling();
            expression.value = task.result;
            loading.value = false;
          } else if (task.status === "failed") {
            stopPolling();
            error.value = task.error || "生成表达式失败";
            loading.value = false;
          }
        } catch {
          stopPolling();
          error.value = "轮询状态失败";
          loading.value = false;
        }
      }, POLL_INTERVAL_MS);
    } catch (e: unknown) {
      const err = e as { response?: { data?: { detail?: string } } };
      error.value = err.response?.data?.detail || "生成表达式失败";
      loading.value = false;
    }
  }

  async function simplifyExpression(exprId: string) {
    const res = await safeRequest(
      () => apiClient.post<ExpressionResponse>(
        `/expressions/simplify/${exprId}`,
      ),
      loading, error, "精简失败",
    );
    if (res) expression.value = res.data;
  }

  async function optimizeExpression(exprId: string) {
    const res = await safeRequest(
      () => apiClient.post<ExpressionResponse>(
        `/expressions/optimize/${exprId}`,
      ),
      loading, error, "优化失败",
    );
    if (res) expression.value = res.data;
  }

  async function fetchHistory(exprId: string) {
    const res = await safeRequest(
      () => apiClient.get<ExpressionHistoryResponse>(
        `/expressions/history/${exprId}`,
      ),
      loading, error, "获取历史失败",
    );
    if (res) history.value = res.data;
  }

  async function undoExpression(exprId: string, steps = 1) {
    const res = await safeRequest(
      () => apiClient.post<ExpressionResponse>(
        `/expressions/undo/${exprId}`,
        null,
        { params: { steps } },
      ),
      loading, error, "撤销失败",
    );
    if (res) expression.value = res.data;
  }

  return {
    expression,
    history,
    loading,
    error,
    generateExpression,
    simplifyExpression,
    optimizeExpression,
    fetchHistory,
    undoExpression,
    stopPolling,
  };
}
