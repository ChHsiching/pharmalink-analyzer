import { ref } from "vue";
import { apiClient, safeRequest } from "./useApi";
import type {
  ExpressionResponse,
  ExpressionHistoryResponse,
} from "@/types/expression";

export function useExpression() {
  const expression = ref<ExpressionResponse | null>(null);
  const history = ref<ExpressionHistoryResponse | null>(null);
  const loading = ref(false);
  const error = ref("");

  async function generateExpression(checkpointId: string, topK = 10) {
    const res = await safeRequest(
      () => apiClient.post<ExpressionResponse>(
        `/expressions/generate/${checkpointId}`,
        null,
        { params: { top_k: topK } },
      ),
      loading, error, "生成表达式失败",
    );
    if (res) expression.value = res.data;
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
  };
}
