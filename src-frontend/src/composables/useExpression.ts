// src/composables/useExpression.ts
import { ref } from "vue";
import { apiClient } from "./useApi";
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
    loading.value = true;
    error.value = "";
    try {
      const res = await apiClient.post<ExpressionResponse>(
        `/expressions/generate/${checkpointId}`,
        null,
        { params: { top_k: topK } },
      );
      expression.value = res.data;
    } catch (e: unknown) {
      const err = e as { response?: { data?: { detail?: string } } };
      error.value = err.response?.data?.detail || "生成表达式失败";
    } finally {
      loading.value = false;
    }
  }

  async function simplifyExpression(exprId: string) {
    loading.value = true;
    error.value = "";
    try {
      const res = await apiClient.post<ExpressionResponse>(
        `/expressions/simplify/${exprId}`,
      );
      expression.value = res.data;
    } catch (e: unknown) {
      const err = e as { response?: { data?: { detail?: string } } };
      error.value = err.response?.data?.detail || "精简失败";
    } finally {
      loading.value = false;
    }
  }

  async function optimizeExpression(exprId: string) {
    loading.value = true;
    error.value = "";
    try {
      const res = await apiClient.post<ExpressionResponse>(
        `/expressions/optimize/${exprId}`,
      );
      expression.value = res.data;
    } catch (e: unknown) {
      const err = e as { response?: { data?: { detail?: string } } };
      error.value = err.response?.data?.detail || "优化失败";
    } finally {
      loading.value = false;
    }
  }

  async function fetchHistory(exprId: string) {
    try {
      const res = await apiClient.get<ExpressionHistoryResponse>(
        `/expressions/history/${exprId}`,
      );
      history.value = res.data;
    } catch (e: unknown) {
      const err = e as { response?: { data?: { detail?: string } } };
      error.value = err.response?.data?.detail || "获取历史失败";
    }
  }

  async function undoExpression(exprId: string, steps = 1) {
    loading.value = true;
    error.value = "";
    try {
      const res = await apiClient.post<ExpressionResponse>(
        `/expressions/undo/${exprId}`,
        null,
        { params: { steps } },
      );
      expression.value = res.data;
    } catch (e: unknown) {
      const err = e as { response?: { data?: { detail?: string } } };
      error.value = err.response?.data?.detail || "撤销失败";
    } finally {
      loading.value = false;
    }
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
