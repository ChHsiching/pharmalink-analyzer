// src/composables/useEvaluation.ts
import { ref } from "vue";
import { apiClient } from "./useApi";
import type {
  EvaluationMetricsResponse,
  PredictionsResponse,
  ResidualsResponse,
  LossCurveResponse,
} from "@/types/evaluation";

export function useEvaluation() {
  const metrics = ref<EvaluationMetricsResponse | null>(null);
  const predictions = ref<PredictionsResponse | null>(null);
  const residuals = ref<ResidualsResponse | null>(null);
  const lossCurve = ref<LossCurveResponse | null>(null);
  const loading = ref(false);
  const error = ref("");

  async function fetchMetrics(checkpointId: string) {
    loading.value = true;
    error.value = "";
    try {
      const res = await apiClient.get<EvaluationMetricsResponse>(
        `/evaluation/metrics/${checkpointId}`,
      );
      metrics.value = res.data;
    } catch (e: unknown) {
      const err = e as { response?: { data?: { detail?: string } } };
      error.value = err.response?.data?.detail || "获取评估指标失败";
    } finally {
      loading.value = false;
    }
  }

  async function fetchPredictions(checkpointId: string) {
    loading.value = true;
    error.value = "";
    try {
      const res = await apiClient.get<PredictionsResponse>(
        `/evaluation/predictions/${checkpointId}`,
      );
      predictions.value = res.data;
    } catch (e: unknown) {
      const err = e as { response?: { data?: { detail?: string } } };
      error.value = err.response?.data?.detail || "获取预测数据失败";
    } finally {
      loading.value = false;
    }
  }

  async function fetchResiduals(checkpointId: string) {
    loading.value = true;
    error.value = "";
    try {
      const res = await apiClient.get<ResidualsResponse>(
        `/evaluation/residuals/${checkpointId}`,
      );
      residuals.value = res.data;
    } catch (e: unknown) {
      const err = e as { response?: { data?: { detail?: string } } };
      error.value = err.response?.data?.detail || "获取残差数据失败";
    } finally {
      loading.value = false;
    }
  }

  async function fetchLossCurve(checkpointId: string) {
    loading.value = true;
    error.value = "";
    try {
      const res = await apiClient.get<LossCurveResponse>(
        `/evaluation/loss-curve/${checkpointId}`,
      );
      lossCurve.value = res.data;
    } catch (e: unknown) {
      const err = e as { response?: { data?: { detail?: string } } };
      error.value = err.response?.data?.detail || "获取损失曲线失败";
    } finally {
      loading.value = false;
    }
  }

  async function fetchAll(checkpointId: string) {
    await Promise.all([
      fetchMetrics(checkpointId),
      fetchPredictions(checkpointId),
      fetchResiduals(checkpointId),
      fetchLossCurve(checkpointId),
    ]);
  }

  return {
    metrics,
    predictions,
    residuals,
    lossCurve,
    loading,
    error,
    fetchMetrics,
    fetchPredictions,
    fetchResiduals,
    fetchLossCurve,
    fetchAll,
  };
}
