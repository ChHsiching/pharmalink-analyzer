import { ref } from "vue";
import { apiClient, safeRequest } from "./useApi";
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
    const res = await safeRequest(
      () => apiClient.get<EvaluationMetricsResponse>(
        `/evaluation/metrics/${checkpointId}`,
      ),
      loading, error, "获取评估指标失败",
    );
    if (res) metrics.value = res.data;
  }

  async function fetchPredictions(checkpointId: string) {
    const res = await safeRequest(
      () => apiClient.get<PredictionsResponse>(
        `/evaluation/predictions/${checkpointId}`,
      ),
      loading, error, "获取预测数据失败",
    );
    if (res) predictions.value = res.data;
  }

  async function fetchResiduals(checkpointId: string) {
    const res = await safeRequest(
      () => apiClient.get<ResidualsResponse>(
        `/evaluation/residuals/${checkpointId}`,
      ),
      loading, error, "获取残差数据失败",
    );
    if (res) residuals.value = res.data;
  }

  async function fetchLossCurve(checkpointId: string) {
    const res = await safeRequest(
      () => apiClient.get<LossCurveResponse>(
        `/evaluation/loss-curve/${checkpointId}`,
      ),
      loading, error, "获取损失曲线失败",
    );
    if (res) lossCurve.value = res.data;
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
