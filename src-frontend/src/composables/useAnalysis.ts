// src/composables/useAnalysis.ts
import { ref } from "vue";
import { apiClient } from "./useApi";
import type {
  AttentionMatrixResponse,
  HeatmapDataResponse,
  NetworkGraphResponse,
} from "@/types/analysis";

export function useAnalysis() {
  const attentionMatrix = ref<AttentionMatrixResponse | null>(null);
  const heatmap = ref<HeatmapDataResponse | null>(null);
  const network = ref<NetworkGraphResponse | null>(null);
  const loading = ref(false);
  const error = ref("");

  async function fetchAttentionMatrix(
    checkpointId: string,
    topK = 10,
    threshold?: number,
  ) {
    loading.value = true;
    error.value = "";
    try {
      const params: Record<string, string | number> = { top_k: topK };
      if (threshold !== undefined) params.threshold = threshold;
      const res = await apiClient.get<AttentionMatrixResponse>(
        `/analysis/attention/${checkpointId}`,
        { params },
      );
      attentionMatrix.value = res.data;
    } catch (e: unknown) {
      const err = e as { response?: { data?: { detail?: string } } };
      error.value = err.response?.data?.detail || "获取注意力矩阵失败";
    } finally {
      loading.value = false;
    }
  }

  async function fetchHeatmap(checkpointId: string) {
    loading.value = true;
    error.value = "";
    try {
      const res = await apiClient.get<HeatmapDataResponse>(
        `/analysis/attention/${checkpointId}/heatmap`,
      );
      heatmap.value = res.data;
    } catch (e: unknown) {
      const err = e as { response?: { data?: { detail?: string } } };
      error.value = err.response?.data?.detail || "获取热力图数据失败";
    } finally {
      loading.value = false;
    }
  }

  async function fetchNetwork(checkpointId: string, threshold?: number) {
    loading.value = true;
    error.value = "";
    try {
      const params: Record<string, number> = {};
      if (threshold !== undefined) params.threshold = threshold;
      const res = await apiClient.get<NetworkGraphResponse>(
        `/analysis/network/${checkpointId}`,
        { params },
      );
      network.value = res.data;
    } catch (e: unknown) {
      const err = e as { response?: { data?: { detail?: string } } };
      error.value = err.response?.data?.detail || "获取网络图数据失败";
    } finally {
      loading.value = false;
    }
  }

  return {
    attentionMatrix,
    heatmap,
    network,
    loading,
    error,
    fetchAttentionMatrix,
    fetchHeatmap,
    fetchNetwork,
  };
}
