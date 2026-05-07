import { ref } from "vue";
import { apiClient, safeRequest } from "./useApi";
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
    const params: Record<string, string | number> = { top_k: topK };
    if (threshold !== undefined) params.threshold = threshold;
    const res = await safeRequest(
      () => apiClient.get<AttentionMatrixResponse>(
        `/analysis/attention/${checkpointId}`,
        { params },
      ),
      loading, error, "获取注意力矩阵失败",
    );
    if (res) attentionMatrix.value = res.data;
  }

  async function fetchHeatmap(checkpointId: string) {
    const res = await safeRequest(
      () => apiClient.get<HeatmapDataResponse>(
        `/analysis/attention/${checkpointId}/heatmap`,
      ),
      loading, error, "获取热力图数据失败",
    );
    if (res) heatmap.value = res.data;
  }

  async function fetchNetwork(checkpointId: string, threshold?: number) {
    const params: Record<string, number> = {};
    if (threshold !== undefined) params.threshold = threshold;
    const res = await safeRequest(
      () => apiClient.get<NetworkGraphResponse>(
        `/analysis/network/${checkpointId}`,
        { params },
      ),
      loading, error, "获取网络图数据失败",
    );
    if (res) network.value = res.data;
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
