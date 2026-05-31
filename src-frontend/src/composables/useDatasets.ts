import { apiClient } from "./useApi";
import type {
  DatasetMeta,
  DatasetDetail,
  DatasetStatsResponse,
} from "@/types/dataset";

export function useDatasets() {
  async function listDatasets(): Promise<DatasetMeta[]> {
    const res = await apiClient.get<DatasetMeta[]>("/datasets");
    return res.data;
  }

  async function getDataset(id: string): Promise<DatasetDetail> {
    const res = await apiClient.get<DatasetDetail>(`/datasets/${id}`);
    return res.data;
  }

  async function getDatasetStats(id: string): Promise<DatasetStatsResponse> {
    const res = await apiClient.get<DatasetStatsResponse>(
      `/datasets/${id}/stats`,
    );
    return res.data;
  }

  async function uploadDataset(file: File): Promise<DatasetMeta> {
    const formData = new FormData();
    formData.append("file", file);
    const res = await apiClient.post<DatasetMeta>("/datasets", formData);
    return res.data;
  }

  async function deleteDataset(id: string): Promise<void> {
    await apiClient.delete(`/datasets/${id}`);
  }

  async function updateTarget(id: string, targetColumn: string): Promise<DatasetMeta> {
    const res = await apiClient.patch<DatasetMeta>(
      `/datasets/${id}/target`,
      { target_column: targetColumn },
    );
    return res.data;
  }

  return { listDatasets, getDataset, getDatasetStats, uploadDataset, deleteDataset, updateTarget };
}
