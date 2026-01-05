import axios from "axios";
import { API_BASE_URL } from "@/config";
import type { HealthResponse } from "@/types/api";

export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000,
});

export function useApi() {
  async function checkHealth(): Promise<HealthResponse> {
    const response = await apiClient.get<HealthResponse>("/health");
    return response.data;
  }

  return { checkHealth };
}
