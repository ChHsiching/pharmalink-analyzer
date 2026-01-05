import axios from "axios";
import type { HealthResponse } from "@/types/api";

export const apiClient = axios.create({
  baseURL: "http://localhost:8765/api/v1",
  timeout: 10000,
});

export function useApi() {
  async function checkHealth(): Promise<HealthResponse> {
    const response = await apiClient.get<HealthResponse>("/health");
    return response.data;
  }

  return { checkHealth };
}
