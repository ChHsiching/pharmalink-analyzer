import { type Ref } from "vue";
import axios from "axios";
import { API_BASE_URL } from "@/config";
import type { HealthResponse } from "@/types/api";

export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000,
});

export async function safeRequest<T>(
  fn: () => Promise<T>,
  loading: Ref<boolean>,
  error: Ref<string>,
  fallbackMsg: string,
): Promise<T | undefined> {
  loading.value = true;
  error.value = "";
  try {
    return await fn();
  } catch (e: unknown) {
    const err = e as { response?: { data?: { detail?: string } } };
    error.value = err.response?.data?.detail || fallbackMsg;
    return undefined;
  } finally {
    loading.value = false;
  }
}

export function useApi() {
  async function checkHealth(): Promise<HealthResponse> {
    const response = await apiClient.get<HealthResponse>("/health");
    return response.data;
  }

  return { checkHealth };
}
