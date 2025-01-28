import { ref } from "vue";
import { apiClient, safeRequest } from "./useApi";
import type { FormulationResponse } from "@/types/formulation";

export function useFormulation() {
  const result = ref<FormulationResponse | null>(null);
  const loading = ref(false);
  const error = ref("");

  async function formulate(
    exprId: string,
    topK = 5,
    nSamples = 1000,
  ): Promise<boolean> {
    const res = await safeRequest(
      () =>
        apiClient.post<FormulationResponse>("/formulation", {
          expr_id: exprId,
          top_k: topK,
          n_samples: nSamples,
        }),
      loading,
      error,
      "寻优失败",
    );
    if (res) {
      result.value = res.data;
      return true;
    }
    return false;
  }

  function exportCsv() {
    if (!result.value) return;
    const { candidates, feature_names } = result.value;
    const header = ["排名", ...feature_names, "预测药效"].join(",");
    const rows = candidates.map((c) =>
      [
        c.rank,
        ...feature_names.map((f) => c.components[f].toFixed(4)),
        c.predicted_response.toFixed(4),
      ].join(","),
    );
    const csv = [header, ...rows].join("\n");
    const blob = new Blob(["﻿" + csv], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `formulation_${result.value.expr_id}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  }

  return { result, loading, error, formulate, exportCsv };
}
