export interface FormulationRequest {
  expr_id: string;
  top_k?: number;
  n_samples?: number;
}

export interface FormulationCandidate {
  components: Record<string, number>;
  predicted_response: number;
  rank: number;
}

export interface FormulationResponse {
  expr_id: string;
  candidates: FormulationCandidate[];
  feature_names: string[];
  attention_weights: Record<string, number>;
  attention_weak?: boolean;
}
