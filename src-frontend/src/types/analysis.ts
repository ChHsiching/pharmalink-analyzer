// src/types/analysis.ts
export interface AttentionPair {
  source: string;
  target: string;
  weight: number;
  classification: "synergistic" | "antagonistic";
}

export interface AttentionMatrixResponse {
  model_id: string;
  feature_names: string[];
  matrix: number[][];
  pairs: AttentionPair[];
  threshold: number;
}

export interface HeatmapDataResponse {
  model_id: string;
  feature_names: string[];
  values: number[][];
  min_value: number;
  max_value: number;
}

export interface NetworkNode {
  id: string;
  name: string;
}

export interface NetworkEdge {
  source: string;
  target: string;
  weight: number;
  classification: "synergistic" | "antagonistic";
}

export interface NetworkGraphResponse {
  model_id: string;
  nodes: NetworkNode[];
  edges: NetworkEdge[];
  threshold: number;
}
