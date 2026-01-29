export interface ColumnStats {
  column: string;
  mean: number;
  std: number;
  min: number;
  max: number;
}

export interface DatasetMeta {
  id: string;
  name: string;
  filename: string;
  plant_part: string;
  target: string;
  n_samples: number;
  n_features: number;
  feature_names: string[];
  columns: string[];
  is_preset: boolean;
}

export interface DatasetDetail extends DatasetMeta {
  preview: Record<string, number>[];
}

export interface DatasetStatsResponse {
  dataset_id: string;
  stats: ColumnStats[];
}
