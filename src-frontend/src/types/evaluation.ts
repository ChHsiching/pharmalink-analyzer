// src/types/evaluation.ts
export interface FoldMetrics {
  fold: number;
  r2: number;
  mse: number;
  mae: number;
}

export interface AggregateMetrics {
  r2_mean: number;
  r2_std: number;
  mse_mean: number;
  mse_std: number;
  mae_mean: number;
  mae_std: number;
}

export interface EvaluationMetricsResponse {
  model_id: string;
  folds: FoldMetrics[];
  aggregate: AggregateMetrics;
}

export interface PredictionPoint {
  actual: number;
  predicted: number;
}

export interface PredictionsResponse {
  model_id: string;
  predictions: PredictionPoint[];
}

export interface ResidualBin {
  range_start: number;
  range_end: number;
  count: number;
}

export interface ResidualsResponse {
  model_id: string;
  residuals: number[];
  bins: ResidualBin[];
  mean: number;
  std: number;
}

export interface LossCurvePoint {
  epoch: number;
  train_loss: number;
  val_loss: number;
}

export interface FoldLossCurve {
  fold: number;
  points: LossCurvePoint[];
}

export interface LossCurveResponse {
  model_id: string;
  folds: FoldLossCurve[];
}
