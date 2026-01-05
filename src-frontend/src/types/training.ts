export interface AugmentationConfig {
  enabled: boolean;
  gaussian_noise_sigma: number;
  bootstrap_enabled: boolean;
}

export interface TrainingConfig {
  dataset_id: string;
  d_model: number;
  n_heads: number;
  n_layers: number;
  dropout: number;
  learning_rate: number;
  weight_decay: number;
  epochs: number;
  k_folds: number;
  early_stopping_patience: number;
  augmentation: AugmentationConfig;
}

export interface TrainingProgress {
  epoch: number;
  fold: number;
  train_loss: number;
  val_loss: number;
  r2: number;
  status: string;
}

export interface TrainingStatusResponse {
  status: string;
  task_id: string | null;
  config: TrainingConfig | null;
  current_fold: number;
  current_epoch: number;
  progress: TrainingProgress[];
  error: string | null;
}

export interface CheckpointInfo {
  id: string;
  dataset_id: string;
  config: TrainingConfig;
  final_loss: number;
}
