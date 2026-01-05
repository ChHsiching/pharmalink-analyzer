from pydantic import BaseModel


class AugmentationConfig(BaseModel):
    enabled: bool = True
    gaussian_noise_sigma: float = 0.05
    bootstrap_enabled: bool = True


class TrainingConfig(BaseModel):
    dataset_id: str
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.1
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 100
    k_folds: int = 5
    early_stopping_patience: int = 20
    augmentation: AugmentationConfig = AugmentationConfig()


class TrainingProgress(BaseModel):
    epoch: int
    fold: int
    train_loss: float
    val_loss: float
    r2: float
    status: str


class TrainingStatusResponse(BaseModel):
    status: str
    task_id: str | None = None
    config: TrainingConfig | None = None
    current_fold: int = 0
    current_epoch: int = 0
    progress: list[TrainingProgress] = []
    error: str | None = None


class CheckpointInfo(BaseModel):
    id: str
    dataset_id: str
    config: TrainingConfig
    final_loss: float
