import asyncio
import json
import uuid
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from app.config import CHECKPOINT_DIR
from app.models.training import (
    TrainingConfig, TrainingProgress, TrainingStatusResponse,
    CheckpointInfo,
)
from app.ml.transformer import FeatureTransformer
from app.ml.augmentation import augment
from app.services.data_loader import data_loader as default_data_loader
from app.services.data_loader import DataLoader


class TrainingService:
    def __init__(
        self,
        checkpoint_dir: Path | None = None,
        data_loader: DataLoader | None = None,
    ):
        self._checkpoint_dir = checkpoint_dir or CHECKPOINT_DIR
        self._data_loader = data_loader or default_data_loader
        self._status = "idle"
        self._config: TrainingConfig | None = None
        self._task_id: str | None = None
        self._progress: list[TrainingProgress] = []
        self._stop_requested = False
        self._progress_queue: asyncio.Queue | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def get_status(self) -> TrainingStatusResponse:
        fold, epoch = 0, 0
        if self._progress:
            last = self._progress[-1]
            fold, epoch = last.fold, last.epoch
        return TrainingStatusResponse(
            status=self._status,
            task_id=self._task_id,
            config=self._config,
            current_fold=fold,
            current_epoch=epoch,
            progress=self._progress,
        )

    def set_progress_queue(self, queue: asyncio.Queue):
        self._progress_queue = queue

    def clear_progress_queue(self):
        self._progress_queue = None

    async def start_training(self, config: TrainingConfig) -> str:
        if self._status == "running":
            raise ValueError("Training already in progress")

        df = self._data_loader.get_dataframe(config.dataset_id)
        if df is None:
            raise ValueError(f"Dataset '{config.dataset_id}' not found")

        meta = self._data_loader.get_dataset(config.dataset_id)
        self._task_id = uuid.uuid4().hex[:8]
        self._config = config
        self._status = "running"
        self._stop_requested = False
        self._progress = []
        self._loop = asyncio.get_event_loop()

        X = df[meta.feature_names].values.astype(np.float32)
        y = df[meta.target].values.astype(np.float32)
        n_features = len(meta.feature_names)

        self._loop.run_in_executor(
            None, self._train_sync, config, X, y, n_features
        )
        return self._task_id

    def request_stop(self):
        self._stop_requested = True

    def _push_progress(self, progress: TrainingProgress):
        self._progress.append(progress)
        if self._progress_queue and self._loop:
            asyncio.run_coroutine_threadsafe(
                self._progress_queue.put(progress.model_dump()),
                self._loop,
            )

    def _train_sync(
        self,
        config: TrainingConfig,
        X: np.ndarray,
        y: np.ndarray,
        n_features: int,
    ):
        try:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X).astype(np.float32)

            kfold = KFold(
                n_splits=config.k_folds, shuffle=True, random_state=42
            )
            best_state = None
            best_loss = float("inf")

            for fold_idx, (train_idx, val_idx) in enumerate(
                kfold.split(X_scaled)
            ):
                if self._stop_requested:
                    self._finish("stopped")
                    return

                X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                if config.augmentation.enabled:
                    X_train, y_train = augment(
                        X_train, y_train, config.augmentation
                    )

                model = FeatureTransformer(
                    n_features=n_features,
                    d_model=config.d_model,
                    n_heads=config.n_heads,
                    n_layers=config.n_layers,
                    dropout=config.dropout,
                )
                optimizer = AdamW(
                    model.parameters(),
                    lr=config.learning_rate,
                    weight_decay=config.weight_decay,
                )
                criterion = nn.MSELoss()

                X_train_t = torch.tensor(X_train)
                y_train_t = torch.tensor(y_train)
                X_val_t = torch.tensor(X_val)
                y_val_t = torch.tensor(y_val)

                patience_counter = 0
                best_fold_loss = float("inf")

                for epoch in range(config.epochs):
                    if self._stop_requested:
                        self._finish("stopped")
                        return

                    model.train()
                    optimizer.zero_grad()
                    pred, _ = model(X_train_t)
                    loss = criterion(pred, y_train_t)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                    model.eval()
                    with torch.no_grad():
                        val_pred, _ = model(X_val_t)
                        val_loss = criterion(val_pred, y_val_t)
                        ss_res = ((y_val_t - val_pred) ** 2).sum()
                        ss_tot = ((y_val_t - y_val_t.mean()) ** 2).sum()
                        r2 = (1 - ss_res / ss_tot).item() if ss_tot > 0 else 0.0

                    self._push_progress(TrainingProgress(
                        epoch=epoch + 1,
                        fold=fold_idx + 1,
                        train_loss=round(loss.item(), 6),
                        val_loss=round(val_loss.item(), 6),
                        r2=round(r2, 6),
                        status="training",
                    ))

                    if val_loss.item() < best_fold_loss:
                        best_fold_loss = val_loss.item()
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= config.early_stopping_patience:
                            break

                if best_fold_loss < best_loss:
                    best_loss = best_fold_loss
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}

            if not self._stop_requested:
                if best_state:
                    self._save_checkpoint(config, best_state, best_loss)
                self._finish("completed")

        except Exception as e:
            self._finish("error", str(e))

    def _save_checkpoint(self, config: TrainingConfig, model_state: dict, val_loss: float):
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        ckpt_id = f"{config.dataset_id}_{self._task_id}"
        ckpt_path = self._checkpoint_dir / ckpt_id
        ckpt_path.mkdir(exist_ok=True)
        torch.save(model_state, ckpt_path / "model.pt")
        (ckpt_path / "config.json").write_text(config.model_dump_json())
        metrics_data: dict = {"final_val_loss": val_loss}
        if self._progress:
            metrics_data["history"] = [p.model_dump() for p in self._progress]
        (ckpt_path / "metrics.json").write_text(json.dumps(metrics_data))

    def _finish(self, status: str, error: str | None = None):
        self._status = status
        msg: dict = {"status": status}
        if error:
            msg["error"] = error
        if self._progress_queue and self._loop:
            asyncio.run_coroutine_threadsafe(
                self._progress_queue.put(msg), self._loop
            )

    def list_checkpoints(self) -> list[CheckpointInfo]:
        if not self._checkpoint_dir.exists():
            return []
        result = []
        for d in sorted(self._checkpoint_dir.iterdir()):
            if d.is_dir() and (d / "config.json").exists():
                config = TrainingConfig.model_validate_json((d / "config.json").read_text())
                metrics = json.loads((d / "metrics.json").read_text())
                result.append(CheckpointInfo(
                    id=d.name,
                    dataset_id=config.dataset_id,
                    config=config,
                    final_loss=metrics.get("final_val_loss", 0),
                ))
        return result


training_service = TrainingService()
