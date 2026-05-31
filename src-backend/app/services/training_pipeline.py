"""TrainingPipeline — pure ML training orchestration extracted from TrainingService."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from torch.optim import AdamW

from app.ml.augmentation import augment
from app.ml.transformer import FeatureTransformer
from app.models.training import TrainingConfig, TrainingProgress


@dataclass
class TrainingResult:
    model_state: dict[str, torch.Tensor] | None
    scaler: StandardScaler | None
    best_loss: float
    stopped: bool = False
    fold_states: list[dict[str, torch.Tensor]] | None = None
    fold_scalers: list[StandardScaler] | None = None
    best_fold_idx: int | None = None


class TrainingPipeline:
    """Encapsulates the KFold training loop with early stopping and augmentation.

    Pure ML — no I/O, no asyncio, no service state.
    Reports progress via callback; checks stop via callback.
    """

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_features: int,
        config: TrainingConfig,
        on_progress: Callable[[TrainingProgress], None] | None = None,
        is_stopped: Callable[[], bool] | None = None,
    ) -> TrainingResult:
        _stopped = is_stopped or (lambda: False)

        kfold = KFold(
            n_splits=config.k_folds,
            shuffle=config.k_fold_shuffle,
            random_state=config.k_fold_seed,
        )
        best_state = None
        best_loss = float("inf")
        best_scaler = None
        fold_states: list[dict[str, torch.Tensor]] = []
        fold_scalers: list[StandardScaler] = []
        best_fold_idx: int | None = None

        for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X)):
            if _stopped():
                return TrainingResult(
                    model_state=best_state,
                    scaler=best_scaler,
                    best_loss=best_loss,
                    stopped=True,
                    fold_states=fold_states or None,
                    fold_scalers=fold_scalers or None,
                    best_fold_idx=best_fold_idx,
                )

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X[train_idx]).astype(np.float32)
            X_val = scaler.transform(X[val_idx]).astype(np.float32)
            y_train, y_val = y[train_idx], y[val_idx]

            if config.augmentation.enabled:
                X_train, y_train = augment(X_train, y_train, config.augmentation)

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
                if _stopped():
                    return TrainingResult(
                        model_state=best_state,
                        scaler=best_scaler,
                        best_loss=best_loss,
                        stopped=True,
                        fold_states=fold_states or None,
                        fold_scalers=fold_scalers or None,
                        best_fold_idx=best_fold_idx,
                    )

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

                progress = TrainingProgress(
                    epoch=epoch + 1,
                    fold=fold_idx + 1,
                    train_loss=round(loss.item(), 6),
                    val_loss=round(val_loss.item(), 6),
                    r2=round(r2, 6),
                    status="training",
                )
                if on_progress:
                    on_progress(progress)

                if val_loss.item() < best_fold_loss:
                    best_fold_loss = val_loss.item()
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= config.early_stopping_patience:
                        break

            fold_states.append({k: v.clone() for k, v in model.state_dict().items()})
            fold_scalers.append(scaler)

            if best_fold_loss < best_loss:
                best_loss = best_fold_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                best_scaler = scaler
                best_fold_idx = fold_idx

        return TrainingResult(
            model_state=best_state,
            scaler=best_scaler,
            best_loss=best_loss,
            fold_states=fold_states or None,
            fold_scalers=fold_scalers or None,
            best_fold_idx=best_fold_idx,
        )
