from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import KFold

from app.ml.checkpoint_loader import load_model, load_scaler_params
from app.models.training import TrainingConfig


def compute_fold_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    mse = float(np.mean((y_true - y_pred) ** 2))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    return {"r2": r2, "mse": mse, "mae": mae}


def compute_residual_bins(
    residuals: np.ndarray, n_bins: int = 20,
) -> list[dict]:
    counts, edges = np.histogram(residuals, bins=n_bins)
    bins = []
    for i in range(len(counts)):
        bins.append({
            "range_start": round(float(edges[i]), 6),
            "range_end": round(float(edges[i + 1]), 6),
            "count": int(counts[i]),
        })
    return bins


def evaluate_all_folds(
    checkpoint_dir: Path,
    X: np.ndarray,
    y: np.ndarray,
    n_features: int,
    config: TrainingConfig,
) -> list[dict]:
    mean, scale = load_scaler_params(checkpoint_dir / "scaler_params.json")
    safe_scale = np.where(np.abs(scale) < 1e-8, 1.0, scale)
    X_scaled = ((X - mean) / safe_scale).astype(np.float32)

    kfold = KFold(
        n_splits=config.k_folds,
        shuffle=config.k_fold_shuffle,
        random_state=config.k_fold_seed,
    )

    model = load_model(checkpoint_dir, n_features, config)

    fold_results = []
    with torch.no_grad():
        for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X_scaled)):
            X_val = torch.tensor(X_scaled[val_idx])
            y_val = y[val_idx]
            predictions, _ = model(X_val)
            preds_np = predictions.numpy()

            metrics = compute_fold_metrics(y_val, preds_np)
            metrics["fold"] = fold_idx + 1
            metrics["predictions"] = preds_np.tolist()
            metrics["actuals"] = y_val.tolist()
            metrics["residuals"] = (y_val - preds_np).tolist()
            fold_results.append(metrics)

    return fold_results
