# src-backend/app/services/evaluation.py
import json
from pathlib import Path

import numpy as np
from fastapi import HTTPException

from app.config import CHECKPOINT_DIR
from app.ml.evaluator import compute_residual_bins, evaluate_all_folds
from app.models.evaluation import (
    AggregateMetrics,
    EvaluationMetricsResponse,
    FoldLossCurve,
    FoldMetrics,
    LossCurvePoint,
    LossCurveResponse,
    PredictionPoint,
    PredictionsResponse,
    ResidualBin,
    ResidualsResponse,
)
from app.models.training import TrainingConfig
from app.services.data_loader import data_loader as _default_data_loader


class EvaluationService:
    def __init__(self, data_loader=None, checkpoint_dir=None):
        self._data_loader = data_loader or _default_data_loader
        self._checkpoint_dir = checkpoint_dir or CHECKPOINT_DIR

    def _resolve_checkpoint(self, model_id: str):
        cp_dir = self._checkpoint_dir / model_id
        if not cp_dir.is_dir():
            raise HTTPException(status_code=404, detail=f"Checkpoint not found: {model_id}")
        config_path = cp_dir / "config.json"
        if not config_path.exists():
            raise HTTPException(status_code=404, detail=f"Config not found: {model_id}")
        config = TrainingConfig.model_validate_json(config_path.read_text())
        return cp_dir, config

    def _get_features(self, dataset_id: str):
        dataset = self._data_loader.get_dataset(dataset_id)
        if dataset is None:
            raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_id}")
        df = self._data_loader.get_dataframe(dataset_id)
        if df is None:
            raise HTTPException(status_code=404, detail=f"Dataset data not found: {dataset_id}")
        feature_names = dataset.feature_names
        X = df[feature_names].to_numpy().astype(np.float32)
        y = df["target"].to_numpy().astype(np.float32)
        return X, y, feature_names

    def _evaluate(self, model_id: str):
        cp_dir, config = self._resolve_checkpoint(model_id)
        X, y, _ = self._get_features(config.dataset_id)
        n_features = X.shape[1]
        return cp_dir, config, evaluate_all_folds(cp_dir, X, y, n_features, config)

    def get_metrics(self, model_id: str) -> EvaluationMetricsResponse:
        _, _, fold_results = self._evaluate(model_id)
        folds = [
            FoldMetrics(fold=r["fold"], r2=round(r["r2"], 6), mse=round(r["mse"], 6), mae=round(r["mae"], 6))
            for r in fold_results
        ]
        r2_vals = [f.r2 for f in folds]
        mse_vals = [f.mse for f in folds]
        mae_vals = [f.mae for f in folds]
        aggregate = AggregateMetrics(
            r2_mean=round(float(np.mean(r2_vals)), 6),
            r2_std=round(float(np.std(r2_vals)), 6),
            mse_mean=round(float(np.mean(mse_vals)), 6),
            mse_std=round(float(np.std(mse_vals)), 6),
            mae_mean=round(float(np.mean(mae_vals)), 6),
            mae_std=round(float(np.std(mae_vals)), 6),
        )
        return EvaluationMetricsResponse(model_id=model_id, folds=folds, aggregate=aggregate)

    def get_predictions(self, model_id: str) -> PredictionsResponse:
        _, _, fold_results = self._evaluate(model_id)
        predictions = []
        for r in fold_results:
            for actual, predicted in zip(r["actuals"], r["predictions"]):
                predictions.append(PredictionPoint(actual=round(actual, 6), predicted=round(predicted, 6)))
        return PredictionsResponse(model_id=model_id, predictions=predictions)

    def get_residuals(self, model_id: str) -> ResidualsResponse:
        _, _, fold_results = self._evaluate(model_id)
        all_residuals = []
        for r in fold_results:
            all_residuals.extend(r["residuals"])
        residuals_arr = np.array(all_residuals, dtype=np.float64)
        bins_raw = compute_residual_bins(residuals_arr)
        bins = [ResidualBin(**b) for b in bins_raw]
        return ResidualsResponse(
            model_id=model_id,
            residuals=[round(float(v), 6) for v in all_residuals],
            bins=bins,
            mean=round(float(np.mean(residuals_arr)), 6),
            std=round(float(np.std(residuals_arr)), 6),
        )

    def get_loss_curve(self, model_id: str) -> LossCurveResponse:
        cp_dir, _ = self._resolve_checkpoint(model_id)
        metrics_raw = json.loads((cp_dir / "metrics.json").read_text())
        history = metrics_raw.get("history", [])
        folds_dict: dict[int, list[LossCurvePoint]] = {}
        for entry in history:
            fold_num = entry["fold"]
            if fold_num not in folds_dict:
                folds_dict[fold_num] = []
            folds_dict[fold_num].append(LossCurvePoint(
                epoch=entry["epoch"],
                train_loss=entry["train_loss"],
                val_loss=entry["val_loss"],
            ))
        folds = [FoldLossCurve(fold=k, points=v) for k, v in sorted(folds_dict.items())]
        return LossCurveResponse(model_id=model_id, folds=folds)


evaluation_service = EvaluationService()
