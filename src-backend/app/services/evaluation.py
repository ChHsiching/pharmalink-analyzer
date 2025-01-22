import json
import time
from pathlib import Path

import numpy as np

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
from app.services.checkpoint_resolver import CheckpointResolver
from app.services.data_loader import data_loader as _default_data_loader


class EvaluationService:
    _CACHE_TTL = 300.0  # seconds

    def __init__(self, resolver: CheckpointResolver):
        self._resolver = resolver
        self._cache: dict[str, tuple[float, tuple]] = {}

    def _evaluate(self, model_id: str):
        now = time.time()
        if model_id in self._cache:
            ts, cached = self._cache[model_id]
            if now - ts < self._CACHE_TTL:
                return cached

        cp_dir, config = self._resolver.resolve(model_id)
        X, y, _ = self._resolver.get_features_with_target(config.dataset_id)
        n_features = X.shape[1]
        result = (cp_dir, config, evaluate_all_folds(cp_dir, X, y, n_features, config))
        self._cache[model_id] = (now, result)
        return result

    def invalidate(self, model_id: str) -> None:
        self._cache.pop(model_id, None)

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
        cp_dir, _ = self._resolver.resolve(model_id)
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
