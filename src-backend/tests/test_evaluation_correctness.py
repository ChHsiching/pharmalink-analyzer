"""Correctness tests: evaluator metrics match sklearn."""

import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from app.ml.evaluator import compute_fold_metrics, compute_residual_bins


def test_r2_matches_sklearn():
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 1.9, 3.1, 4.0, 5.2])
    result = compute_fold_metrics(y_true, y_pred)
    assert abs(result["r2"] - r2_score(y_true, y_pred)) < 1e-10


def test_mse_matches_sklearn():
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.5, 2.5, 2.5, 4.5, 4.5])
    result = compute_fold_metrics(y_true, y_pred)
    assert abs(result["mse"] - mean_squared_error(y_true, y_pred)) < 1e-10


def test_mae_matches_sklearn():
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.5, 2.5, 2.5, 4.5, 4.5])
    result = compute_fold_metrics(y_true, y_pred)
    assert abs(result["mae"] - mean_absolute_error(y_true, y_pred)) < 1e-10


def test_perfect_predictions():
    y = np.array([1.0, 2.0, 3.0])
    result = compute_fold_metrics(y, y)
    assert abs(result["r2"] - 1.0) < 1e-10
    assert abs(result["mse"]) < 1e-10


def test_constant_predictions():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.full(4, y_true.mean())
    result = compute_fold_metrics(y_true, y_pred)
    assert abs(result["r2"]) < 1e-10


def test_residual_bins():
    residuals = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    bins = compute_residual_bins(residuals, n_bins=5)
    assert len(bins) == 5
    total = sum(b["count"] for b in bins)
    assert total == len(residuals)
