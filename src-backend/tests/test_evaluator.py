import numpy as np
import pytest

from app.ml.evaluator import (
    compute_fold_metrics,
    evaluate_all_folds,
    compute_residual_bins,
)


class TestComputeFoldMetrics:
    def test_perfect_predictions(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        m = compute_fold_metrics(y_true, y_pred)
        assert abs(m["r2"] - 1.0) < 1e-6
        assert abs(m["mse"]) < 1e-6
        assert abs(m["mae"]) < 1e-6

    def test_imperfect_predictions(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.2, 2.8, 4.1, 4.9])
        m = compute_fold_metrics(y_true, y_pred)
        assert 0.9 < m["r2"] < 1.0
        assert m["mse"] > 0
        assert m["mae"] > 0

    def test_metrics_non_negative(self):
        rng = np.random.default_rng(42)
        y_true = rng.standard_normal(50).astype(np.float32)
        y_pred = rng.standard_normal(50).astype(np.float32)
        m = compute_fold_metrics(y_true, y_pred)
        assert m["mse"] >= 0
        assert m["mae"] >= 0

    def test_constant_predictions(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 2.0, 2.0])
        m = compute_fold_metrics(y_true, y_pred)
        assert m["r2"] <= 0
        assert m["mse"] > 0


class TestComputeResidualBins:
    def test_bins_cover_all_residuals(self):
        residuals = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        bins = compute_residual_bins(residuals, n_bins=5)
        total = sum(b["count"] for b in bins)
        assert total == 5

    def test_bins_have_correct_ranges(self):
        residuals = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        bins = compute_residual_bins(residuals, n_bins=4)
        assert len(bins) == 4
        assert bins[0]["range_start"] < bins[0]["range_end"]
        assert abs(bins[0]["range_start"] - 0.0) < 1e-6
        assert abs(bins[-1]["range_end"] - 4.0) < 1e-6


class TestEvaluateAllFolds:
    @pytest.fixture
    def eval_env(self, make_checkpoint):
        return make_checkpoint(
            n_samples=30,
            training_epochs=30,
            checkpoint_name="checkpoints/test_task1",
            dataset_id="test",
            k_folds=3,
        )

    def test_returns_per_fold_results(self, eval_env):
        results = evaluate_all_folds(
            eval_env["cp_dir"], eval_env["X"], eval_env["y"],
            eval_env["n_features"], eval_env["config"],
        )
        assert len(results) == 3
        for i, r in enumerate(results):
            assert r["fold"] == i + 1
            assert "r2" in r
            assert "mse" in r
            assert "mae" in r
            assert "predictions" in r
            assert "actuals" in r
            assert "residuals" in r

    def test_predictions_match_actuals_length(self, eval_env):
        results = evaluate_all_folds(
            eval_env["cp_dir"], eval_env["X"], eval_env["y"],
            eval_env["n_features"], eval_env["config"],
        )
        for r in results:
            assert len(r["predictions"]) == len(r["actuals"])
            assert len(r["residuals"]) == len(r["actuals"])

    def test_residuals_are_correct(self, eval_env):
        results = evaluate_all_folds(
            eval_env["cp_dir"], eval_env["X"], eval_env["y"],
            eval_env["n_features"], eval_env["config"],
        )
        for r in results:
            for actual, pred, res in zip(r["actuals"], r["predictions"], r["residuals"]):
                assert abs(res - (actual - pred)) < 1e-5

import torch
from app.ml.checkpoint_loader import save_scaler_params


class TestEvaluateUsesSavedScaler:
    def test_loads_scaler_from_checkpoint(self, tmp_path):
        """Evaluator calls load_scaler_params -- never fit_transform."""
        from unittest.mock import patch

        from app.ml.evaluator import evaluate_all_folds
        from app.ml.transformer import FeatureTransformer
        from app.models.training import TrainingConfig

        rng = np.random.default_rng(42)
        n_features = 5
        X = rng.standard_normal((30, n_features)).astype(np.float32)
        y = rng.standard_normal(30).astype(np.float32)

        saved_mean = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        saved_scale = np.array([0.5, 1.0, 1.5, 2.0, 2.5], dtype=np.float32)

        config = TrainingConfig(
            dataset_id="test", d_model=16, n_heads=2, n_layers=1,
            k_folds=3,
        )
        cp_dir = tmp_path / "test_ckpt"
        cp_dir.mkdir()
        save_scaler_params(cp_dir / "scaler_params.json", saved_mean, saved_scale)

        model = FeatureTransformer(
            n_features=n_features, d_model=16, n_heads=2,
            n_layers=1, dropout=0.0,
        )
        torch.save(model.state_dict(), cp_dir / "model.pt")
        (cp_dir / "config.json").write_text(config.model_dump_json())
        (cp_dir / "metrics.json").write_text('{"final_val_loss": 0.1}')

        with patch(
            "app.ml.evaluator.load_scaler_params",
            wraps=__import__(
                "app.ml.checkpoint_loader", fromlist=["load_scaler_params"]
            ).load_scaler_params,
        ) as mock_load:
            results = evaluate_all_folds(cp_dir, X, y, n_features, config)
            mock_load.assert_called_once()
            assert "scaler_params.json" in str(mock_load.call_args[0][0])

        assert len(results) == 3
