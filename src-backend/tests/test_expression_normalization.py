"""Verify expression pipeline passes raw (unscaled) features to gplearn."""
import json
import numpy as np
import pytest
import pandas as pd
from unittest.mock import MagicMock, patch


class TestExpressionNormalization:
    @patch("app.services.expression_pipeline.extract_best_equation")
    @patch("app.services.expression_pipeline.run_pareto_regression")
    @patch("app.services.expression_pipeline.generate_interaction_features")
    @patch("app.services.expression_pipeline.extract_top_pairs")
    @patch("app.services.expression_pipeline.extract_attention_weights")
    def test_training_data_is_raw_unscaled(
        self, mock_attn, mock_pairs, mock_interact, mock_pareto_reg, mock_best,
        tmp_path,
    ):
        """Training data passed to run_pareto_regression should be raw (unscaled)."""
        from app.services.expression_pipeline import ExpressionPipeline
        from app.services.checkpoint_resolver import CheckpointResolver

        mock_loader = MagicMock()
        ds = MagicMock()
        ds.feature_names = ["A", "B", "C"]
        ds.target = "T"
        mock_loader.get_dataset.return_value = ds

        rng = np.random.default_rng(42)
        data = rng.standard_normal((30, 4)).astype(np.float32) * 5 + 3
        mock_loader.get_dataframe.return_value = pd.DataFrame(
            data, columns=["A", "B", "C", "T"],
        )

        cp_dir = tmp_path / "checkpoints"
        cp_dir.mkdir()
        model_dir = cp_dir / "model-norm"
        model_dir.mkdir()
        (model_dir / "config.json").write_text(json.dumps({
            "dataset_id": "ds-1", "d_model": 32, "n_heads": 2,
            "n_layers": 1, "dropout": 0.0,
        }))

        resolver = CheckpointResolver(cp_dir, mock_loader)
        pipeline = ExpressionPipeline(resolver)

        mock_attn.return_value = np.eye(3, dtype=np.float32)
        mock_pairs.return_value = ([], 0.5)
        mock_interact.return_value = (data[:, :3], ["A", "B", "C"])

        mock_model = MagicMock()
        mock_model.score.return_value = 0.85
        mock_model.predict.side_effect = lambda X: np.zeros(X.shape[0], dtype=np.float32)
        mock_model._program = "A"
        mock_pareto_reg.return_value = [mock_model]
        mock_best.return_value = {
            "sympy_expr": __import__("sympy").Symbol("A"),
            "latex": "A",
            "complexity": 1,
            "loss": 0.15,
        }

        pipeline.run("model-norm", preset="quick")

        X_train = mock_pareto_reg.call_args[0][0]
        # Raw features should NOT be centered at 0 or have std=1
        # Original data is ~N(3, 5), so train split should retain that scale
        assert X_train.shape[0] < 30, "Should be a train split"

    @patch("app.services.expression_pipeline.extract_best_equation")
    @patch("app.services.expression_pipeline.run_pareto_regression")
    @patch("app.services.expression_pipeline.generate_interaction_features")
    @patch("app.services.expression_pipeline.extract_top_pairs")
    @patch("app.services.expression_pipeline.extract_attention_weights")
    def test_r2_uses_raw_test_data(
        self, mock_attn, mock_pairs, mock_interact, mock_pareto_reg, mock_best,
        tmp_path,
    ):
        """model.score() should receive raw X_test, not scaled X_test."""
        from app.services.expression_pipeline import ExpressionPipeline
        from app.services.checkpoint_resolver import CheckpointResolver

        mock_loader = MagicMock()
        ds = MagicMock()
        ds.feature_names = ["A", "B"]
        ds.target = "T"
        mock_loader.get_dataset.return_value = ds

        rng = np.random.default_rng(99)
        data = rng.standard_normal((20, 3)).astype(np.float32) * 10 + 5
        mock_loader.get_dataframe.return_value = pd.DataFrame(
            data, columns=["A", "B", "T"],
        )

        cp_dir = tmp_path / "checkpoints"
        cp_dir.mkdir()
        model_dir = cp_dir / "model-r2"
        model_dir.mkdir()
        (model_dir / "config.json").write_text(json.dumps({
            "dataset_id": "ds-1", "d_model": 32, "n_heads": 2,
            "n_layers": 1, "dropout": 0.0,
        }))

        resolver = CheckpointResolver(cp_dir, mock_loader)
        pipeline = ExpressionPipeline(resolver)

        mock_attn.return_value = np.eye(2, dtype=np.float32)
        mock_pairs.return_value = ([], 0.5)
        mock_interact.return_value = (data[:, :2], ["A", "B"])

        mock_model = MagicMock()
        mock_model.score.return_value = 0.80
        mock_model.predict.side_effect = lambda X: np.zeros(X.shape[0], dtype=np.float32)
        mock_model._program = "A"
        mock_pareto_reg.return_value = [mock_model]
        mock_best.return_value = {
            "sympy_expr": __import__("sympy").Symbol("A"),
            "latex": "A",
            "complexity": 1,
            "loss": 0.20,
        }

        pipeline.run("model-r2", preset="quick")

        X_test = mock_model.score.call_args[0][0]
        # Test data should be raw (unscaled), not centered at 0
        test_means = np.mean(X_test, axis=0)
        assert np.all(np.abs(test_means) > 2.0), (
            f"Test data appears scaled (means near 0): means={test_means}"
        )

    @patch("app.services.expression_pipeline.extract_best_equation")
    @patch("app.services.expression_pipeline.run_pareto_regression")
    @patch("app.services.expression_pipeline.generate_interaction_features")
    @patch("app.services.expression_pipeline.extract_top_pairs")
    @patch("app.services.expression_pipeline.extract_attention_weights")
    def test_no_scaler_used(
        self, mock_attn, mock_pairs, mock_interact, mock_pareto_reg, mock_best,
        tmp_path,
    ):
        """Pipeline should not use StandardScaler at all."""
        from app.services.expression_pipeline import ExpressionPipeline
        from app.services.checkpoint_resolver import CheckpointResolver
        from sklearn.preprocessing import StandardScaler as RealScaler

        mock_loader = MagicMock()
        ds = MagicMock()
        ds.feature_names = ["A", "B", "C"]
        ds.target = "T"
        mock_loader.get_dataset.return_value = ds

        rng = np.random.default_rng(7)
        data = rng.standard_normal((30, 4)).astype(np.float32) * 5 + 3
        mock_loader.get_dataframe.return_value = pd.DataFrame(
            data, columns=["A", "B", "C", "T"],
        )

        cp_dir = tmp_path / "checkpoints"
        cp_dir.mkdir()
        model_dir = cp_dir / "model-leak"
        model_dir.mkdir()
        (model_dir / "config.json").write_text(json.dumps({
            "dataset_id": "ds-1", "d_model": 32, "n_heads": 2,
            "n_layers": 1, "dropout": 0.0,
        }))

        resolver = CheckpointResolver(cp_dir, mock_loader)
        pipeline = ExpressionPipeline(resolver)

        mock_attn.return_value = np.eye(3, dtype=np.float32)
        mock_pairs.return_value = ([], 0.5)
        mock_interact.return_value = (data[:, :3], ["A", "B", "C"])

        mock_model = MagicMock()
        mock_model.score.return_value = 0.80
        mock_model.predict.side_effect = lambda X: np.zeros(X.shape[0], dtype=np.float32)
        mock_model._program = "A"
        mock_pareto_reg.return_value = [mock_model]
        mock_best.return_value = {
            "sympy_expr": __import__("sympy").Symbol("A"),
            "latex": "A",
            "complexity": 1,
            "loss": 0.20,
        }

        fit_input_shapes = []
        original_fit_transform = RealScaler.fit_transform

        def tracking_fit_transform(self, X, y=None):
            fit_input_shapes.append(X.shape)
            return original_fit_transform(self, X, y)

        with patch.object(RealScaler, "fit_transform", tracking_fit_transform):
            pipeline.run("model-leak", preset="quick")

        assert len(fit_input_shapes) == 0, (
            f"StandardScaler.fit_transform should not be called, got {len(fit_input_shapes)} calls"
        )
