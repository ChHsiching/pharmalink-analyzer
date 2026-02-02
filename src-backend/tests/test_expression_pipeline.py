"""Tests for ExpressionPipeline — validates extracted ML orchestration."""

from __future__ import annotations

import json
from dataclasses import is_dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import sympy

from app.exceptions import (
    CheckpointNotFoundError,
    DatasetNotFoundError,
    DomainError,
    SymbolicRegressionError,
)
from app.services.checkpoint_resolver import CheckpointResolver
from app.services.expression_pipeline import ExpressionPipeline, PipelineResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_dataset():
    """A DatasetDetail-like object with feature_names and target."""
    ds = MagicMock()
    ds.feature_names = ["A", "B", "C"]
    ds.target = "TC"
    return ds


@pytest.fixture
def fake_df():
    """A minimal DataFrame with columns A, B, C, TC."""
    import pandas as pd
    rng = np.random.default_rng(0)
    data = rng.standard_normal((20, 4)).astype(np.float32)
    return pd.DataFrame(data, columns=["A", "B", "C", "TC"])


@pytest.fixture
def mock_loader(fake_dataset, fake_df):
    loader = MagicMock()
    loader.get_dataset.return_value = fake_dataset
    loader.get_dataframe.return_value = fake_df
    return loader


@pytest.fixture
def checkpoint_dir(tmp_path):
    cp = tmp_path / "checkpoints"
    cp.mkdir()
    return cp


@pytest.fixture
def pipeline(mock_loader, checkpoint_dir):
    return ExpressionPipeline(resolver=CheckpointResolver(checkpoint_dir, mock_loader))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_checkpoint(cp_dir: Path, dataset_id: str = "ds-1"):
    """Create a minimal checkpoint directory with config.json."""
    model_dir = cp_dir / "model-abc"
    model_dir.mkdir()
    config = {
        "dataset_id": dataset_id,
        "d_model": 32,
        "n_heads": 2,
        "n_layers": 1,
        "dropout": 0.0,
    }
    (model_dir / "config.json").write_text(json.dumps(config))
    return model_dir


def _fake_attention_matrix(n=3):
    """Return a deterministic NxN attention matrix."""
    rng = np.random.default_rng(1)
    m = rng.standard_normal((n, n)).astype(np.float32)
    return (m + m.T) / 2  # symmetrize


def _fake_pairs():
    return [
        {"source": "A", "target": "B", "weight": 0.9},
        {"source": "B", "target": "C", "weight": 0.7},
    ]


def _fake_best():
    a = sympy.Symbol("A")
    b = sympy.Symbol("B")
    expr = a ** 2 + b ** 2 + 1
    return {
        "sympy_expr": expr,
        "latex": "A^{2} + B^{2} + 1",
        "complexity": 7,
        "loss": 0.005,
    }


def _fake_pareto():
    a = sympy.Symbol("A")
    b = sympy.Symbol("B")
    return [
        {"index": 0, "latex": "A + B", "complexity": 3, "loss": 0.05, "r2_score": 0.80, "sympy_expr": a + b},
        {"index": 1, "latex": "A^{2} + 1", "complexity": 5, "loss": 0.01, "r2_score": 0.90, "sympy_expr": a ** 2 + 1},
        {"index": 2, "latex": "A^{2} + B^{2} + 1", "complexity": 7, "loss": 0.005, "r2_score": 0.95, "sympy_expr": a ** 2 + b ** 2 + 1},
    ]


def _fake_model():
    """Return a mock gplearn model with .score() returning R2."""
    model = MagicMock()
    model.score.return_value = 0.95
    model.predict.side_effect = lambda X: np.zeros(X.shape[0], dtype=np.float32)
    model._program = "X0 + X1 * X2"
    return model


def _setup_pareto_mocks(mock_pareto_reg, mock_best):
    """Configure mocks for run_pareto_regression + extract_best_equation."""
    eqs = _fake_pareto()
    mock_models = [_fake_model(), _fake_model(), _fake_model()]
    mock_models[0].score.return_value = eqs[0]["r2_score"]
    mock_models[1].score.return_value = eqs[1]["r2_score"]
    mock_models[2].score.return_value = eqs[2]["r2_score"]
    mock_pareto_reg.return_value = mock_models
    mock_best.side_effect = [
        {"sympy_expr": eqs[0]["sympy_expr"], "latex": eqs[0]["latex"], "complexity": eqs[0]["complexity"], "loss": eqs[0]["loss"]},
        {"sympy_expr": eqs[1]["sympy_expr"], "latex": eqs[1]["latex"], "complexity": eqs[1]["complexity"], "loss": eqs[1]["loss"]},
        {"sympy_expr": eqs[2]["sympy_expr"], "latex": eqs[2]["latex"], "complexity": eqs[2]["complexity"], "loss": eqs[2]["loss"]},
    ]
    return mock_models


# ---------------------------------------------------------------------------
# TestPipelineResult
# ---------------------------------------------------------------------------

class TestPipelineResult:
    def test_is_dataclass(self):
        assert is_dataclass(PipelineResult)

    def test_holds_pipeline_output(self):
        x = sympy.Symbol("x")
        result = PipelineResult(
            sympy_expr=x ** 2 + 1,
            latex="x^{2} + 1",
            complexity=3,
            loss=0.01,
            r2_score=0.95,
            pareto_equations=[{"latex": "x"}],
        )
        assert result.sympy_expr == x ** 2 + 1
        assert result.latex == "x^{2} + 1"
        assert result.complexity == 3
        assert result.loss == 0.01
        assert result.r2_score == 0.95
        assert len(result.pareto_equations) == 1


# ---------------------------------------------------------------------------
# TestPipelineRun
# ---------------------------------------------------------------------------

class TestPipelineRun:
    @patch("app.services.expression_pipeline.compute_impact")
    @patch("app.services.expression_pipeline.extract_best_equation")
    @patch("app.services.expression_pipeline.run_pareto_regression")
    @patch("app.services.expression_pipeline.generate_interaction_features")
    @patch("app.services.expression_pipeline.extract_top_pairs")
    @patch("app.services.expression_pipeline.extract_attention_weights")
    def test_run_returns_pipeline_result(
        self, mock_attn, mock_pairs, mock_interact, mock_pareto_reg, mock_best, mock_impact,
        pipeline, checkpoint_dir,
    ):
        _write_checkpoint(checkpoint_dir)
        mock_attn.return_value = _fake_attention_matrix()
        mock_pairs.return_value = (_fake_pairs(), 0.5)
        mock_interact.return_value = (
            np.zeros((20, 5)),
            ["A", "B", "C", "A_mul_B", "B_mul_C"],
        )
        _setup_pareto_mocks(mock_pareto_reg, mock_best)
        mock_impact.return_value = {"A": 0.5, "B": 0.3, "C": 0.2}

        result = pipeline.run("model-abc")

        assert isinstance(result, PipelineResult)
        assert result.latex == "A^{2} + 1"
        assert result.complexity == 5
        assert result.loss == 0.01
        assert result.r2_score == 0.90
        assert len(result.pareto_equations) == 3

    def test_run_raises_on_missing_checkpoint(self, pipeline, checkpoint_dir):
        with pytest.raises(CheckpointNotFoundError):
            pipeline.run("nonexistent")

    @patch("app.services.expression_pipeline.extract_attention_weights")
    def test_run_raises_on_missing_dataset(
        self, mock_attn, pipeline, checkpoint_dir, mock_loader,
    ):
        _write_checkpoint(checkpoint_dir)
        mock_loader.get_dataset.return_value = None

        with pytest.raises(DatasetNotFoundError):
            pipeline.run("model-abc")

    @patch("app.services.expression_pipeline.compute_impact")
    @patch("app.services.expression_pipeline.extract_best_equation")
    @patch("app.services.expression_pipeline.run_pareto_regression")
    @patch("app.services.expression_pipeline.generate_interaction_features")
    @patch("app.services.expression_pipeline.extract_top_pairs")
    @patch("app.services.expression_pipeline.extract_attention_weights")
    def test_uses_train_test_split_for_r2(
        self, mock_attn, mock_pairs, mock_interact, mock_pareto_reg, mock_best, mock_impact,
        pipeline, checkpoint_dir,
    ):
        """Verify the pipeline splits data into train/test before fitting and scoring.

        The pipeline must evaluate out-of-sample R2 on a held-out test set,
        not on the training data used for symbolic regression.
        """
        _write_checkpoint(checkpoint_dir)
        mock_attn.return_value = _fake_attention_matrix()
        mock_pairs.return_value = (_fake_pairs(), 0.5)
        mock_interact.return_value = (
            np.zeros((20, 5)),
            ["A", "B", "C", "A_mul_B", "B_mul_C"],
        )
        mock_models = _setup_pareto_mocks(mock_pareto_reg, mock_best)
        mock_impact.return_value = {"A": 0.5, "B": 0.3, "C": 0.2}

        pipeline.run("model-abc")

        # run_pareto_regression should receive a training subset (< 20 rows)
        X_train = mock_pareto_reg.call_args[0][0]
        y_train = mock_pareto_reg.call_args[0][1]
        assert X_train.shape[0] < 20, (
            f"Expected train rows < 20, got {X_train.shape[0]}"
        )

        # model.score should receive the held-out test subset
        X_test = mock_models[0].score.call_args[0][0]
        y_test = mock_models[0].score.call_args[0][1]

        # train + test must equal the full dataset (20 rows)
        assert X_train.shape[0] + X_test.shape[0] == 20, (
            f"train ({X_train.shape[0]}) + test ({X_test.shape[0]}) != 20"
        )

    @patch("app.services.expression_pipeline.compute_impact")
    @patch("app.services.expression_pipeline.extract_best_equation")
    @patch("app.services.expression_pipeline.run_pareto_regression")
    @patch("app.services.expression_pipeline.generate_interaction_features")
    @patch("app.services.expression_pipeline.extract_top_pairs")
    @patch("app.services.expression_pipeline.extract_attention_weights")
    def test_run_forwards_preset_to_regressor(
        self, mock_attn, mock_pairs, mock_interact, mock_pareto_reg, mock_best, mock_impact,
        pipeline, checkpoint_dir,
    ):
        _write_checkpoint(checkpoint_dir)
        mock_attn.return_value = _fake_attention_matrix()
        mock_pairs.return_value = (_fake_pairs(), 0.5)
        mock_interact.return_value = (
            np.zeros((20, 5)),
            ["A", "B", "C", "A_mul_B", "B_mul_C"],
        )
        _setup_pareto_mocks(mock_pareto_reg, mock_best)
        mock_impact.return_value = {"A": 0.5, "B": 0.3, "C": 0.2}

        pipeline.run("model-abc", preset="quick")

        mock_pareto_reg.assert_called_once()
        assert mock_pareto_reg.call_args[1]["preset"] == "quick"

    @patch("app.services.expression_pipeline.compute_impact")
    @patch("app.services.expression_pipeline.extract_best_equation")
    @patch("app.services.expression_pipeline.run_pareto_regression")
    @patch("app.services.expression_pipeline.generate_interaction_features")
    @patch("app.services.expression_pipeline.extract_top_pairs")
    @patch("app.services.expression_pipeline.extract_attention_weights")
    def test_run_uses_pareto_regression(
        self, mock_attn, mock_pairs, mock_interact, mock_pareto_reg, mock_best, mock_impact,
        pipeline, checkpoint_dir,
    ):
        _write_checkpoint(checkpoint_dir)
        mock_attn.return_value = _fake_attention_matrix()
        mock_pairs.return_value = (_fake_pairs(), 0.5)
        mock_interact.return_value = (
            np.zeros((20, 5)),
            ["A", "B", "C", "A_mul_B", "B_mul_C"],
        )
        _setup_pareto_mocks(mock_pareto_reg, mock_best)
        mock_impact.return_value = {"A": 0.5, "B": 0.3, "C": 0.2}

        result = pipeline.run("model-abc")

        mock_pareto_reg.assert_called_once()
        assert len(result.pareto_equations) == 3
        for eq in result.pareto_equations:
            assert "r2_score" in eq
        assert result.complexity == 5

    @patch("app.services.expression_pipeline.compute_impact")
    @patch("app.services.expression_pipeline.extract_best_equation")
    @patch("app.services.expression_pipeline.run_pareto_regression")
    @patch("app.services.expression_pipeline.generate_interaction_features")
    @patch("app.services.expression_pipeline.extract_top_pairs")
    @patch("app.services.expression_pipeline.extract_attention_weights")
    def test_run_passes_raw_features_to_regressor(
        self, mock_attn, mock_pairs, mock_interact, mock_pareto_reg, mock_best, mock_impact,
        pipeline, checkpoint_dir,
    ):
        """gplearn should receive raw (unscaled) features, not StandardScaler output."""
        _write_checkpoint(checkpoint_dir)
        mock_attn.return_value = _fake_attention_matrix()
        mock_pairs.return_value = (_fake_pairs(), 0.5)
        mock_interact.return_value = (
            np.ones((20, 5)),
            ["A", "B", "C", "A_mul_B", "B_mul_C"],
        )
        _setup_pareto_mocks(mock_pareto_reg, mock_best)
        mock_impact.return_value = {"A": 0.5, "B": 0.3, "C": 0.2}

        pipeline.run("model-abc")

        X_received = mock_pareto_reg.call_args[0][0]
        # Raw features: train split from 20 rows of ones (not scaled to mean=0, std=1)
        assert X_received.shape[0] < 20
        # If scaled, means would be ~0 and stds ~1; raw ones should have different stats
        assert np.all(np.abs(X_received.mean(axis=0) - 1.0) < 0.5), (
            f"Features appear scaled (mean ~0): mean={X_received.mean(axis=0)}"
        )

    @patch("app.services.expression_pipeline.compute_impact")
    @patch("app.services.expression_pipeline.extract_best_equation")
    @patch("app.services.expression_pipeline.run_pareto_regression")
    @patch("app.services.expression_pipeline.generate_interaction_features")
    @patch("app.services.expression_pipeline.extract_top_pairs")
    @patch("app.services.expression_pipeline.extract_attention_weights")
    def test_selection_picks_best_r2_complexity_tradeoff(
        self, mock_attn, mock_pairs, mock_interact, mock_pareto_reg, mock_best, mock_impact,
        pipeline, checkpoint_dir,
    ):
        """Pipeline should select the simplest equation with R2 within 10% of best R2."""
        _write_checkpoint(checkpoint_dir)
        mock_attn.return_value = _fake_attention_matrix()
        mock_pairs.return_value = (_fake_pairs(), 0.5)
        mock_interact.return_value = (
            np.zeros((20, 5)),
            ["A", "B", "C", "A_mul_B", "B_mul_C"],
        )
        _setup_pareto_mocks(mock_pareto_reg, mock_best)
        mock_impact.return_value = {"A": 0.5, "B": 0.3, "C": 0.2}

        result = pipeline.run("model-abc")

        # eq[0]: R2=0.80, eq[1]: R2=0.90, eq[2]: R2=0.95
        # threshold = 0.95 * 0.9 = 0.855
        # eq[1] (R2=0.90 >= 0.855) should be selected, NOT eq[2] (most complex)
        assert result.complexity == 5
        assert result.r2_score == pytest.approx(0.90)
        assert result.latex == "A^{2} + 1"


# ---------------------------------------------------------------------------
# TestPipelineErrorWrapping
# ---------------------------------------------------------------------------

class TestPipelineErrorWrapping:
    @patch("app.services.expression_pipeline.extract_attention_weights")
    def test_wraps_attention_error(self, mock_attn, pipeline, checkpoint_dir):
        _write_checkpoint(checkpoint_dir)
        mock_attn.side_effect = RuntimeError("CUDA out of memory")

        with pytest.raises(SymbolicRegressionError):
            pipeline.run("model-abc")

    @patch("app.services.expression_pipeline.extract_top_pairs")
    @patch("app.services.expression_pipeline.extract_attention_weights")
    def test_wraps_pairs_error(
        self, mock_attn, mock_pairs, pipeline, checkpoint_dir,
    ):
        _write_checkpoint(checkpoint_dir)
        mock_attn.return_value = _fake_attention_matrix()
        mock_pairs.side_effect = ValueError("bad pairs")

        with pytest.raises(SymbolicRegressionError):
            pipeline.run("model-abc")

    @patch("app.services.expression_pipeline.generate_interaction_features")
    @patch("app.services.expression_pipeline.extract_top_pairs")
    @patch("app.services.expression_pipeline.extract_attention_weights")
    def test_wraps_feature_error(
        self, mock_attn, mock_pairs, mock_interact, pipeline, checkpoint_dir,
    ):
        _write_checkpoint(checkpoint_dir)
        mock_attn.return_value = _fake_attention_matrix()
        mock_pairs.return_value = (_fake_pairs(), 0.5)
        mock_interact.side_effect = RuntimeError("feature gen failed")

        with pytest.raises(SymbolicRegressionError):
            pipeline.run("model-abc")

    @patch("app.services.expression_pipeline.compute_impact")
    @patch("app.services.expression_pipeline.extract_best_equation")
    @patch("app.services.expression_pipeline.run_pareto_regression")
    @patch("app.services.expression_pipeline.generate_interaction_features")
    @patch("app.services.expression_pipeline.extract_top_pairs")
    @patch("app.services.expression_pipeline.extract_attention_weights")
    def test_wraps_equation_extraction_error(
        self, mock_attn, mock_pairs, mock_interact, mock_pareto_reg, mock_best, mock_impact,
        pipeline, checkpoint_dir,
    ):
        _write_checkpoint(checkpoint_dir)
        mock_attn.return_value = _fake_attention_matrix()
        mock_pairs.return_value = (_fake_pairs(), 0.5)
        mock_interact.return_value = (
            np.zeros((20, 5)),
            ["A", "B", "C", "A_mul_B", "B_mul_C"],
        )
        mock_models = [_fake_model(), _fake_model(), _fake_model()]
        mock_pareto_reg.return_value = mock_models
        mock_best.side_effect = KeyError("no best equation")

        with pytest.raises(SymbolicRegressionError):
            pipeline.run("model-abc")
