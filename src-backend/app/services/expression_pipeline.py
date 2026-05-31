"""ExpressionPipeline — extracted ML orchestration from ExpressionService.

This module encapsulates the pure ML pipeline (attention extraction,
feature engineering, symbolic regression) without any state management.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.ml.expression_impact import compute_impact
import numpy as np
import sympy

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

from app.exceptions import DomainError, SymbolicRegressionError
from app.ml.attention_extractor import extract_attention_weights, extract_top_pairs
from app.ml.expression_tree import expr_to_latex
from app.ml.symbolic_regressor import (
    extract_best_equation,
    generate_interaction_features,
    run_pareto_regression,
)
from app.services.checkpoint_resolver import CheckpointResolver


@dataclass
class PipelineResult:
    """Immutable result of the expression pipeline."""

    sympy_expr: sympy.Basic
    latex: str
    complexity: int
    loss: float
    r2_score: float
    pareto_equations: list[dict]
    variable_impact: dict[str, float] = field(default_factory=dict)
    indicators: dict[str, float] = field(default_factory=dict)
    target_name: str = ""
    X_train: list[list[float]] = field(default_factory=list)
    X_test: list[list[float]] = field(default_factory=list)
    y_train: list[float] = field(default_factory=list)
    y_test: list[float] = field(default_factory=list)
    aug_names: list[str] = field(default_factory=list)
    X_raw: list[list[float]] = field(default_factory=list)
    pairs_raw: list[dict] = field(default_factory=list)
    attention_matrix: list[list[float]] = field(default_factory=list)
    feature_names: list[str] = field(default_factory=list)


class ExpressionPipeline:
    """Orchestrates the ML pipeline: attention -> features -> symbolic regression.

    Parameters
    ----------
    resolver : CheckpointResolver
        Resolves model checkpoints and loads dataset features.
    """

    def __init__(self, resolver: CheckpointResolver) -> None:
        self._resolver = resolver

    def run(self, model_id: str, top_k: int = 10, preset: str = "standard") -> PipelineResult:
        """Run the full expression discovery pipeline.

        Steps 1-5: resolve checkpoint, load features, extract attention,
        find top pairs, generate interaction features.
        Step 6: run parallel pareto regression (3 models).
        Steps 7-9: extract equations from all models, compute R2 for each.
        """
        cp_dir, config = self._resolver.resolve(model_id)

        # Steps 1-5: DomainError passthrough, others -> SymbolicRegressionError
        try:
            X, y, feature_names = self._resolver.get_features_with_target(
                config.dataset_id,
            )
            matrix = extract_attention_weights(cp_dir, X, config)
            pairs_raw, _ = extract_top_pairs(matrix, feature_names, top_k)
            X_aug, aug_names = generate_interaction_features(
                X, feature_names, pairs_raw,
            )
        except Exception as e:
            if isinstance(e, DomainError):
                raise
            raise SymbolicRegressionError(
                f"Expression pipeline failed: {e}"
            ) from e

        # Step 6: Split for honest out-of-sample R2 evaluation
        # Raw features passed to gplearn (no scaling) to preserve magnitudes
        X_train, X_test, y_train, y_test = train_test_split(
            X_aug, y, test_size=0.2, random_state=42,
        )

        models = run_pareto_regression(X_train, y_train, aug_names, preset=preset)

        try:
            pareto_equations = []
            for i, model in enumerate(models):
                eq = extract_best_equation(model)
                r2 = float(model.score(X_test, y_test))
                pareto_equations.append({
                    "index": i,
                    "sympy_expr": eq["sympy_expr"],
                    "latex": expr_to_latex(eq["sympy_expr"], feature_names=aug_names),
                    "complexity": eq["complexity"],
                    "loss": eq["loss"],
                    "r2_score": r2,
                })
            best_r2 = max(eq["r2_score"] for eq in pareto_equations)
            threshold = best_r2 * 0.9
            best_idx = next(
                (i for i, eq in enumerate(pareto_equations)
                 if eq["r2_score"] >= threshold),
                max(range(len(pareto_equations)),
                    key=lambda i: pareto_equations[i]["r2_score"]),
            )
            best_eq = pareto_equations[best_idx]
        except Exception as e:
            raise SymbolicRegressionError(
                f"Equation extraction failed: {e}"
            ) from e

        best_model = models[best_idx]
        y_train_pred = best_model.predict(X_train)
        y_test_pred = best_model.predict(X_test)

        train_r2 = float(best_model.score(X_train, y_train))
        test_r2 = best_eq["r2_score"]
        train_mae = float(mean_absolute_error(y_train, y_train_pred))
        test_mae = float(mean_absolute_error(y_test, y_test_pred))
        train_mse = float(mean_squared_error(y_train, y_train_pred))
        test_mse = float(mean_squared_error(y_test, y_test_pred))
        train_nmse = train_mse / float(np.var(y_train))
        test_nmse = test_mse / float(np.var(y_test))
        train_rmse = float(np.sqrt(train_mse))
        test_rmse = float(np.sqrt(test_mse))

        indicators = {
            "train_r2": train_r2, "test_r2": test_r2,
            "train_mae": train_mae, "test_mae": test_mae,
            "train_mse": train_mse, "test_mse": test_mse,
            "train_nmse": train_nmse, "test_nmse": test_nmse,
            "train_rmse": train_rmse, "test_rmse": test_rmse,
            "depth": float(best_eq["complexity"]),
            "length": float(len(str(best_model._program))),
        }

        dataset = self._resolver._data_loader.get_dataset(config.dataset_id)
        target_name = dataset.target if dataset else ""

        variable_impact = compute_impact(
            sympy_expr=best_eq["sympy_expr"],
            feature_names=feature_names,
            X=X,
            pairs_raw=pairs_raw,
            attention_matrix=matrix,
            aug_names=aug_names,
        )

        return PipelineResult(
            sympy_expr=best_eq["sympy_expr"],
            latex=expr_to_latex(best_eq["sympy_expr"], feature_names=aug_names),
            complexity=best_eq["complexity"],
            loss=best_eq["loss"],
            r2_score=best_eq["r2_score"],
            pareto_equations=pareto_equations,
            variable_impact=variable_impact,
            indicators=indicators,
            target_name=target_name,
            X_train=X_train.tolist(),
            X_test=X_test.tolist(),
            y_train=y_train.tolist(),
            y_test=y_test.tolist(),
            aug_names=aug_names,
            X_raw=X.tolist(),
            pairs_raw=pairs_raw,
            attention_matrix=matrix.tolist(),
            feature_names=feature_names,
        )
