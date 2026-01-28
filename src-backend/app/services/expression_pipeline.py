"""ExpressionPipeline — extracted ML orchestration from ExpressionService.

This module encapsulates the pure ML pipeline (attention extraction,
feature engineering, symbolic regression) without any state management.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import sympy

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from app.exceptions import DomainError, SymbolicRegressionError
from app.ml.attention_extractor import extract_attention_weights, extract_top_pairs
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


class ExpressionPipeline:
    """Orchestrates the ML pipeline: attention → features → symbolic regression.

    Parameters
    ----------
    resolver : CheckpointResolver
        Resolves model checkpoints and loads dataset features.
    """

    def __init__(self, resolver: CheckpointResolver) -> None:
        self._resolver = resolver

    def run(self, model_id: str, top_k: int = 10, preset: str = "standard") -> PipelineResult:
        """Run the full expression discovery pipeline.

        Steps 1–5: resolve checkpoint, load features, extract attention,
        find top pairs, generate interaction features.
        Step 6: run parallel pareto regression (3 models).
        Steps 7–9: extract equations from all models, compute R² for each.
        """
        cp_dir, config = self._resolver.resolve(model_id)

        # Steps 1–5: DomainError passthrough, others → SymbolicRegressionError
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

        # Step 6: Split for honest out-of-sample R² evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            X_aug, y, test_size=0.2, random_state=42,
        )

        # Scale features (fit on train only to prevent data leakage)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train).astype(np.float32)
        X_test = scaler.transform(X_test).astype(np.float32)

        models = run_pareto_regression(X_train, y_train, aug_names, preset=preset)

        try:
            pareto_equations = []
            for i, model in enumerate(models):
                eq = extract_best_equation(model)
                r2 = float(model.score(X_test, y_test))
                pareto_equations.append({
                    "index": i,
                    "sympy_expr": eq["sympy_expr"],
                    "latex": eq["latex"],
                    "complexity": eq["complexity"],
                    "loss": eq["loss"],
                    "r2_score": r2,
                })
            best_eq = pareto_equations[-1]
        except Exception as e:
            raise SymbolicRegressionError(
                f"Equation extraction failed: {e}"
            ) from e

        return PipelineResult(
            sympy_expr=best_eq["sympy_expr"],
            latex=best_eq["latex"],
            complexity=best_eq["complexity"],
            loss=best_eq["loss"],
            r2_score=best_eq["r2_score"],
            pareto_equations=pareto_equations,
        )
