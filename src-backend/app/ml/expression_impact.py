"""Variable influence scoring: impact = attention_weight * |sensitivity|."""

import numpy as np
import sympy
from sklearn.preprocessing import StandardScaler

from app.ml.symbolic_regressor import generate_interaction_features


def _find_involved_variables(
    symbols_in_expr: set[sympy.Symbol],
    feature_names: list[str],
    pairs_raw: list[dict],
) -> set[str]:
    """Return feature names that appear in the expression (directly or via interaction terms)."""
    involved: set[str] = set()
    for name in feature_names:
        if sympy.Symbol(name) in symbols_in_expr:
            involved.add(name)
    # Check interaction terms
    for pair in pairs_raw:
        source = pair["source"]
        target = pair["target"]
        mul_name = f"{source}_mul_{target}"
        div_name = f"{source}_div_{target}"
        if sympy.Symbol(mul_name) in symbols_in_expr:
            involved.add(source)
            involved.add(target)
        if sympy.Symbol(div_name) in symbols_in_expr:
            involved.add(source)
            involved.add(target)
    return involved


def _per_variable_attention(
    matrix: np.ndarray,
    feature_names: list[str],
) -> dict[str, float]:
    """Mean attention per variable, excluding self-attention (diagonal)."""
    n = len(feature_names)
    result: dict[str, float] = {}
    for i, name in enumerate(feature_names):
        row = matrix[i]
        mask = np.ones(n, dtype=bool)
        mask[i] = False  # exclude self-attention
        if n > 1:
            result[name] = float(row[mask].mean())
        else:
            result[name] = 0.0
    return result


def compute_impact(
    sympy_expr: sympy.Basic,
    feature_names: list[str],
    X: np.ndarray,
    pairs_raw: list[dict],
    attention_matrix: np.ndarray,
    scaler: StandardScaler,
    aug_names: list[str],
) -> dict[str, float]:
    """Compute per-variable impact = attention_weight * |sensitivity|, normalized by max."""
    symbols_in_expr = sympy_expr.free_symbols
    involved = _find_involved_variables(symbols_in_expr, feature_names, pairs_raw)

    # Baseline: mean of X, augment, scale, evaluate
    baseline_X = X.mean(axis=0, keepdims=True).copy()
    baseline_aug, _ = generate_interaction_features(baseline_X, feature_names, pairs_raw)
    baseline_scaled = scaler.transform(baseline_aug)

    # Build lambdify function
    symbols = [sympy.Symbol(name) for name in aug_names]
    func = sympy.lambdify(symbols, sympy_expr, modules=["numpy"])
    baseline_val = float(func(*baseline_scaled[0].flatten()))

    # Sensitivity per variable
    sensitivity: dict[str, float] = {}
    for i, name in enumerate(feature_names):
        if name not in involved:
            sensitivity[name] = 0.0
            continue
        perturbed_X = baseline_X.copy()
        perturbed_X[0, i] *= 1.01
        perturbed_aug, _ = generate_interaction_features(perturbed_X, feature_names, pairs_raw)
        perturbed_scaled = scaler.transform(perturbed_aug)
        perturbed_val = float(func(*perturbed_scaled[0].flatten()))
        sensitivity[name] = abs(perturbed_val - baseline_val)

    # Attention weights
    attention = _per_variable_attention(attention_matrix, feature_names)

    # Impact = attention * sensitivity
    impact: dict[str, float] = {}
    for name in feature_names:
        impact[name] = attention.get(name, 0.0) * sensitivity.get(name, 0.0)

    # Normalize by max
    max_impact = max(impact.values()) if impact else 0.0
    if max_impact > 0:
        impact = {name: val / max_impact for name, val in impact.items()}

    return impact
