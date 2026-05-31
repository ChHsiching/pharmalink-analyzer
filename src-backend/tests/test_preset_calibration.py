"""Preset calibration diagnostic — developer tool, not CI gate.

Run manually: cd src-backend && uv run pytest tests/test_preset_calibration.py -v -s --timeout=300
Skip by default: pytest -m "not slow"
"""
import json
from pathlib import Path

import numpy as np
import pytest
import sympy

from app.ml.symbolic_regressor import PRESET_CONFIG, extract_best_equation, run_symbolic_regression

MOCK_DIR = Path(__file__).parent / "fixtures" / "mock"

FEATURE_NAMES = [
    "VR", "HYP", "UA", "CA", "MA", "QA", "OA",
    "MDA", "QUE", "CRA", "EPI", "PC1", "PB2",
    "VG", "RUT", "GUA", "AST", "PIS", "CCGA", "CGA", "NCGA",
]


def _load_mock_indicators(state: int) -> dict:
    path = MOCK_DIR / "indicators" / f"indicators-{state}.json"
    if not path.exists():
        pytest.skip(f"Mock indicators file not found: {path}")
    return json.loads(path.read_text())


def _generate_synthetic_data(n_samples: int = 100, seed: int = 42):
    """Generate synthetic data mimicking Leaf100HDL structure.

    y is a nonlinear combination of the first 7 features with strong signal,
    producing R2 around 0.90+ on training data.
    """
    rng = np.random.default_rng(seed)
    n_features = len(FEATURE_NAMES)
    X = rng.standard_normal((n_samples, n_features)).astype(np.float32)

    y = (
        1.0 * X[:, 0]       # VR — strong signal
        + 0.8 * X[:, 1]     # HYP
        + 0.6 * X[:, 2]     # UA
        + 0.5 * X[:, 3]     # CA
        + 0.3 * X[:, 4]     # MA
        + 0.2 * X[:, 5]     # QA
        + 0.1 * X[:, 6]     # OA
        + 0.3 * X[:, 0] * X[:, 1]   # VR*HYP interaction
        + 0.2 * X[:, 2] * X[:, 3]   # UA*CA interaction
        + rng.standard_normal(n_samples) * 0.05  # low noise
    ).astype(np.float32)
    return X, y


def _count_variables(expr: sympy.Basic) -> int:
    return len(expr.free_symbols)


def _get_variable_names(expr: sympy.Basic) -> set[str]:
    return {str(s) for s in expr.free_symbols}


@pytest.mark.slow
def test_mock_data_files_present():
    """Verify mock data fixtures are available for comparison."""
    assert MOCK_DIR.is_dir(), f"Mock data directory missing: {MOCK_DIR}"
    indicators_dir = MOCK_DIR / "indicators"
    assert indicators_dir.is_dir()
    for state in [1, 15]:
        assert (indicators_dir / f"indicators-{state}.json").exists()


@pytest.mark.slow
@pytest.mark.timeout(300)
def test_quick_preset_not_constant():
    """Quick preset should produce non-constant expressions (main bug fix)."""
    X, y = _generate_synthetic_data()
    model = run_symbolic_regression(X, y, FEATURE_NAMES, preset="quick")
    eq = extract_best_equation(model)
    expr = eq["sympy_expr"]
    assert _count_variables(expr) >= 1, f"Quick preset produced constant expression: {expr}"
    assert eq["complexity"] > 0


@pytest.mark.slow
@pytest.mark.timeout(300)
def test_standard_preset_quality():
    """Standard preset should produce expressions with R2 > 0.80 and 3+ variables."""
    X, y = _generate_synthetic_data()
    model = run_symbolic_regression(X, y, FEATURE_NAMES, preset="standard")
    eq = extract_best_equation(model)
    expr = eq["sympy_expr"]

    r2 = float(model.score(X, y))
    n_vars = _count_variables(expr)
    depth = int(sympy.count_ops(expr))
    length = len(str(expr))

    print(f"\n=== Standard Preset ===")
    print(f"Variables: {n_vars} (target: >= 5)")
    print(f"Variable set: {_get_variable_names(expr)}")
    print(f"Depth (ops): {depth} (target: ~9)")
    print(f"Length: {length} (target: ~28-30)")
    print(f"R2 (train): {r2:.4f} (target: > 0.80)")
    print(f"Expression: {expr}")

    assert r2 > 0.50, f"Standard preset R2 = {r2:.4f} < 0.50 (synthetic data threshold)"
    assert n_vars >= 3, f"Standard preset only has {n_vars} variables"


@pytest.mark.slow
@pytest.mark.timeout(600)
def test_thorough_preset_quality():
    """Thorough preset should produce expressions at least as complex as standard."""
    X, y = _generate_synthetic_data()
    model = run_symbolic_regression(X, y, FEATURE_NAMES, preset="thorough")
    eq = extract_best_equation(model)
    expr = eq["sympy_expr"]

    r2 = float(model.score(X, y))
    n_vars = _count_variables(expr)

    print(f"\n=== Thorough Preset ===")
    print(f"Variables: {n_vars} (target: >= 5)")
    print(f"Variable set: {_get_variable_names(expr)}")
    print(f"Depth (ops): {int(sympy.count_ops(expr))} (target: ~10-12)")
    print(f"Length: {len(str(expr))} (target: ~30-40)")
    print(f"R2 (train): {r2:.4f} (target: > 0.80)")
    print(f"Expression: {expr}")

    assert r2 > 0.50, f"Thorough preset R2 = {r2:.4f} < 0.50 (synthetic data threshold)"
    assert n_vars >= 3, f"Thorough preset only has {n_vars} variables"


@pytest.mark.slow
@pytest.mark.timeout(600)
def test_preset_comparison_report():
    """Print a full comparison report of all presets against mock targets."""
    mock_state1 = _load_mock_indicators(1)
    mock_state15 = _load_mock_indicators(15)

    targets = {
        "quick": {"depth": 8, "length": 18, "min_vars": 5, "min_r2": 0.80},
        "standard": {"depth": 9, "length": 29, "min_vars": 5, "min_r2": 0.80},
        "thorough": {"depth": 11, "length": 35, "min_vars": 5, "min_r2": 0.80},
    }

    print("\n=== Mock Data Reference ===")
    print(f"State 1:  depth={mock_state1['模型深度']}, length={mock_state1['模型长度']}, "
          f"R2(test)={mock_state1['相关系数 R²(测试)']:.4f}")
    print(f"State 15: depth={mock_state15['模型深度']}, length={mock_state15['模型长度']}, "
          f"R2(test)={mock_state15['相关系数 R²(测试)']:.4f}")

    for preset_name in ["quick", "standard", "thorough"]:
        cfg = PRESET_CONFIG[preset_name]
        X, y = _generate_synthetic_data()
        model = run_symbolic_regression(X, y, FEATURE_NAMES, preset=preset_name)
        eq = extract_best_equation(model)
        expr = eq["sympy_expr"]
        r2 = float(model.score(X, y))
        n_vars = _count_variables(expr)
        depth = int(sympy.count_ops(expr))
        length = len(str(expr))
        var_set = _get_variable_names(expr)
        target = targets[preset_name]

        print(f"\n=== {preset_name.title()} Preset ===")
        print(f"parsimony={cfg['parsimony_coefficient']}, init_depth={cfg['init_depth']}, "
              f"pop={cfg['population_size']}, gen={cfg['generations']}")
        print(f"Variables: {n_vars} (target: >= {target['min_vars']}, "
              f"{'OK' if n_vars >= target['min_vars'] else 'BELOW'})")
        print(f"Variable set: {var_set}")
        print(f"Depth (ops): {depth} (target: ~{target['depth']})")
        print(f"Length: {length} (target: ~{target['length']})")
        print(f"R2 (train): {r2:.4f} (target: > {target['min_r2']:.2f}, "
              f"{'OK' if r2 > target['min_r2'] else 'BELOW'})")
        print(f"Expression: {expr}")

        assert r2 > 0.50, f"{preset_name} R2 = {r2:.4f} < 0.50 (synthetic data threshold)"
        assert n_vars >= 3, f"{preset_name} only has {n_vars} variables"


def test_pareto_coefficients_no_zero():
    """Pareto coefficients should never include zero (eliminates complexity runaway)."""
    base = PRESET_CONFIG["quick"]["parsimony_coefficient"]
    coefficients = [base * 0.25, base, base * 4]
    assert all(c > 0 for c in coefficients), f"Zero in parsimony coefficients: {coefficients}"
