"""Diagnostic script for PRESET_CONFIG parameter tuning.

Runs gplearn symbolic regression with each preset on Leaf100HDL data,
prints expression structure, variables, complexity, and R² for comparison
against mock data targets.

Usage:
    PYTHONPATH=src-backend uv run --project src-backend python scripts/diagnose_presets.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from app.ml.symbolic_regressor import (
    PRESET_CONFIG,
    extract_best_equation,
    run_pareto_regression,
)

DATA_PATH = Path(__file__).resolve().parent / "docs" / "csv_data" / "Leaf100HDL.csv"


def load_data():
    df = pd.read_csv(DATA_PATH)
    feature_cols = [c for c in df.columns if c != "HDL"]
    X = df[feature_cols].values.astype(np.float32)
    y = df["HDL"].values.astype(np.float32)
    return X, y, feature_cols


def run_diagnostic(preset_name: str, X, y, feature_cols):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
    )
    params = PRESET_CONFIG[preset_name]
    print(f"\n{'=' * 60}")
    print(f"Preset: {preset_name}")
    print(f"Params: pop={params['population_size']}, gen={params['generations']}, "
          f"parsimony={params['parsimony_coefficient']}, depth={params['init_depth']}")
    print(f"{'=' * 60}")

    try:
        models = run_pareto_regression(X_train, y_train, feature_cols, preset=preset_name)
    except Exception as e:
        print(f"FAILED: {e}")
        return None

    best_r2 = -999
    best_expr = ""
    best_vars = []

    for i, model in enumerate(models):
        eq = extract_best_equation(model)
        r2_train = float(model.score(X_train, y_train))
        r2_test = float(model.score(X_test, y_test))
        expr_str = str(eq["sympy_expr"])
        vars_used = sorted([str(s) for s in eq["sympy_expr"].free_symbols])
        complexity = eq["complexity"]
        parsimony = model.parsimony_coefficient

        print(f"\n  Model {i} (parsimony={parsimony:.6f}):")
        print(f"    Expression: {expr_str[:120]}{'...' if len(expr_str) > 120 else ''}")
        print(f"    Variables ({len(vars_used)}): {vars_used}")
        print(f"    Complexity: {complexity}")
        print(f"    R² train: {r2_train:.4f}")
        print(f"    R² test:  {r2_test:.4f}")

        if r2_test > best_r2:
            best_r2 = r2_test
            best_expr = expr_str
            best_vars = vars_used

    print(f"\n  --- Best (R²={best_r2:.4f}) ---")
    print(f"  Expression: {best_expr[:120]}{'...' if len(best_expr) > 120 else ''}")
    print(f"  Variables ({len(best_vars)}): {best_vars}")
    return {"r2": best_r2, "vars": len(best_vars), "expr": best_expr}


def main():
    if not DATA_PATH.exists():
        print(f"Data file not found: {DATA_PATH}")
        return

    print("Loading Leaf100HDL dataset...")
    X, y, feature_cols = load_data()
    print(f"Samples: {len(X)}, Features: {len(feature_cols)}")
    print(f"Features: {feature_cols}")

    print(f"\nMock targets:")
    print(f"  Round 15 (quick target):    ~5 vars, depth ~9, R² ~0.87")
    print(f"  Round 1  (standard target): ~7 vars, depth ~9, R² ~0.87")
    print(f"  Thorough: better than standard")

    for preset in ["quick", "standard", "thorough"]:
        run_diagnostic(preset, X, y, feature_cols)

    print(f"\n{'=' * 60}")
    print("Done. Adjust PRESET_CONFIG in symbolic_regressor.py and re-run.")


if __name__ == "__main__":
    main()
