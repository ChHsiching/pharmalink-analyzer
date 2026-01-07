import numpy as np


def generate_interaction_features(
    X: np.ndarray,
    feature_names: list[str],
    top_pairs: list[dict],
) -> tuple[np.ndarray, list[str]]:
    """Augment feature matrix with interaction terms from top attention pairs."""
    if not top_pairs:
        return X.copy(), list(feature_names)

    interaction_cols = []
    interaction_names = []
    for pair in top_pairs:
        i = feature_names.index(pair["source"])
        j = feature_names.index(pair["target"])
        interaction_cols.append(X[:, i] * X[:, j])
        interaction_names.append(f"{pair['source']}_mul_{pair['target']}")
        safe_j = np.where(np.abs(X[:, j]) < 1e-8, 1e-8, X[:, j])
        interaction_cols.append(X[:, i] / safe_j)
        interaction_names.append(f"{pair['source']}_div_{pair['target']}")

    X_interactions = np.column_stack(interaction_cols).astype(np.float32)
    return np.hstack([X, X_interactions]), list(feature_names) + interaction_names


def run_symbolic_regression(X, y, feature_names, niterations=40):
    """Run PySR symbolic regression on the given data."""
    from pysr import PySRRegressor

    model = PySRRegressor(
        model_selection="best",
        niterations=niterations,
        binary_operators=["+", "-", "*"],
        unary_operators=["sin", "cos", "exp"],
        maxsize=20,
        populations=15,
        population_size=30,
        temp_equation_file=True,
        progress=False,
        verbosity=0,
    )
    model.fit(X, y, variable_names=feature_names)
    return model


def extract_best_equation(model) -> dict:
    """Extract the best equation from a fitted PySR model."""
    best_idx = model.equations_.query("pick == True").index[0]
    row = model.equations_.iloc[best_idx]
    return {
        "sympy_expr": model.sympy(),
        "latex": model.latex(),
        "complexity": int(row["complexity"]),
        "loss": float(row["loss"]),
    }


def extract_pareto_equations(model) -> list[dict]:
    """Extract all Pareto-front equations from a fitted PySR model."""
    results = []
    for idx in range(len(model.equations_)):
        row = model.equations_.iloc[idx]
        results.append({
            "index": idx,
            "latex": model.latex(index=idx),
            "complexity": int(row["complexity"]),
            "loss": float(row["loss"]),
            "sympy_expr": model.sympy(index=idx),
        })
    return results
