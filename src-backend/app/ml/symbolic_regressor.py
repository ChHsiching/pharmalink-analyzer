import re

import numpy as np
import sympy

from app.exceptions import SymbolicRegressionError
from app.ml.pysr_adapter import PySRResultAdapter


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


_OPS = {
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "div": lambda a, b: a / b,
}


def parse_gplearn_program(program_str: str, variable_names: list[str]) -> sympy.Basic:
    var_map = {f"X{i}": sympy.Symbol(name) for i, name in enumerate(variable_names)}
    tokens = _tokenize(program_str)
    result, _ = _parse(tokens, 0, var_map)
    return result


def _tokenize(s: str) -> list[str]:
    tokens = []
    i = 0
    while i < len(s):
        if s[i] in "(), ":
            if s[i] in "(),":
                tokens.append(s[i])
            i += 1
        elif s[i].isalpha() or s[i] == "_":
            j = i
            while i < len(s) and (s[i].isalnum() or s[i] == "_"):
                i += 1
            tokens.append(s[j:i])
        else:
            j = i
            if s[i] in "+-":
                i += 1
            while i < len(s) and (s[i].isdigit() or s[i] == "."):
                i += 1
            tokens.append(s[j:i])
    return tokens


def _parse(tokens: list[str], pos: int, var_map: dict) -> tuple:
    tok = tokens[pos]
    if tok in _OPS:
        assert tokens[pos + 1] == "("
        left, pos = _parse(tokens, pos + 2, var_map)
        assert tokens[pos] == ","
        right, pos = _parse(tokens, pos + 1, var_map)
        assert tokens[pos] == ")"
        return _OPS[tok](left, right), pos + 1
    elif tok in var_map:
        return var_map[tok], pos + 1
    else:
        return sympy.Float(tok), pos + 1


def run_symbolic_regression(X, y, feature_names, niterations=20):
    """Run PySR symbolic regression on the given data."""
    try:
        from pysr import PySRRegressor
    except ImportError as e:
        raise SymbolicRegressionError(
            "Julia backend not installed. "
            "Install with: pip install pysr"
        ) from e

    model = PySRRegressor(
        model_selection="best",
        niterations=niterations,
        binary_operators=["+", "-", "*"],
        unary_operators=["sin", "cos", "exp"],
        maxsize=15,
        populations=5,
        population_size=15,
        tournament_selection_n=10,
        temp_equation_file=True,
        progress=False,
        verbosity=0,
    )

    try:
        model.fit(X, y, variable_names=feature_names)
    except RuntimeError as e:
        raise SymbolicRegressionError(
            f"Symbolic regression failed to converge: {e}"
        ) from e
    except Exception as e:
        raise SymbolicRegressionError(
            f"Symbolic regression error: {e}"
        ) from e

    return model


def extract_best_equation(model) -> dict:
    """Extract the best equation from a fitted PySR model."""
    return PySRResultAdapter(model).get_best_equation()


def extract_pareto_equations(model) -> list[dict]:
    """Extract all Pareto-front equations from a fitted PySR model."""
    return PySRResultAdapter(model).get_pareto_equations()
