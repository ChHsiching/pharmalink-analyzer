import re

import numpy as np
import sympy

from app.exceptions import SymbolicRegressionError


def generate_interaction_features(
    X: np.ndarray,
    feature_names: list[str],
    top_pairs: list[dict],
) -> tuple[np.ndarray, list[str]]:
    """Augment feature matrix with interaction terms from top attention pairs.

    All pairs: mul only (1 feature per pair).
    """
    if not top_pairs:
        return X.copy(), list(feature_names)

    interaction_cols = []
    interaction_names = []
    for pair in top_pairs:
        i = feature_names.index(pair["source"])
        j = feature_names.index(pair["target"])
        interaction_cols.append(X[:, i] * X[:, j])
        interaction_names.append(f"{pair['source']}_mul_{pair['target']}")

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


PRESET_CONFIG = {
    "quick": {"population_size": 2000, "generations": 60, "parsimony_coefficient": 0.0001, "init_depth": (2, 10), "const_range": (-2.0, 2.0)},
    "standard": {"population_size": 3000, "generations": 80, "parsimony_coefficient": 0.00005, "init_depth": (2, 12), "const_range": (-2.0, 2.0)},
    "thorough": {"population_size": 4000, "generations": 120, "parsimony_coefficient": 0.00003, "init_depth": (2, 14), "const_range": (-2.0, 2.0)},
}


def run_symbolic_regression(X, y, feature_names, preset="standard"):
    """Run gplearn symbolic regression on the given data."""
    try:
        from gplearn.genetic import SymbolicRegressor
    except ImportError as e:
        raise SymbolicRegressionError(
            "gplearn not installed. Install with: pip install gplearn"
        ) from e

    preset_params = PRESET_CONFIG.get(preset, PRESET_CONFIG["standard"])

    model = SymbolicRegressor(
        function_set=("add", "sub", "mul", "div"),
        population_size=preset_params["population_size"],
        generations=preset_params["generations"],
        parsimony_coefficient=preset_params["parsimony_coefficient"],
        tournament_size=20,
        init_depth=preset_params.get("init_depth", (2, 8)),
        const_range=preset_params.get("const_range", (-2.0, 2.0)),
        p_crossover=0.7,
        p_subtree_mutation=0.1,
        p_hoist_mutation=0.05,
        p_point_mutation=0.1,
        verbose=0,
        random_state=42,
    )

    try:
        model.fit(X, y)
    except Exception as e:
        raise SymbolicRegressionError(
            f"Symbolic regression failed: {e}"
        ) from e

    model._feature_names = feature_names
    model._training_score = float(model.score(X, y))
    return model


def extract_best_equation(model) -> dict:
    """Extract the best equation from a fitted gplearn model."""
    program_str = str(model._program)
    expr = parse_gplearn_program(program_str, model._feature_names)
    return {
        "sympy_expr": expr,
        "latex": sympy.latex(expr),
        "complexity": sympy.count_ops(expr),
        "loss": 1.0 - model._training_score,
    }


def extract_pareto_equations(model) -> list[dict]:
    """Extract Pareto-front equations. Returns single-element list (best only).
    Full Pareto front via multi-run deferred to Issue #40."""
    best = extract_best_equation(model)
    return [{"index": 0, **best}]


def run_pareto_regression(X, y, feature_names, preset="standard"):
    """Run 3 gplearn symbolic regressions with different parsimony coefficients.

    Uses ThreadPoolExecutor to run all 3 fits in parallel. Returns list of
    fitted models sorted by equation complexity (ascending).
    """
    try:
        from gplearn.genetic import SymbolicRegressor
    except ImportError as e:
        raise SymbolicRegressionError(
            "gplearn not installed. Install with: pip install gplearn"
        ) from e

    from concurrent.futures import ThreadPoolExecutor

    preset_params = PRESET_CONFIG.get(preset, PRESET_CONFIG["standard"])
    base_parsimony = preset_params["parsimony_coefficient"]
    parsimony_coefficients = [base_parsimony * 0.25, base_parsimony, base_parsimony * 4]

    def _fit_single(parsimony_coefficient):
        model = SymbolicRegressor(
            function_set=("add", "sub", "mul", "div"),
            population_size=preset_params["population_size"],
            generations=preset_params["generations"],
            parsimony_coefficient=parsimony_coefficient,
            tournament_size=20,
            init_depth=preset_params.get("init_depth", (2, 8)),
            const_range=preset_params.get("const_range", (-2.0, 2.0)),
            p_crossover=0.7,
            p_subtree_mutation=0.1,
            p_hoist_mutation=0.05,
            p_point_mutation=0.1,
            verbose=0,
            random_state=42,
        )
        try:
            model.fit(X, y)
        except Exception as e:
            raise SymbolicRegressionError(
                f"Symbolic regression failed: {e}"
            ) from e
        model._feature_names = feature_names
        model._training_score = float(model.score(X, y))
        return model

    try:
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(_fit_single, pc) for pc in parsimony_coefficients
            ]
            models = [f.result() for f in futures]
    except SymbolicRegressionError:
        raise
    except Exception as e:
        raise SymbolicRegressionError(
            f"Pareto regression failed: {e}"
        ) from e

    models.sort(key=lambda m: extract_best_equation(m)["complexity"])
    return models
