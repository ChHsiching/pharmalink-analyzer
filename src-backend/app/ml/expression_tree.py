import signal

import numpy as np
import sympy
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sympy import Basic

from app.exceptions import SimplifyTimeoutError

SIMPLIFY_TIMEOUT_SECONDS = 10


def _tree_simplify(expr: Basic) -> Basic:
    """Conservative tree-level simplification: identity removal + constant/like-term merge."""
    expr = sympy.sympify(expr)
    if not expr.args:
        return expr

    args = [_tree_simplify(a) for a in expr.args]

    if isinstance(expr, sympy.Add):
        constants = sum(a for a in args if a.is_Number)
        variables = [a for a in args if not a.is_Number]
        if not variables:
            return sympy.simplify(sympy.Add(*args))
        coeff_map: dict[Basic, float] = {}
        for a in variables:
            if isinstance(a, sympy.Mul):
                coeff = 1.0
                base_parts = []
                for factor in a.args:
                    if factor.is_Number:
                        coeff *= float(factor)
                    else:
                        base_parts.append(factor)
                base = sympy.Mul(*base_parts) if base_parts else sympy.Integer(1)
                coeff_map[base] = coeff_map.get(base, 0) + coeff
            else:
                coeff_map[a] = coeff_map.get(a, 0) + 1.0
        merged = []
        for b, c in coeff_map.items():
            if c == 0:
                continue
            if c == 1.0:
                merged.append(b)
            elif c == int(c):
                merged.append(sympy.Mul(sympy.Integer(int(c)), b))
            else:
                merged.append(sympy.Mul(sympy.Float(c), b))
        if constants:
            v = float(constants)
            merged.append(sympy.Integer(int(v)) if v == int(v) else sympy.Float(v))
        return sympy.Add(*merged) if len(merged) > 1 else merged[0] if merged else sympy.Integer(0)

    if isinstance(expr, sympy.Mul):
        if any(a == sympy.Integer(0) for a in args):
            return sympy.Integer(0)
        args = [a for a in args if a != sympy.Integer(1)]
        if not args:
            return sympy.Integer(1)
        constants = 1.0
        variables = []
        for a in args:
            if a.is_Number:
                constants *= float(a)
            else:
                variables.append(a)
        if not variables:
            return sympy.simplify(sympy.Mul(*args))
        result_args = [sympy.Integer(int(constants)) if constants == int(constants) else sympy.Float(constants)] if constants != 1.0 else []
        result_args.extend(variables)
        return sympy.Mul(*result_args) if len(result_args) > 1 else result_args[0]

    if isinstance(expr, sympy.Pow):
        if args[1] == sympy.Integer(1):
            return args[0]
        if args[1] == sympy.Integer(0):
            return sympy.Integer(1)

    return expr.func(*args)


def sympy_to_tree(expr: Basic) -> dict:
    """Convert a SymPy expression to a nested dictionary tree representation."""
    if expr.is_Number:
        value = str(float(expr)) if isinstance(expr, sympy.Float) else str(expr)
        return {"type": "constant", "value": value, "children": []}
    if expr.is_Symbol:
        return {"type": "variable", "value": str(expr), "children": []}
    if isinstance(expr, sympy.Add):
        return {"type": "operator", "value": "+", "children": [sympy_to_tree(a) for a in expr.args]}
    if isinstance(expr, sympy.Mul):
        return {"type": "operator", "value": "*", "children": [sympy_to_tree(a) for a in expr.args]}
    if isinstance(expr, sympy.Pow):
        return {"type": "operator", "value": "^", "children": [sympy_to_tree(a) for a in expr.args]}
    if hasattr(expr, "func") and expr.args:
        return {"type": "function", "value": expr.func.__name__, "children": [sympy_to_tree(a) for a in expr.args]}
    return {"type": "constant", "value": str(expr), "children": []}


def simplify_expr(expr: Basic) -> Basic:
    """Two-pass hybrid simplification: tree-level rules + guarded sympy.simplify."""
    original_symbols = expr.free_symbols
    original_complexity = sympy.count_ops(expr)

    # Pass 1: Conservative tree-level simplification
    expr = _tree_simplify(expr)

    # Pass 2: Optional sympy.simplify() with guard
    def _timeout_handler(signum, frame):
        raise SimplifyTimeoutError()

    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(5)
    try:
        candidate = sympy.simplify(expr)
        signal.alarm(0)
        if (len(candidate.free_symbols) >= len(original_symbols)
                and sympy.count_ops(candidate) <= original_complexity):
            expr = candidate
    except SimplifyTimeoutError:
        pass
    except Exception:
        pass
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

    return expr


def compute_indicators_from_expr(
    expr: Basic,
    X_train: list[list[float]],
    X_test: list[list[float]],
    y_train: list[float],
    y_test: list[float],
    aug_names: list[str],
) -> dict[str, float]:
    """Compute regression indicators by evaluating a sympy expression on train/test data."""
    if not X_train or not aug_names:
        return {}

    symbols_list = [sympy.Symbol(n) for n in aug_names]
    func = sympy.lambdify(symbols_list, expr, modules=["numpy"])

    X_train_np = np.array(X_train)
    X_test_np = np.array(X_test)
    y_train_np = np.array(y_train)
    y_test_np = np.array(y_test)

    y_train_pred = np.array([float(func(*row)) for row in X_train_np])
    y_test_pred = np.array([float(func(*row)) for row in X_test_np])

    train_r2 = float(r2_score(y_train_np, y_train_pred))
    test_r2 = float(r2_score(y_test_np, y_test_pred))
    train_mae = float(mean_absolute_error(y_train_np, y_train_pred))
    test_mae = float(mean_absolute_error(y_test_np, y_test_pred))
    train_mse = float(mean_squared_error(y_train_np, y_train_pred))
    test_mse = float(mean_squared_error(y_test_np, y_test_pred))
    train_nmse = train_mse / float(np.var(y_train_np)) if np.var(y_train_np) > 0 else 0.0
    test_nmse = test_mse / float(np.var(y_test_np)) if np.var(y_test_np) > 0 else 0.0
    train_rmse = float(np.sqrt(train_mse))
    test_rmse = float(np.sqrt(test_mse))

    return {
        "train_r2": train_r2, "test_r2": test_r2,
        "train_mae": train_mae, "test_mae": test_mae,
        "train_mse": train_mse, "test_mse": test_mse,
        "train_nmse": train_nmse, "test_nmse": test_nmse,
        "train_rmse": train_rmse, "test_rmse": test_rmse,
        "depth": float(sympy.count_ops(expr)),
        "length": float(len(str(expr))),
    }


def get_complexity(expr: Basic) -> int:
    """Return the operation count of a SymPy expression as a complexity measure."""
    return sympy.count_ops(expr)


def expr_to_latex(expr: Basic) -> str:
    """Convert a SymPy expression to its LaTeX string representation."""
    return sympy.latex(expr)
