import signal

import sympy
from sympy import Basic

from app.exceptions import SimplifyTimeoutError

SIMPLIFY_TIMEOUT_SECONDS = 10


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
    """Simplify a SymPy expression with a 10-second timeout safety net."""
    def _timeout_handler(signum, frame):
        raise SimplifyTimeoutError()

    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(SIMPLIFY_TIMEOUT_SECONDS)
    try:
        return sympy.simplify(expr)
    except SimplifyTimeoutError:
        return expr
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def get_complexity(expr: Basic) -> int:
    """Return the operation count of a SymPy expression as a complexity measure."""
    return sympy.count_ops(expr)


def expr_to_latex(expr: Basic) -> str:
    """Convert a SymPy expression to its LaTeX string representation."""
    return sympy.latex(expr)
