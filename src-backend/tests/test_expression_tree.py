import time

import sympy
from sympy import symbols, sin, cos

from app.exceptions import SimplifyTimeoutError
from app.ml.expression_tree import (
    sympy_to_tree,
    simplify_expr,
    get_complexity,
    expr_to_latex,
)

x0, x1, x2 = symbols("x0 x1 x2")


def test_variable_to_tree():
    result = sympy_to_tree(x0)
    assert result == {"type": "variable", "value": "x0", "children": []}


def test_constant_to_tree():
    result = sympy_to_tree(sympy.Float(2.5))
    assert result == {"type": "constant", "value": "2.5", "children": []}


def test_integer_to_tree():
    result = sympy_to_tree(sympy.Integer(3))
    assert result["type"] == "constant"
    assert result["value"] == "3"


def test_add_to_tree():
    result = sympy_to_tree(x0 + x1)
    assert result["type"] == "operator"
    assert result["value"] == "+"
    assert len(result["children"]) == 2


def test_mul_to_tree():
    result = sympy_to_tree(x0 * x1)
    assert result["type"] == "operator"
    assert result["value"] == "*"


def test_pow_to_tree():
    result = sympy_to_tree(x0 ** 2)
    assert result["type"] == "operator"
    assert result["value"] == "^"
    assert len(result["children"]) == 2


def test_function_to_tree():
    result = sympy_to_tree(sin(x0))
    assert result["type"] == "function"
    assert result["value"] == "sin"
    assert len(result["children"]) == 1


def test_complex_expression_to_tree():
    expr = 2.5 * cos(x0) + x1 ** 2
    result = sympy_to_tree(expr)
    assert result["type"] == "operator"
    assert result["value"] == "+"


def test_simplify_expr():
    expr = x0 + x0 * 0
    result = simplify_expr(expr)
    assert result == x0


def test_simplify_redundant():
    expr = (x0 ** 2 - x0 ** 2) + x1
    result = simplify_expr(expr)
    assert result == x1


def test_get_complexity():
    expr = x0 + x1 * x2
    c = get_complexity(expr)
    assert c > 0
    assert isinstance(c, int)


def test_complexity_simpler_is_lower():
    assert get_complexity(x0) < get_complexity(x0 + x1 * sin(x2))


def test_expr_to_latex():
    latex = expr_to_latex(x0 ** 2 + x1)
    assert "x_{0}" in latex
    assert "x_{1}" in latex


def test_nested_function():
    expr = sin(cos(x0))
    result = sympy_to_tree(expr)
    assert result["type"] == "function"
    assert result["value"] == "sin"
    assert result["children"][0]["value"] == "cos"


def test_simplify_expr_timeout_returns_original():
    """Complex expression (>100 ops) triggers timeout and returns original within 12s."""
    x = symbols("x")
    expr = 1
    for i in range(20):
        expr = expr * (sin(x + i) + cos(x - i))

    assert get_complexity(expr) > 100, f"Expression too simple: {get_complexity(expr)} ops"

    start = time.monotonic()
    result = simplify_expr(expr)
    elapsed = time.monotonic() - start

    assert result == expr, "Should return original expression on timeout"
    assert elapsed < 12, f"Should have timed out within 12s, took {elapsed:.1f}s"


def test_simplify_expr_normal_no_timeout():
    """Simple expression completes normally without triggering timeout."""
    expr = x0 + x0 * 0
    result = simplify_expr(expr)
    assert result == x0
