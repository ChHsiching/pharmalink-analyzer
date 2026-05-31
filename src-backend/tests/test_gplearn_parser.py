"""Tests for gplearn program string to SymPy expression parser."""
import sympy
from app.ml.symbolic_regressor import parse_gplearn_program


class TestParseVariable:
    def test_single_variable(self):
        expr = parse_gplearn_program("X0", ["A", "B"])
        assert expr == sympy.Symbol("A")

    def test_variable_mapping(self):
        expr = parse_gplearn_program("X2", ["comp_0", "comp_1", "comp_2"])
        assert expr == sympy.Symbol("comp_2")


class TestParseConstant:
    def test_float_constant(self):
        expr = parse_gplearn_program("0.5", [])
        assert expr == sympy.Float(0.5)

    def test_integer_constant(self):
        expr = parse_gplearn_program("3", [])
        assert expr == sympy.Float(3.0)

    def test_negative_constant(self):
        expr = parse_gplearn_program("-0.3", [])
        assert expr == sympy.Float(-0.3)


class TestParseBinaryOps:
    def test_add(self):
        expr = parse_gplearn_program("add(X0, X1)", ["A", "B"])
        assert expr == sympy.Symbol("A") + sympy.Symbol("B")

    def test_sub(self):
        expr = parse_gplearn_program("sub(X0, X1)", ["A", "B"])
        assert expr == sympy.Symbol("A") - sympy.Symbol("B")

    def test_mul(self):
        expr = parse_gplearn_program("mul(X0, X1)", ["A", "B"])
        assert expr == sympy.Symbol("A") * sympy.Symbol("B")

    def test_div(self):
        expr = parse_gplearn_program("div(X0, X1)", ["A", "B"])
        a, b = sympy.symbols("A B")
        assert expr == a / b


class TestParseNested:
    def test_nested_add_mul(self):
        expr = parse_gplearn_program("add(mul(X0, X1), X2)", ["A", "B", "C"])
        a, b, c = sympy.symbols("A B C")
        assert expr == a * b + c

    def test_variable_and_constant(self):
        expr = parse_gplearn_program("add(X0, 0.5)", ["A"])
        assert expr == sympy.Symbol("A") + sympy.Float(0.5)

    def test_complex_nested(self):
        expr = parse_gplearn_program(
            "mul(add(X0, X1), sub(X2, 0.5))",
            ["A", "B", "C"],
        )
        a, b, c = sympy.symbols("A B C")
        assert expr == (a + b) * (c - 0.5)

    def test_deeply_nested(self):
        expr = parse_gplearn_program(
            "div(add(X0, mul(X1, X2)), sub(X3, 1.0))",
            ["A", "B", "C", "D"],
        )
        a, b, c, d = sympy.symbols("A B C D")
        assert expr == (a + b * c) / (d - 1.0)
