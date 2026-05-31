"""Tests for attention-driven layered constraint sampling formulator."""

import numpy as np
import pytest
import sympy

from app.ml.formulator import (
    build_layer1_constraints, build_layer2_constraints,
    sample_candidates, evaluate_candidates, formulate,
)


def test_layer1_high_attention_wider():
    attention = {"QA": 0.8, "QB": 0.1}
    ranges = {"QA": (0, 10), "QB": (0, 10)}
    c = build_layer1_constraints(attention, ranges)
    assert (c["QA"][1] - c["QA"][0]) > (c["QB"][1] - c["QB"][0])


def test_layer1_default_ranges():
    c = build_layer1_constraints({"X0": 0.5})
    assert c["X0"] == (0.0, 1.0)


def test_layer2_synergistic():
    pairs = [{"source": "A", "target": "B", "weight": 0.9, "classification": "synergistic"}]
    c = build_layer2_constraints(pairs)
    assert len(c) == 1
    assert c[0]["pair"] == ("A", "B")


def test_layer2_skips_antagonistic():
    pairs = [{"source": "A", "target": "B", "weight": 0.2, "classification": "antagonistic"}]
    assert len(build_layer2_constraints(pairs)) == 0


def test_sample_candidates_shape():
    samples = sample_candidates(["X0", "X1"], {"X0": (0.1, 0.9), "X1": (0.2, 0.8)}, [], n_samples=100)
    assert samples.shape == (100, 2)
    assert np.all(samples[:, 0] >= 0.1)


def test_evaluate_candidates():
    x0, x1 = sympy.symbols("X0 X1")
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    scores = evaluate_candidates(x0 + x1, ["X0", "X1"], X)
    assert np.isclose(scores[0], 3.0) and np.isclose(scores[1], 7.0)


def test_formulate_returns_top_k():
    x0, x1 = sympy.symbols("X0 X1")
    result = formulate(x0 + x1, ["X0", "X1"], {"X0": 0.6, "X1": 0.4},
                       [{"source": "X0", "target": "X1", "weight": 0.8, "classification": "synergistic"}],
                       top_k=3, n_samples=500)
    assert len(result["candidates"]) == 3
    assert result["candidates"][0]["predicted_response"] >= result["candidates"][1]["predicted_response"]


def test_formulate_filters_nan():
    x0 = sympy.symbols("X0")
    result = formulate(1 / x0, ["X0"], {"X0": 0.5}, [], top_k=2, n_samples=200)
    for c in result["candidates"]:
        assert np.isfinite(c["predicted_response"])
