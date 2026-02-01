"""Tests for the reference data loader utility."""

import json
from pathlib import Path

import pytest

from tests.reference_loader import load_reference

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures" / "mock"


class TestLoadReference:
    """Verify load_reference reads mock data from all 15 rounds."""

    def test_returns_dict_with_three_keys(self):
        ref = load_reference(round=1)
        assert set(ref.keys()) == {"impact_tree", "indicators", "weights"}

    def test_impact_tree_is_dict(self):
        ref = load_reference(round=1)
        assert isinstance(ref["impact_tree"], dict)

    def test_indicators_has_14_keys(self):
        ref = load_reference(round=1)
        assert len(ref["indicators"]) == 14

    def test_indicators_contains_r2_test(self):
        ref = load_reference(round=1)
        assert "相关系数 R²(测试)" in ref["indicators"]

    def test_weights_is_dict_of_floats(self):
        ref = load_reference(round=1)
        assert isinstance(ref["weights"], dict)
        for k, v in ref["weights"].items():
            assert isinstance(k, str)
            assert isinstance(v, float)

    def test_weights_has_21_features(self):
        ref = load_reference(round=1)
        assert len(ref["weights"]) == 21

    def test_all_15_rounds_loadable(self):
        for r in range(1, 16):
            ref = load_reference(round=r)
            assert "impact_tree" in ref
            assert "indicators" in ref
            assert "weights" in ref

    def test_round_15_differs_from_round_1(self):
        ref1 = load_reference(round=1)
        ref15 = load_reference(round=15)
        assert ref1["indicators"] != ref15["indicators"]

    def test_invalid_round_raises(self):
        with pytest.raises(ValueError):
            load_reference(round=0)
        with pytest.raises(ValueError):
            load_reference(round=16)
