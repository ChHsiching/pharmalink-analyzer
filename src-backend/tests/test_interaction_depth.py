"""Verify interaction features differ by pair classification."""
import numpy as np

from app.ml.symbolic_regressor import generate_interaction_features


class TestInteractionDepth:
    def test_synergistic_pair_gets_two_features(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        names = ["A", "B"]
        pairs = [{"source": "A", "target": "B", "weight": 0.8, "classification": "synergistic"}]
        X_aug, aug_names = generate_interaction_features(X, names, pairs)
        assert X_aug.shape[1] == 4  # 2 original + 2 interaction (mul + div)
        assert "A_mul_B" in aug_names
        assert "A_div_B" in aug_names

    def test_antagonistic_pair_gets_one_feature(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        names = ["A", "B"]
        pairs = [{"source": "A", "target": "B", "weight": 0.1, "classification": "antagonistic"}]
        X_aug, aug_names = generate_interaction_features(X, names, pairs)
        assert X_aug.shape[1] == 3  # 2 original + 1 interaction (mul only)
        assert "A_mul_B" in aug_names
        assert "A_div_B" not in aug_names

    def test_mixed_pairs_correct_count(self):
        X = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        names = ["A", "B", "C"]
        pairs = [
            {"source": "A", "target": "B", "weight": 0.9, "classification": "synergistic"},
            {"source": "A", "target": "C", "weight": 0.1, "classification": "antagonistic"},
        ]
        X_aug, aug_names = generate_interaction_features(X, names, pairs)
        # 3 original + 2 (syn mul+div) + 1 (antag mul) = 6
        assert X_aug.shape[1] == 6

    def test_no_pairs_returns_original(self):
        X = np.array([[1.0, 2.0]], dtype=np.float32)
        names = ["A", "B"]
        X_aug, aug_names = generate_interaction_features(X, names, [])
        assert X_aug.shape[1] == 2
        assert aug_names == names
