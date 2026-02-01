"""Reference data loader for tcm-homogenizer mock data.

Loads one of 15 rounds of mock reference data from tests/fixtures/mock/.
Each round contains:
  - impact tree (JSON): variable impact tree structure
  - indicators (JSON): 14 evaluation metrics with Chinese keys
  - weights (TXT): per-feature attention weights (tab-separated)
"""

import json
from pathlib import Path

MOCK_DIR = Path(__file__).resolve().parent / "fixtures" / "mock"


def load_reference(round: int) -> dict:
    """Load reference data for a given round (1-15).

    Returns:
        dict with keys:
            impact_tree: dict -- nested impact tree structure
            indicators: dict -- 14 evaluation metrics (Chinese keys)
            weights: dict[str, float] -- feature name -> weight value
    """
    if not 1 <= round <= 15:
        raise ValueError(f"Round must be 1-15, got {round}")

    with open(MOCK_DIR / "impact" / f"impact-{round}.json") as f:
        impact_tree = json.load(f)

    with open(MOCK_DIR / "indicators" / f"indicators-{round}.json") as f:
        indicators = json.load(f)

    weights = {}
    with open(MOCK_DIR / "weight" / f"weight-{round}.txt") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) == 2:
                weights[parts[0]] = float(parts[1])

    return {
        "impact_tree": impact_tree,
        "indicators": indicators,
        "weights": weights,
    }
