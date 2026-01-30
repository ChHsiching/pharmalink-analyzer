"""E2E validated tests — Leaf100HDL pipeline vs tcm-homogenizer reference baseline.

Uses real Leaf100HDL data with service-layer calls (no HTTP).
Marked @pytest.mark.slow — expected runtime 3-5 minutes.
Run with:  cd src-backend && uv run pytest tests/test_e2e_leaf100_validated.py -v -m slow
Skip with: cd src-backend && uv run pytest tests/ -m "not slow" -q
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from app.ml.attention_extractor import extract_attention_weights
from app.models.dataset import DatasetMeta, DatasetDetail
from app.services.checkpoint_resolver import CheckpointResolver

LEAF100_CSV = (
    Path(__file__).resolve().parent.parent.parent / "docs" / "csv_data" / "Leaf100HDL.csv"
)

# tcm-homogenizer reference baseline
REFERENCE_TOP_VARS = {"VR", "HYP", "UA", "CA", "MA"}


def _load_leaf100():
    """Load Leaf100HDL.csv into feature matrix X, target y, and feature names."""
    df = pd.read_csv(LEAF100_CSV)
    df.columns = df.columns.str.strip()
    target_col = "HDL"
    feature_names = [c for c in df.columns if c != target_col]
    X = df[feature_names].to_numpy().astype(np.float32)
    y = df[target_col].to_numpy().astype(np.float32)
    return X, y, feature_names


@pytest.fixture
def e2e_env(make_checkpoint):
    """Train a real model with Leaf100HDL data for validated E2E tests."""
    X, y, feature_names = _load_leaf100()
    ckpt = make_checkpoint(
        X=X,
        y=y,
        training_epochs=50,
        k_folds=3,
        checkpoint_name="leaf100hdl_e2e",
        dataset_id="Leaf100HDL",
    )

    # Build a mock DataLoader that returns Leaf100HDL data
    df = pd.read_csv(LEAF100_CSV)
    df.columns = df.columns.str.strip()
    target_col = "HDL"
    all_feature_names = [c for c in df.columns if c != target_col]

    meta = DatasetMeta(
        id="Leaf100HDL",
        name="Leaf100HDL",
        filename="Leaf100HDL.csv",
        plant_part="Leaf",
        target=target_col,
        n_samples=len(df),
        n_features=len(all_feature_names),
        feature_names=all_feature_names,
        columns=list(df.columns),
        is_preset=True,
    )
    detail = DatasetDetail(
        **meta.model_dump(),
        preview=df.head(10).round(4).to_dict(orient="records"),
    )

    mock_loader = MagicMock()
    mock_loader.get_dataset.return_value = detail
    mock_loader.get_dataframe.return_value = df.copy()

    resolver = CheckpointResolver(
        checkpoint_dir=ckpt["cp_dir"].parent,
        data_loader=mock_loader,
    )

    return {
        "ckpt": ckpt,
        "X": X,
        "y": y,
        "feature_names": feature_names,
        "resolver": resolver,
        "checkpoint_id": ckpt["cp_dir"].name,
    }


@pytest.mark.slow
@pytest.mark.timeout(300)
def test_attention_weights_nonuniform(e2e_env):
    """Attention weight variance must exceed 1e-6 — weights are not uniform."""
    ckpt = e2e_env["ckpt"]
    matrix = extract_attention_weights(
        ckpt["cp_dir"],
        e2e_env["X"],
        ckpt["config"],
    )
    flat = np.array(matrix).flatten()
    assert np.var(flat) > 1e-6, (
        f"Attention weights are near-uniform (var={np.var(flat):.2e})"
    )
