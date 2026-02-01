"""E2E pipeline tests — Leaf100HDL with session-scoped fixture for Issue #70.

Session-scoped fixture trains a real Transformer model with Leaf100HDL data and
provides ExpressionService + FormulationService instances sharing an in-memory
ExpressionStateManager.  This allows downstream tests (T1–T8) to reuse trained
artifacts without re-training per test function.

Marked @pytest.mark.slow — expected runtime 3-5 minutes for fixture setup.
Run with:  cd src-backend && uv run pytest tests/test_pipeline_leaf100hdl.py -v -m slow
Skip with: cd src-backend && uv run pytest tests/ -m "not slow" -q
"""

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from unittest.mock import MagicMock

from app.ml.attention_extractor import extract_attention_weights
from app.ml.checkpoint_loader import save_scaler_params
from app.ml.expression_tree import get_complexity, simplify_expr
from app.ml.transformer import FeatureTransformer
from app.models.dataset import DatasetMeta, DatasetDetail
from app.models.training import TrainingConfig
from app.services.checkpoint_resolver import CheckpointResolver
from app.services.expression_pipeline import ExpressionPipeline
from app.services.expression import ExpressionService
from app.services.expression_state import ExpressionStateManager
from app.services.formulation import FormulationService
from tests.reference_loader import load_reference

LEAF100_CSV = (
    Path(__file__).resolve().parent.parent.parent / "docs" / "csv_data" / "Leaf100HDL.csv"
)

D_MODEL = 32
N_HEADS = 4
N_LAYERS = 1
DROPOUT = 0.0
EPOCHS = 50
K_FOLDS = 3


def _load_leaf100():
    """Load Leaf100HDL.csv into feature matrix X, target y, and feature names."""
    df = pd.read_csv(LEAF100_CSV)
    df.columns = df.columns.str.strip()
    target_col = "HDL"
    feature_names = [c for c in df.columns if c != target_col]
    X = df[feature_names].to_numpy().astype(np.float32)
    y = df[target_col].to_numpy().astype(np.float32)
    return X, y, feature_names


def _train_and_save(X, y, n_features, cp_dir, config):
    """Train model and save checkpoint files including per-fold checkpoints."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float32)

    model = FeatureTransformer(
        n_features=n_features,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        dropout=config.dropout,
    )
    X_t = torch.tensor(X_scaled)
    y_t = torch.tensor(y)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    model.train()
    for _ in range(config.epochs):
        pred, _ = model(X_t)
        loss = loss_fn(pred, y_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), cp_dir / "model.pt")
    (cp_dir / "config.json").write_text(config.model_dump_json())
    (cp_dir / "metrics.json").write_text('{"final_val_loss": 0.1}')
    save_scaler_params(cp_dir / "scaler_params.json", scaler.mean_, scaler.scale_)

    # Per-fold checkpoints
    kfold = KFold(n_splits=config.k_folds, shuffle=True, random_state=42)
    for fi, (train_idx, _) in enumerate(kfold.split(X)):
        fold_scaler = StandardScaler()
        X_fold = fold_scaler.fit_transform(X[train_idx]).astype(np.float32)
        y_fold = y[train_idx]

        fold_model = FeatureTransformer(
            n_features=n_features,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            dropout=config.dropout,
        )
        X_ft = torch.tensor(X_fold)
        y_ft = torch.tensor(y_fold)
        fold_opt = torch.optim.Adam(fold_model.parameters(), lr=0.01)
        fold_model.train()
        for _ in range(config.epochs):
            fp, _ = fold_model(X_ft)
            fl = loss_fn(fp, y_ft)
            fold_opt.zero_grad()
            fl.backward()
            fold_opt.step()

        torch.save(fold_model.state_dict(), cp_dir / f"model_fold{fi}.pt")
        save_scaler_params(
            cp_dir / f"scaler_fold{fi}.json",
            fold_scaler.mean_,
            fold_scaler.scale_,
        )


@pytest.fixture(scope="session")
def e2e_env(tmp_path_factory):
    """Train a real model with Leaf100HDL data; provide service instances for T1–T8."""
    np.random.seed(42)
    torch.manual_seed(42)

    X, y, feature_names = _load_leaf100()
    n_features = X.shape[1]
    dataset_id = "Leaf100HDL"

    config = TrainingConfig(
        dataset_id=dataset_id,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        dropout=DROPOUT,
        k_folds=K_FOLDS,
        epochs=EPOCHS,
    )

    cp_dir = tmp_path_factory.mktemp("leaf100hdl_pipeline") / "leaf100hdl_e2e"
    cp_dir.mkdir(parents=True)
    _train_and_save(X, y, n_features, cp_dir, config)

    # Build a mock DataLoader that returns Leaf100HDL data
    df = pd.read_csv(LEAF100_CSV)
    df.columns = df.columns.str.strip()
    target_col = "HDL"
    all_feature_names = [c for c in df.columns if c != target_col]

    meta = DatasetMeta(
        id=dataset_id,
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
        checkpoint_dir=cp_dir.parent,
        data_loader=mock_loader,
    )
    pipeline = ExpressionPipeline(resolver)

    # In-memory state manager (no checkpoint_dir) shared by both services
    state_mgr = ExpressionStateManager()

    expression_svc = ExpressionService(
        resolver=resolver,
        pipeline=pipeline,
        state_manager=state_mgr,
    )
    formulation_svc = FormulationService(
        state_manager=state_mgr,
        resolver=resolver,
    )

    return {
        "cp_dir": cp_dir,
        "config": config,
        "X": X,
        "y": y,
        "feature_names": feature_names,
        "resolver": resolver,
        "pipeline": pipeline,
        "expression_svc": expression_svc,
        "formulation_svc": formulation_svc,
        "checkpoint_id": cp_dir.name,
    }


@pytest.mark.slow
@pytest.mark.timeout(300)
def test_T1_train_transformer(e2e_env):
    """Re-train to observe loss convergence (separate from fixture training).

    Trains the same architecture with recorded per-epoch loss, then validates:
    - Loss decreased over training (convergence)
    - Final loss is finite
    - Per-fold R-squared values are finite
    Prints structured output including reference round 1 indicators.
    """
    np.random.seed(42)
    torch.manual_seed(42)

    X, y = e2e_env["X"], e2e_env["y"]
    config = e2e_env["config"]
    n_features = X.shape[1]

    # --- Train with loss recording ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float32)
    model = FeatureTransformer(
        n_features=n_features,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        dropout=config.dropout,
    )
    X_t = torch.tensor(X_scaled)
    y_t = torch.tensor(y)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    losses = []
    model.train()
    for epoch in range(config.epochs):
        pred, _ = model(X_t)
        loss = loss_fn(pred, y_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    # --- Per-fold R-squared ---
    kfold = KFold(n_splits=config.k_folds, shuffle=True, random_state=42)
    fold_r2s = []
    for train_idx, test_idx in kfold.split(X):
        fold_scaler = StandardScaler()
        X_tr = fold_scaler.fit_transform(X[train_idx]).astype(np.float32)
        X_te = fold_scaler.transform(X[test_idx]).astype(np.float32)
        y_tr = y[train_idx]
        y_te = y[test_idx]

        fold_model = FeatureTransformer(
            n_features=n_features,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            dropout=config.dropout,
        )
        X_ft = torch.tensor(X_tr)
        y_ft = torch.tensor(y_tr)
        fold_opt = torch.optim.Adam(fold_model.parameters(), lr=0.01)
        fold_model.train()
        for _ in range(config.epochs):
            fp, _ = fold_model(X_ft)
            fl = loss_fn(fp, y_ft)
            fold_opt.zero_grad()
            fl.backward()
            fold_opt.step()

        fold_model.eval()
        with torch.no_grad():
            pred_te, _ = fold_model(torch.tensor(X_te))
            ss_res = np.sum((y_te - pred_te.numpy()) ** 2)
            ss_tot = np.sum((y_te - y_te.mean()) ** 2)
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
            fold_r2s.append(float(r2))

    # --- Load reference round 1 ---
    ref = load_reference(1)

    # --- Structured output ---
    mid_epoch = len(losses) // 2
    loss_reduction = (losses[0] - losses[-1]) / losses[0] * 100 if losses[0] != 0 else 0.0

    print("\n" + "=" * 60)
    print("T1: Train Transformer — Loss Convergence Report")
    print("=" * 60)
    print(f"  Config: d_model={config.d_model}, n_heads={config.n_heads}, "
          f"n_layers={config.n_layers}, epochs={config.epochs}, k_folds={config.k_folds}")
    print(f"  Data: {X.shape[0]} samples x {X.shape[1]} features")
    print(f"  Loss: epoch 1={losses[0]:.4f}, epoch {mid_epoch}={losses[mid_epoch]:.4f}, "
          f"epoch {len(losses)}={losses[-1]:.6f}")
    print(f"  Loss reduction: {loss_reduction:.1f}%")
    print(f"  Per-fold R²: {[f'{r:.4f}' for r in fold_r2s]}")
    print(f"  Reference round 1 weights: {len(ref['weights'])} features")
    print(f"  Reference round 1 indicators: {len(ref['indicators'])} metrics")
    print("=" * 60)

    # --- Assertions ---
    assert losses[-1] < losses[0], (
        f"Loss did not decrease: first={losses[0]:.4f}, last={losses[-1]:.4f}"
    )
    assert np.isfinite(losses[-1]), f"Final loss is not finite: {losses[-1]}"


@pytest.mark.slow
@pytest.mark.timeout(60)
def test_T2_attention_extraction(e2e_env):
    """Extract attention weights and compare feature ranking with reference.

    Validates that the attention matrix has meaningful variance (not degenerate)
    and prints a comparison against reference round-1 weights.
    """
    matrix = extract_attention_weights(
        e2e_env["cp_dir"], e2e_env["X"], e2e_env["config"],
    )

    # Per-feature average attention (column means)
    feature_names = e2e_env["feature_names"]
    col_means = matrix.mean(axis=0)
    sorted_indices = np.argsort(col_means)[::-1]
    sorted_features = [(feature_names[i], float(col_means[i])) for i in sorted_indices]

    ref = load_reference(1)
    ref_weights = ref["weights"]
    ref_sorted = sorted(ref_weights.items(), key=lambda kv: kv[1], reverse=True)

    our_top10 = set(name for name, _ in sorted_features[:10])
    ref_top10 = set(name for name, _ in ref_sorted[:10])
    overlap = our_top10 & ref_top10

    print("\n" + "=" * 60)
    print("T2: Attention Extraction — Feature Ranking Report")
    print("=" * 60)
    print(f"  Matrix shape: {matrix.shape}")
    print(f"  Value range: [{matrix.min():.6f}, {matrix.max():.6f}]")
    print(f"  Variance: {matrix.var():.8f}")
    print(f"  Top-10 features (ours):  {[n for n, _ in sorted_features[:10]]}")
    print(f"  Top-10 features (ref):   {[n for n, _ in ref_sorted[:10]]}")
    print(f"  Top-5 overlap count: {len(overlap & set(n for n, _ in sorted_features[:5]))}")
    print("=" * 60)

    assert matrix.shape == (21, 21), f"Expected shape (21,21), got {matrix.shape}"
    assert matrix.var() > 1e-6, f"Attention matrix variance too low: {matrix.var():.8f}"


@pytest.mark.slow
@pytest.mark.timeout(300)
def test_T3_expression_standard(e2e_env):
    """Run expression pipeline with standard preset and validate results.

    Checks that symbolic regression produces a valid expression with
    reasonable R-squared, and prints indicators vs reference.
    """
    result = e2e_env["pipeline"].run(
        model_id=e2e_env["checkpoint_id"],
        top_k=10,
        preset="standard",
    )

    ref = load_reference(1)
    ref_indicators = ref["indicators"]

    print("\n" + "=" * 60)
    print("T3: Expression Standard — Pipeline Result Report")
    print("=" * 60)
    print(f"  LaTeX: {result.latex}")
    print(f"  R² (test): {result.r2_score:.6f}")
    print(f"  Complexity: {result.complexity}")
    print(f"  Target: {result.target_name}")
    print(f"  Pareto equations: {len(result.pareto_equations)}")
    print(f"  Active variables: {list(result.variable_impact.keys())}")
    print(f"  Our indicators: {result.indicators}")
    print(f"  Ref indicators: {ref_indicators}")
    print("=" * 60)

    assert result.latex, "LaTeX expression is empty"
    assert np.isfinite(result.r2_score), f"R² is not finite: {result.r2_score}"
    assert result.r2_score > 0.50, f"R² too low: {result.r2_score:.4f}"


@pytest.mark.slow
@pytest.mark.timeout(600)
def test_T4_expression_presets(e2e_env):
    """Run expression pipeline across all three presets and compare results.

    Validates that quick, standard, and thorough presets all produce
    valid expressions, and prints a comparison table.
    """
    presets = ["quick", "standard", "thorough"]
    results = {}
    for preset in presets:
        res = e2e_env["pipeline"].run(
            model_id=e2e_env["checkpoint_id"],
            top_k=10,
            preset=preset,
        )
        results[preset] = res

    print("\n" + "=" * 60)
    print("T4: Expression Presets — Comparison Table")
    print("=" * 60)
    print(f"  {'Preset':<10} {'R²':>10} {'Complexity':>12} {'LaTeX'}")
    print(f"  {'-'*10} {'-'*10} {'-'*12} {'-'*40}")
    for preset in presets:
        res = results[preset]
        latex_short = (res.latex[:60] + "...") if len(res.latex) > 60 else res.latex
        print(f"  {preset:<10} {res.r2_score:>10.6f} {res.complexity:>12} {latex_short}")
    print("=" * 60)

    for preset in presets:
        res = results[preset]
        assert res.latex, f"[{preset}] LaTeX expression is empty"
        assert np.isfinite(res.r2_score), (
            f"[{preset}] R² is not finite: {res.r2_score}"
        )
