"""E2E integration tests — cross-service data consistency validation.

These tests verify that data flows correctly across the analysis, evaluation,
and expression API services. They exercise real service instances (not mock
services) with a shared checkpoint and dataset, using monkeypatching only for
the symbolic-regression backend (PySR/Julia).
"""

import json
import math
import time

import numpy as np
import pytest
import sympy
import torch
from httpx import ASGITransport, AsyncClient
from unittest.mock import MagicMock, patch

from app.main import app
from app.dependencies import (
    get_analysis_service,
    get_evaluation_service,
    get_expression_service,
)
from app.services.analysis import AnalysisService
from app.services.evaluation import EvaluationService
from app.services.expression import ExpressionService
from app.services.checkpoint_resolver import CheckpointResolver
from app.services.data_loader import DataLoader


# ---------------------------------------------------------------------------
# MockSymbolicRegressor helpers
# ---------------------------------------------------------------------------

def _mock_best_expression():
    """Return a simple best-equation dict (comp_0, complexity=1)."""
    x0 = sympy.Symbol("comp_0")
    return {
        "sympy_expr": x0,
        "latex": "x_{0}",
        "complexity": 1,
        "loss": 0.05,
    }


def _mock_pareto_equations():
    """Return a minimal Pareto front with two equations."""
    x0 = sympy.Symbol("comp_0")
    x1 = sympy.Symbol("comp_1")
    return [
        {"index": 0, "latex": "x_{0}", "complexity": 1, "loss": 0.05, "sympy_expr": x0},
        {"index": 1, "latex": "x_{0} + x_{1}", "complexity": 3, "loss": 0.02, "sympy_expr": x0 + x1},
    ]


def _mock_pysr_model():
    """Return a mock PySR model object with .score()."""
    model = MagicMock()
    model.score.return_value = 0.85
    return model


# ---------------------------------------------------------------------------
# e2e_env fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def e2e_env(tmp_path, make_checkpoint):
    np.random.seed(42)
    torch.manual_seed(42)

    n_samples = 30
    n_features = 5
    feature_names = [f"comp_{i}" for i in range(n_features)]

    rng = np.random.default_rng(42)
    header = ",".join(feature_names + ["target"])
    rows_data = rng.standard_normal((n_samples, n_features + 1)).astype(np.float32)
    rows_data[:, n_features] = (
        rows_data[:, 0] * 2
        + rows_data[:, 1] * 0.5
        + rng.standard_normal(n_samples) * 0.1
    )
    rows = "\n".join(",".join(f"{v:.4f}" for v in row) for row in rows_data)
    csv_path = tmp_path / "test_e2e.csv"
    csv_path.write_text(header + "\n" + rows)

    dl = DataLoader()
    meta, df = dl._load_csv(csv_path)
    dl._datasets[meta.id] = (meta, df)

    X = df[meta.feature_names].to_numpy().astype(np.float32)
    y = df["target"].to_numpy().astype(np.float32)

    metrics_content = json.dumps({
        "final_val_loss": 0.1,
        "history": [
            {"epoch": 1, "fold": 1, "train_loss": 0.5, "val_loss": 0.6, "r2": 0.5, "status": "training"},
            {"epoch": 2, "fold": 1, "train_loss": 0.3, "val_loss": 0.4, "r2": 0.7, "status": "training"},
        ],
    })

    env = make_checkpoint(
        X=X,
        y=y,
        n_features=n_features,
        training_epochs=30,
        checkpoint_name=f"checkpoints/{meta.id}_e2e",
        dataset_id=meta.id,
        k_folds=3,
        metrics_content=metrics_content,
    )

    resolver = CheckpointResolver(tmp_path / "checkpoints", dl)

    analysis_svc = AnalysisService(resolver=resolver)
    evaluation_svc = EvaluationService(resolver=resolver)

    # ExpressionService needs heavy monkeypatching — we patch at the
    # app.services.expression module level so generate() uses our stubs.
    expression_svc = ExpressionService(resolver=resolver)

    app.dependency_overrides[get_analysis_service] = lambda: analysis_svc
    app.dependency_overrides[get_evaluation_service] = lambda: evaluation_svc
    app.dependency_overrides[get_expression_service] = lambda: expression_svc

    checkpoint_id = env["cp_dir"].name

    yield {
        "checkpoint_id": checkpoint_id,
        "feature_count": env["n_features"],
        "n_samples": n_samples,
        "expression_svc": expression_svc,
    }

    del app.dependency_overrides[get_analysis_service]
    del app.dependency_overrides[get_evaluation_service]
    del app.dependency_overrides[get_expression_service]


# ---------------------------------------------------------------------------
# Expression helper: run async generate + poll until completed
# ---------------------------------------------------------------------------

async def _generate_expression(client, checkpoint_id, timeout=10.0):
    """POST generate, then poll GET result until completed. Returns (expr_id, result)."""
    resp = await client.post(f"/api/v1/expressions/generate/{checkpoint_id}")
    assert resp.status_code == 200
    task_data = resp.json()
    task_id = task_data["task_id"]

    deadline = time.time() + timeout
    while time.time() < deadline:
        resp = await client.get(f"/api/v1/expressions/result/{task_id}")
        assert resp.status_code == 200
        status_data = resp.json()
        if status_data["status"] == "completed":
            result = status_data["result"]
            return result["expr_id"], result
        if status_data["status"] == "failed":
            pytest.fail(f"Expression generation failed: {status_data.get('error')}")
        time.sleep(0.2)
    pytest.fail("Expression generation timed out")


# ===========================================================================
# Analysis cross-service tests (3)
# ===========================================================================

@pytest.mark.asyncio
async def test_analysis_heatmap_matches_attention_matrix(e2e_env):
    """Heatmap values == attention matrix values, feature names match."""
    info = e2e_env
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        matrix_resp = await client.get(f"/api/v1/analysis/attention/{info['checkpoint_id']}")
        heatmap_resp = await client.get(f"/api/v1/analysis/attention/{info['checkpoint_id']}/heatmap")

    assert matrix_resp.status_code == 200
    assert heatmap_resp.status_code == 200

    matrix_data = matrix_resp.json()
    heatmap_data = heatmap_resp.json()

    # Feature names must match
    assert matrix_data["feature_names"] == heatmap_data["feature_names"]

    # Heatmap values must equal the attention matrix
    assert heatmap_data["values"] == matrix_data["matrix"]


@pytest.mark.asyncio
async def test_network_nodes_match_attention_features(e2e_env):
    """Network node IDs == attention feature_names (as sets)."""
    info = e2e_env
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        attn_resp = await client.get(f"/api/v1/analysis/attention/{info['checkpoint_id']}")
        net_resp = await client.get(f"/api/v1/analysis/network/{info['checkpoint_id']}")

    assert attn_resp.status_code == 200
    assert net_resp.status_code == 200

    attn = attn_resp.json()
    net = net_resp.json()

    node_ids = {n["id"] for n in net["nodes"]}
    feature_names = set(attn["feature_names"])
    assert node_ids == feature_names


@pytest.mark.asyncio
async def test_network_classification_split(e2e_env):
    """At default threshold (50), both synergistic AND antagonistic edges exist."""
    info = e2e_env
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get(f"/api/v1/analysis/network/{info['checkpoint_id']}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["threshold"] == 50.0

    synergistic = [e for e in data["edges"] if e["classification"] == "synergistic"]
    antagonistic = [e for e in data["edges"] if e["classification"] == "antagonistic"]
    assert len(synergistic) > 0, "Expected at least one synergistic edge at threshold 50"
    assert len(antagonistic) > 0, "Expected at least one antagonistic edge at threshold 50"


# ===========================================================================
# Evaluation chain tests (3)
# ===========================================================================

@pytest.mark.asyncio
async def test_evaluation_metrics_are_finite(e2e_env):
    """All aggregate and fold metrics are finite (not inf/nan)."""
    info = e2e_env
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get(f"/api/v1/evaluation/metrics/{info['checkpoint_id']}")
    assert resp.status_code == 200
    data = resp.json()

    # Check aggregate metrics
    agg = data["aggregate"]
    for key in ("r2_mean", "r2_std", "mse_mean", "mse_std", "mae_mean", "mae_std"):
        assert math.isfinite(agg[key]), f"aggregate.{key} is not finite: {agg[key]}"

    # Check each fold
    for fold in data["folds"]:
        for key in ("r2", "mse", "mae"):
            assert math.isfinite(fold[key]), f"fold {fold['fold']}.{key} is not finite: {fold[key]}"


@pytest.mark.asyncio
async def test_evaluation_predictions_exist(e2e_env):
    """Predictions list is non-empty; each has finite actual and predicted."""
    info = e2e_env
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get(f"/api/v1/evaluation/predictions/{info['checkpoint_id']}")
    assert resp.status_code == 200
    data = resp.json()

    predictions = data["predictions"]
    assert len(predictions) > 0, "Expected non-empty predictions list"

    for p in predictions:
        assert "actual" in p
        assert "predicted" in p
        assert math.isfinite(p["actual"]), f"actual not finite: {p['actual']}"
        assert math.isfinite(p["predicted"]), f"predicted not finite: {p['predicted']}"


@pytest.mark.asyncio
async def test_loss_curve_has_history(e2e_env):
    """Folds list is non-empty; each fold has points with train_loss and val_loss."""
    info = e2e_env
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get(f"/api/v1/evaluation/loss-curve/{info['checkpoint_id']}")
    assert resp.status_code == 200
    data = resp.json()

    folds = data["folds"]
    assert len(folds) > 0, "Expected non-empty folds list"
    for fold in folds:
        assert "points" in fold
        assert len(fold["points"]) > 0
        for pt in fold["points"]:
            assert "train_loss" in pt
            assert "val_loss" in pt


# ===========================================================================
# Expression chain tests (4) — uses monkeypatch for symbolic regression
# ===========================================================================

@pytest.mark.asyncio
@patch("app.services.expression.extract_pareto_equations", return_value=_mock_pareto_equations())
@patch("app.services.expression.extract_best_equation", return_value=_mock_best_expression())
@patch("app.services.expression.run_symbolic_regression", return_value=_mock_pysr_model())
@patch("app.services.expression.generate_interaction_features")
@patch("app.services.expression.extract_top_pairs")
@patch("app.services.expression.extract_attention_weights")
async def test_expression_generate_and_simplify_chain(
    mock_attn, mock_pairs, mock_interact, mock_regression,
    mock_best, mock_pareto, e2e_env,
):
    """Generate (async task, poll until completed), then simplify; verify both valid."""
    info = e2e_env

    # Set up mocks for generate pipeline
    n = info["feature_count"]
    rng = np.random.default_rng(42)
    fake_matrix = (rng.standard_normal((n, n))).astype(np.float32)
    mock_attn.return_value = fake_matrix
    mock_pairs.return_value = (
        [{"source": "comp_0", "target": "comp_1", "weight": 0.9, "classification": "synergistic"}],
        0.5,
    )
    mock_interact.return_value = (
        np.zeros((info["n_samples"], n + 2), dtype=np.float32),
        [f"comp_{i}" for i in range(n)] + ["comp_0_mul_comp_1", "comp_0_div_comp_1"],
    )

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        expr_id, gen_result = await _generate_expression(client, info["checkpoint_id"])

        # Verify generate result structure
        assert "expr_id" in gen_result
        assert "latex" in gen_result
        assert "complexity" in gen_result
        assert "r2_score" in gen_result

        # Now simplify
        simplify_resp = await client.post(f"/api/v1/expressions/simplify/{expr_id}")
    assert simplify_resp.status_code == 200
    simp = simplify_resp.json()
    assert "expr_id" in simp
    assert "latex" in simp
    assert "complexity" in simp
    assert "r2_score" in simp


@pytest.mark.asyncio
@patch("app.services.expression.extract_pareto_equations", return_value=_mock_pareto_equations())
@patch("app.services.expression.extract_best_equation", return_value=_mock_best_expression())
@patch("app.services.expression.run_symbolic_regression", return_value=_mock_pysr_model())
@patch("app.services.expression.generate_interaction_features")
@patch("app.services.expression.extract_top_pairs")
@patch("app.services.expression.extract_attention_weights")
async def test_expression_undo_redo_chain(
    mock_attn, mock_pairs, mock_interact, mock_regression,
    mock_best, mock_pareto, e2e_env,
):
    """Generate -> simplify -> undo (original) -> simplify again (redo)."""
    info = e2e_env

    n = info["feature_count"]
    rng = np.random.default_rng(42)
    fake_matrix = (rng.standard_normal((n, n))).astype(np.float32)
    mock_attn.return_value = fake_matrix
    mock_pairs.return_value = (
        [{"source": "comp_0", "target": "comp_1", "weight": 0.9, "classification": "synergistic"}],
        0.5,
    )
    mock_interact.return_value = (
        np.zeros((info["n_samples"], n + 2), dtype=np.float32),
        [f"comp_{i}" for i in range(n)] + ["comp_0_mul_comp_1", "comp_0_div_comp_1"],
    )

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        expr_id, gen_result = await _generate_expression(client, info["checkpoint_id"])
        original_latex = gen_result["latex"]

        # Simplify
        simp_resp = await client.post(f"/api/v1/expressions/simplify/{expr_id}")
        assert simp_resp.status_code == 200
        simplified_latex = simp_resp.json()["latex"]

        # Undo — should return to original
        undo_resp = await client.post(f"/api/v1/expressions/undo/{expr_id}")
        assert undo_resp.status_code == 200
        assert undo_resp.json()["latex"] == original_latex

        # Redo via simplify again — should return to simplified form
        redo_resp = await client.post(f"/api/v1/expressions/simplify/{expr_id}")
        assert redo_resp.status_code == 200
        assert redo_resp.json()["latex"] == simplified_latex


@pytest.mark.asyncio
@patch("app.services.expression.extract_pareto_equations", return_value=_mock_pareto_equations())
@patch("app.services.expression.extract_best_equation", return_value=_mock_best_expression())
@patch("app.services.expression.run_symbolic_regression", return_value=_mock_pysr_model())
@patch("app.services.expression.generate_interaction_features")
@patch("app.services.expression.extract_top_pairs")
@patch("app.services.expression.extract_attention_weights")
async def test_expression_tree_and_history(
    mock_attn, mock_pairs, mock_interact, mock_regression,
    mock_best, mock_pareto, e2e_env,
):
    """After generation, tree has type+value, history has list with current_index."""
    info = e2e_env

    n = info["feature_count"]
    rng = np.random.default_rng(42)
    fake_matrix = (rng.standard_normal((n, n))).astype(np.float32)
    mock_attn.return_value = fake_matrix
    mock_pairs.return_value = (
        [{"source": "comp_0", "target": "comp_1", "weight": 0.9, "classification": "synergistic"}],
        0.5,
    )
    mock_interact.return_value = (
        np.zeros((info["n_samples"], n + 2), dtype=np.float32),
        [f"comp_{i}" for i in range(n)] + ["comp_0_mul_comp_1", "comp_0_div_comp_1"],
    )

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        expr_id, gen_result = await _generate_expression(client, info["checkpoint_id"])

        # Check tree
        tree_resp = await client.get(f"/api/v1/expressions/tree/{expr_id}")
        assert tree_resp.status_code == 200
        tree_data = tree_resp.json()
        tree = tree_data["tree"]
        assert "type" in tree
        assert "value" in tree

        # Check history
        hist_resp = await client.get(f"/api/v1/expressions/history/{expr_id}")
        assert hist_resp.status_code == 200
        hist_data = hist_resp.json()
        assert "history" in hist_data
        assert isinstance(hist_data["history"], list)
        assert len(hist_data["history"]) > 0
        assert "current_index" in hist_data


@pytest.mark.asyncio
@patch("app.services.expression.extract_pareto_equations", return_value=_mock_pareto_equations())
@patch("app.services.expression.extract_best_equation", return_value=_mock_best_expression())
@patch("app.services.expression.run_symbolic_regression", return_value=_mock_pysr_model())
@patch("app.services.expression.generate_interaction_features")
@patch("app.services.expression.extract_top_pairs")
@patch("app.services.expression.extract_attention_weights")
async def test_expression_history_tracks_multiple_operations(
    mock_attn, mock_pairs, mock_interact, mock_regression,
    mock_best, mock_pareto, e2e_env,
):
    """History accumulates entries across generate, simplify, undo; index stays correct."""
    info = e2e_env

    n = info["feature_count"]
    rng = np.random.default_rng(42)
    fake_matrix = (rng.standard_normal((n, n))).astype(np.float32)
    mock_attn.return_value = fake_matrix
    mock_pairs.return_value = (
        [{"source": "comp_0", "target": "comp_1", "weight": 0.9, "classification": "synergistic"}],
        0.5,
    )
    mock_interact.return_value = (
        np.zeros((info["n_samples"], n + 2), dtype=np.float32),
        [f"comp_{i}" for i in range(n)] + ["comp_0_mul_comp_1", "comp_0_div_comp_1"],
    )

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        expr_id, _ = await _generate_expression(client, info["checkpoint_id"])

        # After generate: history has 1 entry at index 0
        hist_resp = await client.get(f"/api/v1/expressions/history/{expr_id}")
        assert hist_resp.status_code == 200
        hist = hist_resp.json()
        assert len(hist["history"]) == 1
        assert hist["current_index"] == 0
        assert hist["history"][0]["operation"] == "generate"

        # Simplify: history grows to 2
        await client.post(f"/api/v1/expressions/simplify/{expr_id}")
        hist_resp = await client.get(f"/api/v1/expressions/history/{expr_id}")
        hist = hist_resp.json()
        assert len(hist["history"]) == 2
        assert hist["current_index"] == 1
        assert hist["history"][1]["operation"] == "simplify"

        # Undo: index moves back to 0, history stays at 2
        await client.post(f"/api/v1/expressions/undo/{expr_id}")
        hist_resp = await client.get(f"/api/v1/expressions/history/{expr_id}")
        hist = hist_resp.json()
        assert len(hist["history"]) == 2
        assert hist["current_index"] == 0
