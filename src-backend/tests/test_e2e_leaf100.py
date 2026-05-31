"""E2E integration test — full pipeline with real Leaf100HDL data.

Exercises all 9 pipeline stages end-to-end using production services
(no mocks). Marked @pytest.mark.slow — expected runtime 3-5 minutes.
Run with:  pytest tests/test_e2e_leaf100.py -v
Skip with: pytest -m "not slow"
"""

import asyncio
import math
import time
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app

LEAF100_CSV = Path(__file__).resolve().parent.parent.parent / "docs" / "csv_data" / "Leaf100HDL.csv"
TRAINING_TIMEOUT = 180
EXPRESSION_TIMEOUT = 120


async def _poll_training(client, timeout=TRAINING_TIMEOUT):
    """Poll GET /models/train/status until completed. Returns task_id."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        resp = await client.get("/api/v1/models/train/status")
        assert resp.status_code == 200
        data = resp.json()
        if data["status"] == "completed":
            return data["task_id"]
        if data["status"] == "error":
            pytest.fail(f"Training failed: {data.get('error')}")
        await asyncio.sleep(1)
    pytest.fail("Training timed out")


async def _poll_expression(client, task_id, timeout=EXPRESSION_TIMEOUT):
    """Poll GET /expressions/result/{task_id} until completed. Returns result dict."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        resp = await client.get(f"/api/v1/expressions/result/{task_id}")
        assert resp.status_code == 200
        data = resp.json()
        if data["status"] == "completed":
            return data["result"]
        if data["status"] == "failed":
            pytest.fail(f"Expression generation failed: {data.get('error')}")
        await asyncio.sleep(1)
    pytest.fail("Expression generation timed out")


@pytest.mark.slow
@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_full_pipeline_leaf100_hdl():
    """9-stage E2E: upload -> train -> attention -> expression -> simplify
    -> optimize -> undo -> history -> formulation."""
    assert LEAF100_CSV.exists(), f"Test data not found: {LEAF100_CSV}"

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:

        # ---- Stage 1: Upload dataset ----
        csv_bytes = LEAF100_CSV.read_bytes()
        upload_resp = await client.post(
            "/api/v1/datasets",
            files={"file": ("Leaf100HDL.csv", csv_bytes, "text/csv")},
        )
        assert upload_resp.status_code == 201
        dataset = upload_resp.json()
        dataset_id = dataset["id"]
        assert dataset["n_samples"] == 60
        assert dataset["n_features"] == 21
        assert dataset["target"] == "HDL"

        # ---- Stage 2: Train model (async, poll until completed) ----
        train_resp = await client.post(
            "/api/v1/models/train",
            json={
                "dataset_id": dataset_id,
                "d_model": 32,
                "n_heads": 4,
                "n_layers": 1,
                "dropout": 0.0,
                "epochs": 30,
                "k_folds": 3,
                "early_stopping_patience": 10,
            },
        )
        assert train_resp.status_code == 200
        task_id = train_resp.json()["task_id"]

        await _poll_training(client)
        # CheckpointManager saves as {dataset_id}_{task_id}
        checkpoint_id = f"{dataset_id}_{task_id}"

        # Verify loss decreased during training
        status_resp = await client.get("/api/v1/models/train/status")
        progress = status_resp.json()["progress"]
        if len(progress) >= 2:
            first_loss = progress[0]["train_loss"]
            last_loss = progress[-1]["train_loss"]
            assert last_loss < first_loss, (
                f"Loss did not decrease: first={first_loss}, last={last_loss}"
            )

        # ---- Stage 3: Attention matrix ----
        attn_resp = await client.get(
            f"/api/v1/analysis/attention/{checkpoint_id}"
        )
        assert attn_resp.status_code == 200
        attn = attn_resp.json()
        matrix = attn["matrix"]
        assert len(matrix) == 21
        assert all(len(row) == 21 for row in matrix)
        assert all(v >= 0 for row in matrix for v in row)

        # ---- Stage 4: Generate expression (async, poll) ----
        gen_resp = await client.post(
            f"/api/v1/expressions/generate/{checkpoint_id}?preset=quick&top_k=3"
        )
        assert gen_resp.status_code == 200
        gen_task_id = gen_resp.json()["task_id"]

        expr_result = await _poll_expression(client, gen_task_id)
        expr_id = expr_result["expr_id"]
        r2 = expr_result["r2_score"]
        assert math.isfinite(r2), f"R² is not finite: {r2}"
        assert expr_result["latex"], "LaTeX expression is empty"
        original_complexity = expr_result["complexity"]
        original_r2 = r2

        # ---- Stage 5: Simplify expression ----
        simp_resp = await client.post(
            f"/api/v1/expressions/simplify/{expr_id}"
        )
        assert simp_resp.status_code == 200
        simp = simp_resp.json()
        assert abs(simp["r2_score"] - original_r2) <= 0.01, (
            f"R² drifted after simplify: {original_r2} -> {simp['r2_score']}"
        )
        assert simp["complexity"] > 0
        simplified_latex = simp["latex"]

        # ---- Stage 6: Optimize expression (Pareto cycle) ----
        opt_resp = await client.post(
            f"/api/v1/expressions/optimize/{expr_id}"
        )
        assert opt_resp.status_code == 200
        opt = opt_resp.json()
        assert opt["pareto_count"] > 0, "Expected at least one Pareto equation"

        # ---- Stage 7: Undo (restore to simplified state) ----
        undo_resp = await client.post(
            f"/api/v1/expressions/undo/{expr_id}?steps=1"
        )
        assert undo_resp.status_code == 200
        undo = undo_resp.json()
        assert "expr_id" in undo
        assert "latex" in undo
        assert undo["latex"]

        # ---- Stage 8: History ----
        hist_resp = await client.get(
            f"/api/v1/expressions/history/{expr_id}"
        )
        assert hist_resp.status_code == 200
        hist = hist_resp.json()
        assert len(hist["history"]) >= 2, (
            f"Expected >= 2 history entries, got {len(hist['history'])}"
        )
        assert 0 <= hist["current_index"] < len(hist["history"])
        operations = [h["operation"] for h in hist["history"]]
        assert "generate" in operations
        assert "simplify" in operations

        # ---- Stage 9: Optimal formulation ----
        form_resp = await client.post(
            "/api/v1/formulation",
            json={"expr_id": expr_id, "top_k": 20, "n_samples": 5000},
        )
        assert form_resp.status_code == 200
        form = form_resp.json()
        assert form["expr_id"] == expr_id
        assert isinstance(form["candidates"], list)
        assert len(form["feature_names"]) == 21
        assert len(form["attention_weights"]) == 21
        for cand in form["candidates"]:
            assert "components" in cand
            assert "predicted_response" in cand
            assert cand["rank"] >= 1
            assert all(v >= 0 for v in cand["components"].values())
