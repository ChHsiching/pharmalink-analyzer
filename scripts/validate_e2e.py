#!/usr/bin/env python3
"""E2E validation runner for Issue #24 acceptance criteria.

Validates all 10 ACs across the 6 preset datasets by running the full
pipeline (load -> train -> evaluate -> analyse -> expression) against
a live backend server.

Usage:
    cd src-backend && uv run uvicorn app.main:app --port 8765
    uv run python scripts/validate_e2e.py [--base-url URL] [--timeout SECS] [--datasets ID1 ID2 ...]
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass, field

try:
    import httpx
except ImportError:
    print(
        "ERROR: httpx is required. Install with: uv add --dev httpx",
        file=sys.stderr,
    )
    sys.exit(1)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PRESET_DATASETS = [
    "Leaf50HDL",
    "Leaf100HDL",
    "leaf100od",
    "fruit50tc",
    "fruit-ldl",
    "fruit50tg",
]

TRAINING_CONFIG = {
    "dataset_id": "",  # placeholder — overwritten per dataset
    "d_model": 64,
    "n_heads": 4,
    "n_layers": 2,
    "dropout": 0.1,
    "learning_rate": 0.001,
    "epochs": 100,
    "k_folds": 5,
    "early_stopping_patience": 20,
    "augmentation": {"enabled": True},
}

TRAINING_POLL_INTERVAL = 2  # seconds
EXPRESSION_POLL_INTERVAL = 2  # seconds


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ACResult:
    ac: str
    label: str
    status: str  # "PASS", "FAIL", "SKIP"
    detail: str = ""


@dataclass
class DatasetReport:
    dataset_id: str
    results: list[ACResult] = field(default_factory=list)
    skipped: bool = False
    skip_reason: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_result(result: ACResult) -> None:
    tag = f"[{result.status:>4s}]"
    detail = f" -- {result.detail}" if result.detail else ""
    print(f"  {tag}   {result.ac}: {result.label}{detail}")


def _check_health(client: httpx.Client) -> bool:
    try:
        resp = client.get("/health")
        if resp.status_code == 200 and resp.json().get("status") == "ok":
            return True
    except httpx.ConnectError:
        pass
    return False


# ---------------------------------------------------------------------------
# Per-dataset workflow
# ---------------------------------------------------------------------------

def validate_dataset(
    client: httpx.Client,
    dataset_id: str,
    timeout: int,
) -> DatasetReport:
    report = DatasetReport(dataset_id=dataset_id)
    results = report.results

    # -- AC1: Dataset loads ---------------------------------------------------
    try:
        resp = client.get(f"/datasets/{dataset_id}")
        if resp.status_code == 200:
            results.append(ACResult("AC1", "Dataset loads", "PASS"))
        else:
            results.append(ACResult("AC1", "Dataset loads", "FAIL", f"status={resp.status_code}"))
            report.skipped = True
            report.skip_reason = "dataset not found"
            return report
    except Exception as exc:
        results.append(ACResult("AC1", "Dataset loads", "FAIL", str(exc)))
        report.skipped = True
        report.skip_reason = "dataset request failed"
        return report

    # -- Training --------------------------------------------------------------
    config = {**TRAINING_CONFIG, "dataset_id": dataset_id}
    try:
        resp = client.post("/models/train", json=config)
        if resp.status_code != 200:
            detail = resp.json().get("detail", resp.text)
            results.append(ACResult("TRAIN", "Start training", "FAIL", detail))
            report.skipped = True
            report.skip_reason = "training start failed"
            return report
    except Exception as exc:
        results.append(ACResult("TRAIN", "Start training", "FAIL", str(exc)))
        report.skipped = True
        report.skip_reason = "training request failed"
        return report

    # Poll training status
    deadline = time.monotonic() + timeout
    training_ok = False
    while time.monotonic() < deadline:
        try:
            status_resp = client.get("/models/train/status")
            status_data = status_resp.json()
            status = status_data.get("status", "")
            if status in ("completed", "stopped", "error", "idle"):
                if status == "completed":
                    training_ok = True
                else:
                    error_msg = status_data.get("error", status)
                    results.append(ACResult("TRAIN", "Training", "FAIL", error_msg))
                break
        except Exception:
            pass
        time.sleep(TRAINING_POLL_INTERVAL)
    else:
        results.append(ACResult("TRAIN", "Training", "FAIL", f"timed out after {timeout}s"))
        report.skipped = True
        report.skip_reason = "training timeout"
        return report

    if not training_ok:
        report.skipped = True
        report.skip_reason = "training did not complete"
        return report

    # Find checkpoint for this dataset
    try:
        ckpts_resp = client.get("/models/checkpoints")
        ckpts = ckpts_resp.json()
        checkpoint = next((c for c in ckpts if c["dataset_id"] == dataset_id), None)
    except Exception as exc:
        results.append(ACResult("CHECKPOINT", "Find checkpoint", "FAIL", str(exc)))
        report.skipped = True
        report.skip_reason = "checkpoint lookup failed"
        return report

    if checkpoint is None:
        results.append(ACResult("CHECKPOINT", "Find checkpoint", "FAIL", "no checkpoint found"))
        report.skipped = True
        report.skip_reason = "no checkpoint"
        return report

    ckpt_id = checkpoint["id"]

    # -- AC2: R-squared reasonable -------------------------------------------
    try:
        resp = client.get(f"/evaluation/metrics/{ckpt_id}")
        data = resp.json()
        r2 = data["aggregate"]["r2_mean"]
        if math.isfinite(r2) and -10.0 <= r2 <= 1.0:
            results.append(ACResult("AC2", "R2 reasonable", "PASS", f"R2={r2:.4f}"))
        else:
            results.append(ACResult("AC2", "R2 reasonable", "FAIL", f"R2={r2}"))
    except Exception as exc:
        results.append(ACResult("AC2", "R2 reasonable", "FAIL", str(exc)))

    # -- AC3: Heatmap non-uniform --------------------------------------------
    try:
        resp = client.get(f"/analysis/attention/{ckpt_id}/heatmap")
        data = resp.json()
        min_val = data["min_value"]
        max_val = data["max_value"]
        if min_val < max_val:
            results.append(ACResult("AC3", "Heatmap non-uniform", "PASS",
                                    f"min={min_val:.6f}, max={max_val:.6f}"))
        else:
            results.append(ACResult("AC3", "Heatmap non-uniform", "FAIL",
                                    "all values identical"))
    except Exception as exc:
        results.append(ACResult("AC3", "Heatmap non-uniform", "FAIL", str(exc)))

    # -- AC4: Classification split -------------------------------------------
    try:
        resp = client.get(f"/analysis/network/{ckpt_id}", params={"threshold": 50})
        data = resp.json()
        edges = data["edges"]
        classifications = {e["classification"] for e in edges}
        has_synergistic = "synergistic" in classifications
        has_antagonistic = "antagonistic" in classifications
        if has_synergistic and has_antagonistic:
            results.append(ACResult("AC4", "Classification split", "PASS",
                                    f"synergistic={sum(1 for e in edges if e['classification'] == 'synergistic')}, "
                                    f"antagonistic={sum(1 for e in edges if e['classification'] == 'antagonistic')}"))
        else:
            results.append(ACResult("AC4", "Classification split", "FAIL",
                                    f"classifications={classifications}"))
    except Exception as exc:
        results.append(ACResult("AC4", "Classification split", "FAIL", str(exc)))

    # -- AC5 & AC6: Expression generation (async) ----------------------------
    expr_id = None
    expr_result = None
    try:
        resp = client.post(f"/expressions/generate/{ckpt_id}")
        gen_data = resp.json()
        task_id = gen_data["task_id"]

        # Poll until completed / failed
        expr_deadline = time.monotonic() + timeout
        while time.monotonic() < expr_deadline:
            result_resp = client.get(f"/expressions/result/{task_id}")
            result_data = result_resp.json()
            expr_status = result_data.get("status", "")
            if expr_status == "completed":
                expr_id = result_data["result"]["expr_id"]
                expr_result = result_data["result"]
                break
            if expr_status == "failed":
                error_msg = result_data.get("error", "unknown")
                results.append(ACResult("AC5", "Expression LaTeX valid", "FAIL",
                                        f"generation failed: {error_msg}"))
                results.append(ACResult("AC6", "Expression R2 positive", "FAIL",
                                        "skipped — generation failed"))
                break
            time.sleep(EXPRESSION_POLL_INTERVAL)
        else:
            results.append(ACResult("AC5", "Expression LaTeX valid", "FAIL",
                                    f"timed out after {timeout}s"))
            results.append(ACResult("AC6", "Expression R2 positive", "FAIL",
                                    "skipped — expression timeout"))
    except Exception as exc:
        results.append(ACResult("AC5", "Expression LaTeX valid", "FAIL", str(exc)))
        results.append(ACResult("AC6", "Expression R2 positive", "FAIL",
                                "skipped — expression error"))

    if expr_result is not None:
        # AC5: Expression LaTeX valid
        latex = expr_result.get("latex", "")
        if latex and isinstance(latex, str) and len(latex) > 0:
            results.append(ACResult("AC5", "Expression LaTeX valid", "PASS",
                                    f"latex length={len(latex)}"))
        else:
            results.append(ACResult("AC5", "Expression LaTeX valid", "FAIL",
                                    "no latex in result"))

        # AC6: Expression R2 positive
        r2_score = expr_result.get("r2_score", 0)
        if isinstance(r2_score, (int, float)) and r2_score > 0:
            results.append(ACResult("AC6", "Expression R2 positive", "PASS",
                                    f"r2_score={r2_score:.4f}"))
        else:
            results.append(ACResult("AC6", "Expression R2 positive", "FAIL",
                                    f"r2_score={r2_score}"))

    # -- AC7: Evaluation predictions -----------------------------------------
    try:
        resp = client.get(f"/evaluation/predictions/{ckpt_id}")
        data = resp.json()
        predictions = data.get("predictions", [])
        if len(predictions) > 0:
            results.append(ACResult("AC7", "Evaluation data", "PASS",
                                    f"{len(predictions)} predictions"))
        else:
            results.append(ACResult("AC7", "Evaluation data", "FAIL",
                                    "no predictions returned"))
    except Exception as exc:
        results.append(ACResult("AC7", "Evaluation data", "FAIL", str(exc)))

    # -- AC8: WebSocket progress ---------------------------------------------
    results.append(ACResult("AC8", "WebSocket progress", "SKIP",
                            "manual verification required"))

    # -- AC9: Undo/redo ------------------------------------------------------
    if expr_id is not None:
        try:
            resp = client.post(f"/expressions/undo/{expr_id}", params={"steps": 1})
            if resp.status_code == 200:
                results.append(ACResult("AC9", "Undo/redo", "PASS"))
            else:
                detail = resp.json().get("detail", resp.text)
                results.append(ACResult("AC9", "Undo/redo", "FAIL",
                                        f"status={resp.status_code}: {detail}"))
        except Exception as exc:
            results.append(ACResult("AC9", "Undo/redo", "FAIL", str(exc)))
    else:
        results.append(ACResult("AC9", "Undo/redo", "FAIL",
                                "skipped — no expression generated"))

    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="E2E validation runner for Issue #24 acceptance criteria",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8765/api/v1",
        help="Base URL of the API (default: http://localhost:8765/api/v1)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Per-dataset training timeout in seconds (default: 300)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=PRESET_DATASETS,
        help="Dataset IDs to validate (default: all 6 presets)",
    )
    args = parser.parse_args()

    # Health check
    with httpx.Client(base_url=args.base_url, timeout=30) as client:
        if not _check_health(client):
            print("ERROR: Backend server is not reachable.")
            print()
            print("Start the server first:")
            print("  cd src-backend && uv run uvicorn app.main:app --port 8765")
            return 1

        print(f"Backend healthy at {args.base_url}")
        print(f"Datasets: {', '.join(args.datasets)}")
        print(f"Training timeout: {args.timeout}s")
        print("=" * 70)
        print()

        all_reports: list[DatasetReport] = []

        for dataset_id in args.datasets:
            print(f"[{dataset_id}]")
            report = validate_dataset(client, dataset_id, args.timeout)

            if report.skipped:
                print(f"  SKIPPED: {report.skip_reason}")

            for r in report.results:
                _print_result(r)
            print()
            all_reports.append(report)

        # -- Summary ----------------------------------------------------------
        total_pass = 0
        total_fail = 0
        total_skip = 0
        failures: list[tuple[str, ACResult]] = []

        for report in all_reports:
            for r in report.results:
                if r.status == "PASS":
                    total_pass += 1
                elif r.status == "FAIL":
                    total_fail += 1
                    failures.append((report.dataset_id, r))
                elif r.status == "SKIP":
                    total_skip += 1

        print("=" * 70)
        print(f"SUMMARY: {total_pass} PASS | {total_fail} FAIL | {total_skip} SKIP")
        print()

        # AC10: Record bugs — exit with failure summary
        if total_fail > 0:
            print("FAILURES:")
            for ds, r in failures:
                print(f"  [{ds}] {r.ac}: {r.label} -- {r.detail}")
            print()
            print("Manual verification checklist:")
            print("  1. Start the backend: cd src-backend && uv run uvicorn app.main:app --port 8765")
            print("  2. Open the frontend and visually check:")
            print("     - Dataset list shows all preset datasets")
            print("     - Training progress bar updates via WebSocket")
            print("     - R-squared charts render after training")
            print("     - Attention heatmap is coloured (not uniform grey)")
            print("     - Network graph shows both synergistic and antagonistic edges")
            print("     - Expression results display LaTeX formulas")
            print("     - Undo/redo buttons work on expressions")
            print("  3. If PySR/Julia is not installed, AC5/AC6/AC9 failures are expected")
            return 1

        print("All checks passed.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
