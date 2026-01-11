import json
import numpy as np
import pytest
import torch

from app.models.training import TrainingConfig
from app.services.evaluation import EvaluationService
from app.services.checkpoint_resolver import CheckpointResolver
from app.services.data_loader import DataLoader


@pytest.fixture
def eval_service_env(tmp_path, make_checkpoint):
    np.random.seed(42)
    torch.manual_seed(42)

    n_samples = 30
    n_features = 5
    rng = np.random.default_rng(42)
    header = ",".join([f"f{i}" for i in range(n_features)] + ["target"])
    rows_data = rng.standard_normal((n_samples, n_features + 1)).astype(np.float32)
    rows_data[:, n_features] = rows_data[:, 0] * 2 + rows_data[:, 1] * 0.5 + rng.standard_normal(n_samples) * 0.1
    rows = "\n".join(",".join(f"{v:.4f}" for v in row) for row in rows_data)
    csv_path = tmp_path / "test_eval.csv"
    csv_path.write_text(header + "\n" + rows)

    dl = DataLoader()
    meta, df = dl._load_csv(csv_path)
    dl._datasets[meta.id] = (meta, df)

    X = df[meta.feature_names].to_numpy().astype(np.float32)
    y = df["target"].to_numpy().astype(np.float32)

    history = [
        {"epoch": 1, "fold": 1, "train_loss": 0.5, "val_loss": 0.6, "r2": 0.5, "status": "training"},
        {"epoch": 2, "fold": 1, "train_loss": 0.3, "val_loss": 0.4, "r2": 0.7, "status": "training"},
        {"epoch": 1, "fold": 2, "train_loss": 0.4, "val_loss": 0.5, "r2": 0.6, "status": "training"},
        {"epoch": 2, "fold": 2, "train_loss": 0.25, "val_loss": 0.35, "r2": 0.75, "status": "training"},
    ]
    metrics_content = json.dumps({"final_val_loss": 0.1, "history": history})

    env = make_checkpoint(
        X=X, y=y,
        n_features=n_features,
        training_epochs=30,
        checkpoint_name=f"checkpoints/{meta.id}_task1",
        dataset_id=meta.id,
        k_folds=3,
        metrics_content=metrics_content,
    )

    svc = EvaluationService(
        resolver=CheckpointResolver(tmp_path / "checkpoints", dl),
    )
    return {"checkpoint_id": env["cp_dir"].name, "service": svc}


def test_get_metrics(eval_service_env):
    svc = eval_service_env["service"]
    result = svc.get_metrics(eval_service_env["checkpoint_id"])
    assert result.model_id == eval_service_env["checkpoint_id"]
    assert len(result.folds) == 3
    for f in result.folds:
        assert f.r2 != 0 or f.mse > 0
    assert result.aggregate.r2_mean != 0
    assert result.aggregate.r2_std >= 0


def test_get_predictions(eval_service_env):
    svc = eval_service_env["service"]
    result = svc.get_predictions(eval_service_env["checkpoint_id"])
    assert result.model_id == eval_service_env["checkpoint_id"]
    assert len(result.predictions) > 0
    for p in result.predictions:
        assert isinstance(p.actual, float)
        assert isinstance(p.predicted, float)


def test_get_residuals(eval_service_env):
    svc = eval_service_env["service"]
    result = svc.get_residuals(eval_service_env["checkpoint_id"])
    assert result.model_id == eval_service_env["checkpoint_id"]
    assert len(result.residuals) > 0
    assert len(result.bins) > 0
    total = sum(b.count for b in result.bins)
    assert total == len(result.residuals)
    assert isinstance(result.mean, float)
    assert result.std >= 0


def test_get_loss_curve(eval_service_env):
    svc = eval_service_env["service"]
    result = svc.get_loss_curve(eval_service_env["checkpoint_id"])
    assert result.model_id == eval_service_env["checkpoint_id"]
    assert len(result.folds) == 2
    assert result.folds[0].fold == 1
    assert len(result.folds[0].points) == 2
    assert result.folds[0].points[0].train_loss == 0.5


def test_get_loss_curve_no_history(tmp_path):
    cp_base = tmp_path / "checkpoints"
    cp_dir = cp_base / "nohist_m1"
    cp_dir.mkdir(parents=True)
    config = TrainingConfig(dataset_id="none")
    (cp_dir / "config.json").write_text(config.model_dump_json())
    (cp_dir / "metrics.json").write_text('{"final_val_loss": 0.1}')
    (cp_dir / "model.pt").write_text("")

    svc = EvaluationService(resolver=CheckpointResolver(cp_base, DataLoader()))
    result = svc.get_loss_curve("nohist_m1")
    assert result.folds == []


def test_checkpoint_not_found(tmp_path):
    from app.exceptions import CheckpointNotFoundError
    resolver = CheckpointResolver(tmp_path, DataLoader())
    svc = EvaluationService(resolver=resolver)
    with pytest.raises(CheckpointNotFoundError) as exc_info:
        svc.get_metrics("nonexistent")
    assert exc_info.value.model_id == "nonexistent"
