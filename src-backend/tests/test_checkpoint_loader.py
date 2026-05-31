import json

import numpy as np
import torch
import pytest

from app.ml.checkpoint_loader import load_model, load_scaler_params, save_scaler_params
from app.ml.transformer import FeatureTransformer
from app.models.training import TrainingConfig


def _save_checkpoint(tmp_path, n_features: int, config: TrainingConfig):
    """Create a model, save its state_dict, and return both model and path."""
    model = FeatureTransformer(
        n_features=n_features,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        dropout=config.dropout,
    )
    checkpoint_path = tmp_path / "model.pt"
    torch.save(model.state_dict(), checkpoint_path)
    return model, tmp_path


def test_load_model_returns_correct_type(tmp_path):
    config = TrainingConfig(dataset_id="test")
    original, checkpoint_dir = _save_checkpoint(tmp_path, n_features=10, config=config)

    loaded = load_model(
        checkpoint_dir=checkpoint_dir,
        n_features=10,
        config=config,
    )

    assert isinstance(loaded, FeatureTransformer)


def test_load_model_weights_match_original(tmp_path):
    config = TrainingConfig(dataset_id="test")
    original, checkpoint_dir = _save_checkpoint(tmp_path, n_features=10, config=config)

    loaded = load_model(
        checkpoint_dir=checkpoint_dir,
        n_features=10,
        config=config,
    )

    for (name_orig, param_orig), (name_loaded, param_loaded) in zip(
        original.named_parameters(), loaded.named_parameters()
    ):
        assert name_orig == name_loaded
        assert torch.equal(param_orig, param_loaded)


def test_load_model_is_eval_mode(tmp_path):
    config = TrainingConfig(dataset_id="test")
    original, checkpoint_dir = _save_checkpoint(tmp_path, n_features=10, config=config)

    loaded = load_model(
        checkpoint_dir=checkpoint_dir,
        n_features=10,
        config=config,
    )

    assert not loaded.training


class TestSaveScalerParams:
    def test_creates_json_file(self, tmp_path):
        mean = np.array([1.0, 2.0, 3.0])
        scale = np.array([0.5, 1.0, 1.5])
        path = tmp_path / "scaler_params.json"
        save_scaler_params(path, mean, scale)
        assert path.exists()
        data = json.loads(path.read_text())
        assert "mean" in data
        assert "scale" in data

    def test_values_round_trip(self, tmp_path):
        mean = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        scale = np.array([0.5, 1.0, 1.5], dtype=np.float32)
        path = tmp_path / "scaler_params.json"
        save_scaler_params(path, mean, scale)
        loaded_mean, loaded_scale = load_scaler_params(path)
        np.testing.assert_allclose(loaded_mean, mean, rtol=1e-5)
        np.testing.assert_allclose(loaded_scale, scale, rtol=1e-5)


class TestLoadScalerParams:
    def test_returns_numpy_arrays(self, tmp_path):
        data = {"mean": [1.0, 2.0], "scale": [0.5, 1.0]}
        path = tmp_path / "scaler_params.json"
        path.write_text(json.dumps(data))
        mean, scale = load_scaler_params(path)
        assert isinstance(mean, np.ndarray)
        assert isinstance(scale, np.ndarray)
        assert mean.dtype == np.float32
        assert scale.dtype == np.float32

    def test_raises_on_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_scaler_params(tmp_path / "nonexistent.json")


# --- per-fold loader tests ---------------------------------------------------

from sklearn.preprocessing import StandardScaler
from app.ml.checkpoint_loader import load_fold_model, load_fold_scaler


@pytest.fixture
def fold_ckpt(tmp_path):
    """Create a checkpoint dir with per-fold model and scaler files."""
    cp_dir = tmp_path / "fold_ckpt"
    cp_dir.mkdir()
    n_features = 5
    config = TrainingConfig(
        dataset_id="test", d_model=16, n_heads=2, n_layers=1, dropout=0.0, k_folds=3,
    )
    (cp_dir / "config.json").write_text(config.model_dump_json())
    (cp_dir / "metrics.json").write_text('{"final_val_loss": 0.1}')

    for i in range(3):
        model = FeatureTransformer(
            n_features=n_features, d_model=16, n_heads=2, n_layers=1, dropout=0.0,
        )
        torch.save(model.state_dict(), cp_dir / f"model_fold{i}.pt")
        scaler = StandardScaler()
        scaler.fit(np.random.randn(20, n_features).astype(np.float32) * (i + 1))
        save_scaler_params(cp_dir / f"scaler_fold{i}.json", scaler.mean_, scaler.scale_)

    return cp_dir, n_features, config


def test_load_fold_model_returns_feature_transformer(fold_ckpt):
    cp_dir, n_features, config = fold_ckpt
    model = load_fold_model(cp_dir, n_features, config, fold_idx=0)
    assert isinstance(model, FeatureTransformer)


def test_load_fold_model_missing_raises(fold_ckpt):
    cp_dir, n_features, config = fold_ckpt
    with pytest.raises(FileNotFoundError):
        load_fold_model(cp_dir, n_features, config, fold_idx=99)


def test_load_fold_scaler_returns_arrays(fold_ckpt):
    cp_dir, n_features, config = fold_ckpt
    mean, scale = load_fold_scaler(cp_dir, fold_idx=0)
    assert mean.shape == (n_features,)
    assert scale.shape == (n_features,)


def test_load_fold_scaler_different_per_fold(fold_ckpt):
    cp_dir, n_features, config = fold_ckpt
    mean0, _ = load_fold_scaler(cp_dir, fold_idx=0)
    mean1, _ = load_fold_scaler(cp_dir, fold_idx=1)
    assert not np.allclose(mean0, mean1)


def test_load_fold_scaler_missing_raises(fold_ckpt):
    cp_dir, _, _ = fold_ckpt
    with pytest.raises(FileNotFoundError):
        load_fold_scaler(cp_dir, fold_idx=99)
