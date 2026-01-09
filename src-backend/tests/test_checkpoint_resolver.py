import json

import numpy as np
import pytest
from fastapi import HTTPException
from pathlib import Path

from app.models.training import TrainingConfig
from app.services.checkpoint_resolver import CheckpointResolver
from app.services.data_loader import DataLoader


@pytest.fixture
def resolver_env(tmp_path):
    """Create a checkpoint dir with config.json + model.pt, and a DataLoader loaded with a CSV file."""
    # Create checkpoint directory
    cp_dir = tmp_path / "checkpoints"
    model_dir = cp_dir / "test-model"
    model_dir.mkdir(parents=True)

    config = TrainingConfig(dataset_id="test-dataset")
    (model_dir / "config.json").write_text(config.model_dump_json())
    (model_dir / "model.pt").write_bytes(b"fake-model")

    # Create DataLoader with test CSV
    csv_path = tmp_path / "test-dataset.csv"
    csv_content = "QA,CGA,CA,TC\n1.0,2.0,3.0,10.0\n4.0,5.0,6.0,20.0\n7.0,8.0,9.0,30.0\n"
    csv_path.write_text(csv_content)

    dl = DataLoader()
    meta, df = dl._load_csv(csv_path)
    dl._datasets[meta.id] = (meta, df)

    resolver = CheckpointResolver(checkpoint_dir=cp_dir, data_loader=dl)
    return resolver, cp_dir, config, dl, meta.id


class TestResolve:
    def test_returns_checkpoint_dir_and_config(self, resolver_env):
        resolver, cp_dir, expected_config, _, _ = resolver_env
        result_dir, result_config = resolver.resolve("test-model")

        assert result_dir == cp_dir / "test-model"
        assert result_config.dataset_id == expected_config.dataset_id
        assert result_dir.is_dir()

    def test_raises_404_for_missing_checkpoint(self, resolver_env):
        resolver, _, _, _, _ = resolver_env
        with pytest.raises(HTTPException) as exc_info:
            resolver.resolve("nonexistent-model")
        assert exc_info.value.status_code == 404
        assert "Checkpoint not found" in exc_info.value.detail

    def test_raises_404_for_missing_config(self, resolver_env):
        resolver, cp_dir, _, _, _ = resolver_env
        # Create a checkpoint dir without config.json
        no_config_dir = cp_dir / "no-config-model"
        no_config_dir.mkdir()
        (no_config_dir / "model.pt").write_bytes(b"fake-model")

        with pytest.raises(HTTPException) as exc_info:
            resolver.resolve("no-config-model")
        assert exc_info.value.status_code == 404
        assert "Config not found" in exc_info.value.detail


class TestGetFeatures:
    def test_returns_correct_shape(self, resolver_env):
        resolver, _, _, dl, dataset_id = resolver_env
        X, feature_names = resolver.get_features(dataset_id)

        assert isinstance(X, np.ndarray)
        assert X.dtype == np.float32
        assert X.shape == (3, 3)  # 3 samples, 3 features (QA, CGA, CA)
        assert feature_names == ["QA", "CGA", "CA"]

    def test_raises_404_for_missing_dataset(self, resolver_env):
        resolver, _, _, _, _ = resolver_env
        with pytest.raises(HTTPException) as exc_info:
            resolver.get_features("nonexistent-dataset")
        assert exc_info.value.status_code == 404
        assert "Dataset not found" in exc_info.value.detail


class TestGetFeaturesWithTarget:
    def test_returns_x_y_and_feature_names(self, resolver_env):
        resolver, _, _, dl, dataset_id = resolver_env
        X, y, feature_names = resolver.get_features_with_target(dataset_id)

        assert isinstance(X, np.ndarray)
        assert X.dtype == np.float32
        assert X.shape == (3, 3)

        assert isinstance(y, np.ndarray)
        assert y.dtype == np.float32
        assert y.shape == (3,)

        assert feature_names == ["QA", "CGA", "CA"]

        # Verify actual values
        np.testing.assert_array_almost_equal(X[0], [1.0, 2.0, 3.0])
        np.testing.assert_array_almost_equal(y, [10.0, 20.0, 30.0])


class TestListCheckpoints:
    def test_returns_checkpoint_info_for_valid_dir(self, resolver_env):
        resolver, _, _, _, _ = resolver_env
        result = resolver.list_checkpoints()

        assert len(result) == 1
        assert result[0].id == "test-model"
        assert result[0].dataset_id == "test-dataset"

    def test_skips_dirs_without_config_json(self, resolver_env):
        resolver, cp_dir, _, _, _ = resolver_env
        no_config_dir = cp_dir / "no-config-model"
        no_config_dir.mkdir()
        (no_config_dir / "model.pt").write_bytes(b"fake-model")

        result = resolver.list_checkpoints()

        assert len(result) == 1
        assert result[0].id == "test-model"

    def test_returns_empty_for_nonexistent_dir(self, tmp_path):
        resolver = CheckpointResolver(
            checkpoint_dir=tmp_path / "no-such-dir",
            data_loader=DataLoader(),
        )

        assert resolver.list_checkpoints() == []

    def test_reads_metrics_json_loss(self, resolver_env):
        resolver, cp_dir, _, _, _ = resolver_env
        (cp_dir / "test-model" / "metrics.json").write_text('{"final_val_loss": 0.042}')

        result = resolver.list_checkpoints()

        assert result[0].final_loss == 0.042

    def test_default_loss_when_no_metrics(self, resolver_env):
        resolver, _, _, _, _ = resolver_env
        # resolver_env creates test-model without metrics.json

        result = resolver.list_checkpoints()

        assert result[0].final_loss == 0.0
