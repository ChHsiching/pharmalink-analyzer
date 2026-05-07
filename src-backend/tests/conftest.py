import numpy as np
import pytest
import torch
import torch.nn as nn

from app.ml.transformer import FeatureTransformer
from app.models.training import TrainingConfig


@pytest.fixture
def make_checkpoint(tmp_path):
    def _make(
        n_samples=10,
        n_features=5,
        training_epochs=20,
        checkpoint_name="test_ds_task1",
        metrics_content='{"final_val_loss": 0.1}',
        dataset_id="test_ds",
        k_folds=5,
        X=None,
        y=None,
    ):
        np.random.seed(42)
        torch.manual_seed(42)

        if X is None:
            X = np.random.randn(n_samples, n_features).astype(np.float32)
        if y is None:
            y = np.random.randn(X.shape[0]).astype(np.float32)

        actual_n_features = X.shape[1]
        feature_names = [f"comp_{i}" for i in range(actual_n_features)]

        config = TrainingConfig(
            dataset_id=dataset_id,
            d_model=16, n_heads=2, n_layers=1, dropout=0.0,
            k_folds=k_folds, epochs=training_epochs,
        )

        model = FeatureTransformer(
            n_features=actual_n_features,
            d_model=config.d_model, n_heads=config.n_heads,
            n_layers=config.n_layers, dropout=config.dropout,
        )
        X_t = torch.tensor(X)
        y_t = torch.tensor(y)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        loss_fn = nn.MSELoss()
        model.train()
        for _ in range(training_epochs):
            pred, _ = model(X_t)
            loss = loss_fn(pred, y_t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        cp_dir = tmp_path / checkpoint_name
        cp_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), cp_dir / "model.pt")
        (cp_dir / "config.json").write_text(config.model_dump_json())
        (cp_dir / "metrics.json").write_text(metrics_content)

        return {
            "cp_dir": cp_dir,
            "X": X,
            "y": y,
            "feature_names": feature_names,
            "n_features": actual_n_features,
            "config": config,
        }
    return _make
