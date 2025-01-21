import numpy as np
import pytest
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from app.ml.checkpoint_loader import save_scaler_params
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

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X).astype(np.float32)

        model = FeatureTransformer(
            n_features=actual_n_features,
            d_model=config.d_model, n_heads=config.n_heads,
            n_layers=config.n_layers, dropout=config.dropout,
        )
        X_t = torch.tensor(X_scaled)
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
        save_scaler_params(
            cp_dir / "scaler_params.json", scaler.mean_, scaler.scale_,
        )


        # Per-fold checkpoints for evaluator
        from sklearn.model_selection import KFold as _KFold
        _kfold = _KFold(n_splits=k_folds, shuffle=True, random_state=42)
        for _fi, (_train_idx, _) in enumerate(_kfold.split(X)):
            _fold_scaler = StandardScaler()
            _X_fold = _fold_scaler.fit_transform(X[_train_idx]).astype(np.float32)
            _y_fold = y[_train_idx]

            _fold_model = FeatureTransformer(
                n_features=actual_n_features,
                d_model=config.d_model, n_heads=config.n_heads,
                n_layers=config.n_layers, dropout=config.dropout,
            )
            _X_t = torch.tensor(_X_fold)
            _y_t = torch.tensor(_y_fold)
            _opt = torch.optim.Adam(_fold_model.parameters(), lr=0.01)
            _loss_fn = nn.MSELoss()
            _fold_model.train()
            for _ in range(training_epochs):
                _p, _ = _fold_model(_X_t)
                _l = _loss_fn(_p, _y_t)
                _opt.zero_grad()
                _l.backward()
                _opt.step()

            torch.save(_fold_model.state_dict(), cp_dir / f"model_fold{_fi}.pt")
            save_scaler_params(
                cp_dir / f"scaler_fold{_fi}.json",
                _fold_scaler.mean_, _fold_scaler.scale_,
            )

        return {
            "cp_dir": cp_dir,
            "X": X,
            "y": y,
            "feature_names": feature_names,
            "n_features": actual_n_features,
            "config": config,
        }
    return _make
