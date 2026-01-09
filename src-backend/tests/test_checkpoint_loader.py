import torch

from app.ml.checkpoint_loader import load_model
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
