"""FormulationService — orchestrates state lookup, attention extraction, and formulation."""

import torch

from app.exceptions import ExpressionNotFoundError
from app.ml.attention_extractor import extract_top_pairs
from app.ml.checkpoint_loader import load_model
from app.ml.formulator import formulate
from app.models.formulation import FormulationCandidate, FormulationResponse
from app.services.checkpoint_resolver import CheckpointResolver
from app.services.expression_state import ExpressionStateManager


class FormulationService:
    def __init__(self, state_manager: ExpressionStateManager, resolver: CheckpointResolver):
        self._state_manager = state_manager
        self._resolver = resolver

    def formulate(self, expr_id: str, top_k: int = 5, n_samples: int = 1000) -> FormulationResponse:
        state = self._state_manager.get(expr_id)
        cp_dir, config = self._resolver.resolve(state.model_id)
        features, feature_names = self._resolver.get_features(config.dataset_id)

        model = load_model(cp_dir, len(feature_names), config)
        with torch.no_grad():
            X_tensor = torch.tensor(features, dtype=torch.float32)
            _, attn_weights = model(X_tensor)
            avg_attn = attn_weights.mean(dim=0).numpy()

        # avg_attn is (n_features, n_features) matrix; per-feature attention = row mean
        per_feature_attn = avg_attn.mean(axis=1)
        attn_dict = {name: float(per_feature_attn[i]) for i, name in enumerate(feature_names)}
        pairs, _ = extract_top_pairs(avg_attn, feature_names, top_k=len(feature_names))

        result = formulate(
            expr=state.current_sympy,
            feature_names=feature_names,
            attention=attn_dict,
            pairs=pairs,
            top_k=top_k,
            n_samples=n_samples,
        )
        candidates = [FormulationCandidate(**c) for c in result["candidates"]]
        return FormulationResponse(
            expr_id=expr_id,
            candidates=candidates,
            feature_names=result["feature_names"],
            attention_weights=result["attention_weights"],
        )
