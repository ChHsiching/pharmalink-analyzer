"""FormulationService — orchestrates state lookup and formulation using variable impact."""

import numpy as np

from app.exceptions import ExpressionNotFoundError
from app.ml.formulator import formulate
from app.models.formulation import FormulationCandidate, FormulationResponse
from app.services.checkpoint_resolver import CheckpointResolver
from app.services.expression_state import ExpressionStateManager


def _check_attention_weak(attention: dict[str, float], threshold: float = 2.0) -> bool:
    values = list(attention.values())
    if not values:
        return True
    min_w = min(values)
    if min_w <= 0:
        return True
    return (max(values) / min_w) < threshold


def _softmax_rescue(attention: dict[str, float], temperature: float = 5.0) -> dict[str, float]:
    names = list(attention.keys())
    logits = np.array([attention[n] for n in names]) * temperature
    exp_logits = np.exp(logits - logits.max())
    softmax_weights = exp_logits / exp_logits.sum()
    return {name: float(w) for name, w in zip(names, softmax_weights)}


class FormulationService:
    def __init__(self, state_manager: ExpressionStateManager, resolver: CheckpointResolver):
        self._state_manager = state_manager
        self._resolver = resolver

    def formulate(self, expr_id: str, top_k: int = 5, n_samples: int = 1000) -> FormulationResponse:
        state = self._state_manager.get(expr_id)
        cp_dir, config = self._resolver.resolve(state.model_id)
        _, feature_names = self._resolver.get_features(config.dataset_id)

        attn_dict = dict(state.variable_impact)
        if not attn_dict:
            raise ValueError(f"No variable_impact available for expression {expr_id}")

        attention_weak = _check_attention_weak(attn_dict)
        if attention_weak:
            attn_dict = _softmax_rescue(attn_dict, temperature=5.0)

        result = formulate(
            expr=state.current_sympy,
            feature_names=feature_names,
            attention=attn_dict,
            pairs=[],
            top_k=top_k,
            n_samples=n_samples,
        )
        candidates = [FormulationCandidate(**c) for c in result["candidates"]]
        return FormulationResponse(
            expr_id=expr_id,
            candidates=candidates,
            feature_names=result["feature_names"],
            attention_weights=result["attention_weights"],
            attention_weak=attention_weak,
        )
