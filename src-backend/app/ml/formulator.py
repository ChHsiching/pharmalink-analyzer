"""Attention-driven layered constraint sampling for optimal formulation (最优配比).

Layer 1: Single-component attention magnitude -> dosage range width.
Layer 2: Pairwise interaction attention -> ratio constraints for synergistic pairs.
Sampling: Latin Hypercube Sampling via scipy.stats.qmc.
Evaluation: sympy.lambdify with numpy backend.
"""

import numpy as np
import sympy
from scipy.stats import qmc


def build_layer1_constraints(
    attention: dict[str, float],
    ranges: dict[str, tuple[float, float]] | None = None,
) -> dict[str, tuple[float, float]]:
    if ranges is None:
        ranges = {name: (0.0, 1.0) for name in attention}
    max_attn = max(attention.values()) if attention else 1.0
    if max_attn == 0:
        max_attn = 1.0
    constraints = {}
    for name, attn_w in attention.items():
        lo, hi = ranges[name]
        span = hi - lo
        width_factor = 0.2 + 0.8 * (attn_w / max_attn)
        new_span = span * width_factor
        center = (lo + hi) / 2
        constraints[name] = (max(lo, center - new_span / 2), min(hi, center + new_span / 2))
    return constraints


def build_layer2_constraints(pairs: list[dict]) -> list[dict]:
    constraints = []
    for pair in pairs:
        if pair["classification"] != "synergistic":
            continue
        ratio = max(0.1, pair["weight"])
        constraints.append({
            "pair": (pair["source"], pair["target"]),
            "ratio_min": ratio * 0.5,
            "ratio_max": ratio * 2.0,
        })
    return constraints


def sample_candidates(
    feature_names: list[str],
    layer1: dict[str, tuple[float, float]],
    layer2: list[dict],
    n_samples: int = 1000,
) -> np.ndarray:
    n_dims = len(feature_names)
    sampler = qmc.LatinHypercube(d=n_dims)
    bounds = np.array([layer1.get(name, (0.0, 1.0)) for name in feature_names])
    lower, upper = bounds[:, 0], bounds[:, 1]
    raw = sampler.random(n=n_samples)
    samples = qmc.scale(raw, lower, upper)
    if layer2:
        name_to_idx = {name: i for i, name in enumerate(feature_names)}
        mask = np.ones(n_samples, dtype=bool)
        for c in layer2:
            src_idx = name_to_idx.get(c["pair"][0])
            tgt_idx = name_to_idx.get(c["pair"][1])
            if src_idx is None or tgt_idx is None:
                continue
            safe_tgt = np.where(np.abs(samples[:, tgt_idx]) < 1e-10, 1e-10, samples[:, tgt_idx])
            ratios = samples[:, src_idx] / safe_tgt
            mask &= (ratios >= c["ratio_min"]) & (ratios <= c["ratio_max"])
        samples = samples[mask]
    return samples


def evaluate_candidates(expr: sympy.Basic, feature_names: list[str], X: np.ndarray) -> np.ndarray:
    all_syms = sorted(expr.free_symbols, key=lambda s: s.name)
    base_set = set(feature_names)

    # Augment samples with interaction term columns for symbols like X_mul_Y
    extra_cols = []
    extra_names = []
    for sym in all_syms:
        if sym.name not in base_set and "_mul_" in sym.name:
            parts = sym.name.split("_mul_")
            if len(parts) == 2 and parts[0] in base_set and parts[1] in base_set:
                i = feature_names.index(parts[0])
                j = feature_names.index(parts[1])
                extra_cols.append(X[:, i] * X[:, j])
                extra_names.append(sym.name)

    if extra_cols:
        X = np.hstack([X, np.column_stack(extra_cols)])
        all_names = feature_names + extra_names
        symbols = [sympy.Symbol(n) for n in all_names]
    else:
        symbols = [sympy.Symbol(name) for name in feature_names]

    func = sympy.lambdify(symbols, expr, modules=["numpy"])
    return func(*X.T)


def formulate(
    expr: sympy.Basic,
    feature_names: list[str],
    attention: dict[str, float],
    pairs: list[dict],
    top_k: int = 5,
    n_samples: int = 1000,
    component_ranges: dict[str, tuple[float, float]] | None = None,
) -> dict:
    layer1 = build_layer1_constraints(attention, component_ranges)
    layer2 = build_layer2_constraints(pairs)
    samples = sample_candidates(feature_names, layer1, layer2, n_samples)
    for _ in range(3):
        if len(samples) >= top_k:
            break
        extra = sample_candidates(feature_names, layer1, layer2, n_samples=n_samples * 2)
        samples = np.vstack([samples, extra])
    if len(samples) == 0:
        return {"candidates": [], "feature_names": feature_names, "attention_weights": attention}
    scores = evaluate_candidates(expr, feature_names, samples)
    finite_mask = np.isfinite(scores)
    if not finite_mask.any():
        return {"candidates": [], "feature_names": feature_names, "attention_weights": attention}
    finite_samples, finite_scores = samples[finite_mask], scores[finite_mask]
    top_indices = np.argsort(finite_scores)[::-1][:top_k]
    candidates = []
    for rank, idx in enumerate(top_indices, 1):
        candidates.append({
            "components": {name: float(finite_samples[idx, i]) for i, name in enumerate(feature_names)},
            "predicted_response": float(finite_scores[idx]),
            "rank": rank,
        })
    return {"candidates": candidates, "feature_names": feature_names, "attention_weights": attention}
