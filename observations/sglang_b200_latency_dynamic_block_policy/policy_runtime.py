#!/usr/bin/env python3
"""Pure runtime for the shared-signal, concurrency-conditioned B200 policy."""

from __future__ import annotations

import bisect
import math
from typing import Any, Dict, Mapping, Optional, Tuple


BLOCKS = (8, 16, 32)


def _finite(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _expected_accept(model: Mapping[str, Any], signal: Optional[float]) -> float:
    if signal is None:
        survival = model["missing_survival"]
    else:
        index = bisect.bisect_left(model["edges"], signal)
        survival = model["survival"][index]
    return float(sum(float(value) for value in survival))


def choose_action(
    policy: Mapping[str, Any],
    features: Mapping[str, Any],
    concurrency: int,
    lambda_override: Optional[float] = None,
) -> Tuple[int, Dict[str, float]]:
    """Choose argmax E[A_B|signal] - lambda_C * T_C(B)."""
    if policy.get("policy_type") != "b200_latency_scalar_value":
        raise ValueError("unexpected policy_type")
    concurrency_key = str(int(concurrency))
    profile = (policy.get("concurrency_profiles") or {}).get(concurrency_key)
    if profile is None:
        raise ValueError(f"policy has no concurrency C={concurrency}")
    lam = float(profile["lambda"] if lambda_override is None else lambda_override)
    if lam < 0:
        raise ValueError("policy lambda must be nonnegative")
    signal_spec = policy["signal"]
    raw = _finite(features.get(signal_spec["feature"]))
    signal = None if raw is None else raw * float(signal_spec.get("orientation", 1.0))
    stratum = "all"
    gate = policy.get("full_streak_gate")
    if gate:
        streak = _finite(features.get("full_streak")) or 0.0
        stratum = "full" if streak >= float(gate["threshold"]) else "not_full"
    models = policy["value_models"][stratum]
    costs = {int(key): float(value) for key, value in profile["costs_ms"].items()}
    scores: Dict[str, float] = {
        "signal": 0.0 if signal is None else signal,
        "signal_missing": 1.0 if signal is None else 0.0,
        "lambda": lam,
        "virtual_concurrency": float(concurrency),
        "full_gate": 1.0 if stratum == "full" else 0.0,
    }
    best_block = BLOCKS[0]
    best_utility = -math.inf
    for block in BLOCKS:
        value = _expected_accept(models[str(block)], signal)
        utility = value - lam * costs[block]
        scores[f"v{block}"] = value
        scores[f"t{block}_ms"] = costs[block]
        scores[f"u{block}"] = utility
        if utility > best_utility + 1e-12:
            best_block = block
            best_utility = utility
    return best_block, scores

