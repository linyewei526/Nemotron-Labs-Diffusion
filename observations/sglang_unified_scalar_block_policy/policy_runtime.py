#!/usr/bin/env python3
"""Pure-Python runtime for the exported unified scalar block policy."""

from __future__ import annotations

import bisect
import math
from typing import Any, Dict, Mapping, Optional, Tuple


BLOCKS = (8, 16, 32)


def finite_or_none(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def oriented_signal(policy: Mapping[str, Any], features: Mapping[str, Any]) -> Optional[float]:
    """Read the policy's only primary signal and apply its fixed orientation."""
    spec = policy["signal"]
    value = finite_or_none(features.get(spec["feature"]))
    if value is None:
        return None
    return value * float(spec.get("orientation", 1.0))


def _survival_value(model: Mapping[str, Any], signal: Optional[float]) -> float:
    if signal is None:
        survival = model["missing_survival"]
    else:
        index = bisect.bisect_left(model["edges"], signal)
        survival = model["survival"][index]
    return float(sum(float(value) for value in survival))


def choose_action(
    policy: Mapping[str, Any],
    features: Mapping[str, Any],
    lambda_override: Optional[float] = None,
) -> Tuple[int, Dict[str, float]]:
    """Choose one of L8/L16/L32 from one scalar signal and one lambda knob.

    A tier-2 policy may additionally use ``full_streak`` only to choose between
    two globally fitted value tables.  It never has transition-specific
    classifiers or dataset-specific parameters.
    """
    lam = (
        float(policy["default_lambda"])
        if lambda_override is None
        else float(lambda_override)
    )
    if lam < 0:
        raise ValueError("policy lambda must be nonnegative")
    signal = oriented_signal(policy, features)
    stratum = "all"
    gate = policy.get("full_streak_gate")
    if gate:
        full_streak = finite_or_none(features.get("full_streak")) or 0.0
        stratum = "full" if full_streak >= float(gate["threshold"]) else "not_full"
    models = policy["value_models"][stratum]
    scores: Dict[str, float] = {
        "signal": 0.0 if signal is None else signal,
        "signal_missing": 1.0 if signal is None else 0.0,
        "lambda": lam,
        "full_gate": 1.0 if stratum == "full" else 0.0,
    }
    best_block = BLOCKS[0]
    best_utility = -math.inf
    for block in BLOCKS:
        value = _survival_value(models[str(block)], signal)
        utility = value - lam * block
        scores[f"v{block}"] = value
        scores[f"u{block}"] = utility
        # Iterate from small to large and retain the smaller block on ties.
        if utility > best_utility + 1e-12:
            best_block = block
            best_utility = utility
    return best_block, scores
