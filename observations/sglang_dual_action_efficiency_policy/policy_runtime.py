#!/usr/bin/env python3
"""Pure runtime for the shared-signal S8/S16 efficiency policy."""

from __future__ import annotations

import bisect
import math
from typing import Any, Dict, Mapping, Optional, Tuple


PROFILES = {
    "s8": {"cold_start_block": 8, "allowed_blocks": (8, 16, 32)},
    "s16": {"cold_start_block": 16, "allowed_blocks": (16, 32)},
}


def finite_or_none(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def oriented_signal(
    policy: Mapping[str, Any], features: Mapping[str, Any]
) -> Optional[float]:
    spec = policy["signal"]
    value = finite_or_none(features.get(spec["feature"]))
    return None if value is None else value * float(spec.get("orientation", 1.0))


def survival_value(model: Mapping[str, Any], signal: Optional[float]) -> float:
    if signal is None:
        survival = model["missing_survival"]
    else:
        index = bisect.bisect_left(model["edges"], signal)
        survival = model["survival"][index]
    return float(sum(float(value) for value in survival))


def profile_spec(policy: Mapping[str, Any], profile: str) -> Mapping[str, Any]:
    if profile not in PROFILES:
        raise ValueError(f"unknown efficiency profile: {profile!r}")
    configured = (policy.get("profiles") or {}).get(profile)
    if configured is None:
        raise ValueError(f"policy does not contain profile {profile!r}")
    expected = PROFILES[profile]
    allowed = tuple(int(value) for value in configured["allowed_blocks"])
    if allowed != expected["allowed_blocks"]:
        raise ValueError(
            f"profile {profile} allowed_blocks={allowed}, expected={expected['allowed_blocks']}"
        )
    if int(configured["cold_start_block"]) != expected["cold_start_block"]:
        raise ValueError(f"profile {profile} has an invalid cold-start block")
    return configured


def choose_action(
    policy: Mapping[str, Any],
    features: Mapping[str, Any],
    profile: str,
    lambda_override: Optional[float] = None,
) -> Tuple[int, Dict[str, float]]:
    """Choose a block using one shared signal and a profile-specific lambda."""
    if policy.get("policy_type") != "dual_action_space_scalar_efficiency":
        raise ValueError("unexpected policy_type")
    configured = profile_spec(policy, profile)
    lam = (
        float(configured["default_lambda"])
        if lambda_override is None
        else float(lambda_override)
    )
    if lam < 0:
        raise ValueError("policy lambda must be nonnegative")

    signal = oriented_signal(policy, features)
    stratum = "all"
    gate = policy.get("full_streak_gate")
    if gate:
        streak = finite_or_none(features.get("full_streak")) or 0.0
        stratum = "full" if streak >= float(gate["threshold"]) else "not_full"
    models = policy["value_models"][stratum]
    allowed = tuple(int(value) for value in configured["allowed_blocks"])
    scores: Dict[str, float] = {
        "signal": 0.0 if signal is None else signal,
        "signal_missing": 1.0 if signal is None else 0.0,
        "lambda": lam,
        "full_gate": 1.0 if stratum == "full" else 0.0,
    }
    best_block = allowed[0]
    best_utility = -math.inf
    for block in (8, 16, 32):
        value = survival_value(models[str(block)], signal)
        utility = value - lam * block
        scores[f"v{block}"] = value
        scores[f"u{block}"] = utility
        if block in allowed and utility > best_utility + 1e-12:
            best_block = block
            best_utility = utility
    scores["profile_s16"] = 1.0 if profile == "s16" else 0.0
    return best_block, scores

