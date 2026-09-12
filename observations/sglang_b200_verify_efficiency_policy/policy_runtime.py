#!/usr/bin/env python3
"""Pure-Python runtime for the three frozen verifier-history policy families."""

from __future__ import annotations

import bisect
import math
from typing import Any, Dict, Mapping, Optional, Tuple


BLOCKS = (8, 16, 32)


def finite(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def signal_value(policy: Mapping[str, Any], features: Mapping[str, Any]) -> Optional[float]:
    spec = policy["signal"]
    value = finite(features.get(spec["feature"]))
    return None if value is None else value * float(spec.get("orientation", 1.0))


def censor_stratum(policy: Mapping[str, Any], features: Mapping[str, Any]) -> str:
    if policy.get("secondary_signal") is None:
        return "all"
    previous_block = int(finite(features.get("prev_block")) or 0)
    previous_full = bool((finite(features.get("prev_full")) or 0.0) >= 0.5)
    if previous_block not in BLOCKS:
        return "all"
    return f"b{previous_block}_{'full' if previous_full else 'partial'}"


def _expected_accept(model: Mapping[str, Any], signal: Optional[float]) -> float:
    if signal is None:
        survival = model["missing_survival"]
    else:
        index = bisect.bisect_left(model["edges"], signal)
        survival = model["survival"][index]
    return float(sum(float(item) for item in survival))


def _models_for_stratum(policy: Mapping[str, Any], stratum: str) -> Mapping[str, Any]:
    models = policy["value_models"]
    return models.get(stratum, models["all"])


def _rank_action(
    policy: Mapping[str, Any], profile: Mapping[str, Any], signal: Optional[float], stratum: str
) -> Tuple[int, Dict[str, float]]:
    tables = profile["rank_tables"]
    table = tables.get(stratum, tables["all"])
    if signal is None:
        index = -1
        action = int(table["missing_action"])
    else:
        index = bisect.bisect_left(table["edges"], signal)
        action = int(table["actions"][index])
    scores: Dict[str, float] = {
        "signal": 0.0 if signal is None else signal,
        "signal_missing": float(signal is None),
        "rank_bin": float(index),
    }
    regrets = table.get("conditional_regret")
    if regrets:
        values = regrets[-1] if index < 0 else regrets[index]
        for block, value in zip(BLOCKS, values):
            scores[f"regret{block}"] = float(value)
    return action, scores


def choose_action(
    policy: Mapping[str, Any], features: Mapping[str, Any], concurrency: int
) -> Tuple[int, Dict[str, float]]:
    """Choose a block from verifier history and the full-C B200 latency table."""
    if policy.get("policy_type") != "b200_verify_efficiency":
        raise ValueError("unexpected policy_type")
    profile = (policy.get("concurrency_profiles") or {}).get(str(int(concurrency)))
    if profile is None:
        raise ValueError(f"policy has no profile for C={concurrency}")
    if int(features.get("history_rounds", 0)) == 0:
        block = int(profile["cold_start_block"])
        return block, {"cold_start": 1.0, "virtual_concurrency": float(concurrency)}

    signal = signal_value(policy, features)
    stratum = censor_stratum(policy, features)
    family = str(policy["family"])
    if family == "direct_rank":
        action, scores = _rank_action(policy, profile, signal, stratum)
        scores["virtual_concurrency"] = float(concurrency)
        scores["censor_stratum_id"] = float(
            ("all", "b8_partial", "b8_full", "b16_partial", "b16_full", "b32_partial", "b32_full").index(stratum)
            if stratum in ("all", "b8_partial", "b8_full", "b16_partial", "b16_full", "b32_partial", "b32_full")
            else 0
        )
        return action, scores

    models = _models_for_stratum(policy, stratum)
    costs = {int(key): float(value) for key, value in profile["costs_ms"].items()}
    rho = float(profile.get("rho", 0.0))
    scores = {
        "signal": 0.0 if signal is None else signal,
        "signal_missing": float(signal is None),
        "virtual_concurrency": float(concurrency),
        "rho": rho,
    }
    best_block = BLOCKS[0]
    best_utility = -math.inf
    for block in BLOCKS:
        value = _expected_accept(models[str(block)], signal)
        if family == "local_ratio":
            utility = value / costs[block]
        elif family == "global_fractional":
            utility = value - rho * costs[block]
        else:
            raise ValueError(f"unsupported policy family: {family}")
        scores[f"v{block}"] = value
        scores[f"t{block}_ms"] = costs[block]
        scores[f"u{block}"] = utility
        if utility > best_utility + 1e-12:
            best_block = block
            best_utility = utility
    return best_block, scores
