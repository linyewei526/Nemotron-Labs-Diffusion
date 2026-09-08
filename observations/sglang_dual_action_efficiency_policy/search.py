#!/usr/bin/env python3
"""Eight-dataset-equal search for shared-signal S8/S16 efficiency profiles."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


HERE = Path(__file__).resolve().parent
OBSERVATIONS = HERE.parent
if str(OBSERVATIONS) not in sys.path:
    sys.path.insert(0, str(OBSERVATIONS))

from sglang_unified_scalar_block_policy.search import (  # noqa: E402
    BLOCKS,
    FEATURES,
    SIGNALS,
    SIGNAL_BY_NAME,
    ProgressBar,
    TraceData,
    candidate_oof,
    collect_official_metrics,
    fit_tables,
    indices_weights,
    read_trace as read_base_trace,
    selected_accept,
    summarize_transitions,
    weighted_quantile,
)


EXPECTED_ORDER = (
    "gsm8k",
    "human-eval",
    "mbpp",
    "math-500",
    "aime25",
    "gpqa",
    "ifeval",
    "livecodebench-cpp",
)
EXPECTED_DATASETS = set(EXPECTED_ORDER)
PROFILE_CONFIG = {
    "s8": {"cold_start_block": 8, "allowed_blocks": (8, 16, 32), "baseline": 8},
    "s16": {"cold_start_block": 16, "allowed_blocks": (16, 32), "baseline": 16},
}


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def subset_eight_datasets(
    data: TraceData,
    allow_partial: bool,
    max_invalid_rate: float,
) -> TraceData:
    """Drop MMLU deliberately and reject every other out-of-protocol dataset."""
    source_names = set(data.dataset_names)
    if "aime24" in source_names:
        raise RuntimeError("AIME24 must not participate in this experiment")
    unexpected = source_names - EXPECTED_DATASETS - {"mmlu"}
    if unexpected:
        raise RuntimeError(f"unexpected datasets in trace root: {sorted(unexpected)}")
    present = [name for name in EXPECTED_ORDER if name in source_names]
    if not allow_partial and set(present) != EXPECTED_DATASETS:
        raise RuntimeError(
            "formal search/validation requires exactly eight datasets; "
            f"missing={sorted(EXPECTED_DATASETS-set(present))}"
        )
    old_to_new = {
        data.dataset_names.index(name): new for new, name in enumerate(present)
    }
    mask = np.isin(data.dataset, np.asarray(list(old_to_new), dtype=data.dataset.dtype))
    indices = np.flatnonzero(mask)
    if not len(indices):
        raise RuntimeError("no usable non-AIME24/non-MMLU rows")
    remapped = np.asarray(
        [old_to_new[int(value)] for value in data.dataset[indices]], dtype=np.uint8
    )
    by_dataset = {
        name: dict((data.quality.get("by_dataset") or {}).get(name, {}))
        for name in present
    }
    quality = {
        "invalid_row_policy": data.quality.get("invalid_row_policy"),
        "max_invalid_row_rate": max_invalid_rate,
        "rows_original": sum(int(row.get("rows_original", 0)) for row in by_dataset.values()),
        "rows_usable": sum(int(row.get("rows_usable", 0)) for row in by_dataset.values()),
        "rows_excluded": sum(int(row.get("rows_excluded", 0)) for row in by_dataset.values()),
        "replay_mismatch": sum(int(row.get("replay_mismatch", 0)) for row in by_dataset.values()),
        "cross_block_mismatch": sum(int(row.get("cross_block_mismatch", 0)) for row in by_dataset.values()),
        "by_dataset": by_dataset,
        "explicitly_ignored_datasets": sorted(source_names & {"mmlu"}),
    }
    quality["excluded_rate"] = quality["rows_excluded"] / max(1, quality["rows_original"])
    if quality["excluded_rate"] > max_invalid_rate:
        raise RuntimeError(
            f"eight-dataset trace exclusion {quality['excluded_rate']:.4%} "
            f"exceeds {max_invalid_rate:.4%}"
        )
    return TraceData(
        dataset_names=present,
        dataset=remapped,
        request=data.request[indices],
        fold=data.fold[indices],
        round=data.round[indices],
        decision=data.decision[indices],
        accept=data.accept[indices],
        features={name: values[indices] for name, values in data.features.items()},
        quality=quality,
        request_counts={name: data.request_counts.get(name, 0) for name in present},
    )


def read_eight_trace(args: argparse.Namespace) -> TraceData:
    # The reusable exploration root also contains MMLU.  Let the tested loader
    # audit it, then explicitly rebuild every array and quality total from only
    # the eight in-protocol datasets.
    data = read_base_trace(
        args.trace_root,
        args.split_seed,
        args.cv_folds,
        args.invalid_row_policy,
        1.0,
        args.max_rows_per_dataset,
    )
    return subset_eight_datasets(
        data, args.allow_partial_datasets, args.max_invalid_row_rate
    )


def profile_columns(profile: str) -> Tuple[int, ...]:
    return tuple(BLOCKS.index(block) for block in PROFILE_CONFIG[profile]["allowed_blocks"])


def actions_from_values(
    values: np.ndarray, lam: float, first_round: np.ndarray, profile: str
) -> np.ndarray:
    config = PROFILE_CONFIG[profile]
    allowed = config["allowed_blocks"]
    columns = profile_columns(profile)
    actions = np.full(len(values), allowed[0], dtype=np.uint8)
    best = values[:, columns[0]] - lam * allowed[0]
    for block, column in zip(allowed[1:], columns[1:]):
        utility = values[:, column] - lam * block
        better = utility > best + 1e-12
        actions[better] = block
        best[better] = utility[better]
    actions[first_round] = config["cold_start_block"]
    return actions


def oracle_actions(data: TraceData, lam: float, profile: str) -> Tuple[np.ndarray, np.ndarray]:
    config = PROFILE_CONFIG[profile]
    allowed = config["allowed_blocks"]
    columns = profile_columns(profile)
    actions = np.full(data.size, allowed[0], dtype=np.uint8)
    best = data.accept[:, columns[0]].astype(np.float64) - lam * allowed[0]
    for block, column in zip(allowed[1:], columns[1:]):
        utility = data.accept[:, column].astype(np.float64) - lam * block
        better = utility > best + 1e-12
        actions[better] = block
        best[better] = utility[better]
    first = data.round == 0
    cold = config["cold_start_block"]
    cold_column = BLOCKS.index(cold)
    actions[first] = cold
    best[first] = data.accept[first, cold_column].astype(np.float64) - lam * cold
    return actions, best


def summarize(
    data: TraceData,
    actions: np.ndarray,
    lam: float,
    profile: str,
    min_gain16: float,
    min_gain32: float,
) -> Dict[str, Any]:
    config = PROFILE_CONFIG[profile]
    allowed = set(config["allowed_blocks"])
    invalid = sorted(set(int(value) for value in np.unique(actions)) - allowed)
    if invalid:
        raise RuntimeError(f"profile {profile} contains forbidden actions {invalid}")
    accepted = selected_accept(data, actions)
    oracle, oracle_utility = oracle_actions(data, lam, profile)
    chosen_utility = accepted - lam * actions
    regret = oracle_utility - chosen_utility
    loss32 = np.maximum(data.accept[:, 2].astype(np.float64) - accepted, 0)
    allowed_columns = profile_columns(profile)
    max_allowed = data.accept[:, allowed_columns].max(axis=1).astype(np.float64)
    truncation = np.maximum(max_allowed - accepted, 0)
    controlled = data.round > 0
    if profile == "s8":
        large = controlled & (actions > 8)
        gain = np.where(
            actions == 16,
            data.accept[:, 1].astype(float) - data.accept[:, 0].astype(float),
            np.where(
                actions == 32,
                data.accept[:, 2].astype(float)
                - np.maximum(data.accept[:, 0], data.accept[:, 1]).astype(float),
                0.0,
            ),
        )
        significant = np.where(actions == 16, gain >= min_gain16, gain >= min_gain32)
    else:
        large = controlled & (actions == 32)
        gain = np.where(
            actions == 32,
            data.accept[:, 2].astype(float) - data.accept[:, 1].astype(float),
            0.0,
        )
        significant = gain >= min_gain32
    waste = gain <= 1

    def one(indices: np.ndarray) -> Dict[str, Any]:
        weights = indices_weights(data, indices)
        chosen = actions[indices]
        chosen_accept = accepted[indices]
        chosen_regret = regret[indices]
        mean_block = float(np.average(chosen, weights=weights))
        mean_accept = float(np.average(chosen_accept, weights=weights))
        return {
            "requests": int(len(np.unique(data.request[indices]))),
            "rounds": int(len(indices)),
            "mean_block": mean_block,
            "mean_accept": mean_accept,
            "decode_tpf_logical": mean_accept / 2.0,
            "accept_per_block_token": mean_accept / mean_block,
            "mean_utility": float(np.average(chosen_utility[indices], weights=weights)),
            "mean_regret": float(np.average(chosen_regret, weights=weights)),
            "p90_regret": weighted_quantile(chosen_regret, weights, 0.90),
            "p95_regret": weighted_quantile(chosen_regret, weights, 0.95),
            "oracle_action_accuracy": float(np.average(chosen == oracle[indices], weights=weights)),
            "loss_vs_l32": float(np.average(loss32[indices], weights=weights)),
            "trunc_gt1": float(np.average(truncation[indices] > 1, weights=weights)),
            "trunc_gt2": float(np.average(truncation[indices] > 2, weights=weights)),
            "trunc_gt4": float(np.average(truncation[indices] > 4, weights=weights)),
            "l8_rate": float(np.average(chosen == 8, weights=weights)),
            "l16_rate": float(np.average(chosen == 16, weights=weights)),
            "l32_rate": float(np.average(chosen == 32, weights=weights)),
        }

    by_dataset: Dict[str, Any] = {}
    for dataset_id, name in enumerate(data.dataset_names):
        indices = np.flatnonzero(data.dataset == dataset_id)
        if len(indices):
            by_dataset[name] = one(indices)
    all_indices = np.arange(data.size)
    macro = one(all_indices)
    macro["ratio_of_macro_means"] = macro["accept_per_block_token"]
    macro["accept_per_block_token"] = float(
        np.mean([row["accept_per_block_token"] for row in by_dataset.values()])
    )
    all_weights = indices_weights(data, all_indices)

    def conditional_macro(
        mask: np.ndarray, values: np.ndarray, empty: Optional[float]
    ) -> Optional[float]:
        rows = []
        for dataset_id in range(len(data.dataset_names)):
            indices = np.flatnonzero((data.dataset == dataset_id) & mask)
            if len(indices):
                rows.append(float(np.average(values[indices], weights=all_weights[indices])))
        return float(np.mean(rows)) if rows else empty

    macro.update(
        {
            "profile": profile,
            "cold_start_block": config["cold_start_block"],
            "allowed_blocks": list(config["allowed_blocks"]),
            "large_rate": conditional_macro(controlled, (actions > config["baseline"]).astype(float), 0.0),
            # With no promotion there is no measurable precision/waste rate;
            # reporting 100%/0% would make an inactive policy look successful.
            "large_precision": conditional_macro(large, significant.astype(float), None),
            "large_waste_rate": conditional_macro(large, waste.astype(float), None),
            "compute_saving_vs_l32": 32.0 - macro["mean_block"],
            "compute_delta_vs_baseline": macro["mean_block"] - config["baseline"],
        }
    )
    return {"datasets": by_dataset, "macro": macro}


def aggregate_groups(data: TraceData, values: np.ndarray) -> Dict[str, np.ndarray]:
    """Collapse repeated fold/bin predictions before exact lambda evaluation."""
    rounded = np.round(values.astype(np.float64), 12)
    first = (data.round == 0).astype(np.float64)[:, None]
    keys = np.column_stack((data.dataset.astype(np.float64), first, rounded))
    unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    weights = indices_weights(data, np.arange(data.size))
    count = len(unique)
    group_weight = np.bincount(inverse, weights=weights, minlength=count)
    group_accept = np.column_stack(
        [
            np.bincount(
                inverse,
                weights=weights * data.accept[:, column].astype(np.float64),
                minlength=count,
            )
            for column in range(3)
        ]
    )
    return {
        "dataset": unique[:, 0].astype(np.int16),
        "first": unique[:, 1].astype(bool),
        "values": unique[:, 2:],
        "weight": group_weight,
        "accept": group_accept,
    }


def exact_lambda_candidates(
    group_values: np.ndarray,
    group_first: np.ndarray,
    profile: str,
    lambda_max: float,
) -> List[Dict[str, float]]:
    roots = {0.0, float(lambda_max)}
    allowed = PROFILE_CONFIG[profile]["allowed_blocks"]
    columns = profile_columns(profile)
    values = group_values[~group_first]
    for left_index in range(len(allowed)):
        for right_index in range(left_index + 1, len(allowed)):
            left, right = allowed[left_index], allowed[right_index]
            left_column, right_column = columns[left_index], columns[right_index]
            candidates = (values[:, right_column] - values[:, left_column]) / (right - left)
            for value in candidates[np.isfinite(candidates)]:
                number = float(value)
                if 0.0 <= number <= lambda_max:
                    roots.add(round(number, 12))
    boundaries = sorted(roots)
    result: List[Dict[str, float]] = []
    for index, value in enumerate(boundaries):
        result.append(
            {"lambda": value, "interval_lo": value, "interval_hi": value, "stable_width": 0.0}
        )
        if index + 1 < len(boundaries) and boundaries[index + 1] > value:
            upper = boundaries[index + 1]
            result.append(
                {
                    "lambda": (value + upper) / 2.0,
                    "interval_lo": value,
                    "interval_hi": upper,
                    "stable_width": upper - value,
                }
            )
    return result


def grouped_efficiency(
    grouped: Mapping[str, np.ndarray], lam: float, profile: str, dataset_count: int
) -> Tuple[float, float]:
    config = PROFILE_CONFIG[profile]
    allowed = config["allowed_blocks"]
    columns = profile_columns(profile)
    values = grouped["values"]
    chosen_columns = np.full(len(values), columns[0], dtype=np.int8)
    best = values[:, columns[0]] - lam * allowed[0]
    for block, column in zip(allowed[1:], columns[1:]):
        utility = values[:, column] - lam * block
        better = utility > best + 1e-12
        chosen_columns[better] = column
        best[better] = utility[better]
    cold_column = BLOCKS.index(config["cold_start_block"])
    chosen_columns[grouped["first"]] = cold_column
    rows = np.arange(len(values))
    chosen_accept = grouped["accept"][rows, chosen_columns]
    chosen_block = grouped["weight"] * np.asarray(BLOCKS)[chosen_columns]
    accept_by_dataset = np.bincount(
        grouped["dataset"], weights=chosen_accept, minlength=dataset_count
    )
    block_by_dataset = np.bincount(
        grouped["dataset"], weights=chosen_block, minlength=dataset_count
    )
    efficiencies = np.divide(
        accept_by_dataset,
        block_by_dataset,
        out=np.zeros_like(accept_by_dataset),
        where=block_by_dataset > 0,
    )
    return float(np.mean(efficiencies)), float(np.sum(chosen_block))


def optimize_profile(
    data: TraceData,
    values: np.ndarray,
    profile: str,
    lambda_max: float,
    baseline_efficiency: float,
    min_gain16: float,
    min_gain32: float,
    diagnostic_grid: Sequence[float],
) -> Dict[str, Any]:
    grouped = aggregate_groups(data, values)
    candidates = exact_lambda_candidates(
        grouped["values"], grouped["first"], profile, lambda_max
    )
    best: Optional[Tuple[Tuple[float, float, float], Dict[str, float]]] = None
    for candidate in candidates:
        efficiency, weighted_block = grouped_efficiency(
            grouped, candidate["lambda"], profile, len(data.dataset_names)
        )
        candidate["efficiency"] = efficiency
        candidate["weighted_block"] = weighted_block
        # Exact efficiency first; on exact ties prefer a stable interval, then
        # less compute.  No acceptance/TPF constraint is hidden here.
        key = (efficiency, candidate["stable_width"], -weighted_block)
        if best is None or key > best[0]:
            best = (key, candidate)
    assert best is not None
    selected = best[1]
    lam = float(selected["lambda"])
    actions = actions_from_values(values, lam, data.round == 0, profile)
    summary = summarize(data, actions, lam, profile, min_gain16, min_gain32)
    summary["transitions"] = summarize_transitions(data, actions)
    summary["macro"]["baseline_efficiency"] = baseline_efficiency
    summary["macro"]["relative_efficiency_gain"] = (
        summary["macro"]["accept_per_block_token"] / baseline_efficiency - 1.0
    )
    diagnostics = []
    for value in diagnostic_grid:
        if 0 <= value <= lambda_max:
            point_actions = actions_from_values(values, value, data.round == 0, profile)
            point = summarize(
                data, point_actions, value, profile, min_gain16, min_gain32
            )["macro"]
            diagnostics.append({"lambda": value, **point})
    return {
        "profile": profile,
        "allowed_blocks": list(PROFILE_CONFIG[profile]["allowed_blocks"]),
        "cold_start_block": PROFILE_CONFIG[profile]["cold_start_block"],
        "lambda": lam,
        "lambda_interval": [selected["interval_lo"], selected["interval_hi"]],
        "lambda_exact_candidates": len(candidates),
        "efficiency": summary["macro"]["accept_per_block_token"],
        "baseline_efficiency": baseline_efficiency,
        "relative_efficiency_gain": summary["macro"]["relative_efficiency_gain"],
        "summary": summary,
        "diagnostic_grid": diagnostics,
    }


def candidate_result(
    args: argparse.Namespace,
    data: TraceData,
    spec: Mapping[str, Any],
    gate: Optional[int],
    baseline_efficiencies: Mapping[str, float],
) -> Tuple[Dict[str, Any], np.ndarray]:
    values = candidate_oof(data, spec, gate, args.cv_folds, args.signal_bins)
    profiles = {
        profile: optimize_profile(
            data,
            values,
            profile,
            args.lambda_max,
            baseline_efficiencies[profile],
            args.min_gain16,
            args.min_gain32,
            args.diagnostic_lambda_grid,
        )
        for profile in ("s8", "s16")
    }
    gains = [profiles[name]["relative_efficiency_gain"] for name in ("s8", "s16")]
    regret_sum = sum(profiles[name]["summary"]["macro"]["mean_regret"] for name in profiles)
    result = {
        "tier": 1 if gate is None else 2,
        "signal": spec["name"],
        "feature": spec["feature"],
        "kind": spec["kind"],
        "zh": spec["zh"],
        "gate": gate,
        "joint_min_gain": float(min(gains)),
        "joint_mean_gain": float(np.mean(gains)),
        "joint_regret_sum": float(regret_sum),
        "profiles": profiles,
    }
    return result, values


def candidate_key(result: Mapping[str, Any]) -> Tuple[Any, ...]:
    return (
        -float(result["joint_min_gain"]),
        -float(result["joint_mean_gain"]),
        float(result["joint_regret_sum"]),
        str(result["signal"]),
        int(result.get("gate") or 0),
    )


def compact_candidate(result: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "tier": result["tier"],
        "signal": result["signal"],
        "feature": result["feature"],
        "kind": result["kind"],
        "zh": result["zh"],
        "gate": result.get("gate"),
        "joint_min_gain": result["joint_min_gain"],
        "joint_mean_gain": result["joint_mean_gain"],
        "joint_regret_sum": result["joint_regret_sum"],
        "s8_lambda": result["profiles"]["s8"]["lambda"],
        "s8_gain": result["profiles"]["s8"]["relative_efficiency_gain"],
        "s16_lambda": result["profiles"]["s16"]["lambda"],
        "s16_gain": result["profiles"]["s16"]["relative_efficiency_gain"],
    }


def publish_progress(
    run_dir: Optional[Path], output_dir: Path, payload: Mapping[str, Any]
) -> None:
    atomic_json(output_dir / "progress.json", payload)
    if run_dir is not None:
        try:
            from reporting import render

            render(run_dir)
        except Exception as exc:
            print(f"[报告更新警告] {exc}", file=sys.stderr, flush=True)


def search_policy(args: argparse.Namespace, data: TraceData) -> Dict[str, Any]:
    # Baseline efficiency is independent of lambda; lambda=0 is used only to
    # populate diagnostic regret fields that are not part of signal selection.
    fixed_actions = {
        "s8": np.full(data.size, 8, dtype=np.uint8),
        "s16": np.full(data.size, 16, dtype=np.uint8),
    }
    baseline_summaries = {
        profile: summarize(
            data,
            actions,
            0.0,
            profile,
            args.min_gain16,
            args.min_gain32,
        )
        for profile, actions in fixed_actions.items()
    }
    baseline_efficiencies = {
        profile: summary["macro"]["accept_per_block_token"]
        for profile, summary in baseline_summaries.items()
    }

    tier1: List[Dict[str, Any]] = []
    progress = ProgressBar("Tier1共享单信号+精确lambda", len(SIGNALS))
    for index, spec in enumerate(SIGNALS, 1):
        result, values = candidate_result(
            args, data, spec, None, baseline_efficiencies
        )
        tier1.append(result)
        ranking = sorted(tier1, key=candidate_key)
        publish_progress(
            args.run_dir,
            args.output_dir,
            {
                "stage": "tier1",
                "completed": index,
                "total": len(SIGNALS),
                "latest": compact_candidate(result),
                "best": compact_candidate(ranking[0]),
            },
        )
        progress.update()
    progress.close()
    tier1.sort(key=candidate_key)
    best_single = tier1[0]

    dual_specs = [
        spec for spec in SIGNALS
        if spec["kind"] == "accept" and spec["name"] != "full_streak"
    ]
    dual_candidates = [
        (spec, gate) for spec in dual_specs for gate in args.full_gate_grid
    ]
    tier2: List[Dict[str, Any]] = []
    progress = ProgressBar("Tier2接收+full+精确lambda", len(dual_candidates))
    for index, (spec, gate) in enumerate(dual_candidates, 1):
        try:
            result, values = candidate_result(
                args, data, spec, gate, baseline_efficiencies
            )
        except RuntimeError as exc:
            print(
                f"[Tier2跳过] signal={spec['name']} gate={gate}: {exc}",
                file=sys.stderr,
                flush=True,
            )
            progress.update()
            continue
        tier2.append(result)
        ranking = sorted(tier2, key=candidate_key)
        publish_progress(
            args.run_dir,
            args.output_dir,
            {
                "stage": "tier2",
                "completed": index,
                "total": len(dual_candidates),
                "latest": compact_candidate(result),
                "tier1_best": compact_candidate(best_single),
                "best": compact_candidate(ranking[0]),
            },
        )
        progress.update()
    progress.close()
    tier2.sort(key=candidate_key)
    best_dual = tier2[0] if tier2 else None

    selected = best_single
    dual_improvement = None
    dual_improves_both = False
    if best_dual is not None:
        dual_improvement = (
            best_dual["joint_min_gain"] - best_single["joint_min_gain"]
        )
        dual_improves_both = all(
            best_dual["profiles"][profile]["relative_efficiency_gain"]
            > best_single["profiles"][profile]["relative_efficiency_gain"] + 1e-12
            for profile in ("s8", "s16")
        )
        if dual_improves_both and dual_improvement >= args.min_dual_improvement:
            selected = best_dual

    spec = SIGNAL_BY_NAME[selected["signal"]]
    historical = np.flatnonzero(data.round > 0)
    oriented = data.features[spec["feature"]].astype(np.float64) * float(spec["orientation"])
    final_tables = fit_tables(
        data, historical, oriented, selected.get("gate"), args.signal_bins
    )
    profiles = {}
    for profile in ("s8", "s16"):
        selected_profile = selected["profiles"][profile]
        profiles[profile] = {
            "cold_start_block": PROFILE_CONFIG[profile]["cold_start_block"],
            "allowed_blocks": list(PROFILE_CONFIG[profile]["allowed_blocks"]),
            "default_lambda": selected_profile["lambda"],
            "optimal_lambda_interval": selected_profile["lambda_interval"],
            "search_efficiency": selected_profile["efficiency"],
            "baseline_efficiency": selected_profile["baseline_efficiency"],
            "relative_efficiency_gain": selected_profile["relative_efficiency_gain"],
        }
    policy = {
        "schema_version": 1,
        "policy_type": "dual_action_space_scalar_efficiency",
        "blocks": list(BLOCKS),
        "signal_count": selected["tier"],
        "signal": {
            "name": spec["name"],
            "feature": spec["feature"],
            "orientation": spec["orientation"],
            "kind": spec["kind"],
            "zh": spec["zh"],
        },
        "full_streak_gate": (
            {
                "feature": "full_streak",
                "threshold": selected["gate"],
                "role": "right_censor_stratum",
            }
            if selected["tier"] == 2 else None
        ),
        "value_models": final_tables,
        "profiles": profiles,
        "lambda_semantics": "choose argmax over profile allowed blocks of E[A_L|signal]-lambda*L",
        "primary_objective": "mean of eight per-dataset accept/block ratios",
        "dataset_weighting": "dataset_equal_then_request_equal_then_round_equal",
        "training_rows": int(len(historical)),
    }
    atomic_json(args.output_dir / "policy.json", {"policy": policy})

    # Recompute baselines and restricted same-state oracles at each profile's
    # selected lambda so every reported regret uses the matching objective.
    fixed = {}
    oracles = {}
    for profile in ("s8", "s16"):
        lam = profiles[profile]["default_lambda"]
        baseline_block = PROFILE_CONFIG[profile]["baseline"]
        fixed[profile] = summarize(
            data,
            np.full(data.size, baseline_block, dtype=np.uint8),
            lam,
            profile,
            args.min_gain16,
            args.min_gain32,
        )
        oracle, _ = oracle_actions(data, lam, profile)
        oracles[profile] = summarize(
            data, oracle, lam, profile, args.min_gain16, args.min_gain32
        )

    payload = {
        "protocol": {
            "datasets": data.dataset_names,
            "dataset_count": len(data.dataset_names),
            "excluded_datasets": ["aime24", "mmlu"],
            "requests_by_dataset": data.request_counts,
            "rows": data.size,
            "historical_rows": int(len(historical)),
            "cv_folds": args.cv_folds,
            "signal_bins": args.signal_bins,
            "lambda_search": "exact value-intersection boundaries and open-interval midpoints",
            "lambda_max": args.lambda_max,
            "weighting": "dataset equal -> request equal -> request rounds equal",
            "primary_efficiency": "mean of per-dataset (mean accept / mean block)",
            "trace_quality": data.quality,
        },
        "tier1_ranking": [compact_candidate(row) for row in tier1],
        "tier2_ranking": [compact_candidate(row) for row in tier2],
        "selection": {
            "selected": compact_candidate(selected),
            "best_single": compact_candidate(best_single),
            "best_dual": compact_candidate(best_dual) if best_dual else None,
            "dual_joint_min_improvement": dual_improvement,
            "dual_improves_both_profiles": dual_improves_both,
            "min_dual_improvement": args.min_dual_improvement,
            "selection_order": "max min(G8,G16), then max mean(G8,G16), then lower summed regret",
        },
        "profiles": selected["profiles"],
        "fixed_baselines": fixed,
        "restricted_same_state_oracles": oracles,
        "policy": policy,
    }
    atomic_json(args.output_dir / "search_results.json", payload)
    publish_progress(
        args.run_dir,
        args.output_dir,
        {
            "stage": "completed",
            "completed": len(SIGNALS) + len(dual_candidates),
            "total": len(SIGNALS) + len(dual_candidates),
            "selected": compact_candidate(selected),
        },
    )
    return payload


def validate_policy(args: argparse.Namespace, data: TraceData) -> Dict[str, Any]:
    payload = json.loads(args.policy.read_text(encoding="utf-8"))
    policy = payload.get("policy", payload)
    if policy.get("policy_type") != "dual_action_space_scalar_efficiency":
        raise RuntimeError("validation policy type mismatch")
    configured = policy["profiles"][args.profile]
    lam = float(
        configured["default_lambda"]
        if args.policy_lambda is None else args.policy_lambda
    )
    actions = data.decision.astype(np.uint8)
    summary = summarize(
        data, actions, lam, args.profile, args.min_gain16, args.min_gain32
    )
    summary["transitions"] = summarize_transitions(data, actions)
    baseline_block = PROFILE_CONFIG[args.profile]["baseline"]
    baseline = summarize(
        data,
        np.full(data.size, baseline_block, dtype=np.uint8),
        lam,
        args.profile,
        args.min_gain16,
        args.min_gain32,
    )
    summary["macro"]["baseline_efficiency"] = baseline["macro"]["accept_per_block_token"]
    summary["macro"]["relative_efficiency_gain"] = (
        summary["macro"]["accept_per_block_token"]
        / baseline["macro"]["accept_per_block_token"] - 1.0
    )
    oracle, _ = oracle_actions(data, lam, args.profile)
    result = {
        "protocol": {
            "profile": args.profile,
            "datasets": data.dataset_names,
            "dataset_count": len(data.dataset_names),
            "requests_by_dataset": data.request_counts,
            "rows": data.size,
            "trace_quality": data.quality,
            "policy_path": str(args.policy),
            "policy_lambda": lam,
            "formal_complete": not args.allow_partial_datasets,
        },
        "dynamic_policy": summary,
        "fixed_baseline_same_state": baseline,
        "restricted_same_state_oracle": summarize(
            data, oracle, lam, args.profile, args.min_gain16, args.min_gain32
        ),
        "official_sglang_metrics": collect_official_metrics(args.eval_root),
    }
    atomic_json(args.output_dir / f"validation_{args.profile}.json", result)
    publish_progress(
        args.run_dir,
        args.output_dir,
        {
            "stage": f"validation_{args.profile}_completed"
            if not args.allow_partial_datasets
            else f"validation_{args.profile}_partial",
            "completed": len(data.dataset_names),
            "total": len(EXPECTED_DATASETS),
            "profile": args.profile,
            "lambda": lam,
        },
    )
    return result


def parse_float_list(text: str) -> List[float]:
    values = sorted(set(float(item.strip()) for item in text.split(",") if item.strip()))
    if not values or values[0] < 0:
        raise argparse.ArgumentTypeError("values must be nonempty and nonnegative")
    return values


def parse_int_list(text: str) -> List[int]:
    values = sorted(set(int(item.strip()) for item in text.split(",") if item.strip()))
    if not values or values[0] < 1:
        raise argparse.ArgumentTypeError("values must be positive integers")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("search", "validate"), required=True)
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--profile", choices=("s8", "s16"))
    parser.add_argument("--policy-lambda", type=float)
    parser.add_argument("--eval-root", type=Path)
    parser.add_argument("--split-seed", type=int, default=20260906)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--signal-bins", type=int, default=32)
    parser.add_argument("--lambda-max", type=float, default=1.0)
    parser.add_argument(
        "--diagnostic-lambda-grid",
        type=parse_float_list,
        default=parse_float_list("0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.5,0.6,0.75,1"),
    )
    parser.add_argument(
        "--full-gate-grid", type=parse_int_list, default=parse_int_list("1,2,3")
    )
    parser.add_argument("--min-dual-improvement", type=float, default=0.002)
    parser.add_argument("--min-gain16", type=float, default=2.0)
    parser.add_argument("--min-gain32", type=float, default=4.0)
    parser.add_argument(
        "--invalid-row-policy", choices=("strict", "exclude"), default="exclude"
    )
    parser.add_argument("--max-invalid-row-rate", type=float, default=0.05)
    parser.add_argument("--allow-partial-datasets", action="store_true")
    parser.add_argument("--max-rows-per-dataset", type=int, default=0)
    args = parser.parse_args()
    if args.cv_folds < 2 or args.signal_bins < 2:
        parser.error("--cv-folds and --signal-bins must be at least 2")
    if args.lambda_max <= 0:
        parser.error("--lambda-max must be positive")
    if not 0 <= args.max_invalid_row_rate <= 1:
        parser.error("--max-invalid-row-rate must be in [0,1]")
    if args.mode == "validate" and (args.policy is None or args.profile is None):
        parser.error("--policy and --profile are required in validate mode")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = read_eight_trace(args)
    if args.mode == "search":
        search_policy(args, data)
    else:
        validate_policy(args, data)


if __name__ == "__main__":
    main()
