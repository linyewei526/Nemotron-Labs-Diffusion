#!/usr/bin/env python3
"""Search three verifier-only dynamic-block policy families and validate one winner."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


HERE = Path(__file__).resolve().parent
OBSERVATIONS = HERE.parent
if str(OBSERVATIONS) not in sys.path:
    sys.path.insert(0, str(OBSERVATIONS))

from sglang_b200_verify_efficiency_policy.latency_costs import (  # noqa: E402
    BLOCKS,
    DEFAULT_CONCURRENCIES,
    atomic_json,
    cold_start_block,
    parse_cost_document,
    parse_int_list,
    snapshot,
)
from sglang_b200_verify_efficiency_policy.models import (  # noqa: E402
    EXPECTED_DATASETS,
    FEATURES,
    SIGNALS,
    SIGNAL_BY_NAME,
    ProgressBar,
    TraceData,
    fixed_summaries,
    fit_value_models,
    local_oracle_metrics,
    oof_values,
    orientations,
    predict_value_models,
    read_trace,
    row_weights,
    sample_summary,
    stratum_values,
    weighted_edges,
)
from sglang_b200_verify_efficiency_policy.policy_runtime import (  # noqa: E402
    choose_action as runtime_choose_action,
)


FAMILIES = ("local_ratio", "direct_rank", "global_fractional")
FAMILY_ZH = {
    "local_ratio": "无λ条件期望比值",
    "direct_rank": "直接排序/regret",
    "global_fractional": "全局分式对照",
}


def actions_from_values(
    values: np.ndarray,
    family: str,
    costs: Mapping[int, float],
    first_round: np.ndarray,
    initial_block: int,
    rho: float = 0.0,
) -> np.ndarray:
    cost_vector = np.asarray([costs[block] for block in BLOCKS], dtype=np.float64)
    if family == "local_ratio":
        utilities = values / cost_vector[None, :]
    elif family == "global_fractional":
        utilities = values - float(rho) * cost_vector[None, :]
    else:
        raise ValueError(f"value action unsupported for family={family}")
    # np.argmax resolves exact ties toward the smallest block.
    actions = np.asarray(BLOCKS, dtype=np.uint8)[np.argmax(utilities, axis=1)]
    actions[first_round] = int(initial_block)
    return actions


def baseline_block(concurrency: int) -> int:
    """The user-specified fixed comparator is also the cold-start block."""
    return cold_start_block(concurrency)


def summarize_candidate_profile(
    data: TraceData,
    actions: np.ndarray,
    concurrency: int,
    all_costs: Mapping[int, Mapping[int, float]],
    baselines: Mapping[str, Any],
) -> Dict[str, Any]:
    summary = sample_summary(data, actions, concurrency, all_costs)
    fixed_block = baseline_block(concurrency)
    fixed = baselines[str(fixed_block)]
    score = float(summary["macro"]["score_token_ms_req"])
    fixed_score = float(fixed["macro"]["score_token_ms_req"])
    gain = score / fixed_score - 1.0
    for dataset, row in summary["datasets"].items():
        dataset_fixed = float(fixed["datasets"][dataset]["score_token_ms_req"])
        dataset_score = float(row["score_token_ms_req"])
        row["baseline_block"] = fixed_block
        row["baseline_score"] = dataset_fixed
        row["relative_gain_vs_designated_fixed"] = dataset_score / dataset_fixed - 1.0
        row["throughput_equivalent_time_saving"] = 1.0 - dataset_fixed / dataset_score
    return {
        "score": score,
        "summary": summary,
        "baseline_block": fixed_block,
        "baseline_score": fixed_score,
        "relative_gain_vs_designated_fixed": gain,
        "throughput_equivalent_time_saving": 1.0 - fixed_score / score,
        "oracle": local_oracle_metrics(data, actions, concurrency, all_costs),
    }


class IncrementalRhoEvaluator:
    """Scan every value-cost breakpoint without recomputing all trace rows."""

    def __init__(
        self,
        data: TraceData,
        values: np.ndarray,
        concurrency: int,
        all_costs: Mapping[int, Mapping[int, float]],
    ) -> None:
        self.data = data
        self.values = values
        self.concurrency = concurrency
        self.costs = all_costs[concurrency]
        self.request_count = int(data.request.max()) + 1
        self.request_dataset = np.full(self.request_count, -1, dtype=np.int16)
        self.request_dataset[data.request] = data.dataset.astype(np.int16)
        self.valid_requests = np.flatnonzero(self.request_dataset >= 0)
        self.dataset_request_counts = np.bincount(
            self.request_dataset[self.valid_requests],
            minlength=len(data.dataset_names),
        ).astype(np.float64)

        first = data.round == 0
        historical = ~first
        rounded = np.round(values[historical].astype(np.float64), 12)
        self.group_values, row_groups = np.unique(
            rounded, axis=0, return_inverse=True
        )
        historical_requests = data.request[historical].astype(np.int64)
        compound = row_groups.astype(np.int64) * self.request_count + historical_requests
        unique_cells, inverse = np.unique(compound, return_inverse=True)
        self.cell_group = unique_cells // self.request_count
        self.cell_request = unique_cells % self.request_count
        self.cell_count = np.bincount(inverse).astype(np.float64)
        self.cell_accept = np.column_stack(
            [
                np.bincount(
                    inverse,
                    weights=data.accept[historical, column].astype(np.float64),
                    minlength=len(unique_cells),
                )
                for column in range(len(BLOCKS))
            ]
        )
        self.cell_cost = self.cell_count[:, None] * np.asarray(
            [self.costs[block] for block in BLOCKS], dtype=np.float64
        )[None, :]
        group_ids = np.arange(len(self.group_values))
        self.group_starts = np.searchsorted(self.cell_group, group_ids, side="left")
        self.group_ends = np.searchsorted(self.cell_group, group_ids, side="right")

        initial_block = cold_start_block(concurrency)
        initial_column = BLOCKS.index(initial_block)
        self.fixed_accept = np.bincount(
            data.request[first],
            weights=data.accept[first, initial_column].astype(np.float64),
            minlength=self.request_count,
        )
        self.fixed_cost = np.bincount(
            data.request[first],
            weights=np.full(int(first.sum()), self.costs[initial_block]),
            minlength=self.request_count,
        )

    def group_actions(self, rho: float) -> np.ndarray:
        utilities = self.group_values - rho * np.asarray(
            [self.costs[block] for block in BLOCKS], dtype=np.float64
        )[None, :]
        return np.asarray(BLOCKS, dtype=np.uint8)[np.argmax(utilities, axis=1)]

    @staticmethod
    def columns(actions: np.ndarray) -> np.ndarray:
        return np.where(actions == 8, 0, np.where(actions == 16, 1, 2))

    def scan(self, candidates: Sequence[float]) -> List[float]:
        if not candidates:
            return []
        group_actions = self.group_actions(float(candidates[0]))
        cell_columns = self.columns(group_actions[self.cell_group])
        rows = np.arange(len(self.cell_group))
        request_accept = self.fixed_accept + np.bincount(
            self.cell_request,
            weights=self.cell_accept[rows, cell_columns],
            minlength=self.request_count,
        )
        request_cost = self.fixed_cost + np.bincount(
            self.cell_request,
            weights=self.cell_cost[rows, cell_columns],
            minlength=self.request_count,
        )
        ratios = np.divide(
            request_accept,
            request_cost,
            out=np.zeros_like(request_accept),
            where=request_cost > 0,
        )
        dataset_sums = np.bincount(
            self.request_dataset[self.valid_requests],
            weights=ratios[self.valid_requests],
            minlength=len(self.data.dataset_names),
        )

        def macro() -> float:
            return float(np.mean(dataset_sums / self.dataset_request_counts))

        scores = [macro()]
        for rho in candidates[1:]:
            new_actions = self.group_actions(float(rho))
            changed = np.flatnonzero(new_actions != group_actions)
            if len(changed):
                cells = np.concatenate(
                    [
                        np.arange(self.group_starts[group], self.group_ends[group])
                        for group in changed
                    ]
                )
                affected = np.unique(self.cell_request[cells])
                old_ratios = ratios[affected].copy()
                old_columns = self.columns(group_actions[self.cell_group[cells]])
                new_columns = self.columns(new_actions[self.cell_group[cells]])
                np.add.at(
                    request_accept,
                    self.cell_request[cells],
                    self.cell_accept[cells, new_columns]
                    - self.cell_accept[cells, old_columns],
                )
                np.add.at(
                    request_cost,
                    self.cell_request[cells],
                    self.cell_cost[cells, new_columns]
                    - self.cell_cost[cells, old_columns],
                )
                ratios[affected] = request_accept[affected] / request_cost[affected]
                dataset_sums += np.bincount(
                    self.request_dataset[affected],
                    weights=ratios[affected] - old_ratios,
                    minlength=len(self.data.dataset_names),
                )
                group_actions = new_actions
            scores.append(macro())
        return scores


def optimize_rho(
    data: TraceData,
    values: np.ndarray,
    concurrency: int,
    all_costs: Mapping[int, Mapping[int, float]],
    baselines: Mapping[str, Any],
    grid_size: int,
) -> Tuple[float, np.ndarray, Dict[str, Any], Dict[str, Any]]:
    """Automatically solve the global fractional comparator; rho is not user taste.

    Exact Dinkelbach separability does not hold for a mean of eight dataset
    ratios.  We therefore combine fixed-point proposals with a deterministic
    one-dimensional grid, and choose rho by the exact required sample/dataset
    macro score.
    """
    costs = all_costs[concurrency]
    historical_values = np.unique(
        np.round(values[data.round > 0].astype(np.float64), 12), axis=0
    )
    cost_vector = np.asarray([costs[block] for block in BLOCKS], dtype=np.float64)
    roots = {0.0}
    for left in range(len(BLOCKS)):
        for right in range(left + 1, len(BLOCKS)):
            denominator = cost_vector[right] - cost_vector[left]
            if abs(denominator) <= 1e-15:
                continue
            values_at_root = (
                historical_values[:, right] - historical_values[:, left]
            ) / denominator
            roots.update(
                float(value)
                for value in values_at_root
                if math.isfinite(float(value)) and float(value) >= 0.0
            )
    boundaries = sorted(roots)
    upper = (boundaries[-1] + max(1e-9, boundaries[-1] * 0.05)) if boundaries else 1.0
    candidates = set(boundaries)
    candidates.add(upper)
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        if right > left:
            candidates.add((left + right) / 2.0)
    candidates.update(float(value) for value in np.linspace(0.0, upper, grid_size))
    ordered = sorted(candidates)
    evaluator = IncrementalRhoEvaluator(data, values, concurrency, all_costs)
    scores = evaluator.scan(ordered)
    best_index = max(range(len(ordered)), key=lambda index: (scores[index], -ordered[index]))
    rho = float(ordered[best_index])
    actions = actions_from_values(
        values,
        "global_fractional",
        costs,
        data.round == 0,
        cold_start_block(concurrency),
        rho,
    )
    profile = summarize_candidate_profile(
        data, actions, concurrency, all_costs, baselines
    )
    diagnostics = {
        "candidate_count": len(ordered),
        "exact_utility_intersections": len(boundaries),
        "supplemental_grid_size": grid_size,
        "selection": (
            "all exact utility intersections and stable midpoints, evaluated "
            "incrementally by the exact sample-then-dataset macro score"
        ),
    }
    return rho, actions, profile, diagnostics


def _fit_rank_group(
    data: TraceData,
    indices: np.ndarray,
    signal: np.ndarray,
    concurrency: int,
    all_costs: Mapping[int, Mapping[int, float]],
    bins: int,
) -> Dict[str, Any]:
    weights = row_weights(data, indices)
    local_signal = signal[indices]
    finite_mask = np.isfinite(local_signal)
    if int(finite_mask.sum()) < max(24, bins * 2):
        raise RuntimeError("too few finite rows for direct-rank table")
    edges = weighted_edges(local_signal[finite_mask], weights[finite_mask], bins)
    bin_index = np.searchsorted(edges, local_signal, side="left")
    bin_count = len(edges) + 1
    efficiencies = data.accept[indices].astype(np.float64) / np.asarray(
        [all_costs[concurrency][block] for block in BLOCKS]
    )[None, :]
    oracle = efficiencies.max(axis=1)
    regrets = oracle[:, None] - efficiencies
    tables = np.zeros((bin_count, len(BLOCKS)), dtype=np.float64)
    actions = np.empty(bin_count, dtype=np.uint8)
    for bin_id in range(bin_count):
        mask = (bin_index == bin_id) & finite_mask
        local_weights = weights[mask]
        local_regret = (
            (regrets[mask] * local_weights[:, None]).sum(axis=0)
            / local_weights.sum()
        )
        tables[bin_id] = local_regret
        actions[bin_id] = BLOCKS[int(np.argmin(local_regret))]
    missing = ~finite_mask
    if missing.any():
        missing_regret = (
            (regrets[missing] * weights[missing, None]).sum(axis=0)
            / weights[missing].sum()
        )
    else:
        missing_regret = (regrets * weights[:, None]).sum(axis=0) / weights.sum()
    return {
        "type": "binned_conditional_regret",
        "edges": edges.tolist(),
        "actions": [int(value) for value in actions],
        "conditional_regret": tables.tolist(),
        "missing_action": int(BLOCKS[int(np.argmin(missing_regret))]),
        "missing_regret": missing_regret.tolist(),
    }


def fit_rank_tables(
    data: TraceData,
    indices: np.ndarray,
    spec: Mapping[str, Any],
    use_censor: bool,
    concurrency: int,
    all_costs: Mapping[int, Mapping[int, float]],
    bins: int,
) -> Dict[str, Any]:
    signal = orientations(data, spec)
    indices = np.asarray(indices, dtype=np.int64)
    result = {
        "all": _fit_rank_group(
            data, indices, signal, concurrency, all_costs, bins
        )
    }
    if use_censor:
        strata = stratum_values(data, indices)
        for stratum in (
            "b8_partial",
            "b8_full",
            "b16_partial",
            "b16_full",
            "b32_partial",
            "b32_full",
        ):
            mask = strata == stratum
            if int(mask.sum()) < max(48, bins * 3):
                continue
            local = indices[mask]
            try:
                result[stratum] = _fit_rank_group(
                    data, local, signal, concurrency, all_costs, bins
                )
            except RuntimeError:
                continue
    return result


def predict_rank_tables(
    data: TraceData,
    indices: np.ndarray,
    spec: Mapping[str, Any],
    use_censor: bool,
    tables: Mapping[str, Any],
) -> np.ndarray:
    signal = orientations(data, spec)[indices]
    strata = (
        stratum_values(data, indices)
        if use_censor
        else np.full(len(indices), "all", dtype=object)
    )
    output = np.empty(len(indices), dtype=np.uint8)
    for stratum in np.unique(strata):
        mask = strata == stratum
        table = tables.get(str(stratum), tables["all"])
        values = signal[mask]
        valid = np.isfinite(values)
        selected = np.full(len(values), int(table["missing_action"]), dtype=np.uint8)
        if valid.any():
            bin_ids = np.searchsorted(np.asarray(table["edges"]), values[valid], side="left")
            selected[valid] = np.asarray(table["actions"], dtype=np.uint8)[bin_ids]
        output[mask] = selected
    return output


def oof_rank_actions(
    data: TraceData,
    spec: Mapping[str, Any],
    use_censor: bool,
    concurrency: int,
    all_costs: Mapping[int, Mapping[int, float]],
    folds: int,
    bins: int,
) -> np.ndarray:
    historical = data.round > 0
    output = np.full(data.size, cold_start_block(concurrency), dtype=np.uint8)
    for fold in range(folds):
        train = np.flatnonzero(historical & (data.fold != fold))
        held = np.flatnonzero(historical & (data.fold == fold))
        tables = fit_rank_tables(
            data, train, spec, use_censor, concurrency, all_costs, bins
        )
        output[held] = predict_rank_tables(data, held, spec, use_censor, tables)
    return output


def candidate_compact(candidate: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "family": candidate["family"],
        "family_zh": FAMILY_ZH[candidate["family"]],
        "signal": candidate["signal"],
        "signal_zh": candidate["signal_zh"],
        "use_censor_state": bool(candidate["use_censor_state"]),
        "signal_count": int(candidate["signal_count"]),
        "mean_gain_across_c": float(candidate["mean_gain_across_c"]),
        "min_gain_across_c": float(candidate["min_gain_across_c"]),
        "mean_regret_across_c": float(candidate["mean_regret_across_c"]),
        "profiles": {
            key: {
                "rho": value.get("rho"),
                "score": value["score"],
                "relative_gain_vs_designated_fixed": value[
                    "relative_gain_vs_designated_fixed"
                ],
                "throughput_equivalent_time_saving": value[
                    "throughput_equivalent_time_saving"
                ],
                "oracle_agreement": value["oracle"]["oracle_agreement"],
                "mean_local_regret": value["oracle"]["mean_local_regret"],
                "l8_rate": value["summary"]["macro"]["l8_rate"],
                "l16_rate": value["summary"]["macro"]["l16_rate"],
                "l32_rate": value["summary"]["macro"]["l32_rate"],
            }
            for key, value in candidate["profiles"].items()
        },
    }


def candidate_sort_key(candidate: Mapping[str, Any]) -> Tuple[Any, ...]:
    family_preference = {"local_ratio": 0, "direct_rank": 1, "global_fractional": 2}
    return (
        -float(candidate["mean_gain_across_c"]),
        -float(candidate["min_gain_across_c"]),
        int(candidate["signal_count"]),
        family_preference[candidate["family"]],
        str(candidate["signal"]),
    )


def publish_progress(
    output_dir: Path, run_dir: Optional[Path], payload: Mapping[str, Any]
) -> None:
    atomic_json(output_dir / "progress.json", payload)
    if run_dir is not None:
        try:
            from reporting import render

            render(run_dir)
        except Exception as exc:
            print(f"[报告更新警告] {exc}", file=sys.stderr, flush=True)


def evaluate_candidate(
    args: argparse.Namespace,
    data: TraceData,
    spec: Mapping[str, Any],
    use_censor: bool,
    family: str,
    all_costs: Mapping[int, Mapping[int, float]],
    baselines: Mapping[int, Mapping[str, Any]],
    cached_values: Optional[np.ndarray],
) -> Dict[str, Any]:
    profiles: Dict[str, Any] = {}
    for concurrency in args.concurrencies:
        if family == "direct_rank":
            actions = oof_rank_actions(
                data,
                spec,
                use_censor,
                concurrency,
                all_costs,
                args.cv_folds,
                args.signal_bins,
            )
            profile = summarize_candidate_profile(
                data, actions, concurrency, all_costs, baselines[concurrency]
            )
        else:
            assert cached_values is not None
            if family == "local_ratio":
                actions = actions_from_values(
                    cached_values,
                    family,
                    all_costs[concurrency],
                    data.round == 0,
                    cold_start_block(concurrency),
                )
                profile = summarize_candidate_profile(
                    data, actions, concurrency, all_costs, baselines[concurrency]
                )
            else:
                rho, actions, profile, diagnostics = optimize_rho(
                    data,
                    cached_values,
                    concurrency,
                    all_costs,
                    baselines[concurrency],
                    args.rho_grid_size,
                )
                profile["rho"] = rho
                profile["rho_diagnostics"] = diagnostics
        profiles[str(concurrency)] = profile
    gains = [
        float(profile["relative_gain_vs_designated_fixed"])
        for profile in profiles.values()
    ]
    regrets = [
        float(profile["oracle"]["mean_local_regret"])
        for profile in profiles.values()
    ]
    return {
        "family": family,
        "signal": spec["name"],
        "feature": spec["feature"],
        "orientation": spec["orientation"],
        "kind": spec["kind"],
        "signal_zh": spec["zh"],
        "use_censor_state": use_censor,
        "signal_count": 2 if use_censor else 1,
        "mean_gain_across_c": float(np.mean(gains)),
        "min_gain_across_c": float(np.min(gains)),
        "mean_regret_across_c": float(np.mean(regrets)),
        "profiles": profiles,
    }


def build_frozen_policy(
    args: argparse.Namespace,
    data: TraceData,
    candidate: Mapping[str, Any],
    all_costs: Mapping[int, Mapping[int, float]],
) -> Dict[str, Any]:
    spec = SIGNAL_BY_NAME[str(candidate["signal"])]
    use_censor = bool(candidate["use_censor_state"])
    historical = np.flatnonzero(data.round > 0)
    family = str(candidate["family"])
    policy: Dict[str, Any] = {
        "schema_version": 1,
        "policy_type": "b200_verify_efficiency",
        "family": family,
        "family_zh": FAMILY_ZH[family],
        "model_size": args.model_size,
        "blocks": list(BLOCKS),
        "signal_count": 2 if use_censor else 1,
        "signal": {
            "name": spec["name"],
            "feature": spec["feature"],
            "orientation": spec["orientation"],
            "kind": spec["kind"],
            "zh": spec["zh"],
            "source": "completed causal verify rounds only",
        },
        "secondary_signal": (
            {
                "name": "previous_block_and_full",
                "features": ["prev_block", "prev_full"],
                "zh": "上一轮块长与是否整块通过（截断状态）",
                "source": "previous action metadata plus verifier outcome",
            }
            if use_censor
            else None
        ),
        "concurrency_profiles": {},
        "primary_objective": (
            "sample token/ms within dataset, eight datasets equal, seven "
            "concurrencies equal; every request charged full nominal-C latency"
        ),
        "excluded_datasets": ["aime24", "mmlu"],
        "training_rows": int(len(historical)),
    }
    if family in {"local_ratio", "global_fractional"}:
        policy["value_models"] = fit_value_models(
            data,
            historical,
            orientations(data, spec),
            use_censor,
            args.signal_bins,
        )
    for concurrency in args.concurrencies:
        source_profile = candidate["profiles"][str(concurrency)]
        profile: Dict[str, Any] = {
            "cold_start_block": cold_start_block(concurrency),
            "costs_ms": {
                str(block): all_costs[concurrency][block] for block in BLOCKS
            },
            "offline_score": source_profile["score"],
            "offline_gain_vs_designated_fixed": source_profile[
                "relative_gain_vs_designated_fixed"
            ],
        }
        if family == "global_fractional":
            profile["rho"] = float(source_profile["rho"])
        elif family == "direct_rank":
            profile["rank_tables"] = fit_rank_tables(
                data,
                historical,
                spec,
                use_censor,
                concurrency,
                all_costs,
                args.signal_bins,
            )
        policy["concurrency_profiles"][str(concurrency)] = profile
    return policy


def search(args: argparse.Namespace, data: TraceData) -> Dict[str, Any]:
    all_costs = parse_cost_document(args.cost_document)
    baselines = {
        concurrency: fixed_summaries(data, concurrency, all_costs)
        for concurrency in args.concurrencies
    }
    candidate_specs = [
        (spec, use_censor) for spec in SIGNALS for use_censor in (False, True)
    ]
    rankings: Dict[str, List[Dict[str, Any]]] = {family: [] for family in FAMILIES}
    total = len(candidate_specs) * len(FAMILIES)
    progress = ProgressBar("三策略族×verify信号", total)
    completed = 0
    for spec, use_censor in candidate_specs:
        try:
            values = oof_values(
                data, spec, use_censor, args.cv_folds, args.signal_bins
            )
        except RuntimeError as exc:
            for family in FAMILIES:
                completed += 1
                publish_progress(
                    args.output_dir,
                    args.run_dir,
                    {
                        "stage": "offline_search",
                        "completed": completed,
                        "total": total,
                        "latest": {
                            "family": family,
                            "signal": spec["name"],
                            "use_censor_state": use_censor,
                            "status": "invalid",
                            "reason": str(exc),
                        },
                    },
                )
                progress.update()
            continue
        for family in FAMILIES:
            try:
                candidate = evaluate_candidate(
                    args,
                    data,
                    spec,
                    use_censor,
                    family,
                    all_costs,
                    baselines,
                    values,
                )
                rankings[family].append(candidate)
                latest: Dict[str, Any] = candidate_compact(candidate)
            except RuntimeError as exc:
                latest = {
                    "family": family,
                    "signal": spec["name"],
                    "use_censor_state": use_censor,
                    "status": "invalid",
                    "reason": str(exc),
                }
            completed += 1
            best_so_far = [
                min(rows, key=candidate_sort_key)
                for rows in rankings.values()
                if rows
            ]
            publish_progress(
                args.output_dir,
                args.run_dir,
                {
                    "stage": "offline_search",
                    "completed": completed,
                    "total": total,
                    "latest": latest,
                    "best": (
                        candidate_compact(min(best_so_far, key=candidate_sort_key))
                        if best_so_far
                        else None
                    ),
                },
            )
            progress.update()
        del values
    progress.close()
    best_by_family: Dict[str, Dict[str, Any]] = {}
    for family, rows in rankings.items():
        if not rows:
            raise RuntimeError(f"no valid candidate for {family}")
        rows.sort(key=candidate_sort_key)
        best_by_family[family] = rows[0]
    winner = min(best_by_family.values(), key=candidate_sort_key)

    policies: Dict[str, Dict[str, Any]] = {}
    for family, candidate in best_by_family.items():
        policy = build_frozen_policy(args, data, candidate, all_costs)
        policies[family] = policy
        atomic_json(args.output_dir / f"policy_{family}.json", {"policy": policy})
    winner_policy = policies[str(winner["family"])]
    atomic_json(args.output_dir / "policy_winner.json", {"policy": winner_policy})
    payload = {
        "protocol": {
            "model_size": args.model_size,
            "datasets": data.dataset_names,
            "excluded_datasets": ["aime24", "mmlu"],
            "requests_by_dataset": data.request_counts,
            "rows": data.size,
            "concurrencies": args.concurrencies,
            "concurrency_weighting": "seven concurrencies equal",
            "dataset_weighting": "samples equal within dataset; eight datasets equal",
            "cost_scope": "full nominal C for every request; no tail-C correction",
            "cv_folds": args.cv_folds,
            "signal_bins": args.signal_bins,
            "rho_grid_size": args.rho_grid_size,
            "trace_quality": data.quality,
            "cost_snapshot": snapshot(args.cost_document, args.concurrencies),
        },
        "best_by_family": {
            family: candidate_compact(candidate)
            for family, candidate in best_by_family.items()
        },
        "winner": candidate_compact(winner),
        "offline_winner_profiles": {
            key: {
                name: value
                for name, value in profile.items()
                if name != "rho_diagnostics"
            }
            for key, profile in winner["profiles"].items()
        },
        "rankings": {
            family: [candidate_compact(row) for row in rows[: args.report_top]]
            for family, rows in rankings.items()
        },
        "fixed_baselines": {
            str(concurrency): value for concurrency, value in baselines.items()
        },
        "policy_files": {
            family: str((args.output_dir / f"policy_{family}.json").resolve())
            for family in FAMILIES
        }
        | {"winner": str((args.output_dir / "policy_winner.json").resolve())},
    }
    atomic_json(args.output_dir / "search_results.json", payload)
    publish_progress(
        args.output_dir,
        args.run_dir,
        {
            "stage": "offline_search_completed",
            "completed": total,
            "total": total,
            "winner": candidate_compact(winner),
        },
    )
    return payload


def collect_official_metrics(eval_root: Optional[Path]) -> Dict[str, Any]:
    if eval_root is None or not eval_root.exists():
        return {"datasets": {}}
    latest: Dict[str, Tuple[float, Dict[str, Any], str]] = {}
    for path in eval_root.rglob("sglang_metrics_summary.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        dataset = str(payload.get("benchmark", ""))
        if not dataset:
            continue
        if dataset not in latest or path.stat().st_mtime > latest[dataset][0]:
            latest[dataset] = (path.stat().st_mtime, payload, str(path))
    result = {}
    for dataset, (_mtime, payload, path) in latest.items():
        decode = payload.get("decode") or {}
        result[dataset] = {
            "decode_tpf": decode.get("tokens_per_forward_pass"),
            "decode_forward_passes": decode.get("decode_forward_passes"),
            "decode_tokens": decode.get("decode_tokens"),
            "source": path,
        }
    return {"datasets": result}


def validate(args: argparse.Namespace, data: TraceData) -> Dict[str, Any]:
    payload = json.loads(args.policy.read_text(encoding="utf-8"))
    policy = payload.get("policy", payload)
    if policy.get("policy_type") != "b200_verify_efficiency":
        raise RuntimeError("policy type mismatch")
    if policy.get("model_size") != args.model_size:
        raise RuntimeError("policy/trace model-size mismatch")
    if args.concurrency is None:
        raise RuntimeError("--concurrency is required")
    actions = np.empty(data.size, dtype=np.uint8)
    mismatches = []
    for index in range(data.size):
        features = {name: data.features[name][index] for name in FEATURES}
        features["history_rounds"] = int(data.round[index])
        expected, _scores = runtime_choose_action(policy, features, args.concurrency)
        actions[index] = expected
        actual = int(data.decision[index])
        if expected != actual and len(mismatches) < 20:
            mismatches.append(
                {
                    "row": index,
                    "request": int(data.request[index]),
                    "round": int(data.round[index]),
                    "expected": expected,
                    "actual": actual,
                }
            )
    mismatch_count = int(np.sum(actions != data.decision))
    if mismatch_count:
        raise RuntimeError(
            f"frozen runtime replay mismatch {mismatch_count}/{data.size}: {mismatches[:3]}"
        )
    costs = parse_cost_document(args.cost_document)
    dynamic = sample_summary(data, actions, args.concurrency, costs)
    baselines = fixed_summaries(data, args.concurrency, costs)
    fixed_block = baseline_block(args.concurrency)
    fixed_score = float(baselines[str(fixed_block)]["macro"]["score_token_ms_req"])
    score = float(dynamic["macro"]["score_token_ms_req"])
    for dataset, row in dynamic["datasets"].items():
        fixed_dataset = baselines[str(fixed_block)]["datasets"][dataset]
        dataset_score = float(row["score_token_ms_req"])
        dataset_fixed_score = float(fixed_dataset["score_token_ms_req"])
        row["baseline_block"] = fixed_block
        row["baseline_score"] = dataset_fixed_score
        row["relative_gain_vs_designated_fixed"] = (
            dataset_score / dataset_fixed_score - 1.0
        )
        row["throughput_equivalent_time_saving"] = (
            1.0 - dataset_fixed_score / dataset_score
        )
    dynamic["macro"].update(
        {
            "baseline_block": fixed_block,
            "baseline_score": fixed_score,
            "relative_gain_vs_designated_fixed": score / fixed_score - 1.0,
            "throughput_equivalent_time_saving": 1.0 - fixed_score / score,
            **local_oracle_metrics(data, actions, args.concurrency, costs),
        }
    )
    result = {
        "protocol": {
            "model_size": args.model_size,
            "concurrency": args.concurrency,
            "datasets": data.dataset_names,
            "dataset_count": len(data.dataset_names),
            "requests_by_dataset": data.request_counts,
            "rows": data.size,
            "formal_complete": not args.allow_partial_datasets,
            "policy": str(args.policy),
            "policy_family": policy["family"],
            "policy_replay_mismatches": mismatch_count,
            "trace_quality": data.quality,
            "cost_scope": "full nominal C for every request",
        },
        "dynamic_policy": dynamic,
        "fixed_baselines_same_dynamic_state": baselines,
        "official_sglang_metrics": collect_official_metrics(args.eval_root),
    }
    family = str(policy["family"])
    atomic_json(
        args.output_dir / f"validation_{family}_c{args.concurrency}.json", result
    )
    publish_progress(
        args.output_dir,
        args.run_dir,
        {
            "stage": f"validation_c{args.concurrency}_updated",
            "concurrency": args.concurrency,
            "datasets": len(data.dataset_names),
            "total_datasets": len(EXPECTED_DATASETS),
        },
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("search", "validate", "costs"), required=True)
    parser.add_argument("--trace-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--eval-root", type=Path)
    parser.add_argument("--cost-document", type=Path, required=True)
    parser.add_argument("--model-size", choices=("8b", "14b"), default="8b")
    parser.add_argument(
        "--concurrencies", type=parse_int_list, default=list(DEFAULT_CONCURRENCIES)
    )
    parser.add_argument("--concurrency", type=int)
    parser.add_argument("--split-seed", type=int, default=20260912)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--signal-bins", type=int, default=24)
    parser.add_argument("--rho-grid-size", type=int, default=33)
    parser.add_argument("--report-top", type=int, default=20)
    parser.add_argument("--allow-partial-datasets", action="store_true")
    parser.add_argument("--max-rows-per-dataset", type=int, default=0)
    parser.add_argument("--max-invalid-row-rate", type=float, default=0.05)
    args = parser.parse_args()
    if args.cv_folds < 2 or args.signal_bins < 2 or args.rho_grid_size < 3:
        parser.error("cv folds/bins must be >=2 and rho grid >=3")
    if not set(args.concurrencies).issubset(DEFAULT_CONCURRENCIES):
        parser.error("unsupported concurrency")
    if not 0.0 <= args.max_invalid_row_rate <= 1.0:
        parser.error("--max-invalid-row-rate must be in [0,1]")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(
        args.output_dir / "b200_latency_costs.json",
        snapshot(args.cost_document, args.concurrencies),
    )
    if args.mode == "costs":
        return
    if args.trace_root is None:
        parser.error("--trace-root is required")
    data = read_trace(
        args.trace_root,
        args.split_seed,
        args.cv_folds,
        args.model_size,
        args.allow_partial_datasets,
        args.max_rows_per_dataset,
        committed_only=args.mode == "validate",
        max_invalid_rate=args.max_invalid_row_rate,
    )
    if args.mode == "search":
        search(args, data)
    else:
        if args.policy is None or args.concurrency is None:
            parser.error("validate requires --policy and --concurrency")
        validate(args, data)


if __name__ == "__main__":
    main()
