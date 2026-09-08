#!/usr/bin/env python3
"""Search/validate one shared history signal against measured B200 latency."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from array import array
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


HERE = Path(__file__).resolve().parent
OBSERVATIONS = HERE.parent
if str(OBSERVATIONS) not in sys.path:
    sys.path.insert(0, str(OBSERVATIONS))

from sglang_b200_latency_dynamic_block_policy.signal_models import (  # noqa: E402
    BLOCKS,
    EXPECTED_DATASETS,
    EXPECTED_ORDER,
    FEATURES,
    SIGNALS,
    SIGNAL_BY_NAME,
    ProgressBar,
    TraceData,
    candidate_oof,
    collect_official_metrics,
    finite,
    fit_tables,
    fold_for,
    read_shadow_trace,
    selected_accept,
    subset_eight_datasets,
    summarize_transitions,
)

from sglang_b200_latency_dynamic_block_policy.latency_costs import (
    DEFAULT_CONCURRENCIES,
    atomic_json,
    cold_start_block,
    parse_cost_document,
    parse_int_list,
    snapshot,
)
from sglang_b200_latency_dynamic_block_policy.policy_runtime import (
    choose_action as runtime_choose_action,
)


def read_eight_shadow(args: argparse.Namespace) -> TraceData:
    data = read_shadow_trace(
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


def read_committed_trace(args: argparse.Namespace) -> TraceData:
    """Read canonical validation rows without treating shadow disagreement as fatal.

    Cross-block prefix disagreement invalidates a counterfactual comparison, but
    it does not invalidate the actually committed branch.  This is the audit
    distinction missing in the older validation command.
    """
    order = {name: index for index, name in enumerate(EXPECTED_ORDER)}
    paths = sorted(
        args.trace_root.glob("*.jsonl"),
        key=lambda path: (order.get(path.stem, len(order)), path.stem),
    )
    if not paths:
        raise RuntimeError(f"no trace jsonl files under {args.trace_root}")
    dataset_names: List[str] = []
    dataset_values = array("B")
    request_values = array("I")
    fold_values = array("B")
    round_values = array("I")
    decision_values = array("B")
    accept_values = array("B")
    feature_values = {name: array("f") for name in FEATURES}
    request_ids: Dict[Tuple[str, str], int] = {}
    request_folds: Dict[int, int] = {}
    request_seen: Dict[str, set[str]] = defaultdict(set)
    by_dataset: Dict[str, Dict[str, Any]] = {}
    total_original = total_excluded = replay_bad = cross_bad = 0
    for path in paths:
        dataset = path.stem
        if dataset in {"aime24", "mmlu"}:
            raise RuntimeError(f"excluded dataset appeared in validation: {dataset}")
        if dataset not in EXPECTED_DATASETS:
            raise RuntimeError(f"unexpected validation dataset: {dataset}")
        dataset_id = len(dataset_names)
        dataset_names.append(dataset)
        metrics = {
            "rows_original": 0,
            "rows_usable": 0,
            "rows_excluded": 0,
            "replay_mismatch": 0,
            "cross_block_mismatch": 0,
            "audit_mode": "committed_branch",
        }
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("event") != "sglang_dynamic_block_shadow_round":
                    continue
                if args.max_rows_per_dataset and metrics["rows_original"] >= args.max_rows_per_dataset:
                    break
                branches = record.get("branches") or {}
                decision = int(record.get("decision_block", 0))
                replay_mismatch = not bool(record.get("canonical_replay_match", False))
                cross_mismatch = not bool(record.get("cross_block_common_prefix_match", True))
                invalid = (
                    replay_mismatch
                    or decision not in BLOCKS
                    or str(decision) not in branches
                    or any(str(block) not in branches for block in BLOCKS)
                )
                metrics["rows_original"] += 1
                total_original += 1
                metrics["replay_mismatch"] += int(replay_mismatch)
                metrics["cross_block_mismatch"] += int(cross_mismatch)
                replay_bad += int(replay_mismatch)
                cross_bad += int(cross_mismatch)
                if invalid:
                    metrics["rows_excluded"] += 1
                    total_excluded += 1
                    continue
                fingerprint = str(
                    record.get("prompt_fingerprint") or record["request_id"]
                )
                request_seen[dataset].add(fingerprint)
                key = (dataset, fingerprint)
                request_id = request_ids.setdefault(key, len(request_ids))
                request_folds.setdefault(
                    request_id, fold_for(fingerprint, args.split_seed, args.cv_folds)
                )
                metrics["rows_usable"] += 1
                dataset_values.append(dataset_id)
                request_values.append(request_id)
                fold_values.append(request_folds[request_id])
                round_values.append(int(record["round_index"]))
                decision_values.append(decision)
                for block in BLOCKS:
                    accept_values.append(int(branches[str(block)]["accept_length"]))
                feature_map = record.get("history_before_round") or {}
                for name in FEATURES:
                    feature_values[name].append(finite(feature_map.get(name)))
        metrics["excluded_rate"] = metrics["rows_excluded"] / max(
            1, metrics["rows_original"]
        )
        by_dataset[dataset] = metrics
        print(
            f"[冻结trace读取] C{args.concurrency}/{dataset}: "
            f"原始={metrics['rows_original']} 可用={metrics['rows_usable']} "
            f"提交无效={metrics['rows_excluded']} shadow前缀异={metrics['cross_block_mismatch']}",
            flush=True,
        )
    present = set(dataset_names)
    if not args.allow_partial_datasets and present != EXPECTED_DATASETS:
        raise RuntimeError(
            "formal validation requires exactly eight datasets; "
            f"missing={sorted(EXPECTED_DATASETS-present)}, "
            f"extra={sorted(present-EXPECTED_DATASETS)}"
        )
    quality = {
        "audit_mode": "committed_branch",
        "rows_original": total_original,
        "rows_usable": total_original - total_excluded,
        "rows_excluded": total_excluded,
        "excluded_rate": total_excluded / max(1, total_original),
        "replay_mismatch": replay_bad,
        "cross_block_mismatch_diagnostic_only": cross_bad,
        "by_dataset": by_dataset,
    }
    if total_excluded:
        raise RuntimeError(
            f"committed validation trace contains {total_excluded} invalid canonical rows"
        )
    return TraceData(
        dataset_names=dataset_names,
        dataset=np.asarray(dataset_values, dtype=np.uint8),
        request=np.asarray(request_values, dtype=np.uint32),
        fold=np.asarray(fold_values, dtype=np.uint8),
        round=np.asarray(round_values, dtype=np.uint32),
        decision=np.asarray(decision_values, dtype=np.uint8),
        accept=np.asarray(accept_values, dtype=np.uint8).reshape(-1, 3),
        features={
            name: np.asarray(values, dtype=np.float32)
            for name, values in feature_values.items()
        },
        quality=quality,
        request_counts={name: len(values) for name, values in request_seen.items()},
    )


def effective_request_concurrency(data: TraceData, nominal: int) -> np.ndarray:
    """Assign nominal C, except requests in the final incomplete cohort use r."""
    if data.size == 0:
        return np.zeros(0, dtype=np.uint16)
    output = np.full(int(data.request.max()) + 1, nominal, dtype=np.uint16)
    for dataset_id in range(len(data.dataset_names)):
        requests = np.unique(data.request[data.dataset == dataset_id])
        remainder = len(requests) % nominal
        if remainder:
            output[requests[-remainder:]] = remainder
    return output


def actions_from_values(
    values: np.ndarray,
    lam: float,
    first_round: np.ndarray,
    nominal_costs: Mapping[int, float],
    initial_block: int,
) -> np.ndarray:
    costs = np.asarray([nominal_costs[block] for block in BLOCKS], dtype=np.float64)
    utilities = values - lam * costs
    actions = np.full(len(values), BLOCKS[0], dtype=np.uint8)
    best = utilities[:, 0].copy()
    for column, block in enumerate(BLOCKS[1:], 1):
        better = utilities[:, column] > best + 1e-12
        actions[better] = block
        best[better] = utilities[better, column]
    if initial_block not in BLOCKS:
        raise ValueError(f"invalid initial block: {initial_block}")
    actions[first_round] = initial_block
    return actions


def exact_lambda_candidates(
    values: np.ndarray,
    first_round: np.ndarray,
    nominal_costs: Mapping[int, float],
    lambda_max: float,
) -> List[Dict[str, float]]:
    unique = np.unique(np.round(values[~first_round], 12), axis=0)
    costs = np.asarray([nominal_costs[block] for block in BLOCKS], dtype=np.float64)
    roots = {0.0, float(lambda_max)}
    for left in range(3):
        for right in range(left + 1, 3):
            denominator = costs[right] - costs[left]
            if abs(denominator) <= 1e-12:
                continue
            candidates = (unique[:, right] - unique[:, left]) / denominator
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


def sample_summary(
    data: TraceData,
    actions: np.ndarray,
    nominal_concurrency: int,
    all_costs: Mapping[int, Mapping[int, float]],
) -> Dict[str, Any]:
    """Sample macro inside each dataset, then equal macro across datasets."""
    if len(actions) != data.size:
        raise ValueError("actions length does not match trace")
    accepted = selected_accept(data, actions)
    request_concurrency = effective_request_concurrency(data, nominal_concurrency)
    row_concurrency = request_concurrency[data.request]
    cost_matrix = np.zeros((129, 3), dtype=np.float64)
    for concurrency, costs in all_costs.items():
        cost_matrix[int(concurrency)] = [costs[block] for block in BLOCKS]
    action_columns = np.where(actions == 8, 0, np.where(actions == 16, 1, 2))
    row_cost = cost_matrix[row_concurrency, action_columns]
    request_count = int(data.request.max()) + 1
    rounds_by_request = np.bincount(data.request, minlength=request_count).astype(np.float64)
    accept_by_request = np.bincount(
        data.request, weights=accepted, minlength=request_count
    )
    time_by_request = np.bincount(
        data.request, weights=row_cost, minlength=request_count
    )
    l8_by_request = np.bincount(
        data.request, weights=(actions == 8), minlength=request_count
    )
    l16_by_request = np.bincount(
        data.request, weights=(actions == 16), minlength=request_count
    )
    l32_by_request = np.bincount(
        data.request, weights=(actions == 32), minlength=request_count
    )
    request_dataset = np.full(request_count, -1, dtype=np.int16)
    request_dataset[data.request] = data.dataset.astype(np.int16)
    by_dataset: Dict[str, Any] = {}
    for dataset_id, name in enumerate(data.dataset_names):
        requests = np.flatnonzero(
            (request_dataset == dataset_id) & (rounds_by_request > 0)
        )
        rounds = rounds_by_request[requests]
        accept_sum = accept_by_request[requests]
        time_sum = time_by_request[requests]
        effective_c = request_concurrency[requests].astype(np.float64)
        token_ms_req = accept_sum / time_sum
        by_dataset[name] = {
            "samples": int(len(requests)),
            "rounds": int(rounds.sum()),
            "score_token_ms_req": float(token_ms_req.mean()),
            "pure_forward_token_ms_batch": float((effective_c * token_ms_req).mean()),
            "decode_tpf": float((accept_sum / (2.0 * rounds)).mean()),
            "mean_accept": float((accept_sum / rounds).mean()),
            "mean_forward_ms": float((time_sum / rounds).mean()),
            "l8_rate": float((l8_by_request[requests] / rounds).mean()),
            "l16_rate": float((l16_by_request[requests] / rounds).mean()),
            "l32_rate": float((l32_by_request[requests] / rounds).mean()),
            "tail_samples": int(np.sum(effective_c != nominal_concurrency)),
        }
    metric_names = (
        "score_token_ms_req",
        "pure_forward_token_ms_batch",
        "decode_tpf",
        "mean_accept",
        "mean_forward_ms",
        "l8_rate",
        "l16_rate",
        "l32_rate",
    )
    macro = {
        key: float(np.mean([row[key] for row in by_dataset.values()]))
        for key in metric_names
    }
    macro.update(
        {
            "datasets": len(by_dataset),
            "samples": sum(row["samples"] for row in by_dataset.values()),
            "rounds": sum(row["rounds"] for row in by_dataset.values()),
            "nominal_concurrency": nominal_concurrency,
            "aggregation": "sample mean within dataset, then equal dataset mean",
        }
    )
    return {"datasets": by_dataset, "macro": macro}


def fixed_baselines(
    data: TraceData,
    concurrency: int,
    all_costs: Mapping[int, Mapping[int, float]],
) -> Dict[str, Any]:
    return {
        str(block): sample_summary(
            data, np.full(data.size, block, dtype=np.uint8), concurrency, all_costs
        )
        for block in BLOCKS
    }


class IncrementalLambdaEvaluator:
    """Evaluate every exact lambda while touching each signal-bin cell O(1) times.

    For fixed predicted values and monotonically increasing lambda, the chosen
    latency can only move from a slower block to a faster block.  We aggregate
    historical rounds by (signal-value group, request), then update only groups
    whose action changes at the next exact breakpoint.  The resulting score is
    exactly the same sample-macro objective as :func:`sample_summary`.
    """

    def __init__(
        self,
        data: TraceData,
        values: np.ndarray,
        nominal_concurrency: int,
        all_costs: Mapping[int, Mapping[int, float]],
    ) -> None:
        self.data = data
        self.nominal_concurrency = nominal_concurrency
        self.nominal_costs = all_costs[nominal_concurrency]
        self.initial_block = cold_start_block(nominal_concurrency)
        self.initial_column = BLOCKS.index(self.initial_block)
        self.request_count = int(data.request.max()) + 1
        self.request_concurrency = effective_request_concurrency(
            data, nominal_concurrency
        )
        self.request_dataset = np.full(self.request_count, -1, dtype=np.int16)
        self.request_dataset[data.request] = data.dataset.astype(np.int16)
        self.valid_requests = np.flatnonzero(self.request_dataset >= 0)
        self.dataset_request_counts = np.bincount(
            self.request_dataset[self.valid_requests],
            minlength=len(data.dataset_names),
        ).astype(np.float64)

        first = data.round == 0
        historical = ~first
        historical_values = np.round(values[historical].astype(np.float64), 12)
        self.group_values, row_group = np.unique(
            historical_values, axis=0, return_inverse=True
        )
        historical_requests = data.request[historical].astype(np.int64)
        compound = row_group.astype(np.int64) * self.request_count + historical_requests
        unique_cells, cell_inverse = np.unique(compound, return_inverse=True)
        self.cell_group = unique_cells // self.request_count
        self.cell_request = unique_cells % self.request_count
        self.cell_count = np.bincount(cell_inverse).astype(np.float64)
        self.cell_accept = np.column_stack(
            [
                np.bincount(
                    cell_inverse,
                    weights=data.accept[historical, column].astype(np.float64),
                    minlength=len(unique_cells),
                )
                for column in range(3)
            ]
        )
        self.cell_cost = np.empty((len(unique_cells), 3), dtype=np.float64)
        cell_concurrency = self.request_concurrency[self.cell_request]
        for column, block in enumerate(BLOCKS):
            per_request_cost = np.asarray(
                [all_costs[int(value)][block] for value in cell_concurrency],
                dtype=np.float64,
            )
            self.cell_cost[:, column] = self.cell_count * per_request_cost
        # unique_cells is sorted by group then request, so each group owns one
        # contiguous cell interval.
        self.group_starts = np.searchsorted(
            self.cell_group, np.arange(len(self.group_values)), side="left"
        )
        self.group_ends = np.searchsorted(
            self.cell_group, np.arange(len(self.group_values)), side="right"
        )

        self.fixed_accept = np.bincount(
            data.request[first],
            weights=data.accept[first, self.initial_column].astype(np.float64),
            minlength=self.request_count,
        )
        first_cost_weights = np.asarray(
            [
                all_costs[int(self.request_concurrency[int(request)])][
                    self.initial_block
                ]
                for request in data.request[first]
            ],
            dtype=np.float64,
        )
        self.fixed_cost = np.bincount(
            data.request[first],
            weights=first_cost_weights,
            minlength=self.request_count,
        )

    def _group_actions(self, lam: float) -> np.ndarray:
        return actions_from_values(
            self.group_values,
            lam,
            np.zeros(len(self.group_values), dtype=bool),
            self.nominal_costs,
            self.initial_block,
        )

    @staticmethod
    def _columns(actions: np.ndarray) -> np.ndarray:
        return np.where(actions == 8, 0, np.where(actions == 16, 1, 2))

    def scan(self, candidates: Sequence[Mapping[str, float]]) -> List[float]:
        if not candidates:
            return []
        group_actions = self._group_actions(float(candidates[0]["lambda"]))
        cell_columns = self._columns(group_actions[self.cell_group])
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
        dataset_ratio_sum = np.bincount(
            self.request_dataset[self.valid_requests],
            weights=ratios[self.valid_requests],
            minlength=len(self.data.dataset_names),
        )

        def macro() -> float:
            return float(np.mean(dataset_ratio_sum / self.dataset_request_counts))

        scores = [macro()]
        for candidate in candidates[1:]:
            new_actions = self._group_actions(float(candidate["lambda"]))
            changed = np.flatnonzero(new_actions != group_actions)
            if len(changed):
                intervals = [
                    np.arange(self.group_starts[group], self.group_ends[group])
                    for group in changed
                ]
                cells = np.concatenate(intervals) if intervals else np.zeros(0, dtype=np.int64)
                affected = np.unique(self.cell_request[cells])
                old_ratios = ratios[affected].copy()
                old_columns = self._columns(group_actions[self.cell_group[cells]])
                new_columns = self._columns(new_actions[self.cell_group[cells]])
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
                delta = ratios[affected] - old_ratios
                dataset_ratio_sum += np.bincount(
                    self.request_dataset[affected],
                    weights=delta,
                    minlength=len(self.data.dataset_names),
                )
                group_actions = new_actions
            scores.append(macro())
        return scores


def optimize_lambda(
    data: TraceData,
    values: np.ndarray,
    concurrency: int,
    all_costs: Mapping[int, Mapping[int, float]],
    lambda_max: float,
    baselines: Mapping[str, Any],
) -> Dict[str, Any]:
    nominal_costs = all_costs[concurrency]
    candidates = exact_lambda_candidates(
        values, data.round == 0, nominal_costs, lambda_max
    )
    evaluator = IncrementalLambdaEvaluator(
        data, values, concurrency, all_costs
    )
    scores = evaluator.scan(candidates)
    best: Optional[Tuple[Tuple[float, float, float], Dict[str, Any]]] = None
    for candidate, score in zip(candidates, scores):
        # Mean latency is evaluated only for exact score ties; doing the full
        # sample summary here would defeat the incremental scan.
        key = (score, candidate["stable_width"], -candidate["lambda"])
        row = {**candidate, "score": score}
        if best is None or key > best[0]:
            best = (key, row)
    assert best is not None
    selected = best[1]
    selected_actions = actions_from_values(
        values,
        selected["lambda"],
        data.round == 0,
        nominal_costs,
        cold_start_block(concurrency),
    )
    selected["summary"] = sample_summary(
        data, selected_actions, concurrency, all_costs
    )
    baseline_scores = {
        block: result["macro"]["score_token_ms_req"]
        for block, result in baselines.items()
    }
    best_block, best_score = max(
        baseline_scores.items(), key=lambda item: (item[1], -int(item[0]))
    )
    selected["exact_candidates"] = len(candidates)
    selected["baseline_scores"] = baseline_scores
    selected["best_fixed_block"] = int(best_block)
    selected["best_fixed_score"] = best_score
    selected["relative_gain_vs_best_fixed"] = selected["score"] / best_score - 1.0
    selected["relative_gain_vs_fixed"] = {
        block: selected["score"] / score - 1.0
        for block, score in baseline_scores.items()
    }
    return selected


def compact_candidate(result: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "tier": result["tier"],
        "signal": result["signal"],
        "feature": result["feature"],
        "kind": result["kind"],
        "zh": result["zh"],
        "gate": result.get("gate"),
        "mean_relative_gain": result["mean_relative_gain"],
        "min_relative_gain": result["min_relative_gain"],
        "profiles": {
            key: {
                "lambda": value["lambda"],
                "lambda_interval": [value["interval_lo"], value["interval_hi"]],
                "score": value["score"],
                "best_fixed_block": value["best_fixed_block"],
                "best_fixed_score": value["best_fixed_score"],
                "relative_gain_vs_best_fixed": value["relative_gain_vs_best_fixed"],
                "l8_rate": value["summary"]["macro"]["l8_rate"],
                "l16_rate": value["summary"]["macro"]["l16_rate"],
                "l32_rate": value["summary"]["macro"]["l32_rate"],
            }
            for key, value in result["profiles"].items()
        },
    }


def candidate_result(
    args: argparse.Namespace,
    data: TraceData,
    spec: Mapping[str, Any],
    gate: Optional[int],
    all_costs: Mapping[int, Mapping[int, float]],
    baselines: Mapping[int, Mapping[str, Any]],
) -> Tuple[Dict[str, Any], np.ndarray]:
    values = candidate_oof(data, spec, gate, args.cv_folds, args.signal_bins)
    profiles = {
        str(concurrency): optimize_lambda(
            data,
            values,
            concurrency,
            all_costs,
            args.lambda_max,
            baselines[concurrency],
        )
        for concurrency in args.concurrencies
    }
    gains = [row["relative_gain_vs_best_fixed"] for row in profiles.values()]
    return (
        {
            "tier": 1 if gate is None else 2,
            "signal": spec["name"],
            "feature": spec["feature"],
            "kind": spec["kind"],
            "zh": spec["zh"],
            "gate": gate,
            "mean_relative_gain": float(np.mean(gains)),
            "min_relative_gain": float(np.min(gains)),
            "profiles": profiles,
        },
        values,
    )


def candidate_key(result: Mapping[str, Any]) -> Tuple[Any, ...]:
    return (
        -float(result["mean_relative_gain"]),
        -float(result["min_relative_gain"]),
        int(result["tier"]),
        str(result["signal"]),
        int(result.get("gate") or 0),
    )


def publish_progress(
    run_dir: Optional[Path], output_dir: Path, payload: Mapping[str, Any]
) -> None:
    stage = str(payload.get("stage", ""))
    progress_name = (
        f"{stage}_progress.json" if stage.startswith("validation_") else "progress.json"
    )
    atomic_json(output_dir / progress_name, payload)
    if run_dir is not None:
        try:
            from reporting import render

            render(run_dir)
        except Exception as exc:
            print(f"[报告更新警告] {exc}", file=sys.stderr, flush=True)


def search_policy(args: argparse.Namespace, data: TraceData) -> Dict[str, Any]:
    all_costs = parse_cost_document(args.cost_document)
    baselines = {
        concurrency: fixed_baselines(data, concurrency, all_costs)
        for concurrency in args.concurrencies
    }
    tier1: List[Dict[str, Any]] = []
    progress = ProgressBar(
        f"Tier1单信号×{len(args.concurrencies)}并发精确lambda", len(SIGNALS)
    )
    for index, spec in enumerate(SIGNALS, 1):
        result, _ = candidate_result(
            args, data, spec, None, all_costs, baselines
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
        spec
        for spec in SIGNALS
        if spec["kind"] == "accept" and spec["name"] != "full_streak"
    ]
    dual_candidates = [(spec, gate) for spec in dual_specs for gate in args.full_gate_grid]
    tier2: List[Dict[str, Any]] = []
    progress = ProgressBar(
        f"Tier2单接收信号+full门×{len(args.concurrencies)}并发",
        len(dual_candidates),
    )
    for index, (spec, gate) in enumerate(dual_candidates, 1):
        try:
            result, _ = candidate_result(
                args, data, spec, gate, all_costs, baselines
            )
            tier2.append(result)
            latest: Dict[str, Any] = compact_candidate(result)
        except RuntimeError as exc:
            latest = {
                "tier": 2,
                "signal": spec["name"],
                "gate": gate,
                "status": "invalid",
                "reason": str(exc),
            }
        ranking = sorted(tier2, key=candidate_key)
        publish_progress(
            args.run_dir,
            args.output_dir,
            {
                "stage": "tier2",
                "completed": index,
                "total": len(dual_candidates),
                "latest": latest,
                "best_single": compact_candidate(best_single),
                "best": compact_candidate(ranking[0]) if ranking else compact_candidate(best_single),
            },
        )
        progress.update()
    progress.close()
    tier2.sort(key=candidate_key)
    best_dual = tier2[0] if tier2 else None
    # Search both legal complexities against the same pure throughput objective.
    # Numerical ties prefer one signal, matching the user's simplicity rule.
    selected = best_single
    if best_dual is not None and candidate_key(best_dual) < candidate_key(best_single):
        if best_dual["mean_relative_gain"] > best_single["mean_relative_gain"] + 1e-12:
            selected = best_dual

    spec = SIGNAL_BY_NAME[selected["signal"]]
    values = candidate_oof(
        data, spec, selected.get("gate"), args.cv_folds, args.signal_bins
    )
    historical = np.flatnonzero(data.round > 0)
    oriented = data.features[spec["feature"]].astype(np.float64) * float(spec["orientation"])
    final_tables = fit_tables(
        data, historical, oriented, selected.get("gate"), args.signal_bins
    )
    profiles: Dict[str, Any] = {}
    offline_selected: Dict[str, Any] = {}
    for concurrency in args.concurrencies:
        row = selected["profiles"][str(concurrency)]
        profiles[str(concurrency)] = {
            "lambda": row["lambda"],
            "optimal_lambda_interval": [row["interval_lo"], row["interval_hi"]],
            "cold_start_block": cold_start_block(concurrency),
            "costs_ms": {
                str(block): all_costs[concurrency][block] for block in BLOCKS
            },
            "search_score_token_ms_req": row["score"],
            "best_fixed_block": row["best_fixed_block"],
            "best_fixed_score": row["best_fixed_score"],
            "relative_gain_vs_best_fixed": row["relative_gain_vs_best_fixed"],
        }
        offline_selected[str(concurrency)] = {
            **row,
            "cold_start_block": cold_start_block(concurrency),
            "transitions": summarize_transitions(
                data,
                actions_from_values(
                    values,
                    row["lambda"],
                    data.round == 0,
                    all_costs[concurrency],
                    cold_start_block(concurrency),
                ),
            ),
        }
    policy = {
        "schema_version": 2,
        "policy_type": "b200_latency_scalar_value",
        "blocks": list(BLOCKS),
        "cold_start_block_by_concurrency": {
            str(concurrency): cold_start_block(concurrency)
            for concurrency in args.concurrencies
        },
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
                "role": "global right-censor stratum",
            }
            if selected["tier"] == 2
            else None
        ),
        "value_models": final_tables,
        "concurrency_profiles": profiles,
        "lambda_semantics": "choose argmax_B E[A_B|signal] - lambda_C*T_C(B); larger lambda favors lower latency",
        "primary_objective": "mean over eight datasets of mean over samples of total accepted tokens / total B200 forward ms",
        "dataset_weighting": "eight datasets equal; samples equal inside a dataset",
        "virtual_concurrency": "cost/policy condition only; request actions remain independent",
        "training_rows": int(len(historical)),
    }
    atomic_json(args.output_dir / "policy.json", {"policy": policy})
    payload = {
        "protocol": {
            "datasets": data.dataset_names,
            "dataset_count": len(data.dataset_names),
            "excluded_datasets": ["aime24", "mmlu"],
            "requests_by_dataset": data.request_counts,
            "rows": data.size,
            "historical_rows": int(len(historical)),
            "concurrencies": args.concurrencies,
            "cold_start_block_by_concurrency": {
                str(concurrency): cold_start_block(concurrency)
                for concurrency in args.concurrencies
            },
            "cv_folds": args.cv_folds,
            "signal_bins": args.signal_bins,
            "lambda_search": "exact utility intersections and open-interval midpoints",
            "lambda_max": args.lambda_max,
            "candidate_attempts": {
                "tier1": len(SIGNALS),
                "tier2": len(dual_candidates),
                "total": len(SIGNALS) + len(dual_candidates),
            },
            "weighting": "sample equal inside dataset -> eight datasets equal",
            "tail_cost": "last incomplete virtual cohort uses measured T_r(B)",
            "trace_quality": data.quality,
            "cost_snapshot_sha256": snapshot(args.cost_document, args.concurrencies)["source_sha256"],
        },
        "tier1_ranking": [compact_candidate(row) for row in tier1],
        "tier2_ranking": [compact_candidate(row) for row in tier2],
        "selection": {
            "selected": compact_candidate(selected),
            "best_single": compact_candidate(best_single),
            "best_dual": compact_candidate(best_dual) if best_dual else None,
            "selection_order": "max equal-C mean relative gain vs each C's best fixed block; then max worst-C gain; exact tie prefers one signal",
        },
        "offline_selected": offline_selected,
        "fixed_baselines": {
            str(concurrency): row for concurrency, row in baselines.items()
        },
        "policy": policy,
    }
    atomic_json(args.output_dir / "search_results.json", payload)
    publish_progress(
        args.run_dir,
        args.output_dir,
        {
            "stage": "search_completed",
            "completed": len(SIGNALS) + len(dual_candidates),
            "total": len(SIGNALS) + len(dual_candidates),
            "selected": compact_candidate(selected),
        },
    )
    return payload


def replay_actions(
    data: TraceData, policy: Mapping[str, Any], concurrency: int
) -> Dict[str, Any]:
    mismatches = 0
    examples = []
    for index in range(data.size):
        features = {name: data.features[name][index] for name in FEATURES}
        if int(data.round[index]) == 0:
            expected = int(
                policy["concurrency_profiles"][str(concurrency)][
                    "cold_start_block"
                ]
            )
        else:
            expected, _ = runtime_choose_action(policy, features, concurrency)
        actual = int(data.decision[index])
        if expected != actual:
            mismatches += 1
            if len(examples) < 10:
                examples.append(
                    {
                        "row": index,
                        "request": int(data.request[index]),
                        "round": int(data.round[index]),
                        "expected": expected,
                        "actual": actual,
                    }
                )
    return {"rows": data.size, "mismatches": mismatches, "examples": examples}


def validate_policy(args: argparse.Namespace, data: TraceData) -> Dict[str, Any]:
    payload = json.loads(args.policy.read_text(encoding="utf-8"))
    policy = payload.get("policy", payload)
    if policy.get("policy_type") != "b200_latency_scalar_value":
        raise RuntimeError("validation policy type mismatch")
    if policy.get("schema_version") != 2:
        raise RuntimeError(
            "validation policy predates concurrency-specific cold start; "
            "rerun offline search"
        )
    if args.concurrency is None:
        raise RuntimeError("--concurrency is required for validation")
    all_costs = parse_cost_document(args.cost_document)
    audit = replay_actions(data, policy, args.concurrency)
    if audit["mismatches"]:
        raise RuntimeError(
            f"frozen policy replay mismatch: {audit['mismatches']}/{audit['rows']}"
        )
    actions = data.decision.astype(np.uint8)
    dynamic = sample_summary(data, actions, args.concurrency, all_costs)
    dynamic["transitions"] = summarize_transitions(data, actions)
    baselines = fixed_baselines(data, args.concurrency, all_costs)
    baseline_scores = {
        block: row["macro"]["score_token_ms_req"] for block, row in baselines.items()
    }
    best_block, best_score = max(
        baseline_scores.items(), key=lambda item: (item[1], -int(item[0]))
    )
    dynamic["macro"]["best_same_state_fixed_block"] = int(best_block)
    dynamic["macro"]["best_same_state_fixed_score"] = best_score
    dynamic["macro"]["relative_gain_vs_best_same_state_fixed"] = (
        dynamic["macro"]["score_token_ms_req"] / best_score - 1.0
    )
    result = {
        "protocol": {
            "concurrency": args.concurrency,
            "datasets": data.dataset_names,
            "dataset_count": len(data.dataset_names),
            "requests_by_dataset": data.request_counts,
            "rows": data.size,
            "formal_complete": not args.allow_partial_datasets,
            "trace_quality": data.quality,
            "policy_path": str(args.policy),
            "policy_schema_version": policy["schema_version"],
            "policy_replay": audit,
            "virtual_concurrency": "cost lookup only; actions are independent per request",
            "cold_start_block": int(
                policy["concurrency_profiles"][str(args.concurrency)][
                    "cold_start_block"
                ]
            ),
        },
        "dynamic_policy": dynamic,
        "fixed_baselines_same_dynamic_state": baselines,
        "official_sglang_metrics": collect_official_metrics(args.eval_root),
    }
    atomic_json(args.output_dir / f"validation_c{args.concurrency}.json", result)
    publish_progress(
        args.run_dir,
        args.output_dir,
        {
            "stage": f"validation_c{args.concurrency}_completed"
            if not args.allow_partial_datasets
            else f"validation_c{args.concurrency}_partial",
            "completed": len(data.dataset_names),
            "total": len(EXPECTED_DATASETS),
            "concurrency": args.concurrency,
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
    parser.add_argument(
        "--concurrencies",
        type=parse_int_list,
        default=list(DEFAULT_CONCURRENCIES),
    )
    parser.add_argument("--concurrency", type=int)
    parser.add_argument("--split-seed", type=int, default=20260908)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--signal-bins", type=int, default=32)
    parser.add_argument("--lambda-max", type=float, default=16.0)
    parser.add_argument(
        "--full-gate-grid", type=parse_int_list, default=parse_int_list("1,2,3")
    )
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
    if not set(args.concurrencies).issubset(DEFAULT_CONCURRENCIES):
        parser.error(
            "--concurrencies must be a subset of "
            + ",".join(str(value) for value in DEFAULT_CONCURRENCIES)
        )
    if not 0 <= args.max_invalid_row_rate <= 1:
        parser.error("--max-invalid-row-rate must be in [0,1]")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cost_snapshot = snapshot(args.cost_document, args.concurrencies)
    atomic_json(args.output_dir / "b200_latency_costs.json", cost_snapshot)
    if args.mode == "costs":
        return
    if args.trace_root is None:
        parser.error("--trace-root is required")
    if args.mode == "search":
        data = read_eight_shadow(args)
        search_policy(args, data)
    else:
        if args.policy is None or args.concurrency is None:
            parser.error("--policy and --concurrency are required for validate")
        data = read_committed_trace(args)
        validate_policy(args, data)


if __name__ == "__main__":
    main()
