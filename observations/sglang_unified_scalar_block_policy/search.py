#!/usr/bin/env python3
"""Nine-dataset-equal search for one unified scalar L8/L16/L32 policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from array import array
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


BLOCKS = (8, 16, 32)
EXPECTED_DATASETS = {
    "gsm8k", "human-eval", "mbpp", "math-500", "aime25", "gpqa",
    "ifeval", "livecodebench-cpp", "mmlu",
}

# Every tier-1 candidate is exactly one recorded scalar.  No candidate mixes
# acceptance, confidence, margin, entropy, or the current block.
SIGNALS: Tuple[Dict[str, Any], ...] = (
    {"name": "accept_last", "feature": "prev_accept", "orientation": 1, "kind": "accept", "zh": "上一轮接收长度"},
    {"name": "accept_ma2", "feature": "a_ma2", "orientation": 1, "kind": "accept", "zh": "近2轮接收均值"},
    {"name": "accept_ma4", "feature": "a_ma4", "orientation": 1, "kind": "accept", "zh": "近4轮接收均值"},
    {"name": "accept_ma8", "feature": "a_ma8", "orientation": 1, "kind": "accept", "zh": "近8轮接收均值"},
    {"name": "ratio_last", "feature": "prev_accept_ratio", "orientation": 1, "kind": "accept", "zh": "上一轮接收比例"},
    {"name": "ratio_ma2", "feature": "ratio_ma2", "orientation": 1, "kind": "accept", "zh": "近2轮接收比例均值"},
    {"name": "ratio_ma4", "feature": "ratio_ma4", "orientation": 1, "kind": "accept", "zh": "近4轮接收比例均值"},
    {"name": "ratio_ma8", "feature": "ratio_ma8", "orientation": 1, "kind": "accept", "zh": "近8轮接收比例均值"},
    {"name": "full_streak", "feature": "full_streak", "orientation": 1, "kind": "accept", "zh": "连续整块通过轮数"},
    {"name": "nonfull_streak", "feature": "nonfull_streak", "orientation": -1, "kind": "accept", "zh": "连续未整块通过轮数的相反数"},
    {"name": "head_conf_last", "feature": "prev_head_conf", "orientation": 1, "kind": "confidence", "zh": "上一轮头部confidence均值"},
    {"name": "head_conf_ma2", "feature": "head_conf_ma2", "orientation": 1, "kind": "confidence", "zh": "近2轮头部confidence均值"},
    {"name": "head_conf_ma4", "feature": "head_conf_ma4", "orientation": 1, "kind": "confidence", "zh": "近4轮头部confidence均值"},
    {"name": "head_conf_ma8", "feature": "head_conf_ma8", "orientation": 1, "kind": "confidence", "zh": "近8轮头部confidence均值"},
    {"name": "head_margin_last", "feature": "prev_head_margin", "orientation": 1, "kind": "margin", "zh": "上一轮头部margin均值"},
    {"name": "head_margin_ma2", "feature": "head_margin_ma2", "orientation": 1, "kind": "margin", "zh": "近2轮头部margin均值"},
    {"name": "head_margin_ma4", "feature": "head_margin_ma4", "orientation": 1, "kind": "margin", "zh": "近4轮头部margin均值"},
    {"name": "head_margin_ma8", "feature": "head_margin_ma8", "orientation": 1, "kind": "margin", "zh": "近8轮头部margin均值"},
    {"name": "head_entropy_last", "feature": "prev_head_entropy", "orientation": -1, "kind": "entropy", "zh": "上一轮头部entropy的相反数"},
    {"name": "head_entropy_ma2", "feature": "head_entropy_ma2", "orientation": -1, "kind": "entropy", "zh": "近2轮头部entropy均值的相反数"},
    {"name": "head_entropy_ma4", "feature": "head_entropy_ma4", "orientation": -1, "kind": "entropy", "zh": "近4轮头部entropy均值的相反数"},
    {"name": "head_entropy_ma8", "feature": "head_entropy_ma8", "orientation": -1, "kind": "entropy", "zh": "近8轮头部entropy均值的相反数"},
    {"name": "reject_conf_last", "feature": "prev_rejected_conf", "orientation": 1, "kind": "confidence", "zh": "上一轮首个拒绝位confidence"},
    {"name": "reject_margin_last", "feature": "prev_rejected_margin", "orientation": 1, "kind": "margin", "zh": "上一轮首个拒绝位margin"},
)
SIGNAL_BY_NAME = {item["name"]: item for item in SIGNALS}
FEATURES = tuple(dict.fromkeys(item["feature"] for item in SIGNALS))


class ProgressBar:
    def __init__(self, label: str, total: int) -> None:
        self.label = label
        self.total = max(1, int(total))
        self.current = 0
        self.started = time.monotonic()
        self.tty = sys.stderr.isatty()
        self.last_percent = -1
        self.render(True)

    def render(self, force: bool = False) -> None:
        ratio = min(1.0, self.current / self.total)
        percent = int(100 * ratio)
        if not force and not self.tty and percent == self.last_percent:
            return
        elapsed = max(time.monotonic() - self.started, 1e-9)
        rate = self.current / elapsed
        eta = (self.total - self.current) / rate if rate else math.inf
        eta_text = "--:--" if not math.isfinite(eta) else f"{int(eta)//60:02d}:{int(eta)%60:02d}"
        filled = int(30 * ratio)
        text = (
            f"[{self.label}] |{'#' * filled}{'-' * (30-filled)}| "
            f"{self.current}/{self.total} ({100*ratio:5.1f}%) ETA {eta_text}"
        )
        if self.tty:
            sys.stderr.write("\r" + text + "\033[K")
            sys.stderr.flush()
        else:
            print(text, flush=True)
        self.last_percent = percent

    def update(self) -> None:
        self.current += 1
        self.render(self.current >= self.total)

    def close(self) -> None:
        self.current = self.total
        self.render(True)
        if self.tty:
            sys.stderr.write("\n")


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def finite(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def fold_for(fingerprint: str, seed: int, folds: int) -> int:
    digest = hashlib.sha256(f"{seed}|{fingerprint}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % folds


@dataclass
class TraceData:
    dataset_names: List[str]
    dataset: np.ndarray
    request: np.ndarray
    fold: np.ndarray
    round: np.ndarray
    decision: np.ndarray
    accept: np.ndarray
    features: Dict[str, np.ndarray]
    quality: Dict[str, Any]
    request_counts: Dict[str, int]

    @property
    def size(self) -> int:
        return int(len(self.round))


def read_trace(
    root: Path,
    split_seed: int,
    folds: int,
    invalid_policy: str,
    max_invalid_rate: float,
    max_rows_per_dataset: int = 0,
) -> TraceData:
    if invalid_policy not in {"strict", "exclude"}:
        raise ValueError("invalid row policy must be strict or exclude")
    paths = sorted(root.glob("*.jsonl"))
    if not paths:
        raise RuntimeError(f"no trace jsonl files under {root}")
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
        dataset_id = len(dataset_names)
        dataset_names.append(dataset)
        metrics = {
            "rows_original": 0, "rows_usable": 0, "rows_excluded": 0,
            "replay_mismatch": 0, "cross_block_mismatch": 0,
        }
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("event") != "sglang_dynamic_block_shadow_round":
                    continue
                if max_rows_per_dataset and metrics["rows_original"] >= max_rows_per_dataset:
                    break
                branches = record.get("branches") or {}
                if any(str(block) not in branches for block in BLOCKS):
                    raise RuntimeError(f"{path}:{line_number}: missing L8/L16/L32")
                fingerprint = str(record.get("prompt_fingerprint") or record["request_id"])
                request_seen[dataset].add(fingerprint)
                replay_mismatch = not bool(record.get("canonical_replay_match", False))
                cross_mismatch = not bool(record.get("cross_block_common_prefix_match", True))
                invalid = replay_mismatch or cross_mismatch
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
                metrics["rows_usable"] += 1
                key = (dataset, fingerprint)
                request_id = request_ids.setdefault(key, len(request_ids))
                request_folds.setdefault(
                    request_id, fold_for(fingerprint, split_seed, folds)
                )
                dataset_values.append(dataset_id)
                request_values.append(request_id)
                fold_values.append(request_folds[request_id])
                round_values.append(int(record["round_index"]))
                decision_values.append(int(record["decision_block"]))
                for block in BLOCKS:
                    accept_values.append(int(branches[str(block)]["accept_length"]))
                feature_map = record.get("history_before_round") or {}
                for name in FEATURES:
                    feature_values[name].append(finite(feature_map.get(name)))
        metrics["excluded_rate"] = metrics["rows_excluded"] / max(1, metrics["rows_original"])
        by_dataset[dataset] = metrics
        print(
            f"[trace读取] {dataset}: 原始={metrics['rows_original']} "
            f"可用={metrics['rows_usable']} 排除={metrics['rows_excluded']}",
            flush=True,
        )

    invalid_rate = total_excluded / max(1, total_original)
    quality = {
        "invalid_row_policy": invalid_policy,
        "max_invalid_row_rate": max_invalid_rate,
        "rows_original": total_original,
        "rows_usable": total_original - total_excluded,
        "rows_excluded": total_excluded,
        "excluded_rate": invalid_rate,
        "replay_mismatch": replay_bad,
        "cross_block_mismatch": cross_bad,
        "by_dataset": by_dataset,
    }
    if invalid_rate > max_invalid_rate:
        raise RuntimeError(
            f"trace exclusion {invalid_rate:.4%} exceeds {max_invalid_rate:.4%}"
        )
    if total_excluded and invalid_policy == "strict":
        raise RuntimeError(f"strict trace audit found {total_excluded} ambiguous rows")
    if total_original == total_excluded:
        raise RuntimeError("no usable trace rows")
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


def indices_weights(data: TraceData, indices: np.ndarray) -> np.ndarray:
    """Dataset equal -> request equal -> rounds equal within request."""
    indices = np.asarray(indices, dtype=np.int64)
    weights = np.zeros(len(indices), dtype=np.float64)
    datasets = np.unique(data.dataset[indices])
    for dataset in datasets:
        local = np.flatnonzero(data.dataset[indices] == dataset)
        requests, inverse, counts = np.unique(
            data.request[indices[local]], return_inverse=True, return_counts=True
        )
        weights[local] = 1.0 / len(datasets) / len(requests) / counts[inverse]
    return weights


def weighted_quantile_edges(x: np.ndarray, weights: np.ndarray, bins: int) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    values = x[order]
    ordered_weights = weights[order]
    cumulative = np.cumsum(ordered_weights)
    targets = cumulative[-1] * np.arange(1, bins) / bins
    positions = np.searchsorted(cumulative, targets, side="left")
    edges = np.unique(values[np.minimum(positions, len(values) - 1)])
    # Max-valued boundaries create an unreachable last bin.
    return edges[edges < values[-1]].astype(np.float64)


def pava_increasing(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    blocks: List[List[float]] = []
    for index, (value, weight) in enumerate(zip(values, weights)):
        blocks.append([float(index), float(index), float(weight), float(value * weight)])
        while len(blocks) >= 2:
            left, right = blocks[-2], blocks[-1]
            if left[3] / left[2] <= right[3] / right[2] + 1e-15:
                break
            blocks[-2:] = [[left[0], right[1], left[2] + right[2], left[3] + right[3]]]
    output = np.empty(len(values), dtype=np.float64)
    for start, end, weight, total in blocks:
        output[int(start) : int(end) + 1] = total / weight
    return output


def fit_survival(
    signal: np.ndarray,
    accepted: np.ndarray,
    weights: np.ndarray,
    block: int,
    bins: int,
    shared_edges: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    valid = np.isfinite(signal)
    if not valid.any():
        raise RuntimeError("candidate signal has no finite training values")
    x = signal[valid].astype(np.float64)
    y = accepted[valid].astype(np.int16)
    w = weights[valid].astype(np.float64)
    edges = (
        weighted_quantile_edges(x, w, bins)
        if shared_edges is None
        else np.asarray(shared_edges, dtype=np.float64)
    )
    bin_index = np.searchsorted(edges, x, side="left")
    bin_count = len(edges) + 1
    totals = np.bincount(bin_index, weights=w, minlength=bin_count)
    if np.any(totals <= 0):
        raise RuntimeError("weighted quantile construction produced an empty bin")
    clipped = np.clip(y, 0, block)
    histogram = np.bincount(
        bin_index * (block + 1) + clipped,
        weights=w,
        minlength=bin_count * (block + 1),
    ).reshape(bin_count, block + 1)
    raw_survival = np.cumsum(histogram[:, ::-1], axis=1)[:, ::-1][:, 1:] / totals[:, None]
    survival = np.empty((bin_count, block), dtype=np.float64)
    for token in range(block):
        survival[:, token] = pava_increasing(raw_survival[:, token], totals)
    # A survival function must also be non-increasing in token index.
    survival = np.minimum.accumulate(survival, axis=1)
    # Missing-signal predictions use the complete stratum, including rows in
    # which this particular feature is absent (e.g. no rejected token after a
    # full block), rather than treating missingness as a hidden second signal.
    all_y = accepted.astype(np.int16)
    all_w = weights.astype(np.float64)
    total_weight = float(all_w.sum())
    missing = [float(np.sum(all_w * (all_y >= token)) / total_weight) for token in range(1, block + 1)]
    return {
        "type": "binned_monotone_survival",
        "block": block,
        "edges": edges.tolist(),
        "survival": survival.tolist(),
        "missing_survival": missing,
    }


def predict_survival(model: Mapping[str, Any], signal: np.ndarray) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float64)
    survival = np.asarray(model["survival"], dtype=np.float64)
    missing = np.asarray(model["missing_survival"], dtype=np.float64)
    output = np.full(len(signal), float(missing.sum()), dtype=np.float64)
    valid = np.isfinite(signal)
    if valid.any():
        bins = np.searchsorted(np.asarray(model["edges"]), signal[valid], side="left")
        output[valid] = survival[bins].sum(axis=1)
    return output


def fit_tables(
    data: TraceData,
    indices: np.ndarray,
    signal: np.ndarray,
    gate: Optional[int],
    bins: int,
) -> Dict[str, Dict[str, Any]]:
    strata = {"all": np.ones(len(indices), dtype=bool)}
    if gate is not None:
        streak = data.features["full_streak"][indices]
        strata = {"not_full": streak < gate, "full": streak >= gate}
    result: Dict[str, Dict[str, Any]] = {}
    for name, local_mask in strata.items():
        if int(local_mask.sum()) < max(100, bins * 4):
            raise RuntimeError(f"stratum {name} is too small")
        local_indices = indices[local_mask]
        local_signal = signal[local_indices]
        # Conditioning on the censoring stratum must not let the number of
        # requests/rounds in one dataset determine the fit.  Rebuild the full
        # dataset->request->round weights inside each stratum.
        local_weights = indices_weights(data, local_indices)
        finite_mask = np.isfinite(local_signal)
        if not finite_mask.any():
            raise RuntimeError(f"stratum {name} has no finite signal values")
        shared_edges = weighted_quantile_edges(
            local_signal[finite_mask].astype(np.float64),
            local_weights[finite_mask],
            bins,
        )
        result[name] = {}
        for column, block in enumerate(BLOCKS):
            result[name][str(block)] = fit_survival(
                local_signal,
                data.accept[local_indices, column],
                local_weights,
                block,
                bins,
                shared_edges=shared_edges,
            )
    return result


def predict_tables(
    data: TraceData,
    indices: np.ndarray,
    signal: np.ndarray,
    tables: Mapping[str, Mapping[str, Any]],
    gate: Optional[int],
) -> np.ndarray:
    output = np.empty((len(indices), 3), dtype=np.float64)
    if gate is None:
        for column, block in enumerate(BLOCKS):
            output[:, column] = predict_survival(
                tables["all"][str(block)], signal[indices]
            )
        return output
    streak = data.features["full_streak"][indices]
    for stratum, local_mask in (
        ("not_full", streak < gate), ("full", streak >= gate)
    ):
        for column, block in enumerate(BLOCKS):
            output[local_mask, column] = predict_survival(
                tables[stratum][str(block)], signal[indices[local_mask]]
            )
    return output


def actions_from_values(values: np.ndarray, lam: float, first_round: np.ndarray) -> np.ndarray:
    utilities = values - lam * np.asarray(BLOCKS, dtype=np.float64)
    actions = np.full(len(values), 8, dtype=np.uint8)
    best = utilities[:, 0].copy()
    for column, block in enumerate(BLOCKS[1:], 1):
        better = utilities[:, column] > best + 1e-12
        actions[better] = block
        best[better] = utilities[better, column]
    actions[first_round] = 16
    return actions


def oracle_actions(data: TraceData, lam: float) -> Tuple[np.ndarray, np.ndarray]:
    utilities = data.accept.astype(np.float64) - lam * np.asarray(BLOCKS)
    actions = np.full(data.size, 8, dtype=np.uint8)
    best = utilities[:, 0].copy()
    for column, block in enumerate(BLOCKS[1:], 1):
        better = utilities[:, column] > best + 1e-12
        actions[better] = block
        best[better] = utilities[better, column]
    return actions, best


def selected_accept(data: TraceData, actions: np.ndarray) -> np.ndarray:
    columns = np.where(actions == 8, 0, np.where(actions == 16, 1, 2))
    return data.accept[np.arange(data.size), columns].astype(np.float64)


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    order = np.argsort(values, kind="mergesort")
    cumulative = np.cumsum(weights[order])
    position = np.searchsorted(cumulative, quantile * cumulative[-1], side="left")
    return float(values[order[min(position, len(order) - 1)]])


def summarize(
    data: TraceData,
    actions: np.ndarray,
    lam: float,
    min_gain16: float,
    min_gain32: float,
) -> Dict[str, Any]:
    accepted = selected_accept(data, actions)
    oracle, oracle_utility = oracle_actions(data, lam)
    chosen_utility = accepted - lam * actions
    regret = oracle_utility - chosen_utility
    max_accept = data.accept.max(axis=1).astype(np.float64)
    loss32 = np.maximum(data.accept[:, 2] - accepted, 0)
    truncation = np.maximum(max_accept - accepted, 0)
    controlled = data.round > 0
    # The protocol fixes every request's cold start to L16.  It contributes to
    # actual compute/TPF, but must not be counted as a learned "large-block"
    # prediction when enforcing precision and waste constraints.
    large = controlled & (actions > 8)
    accept_float = data.accept.astype(np.float64)
    incremental_gain = np.where(
        actions == 16,
        accept_float[:, 1] - accept_float[:, 0],
        np.where(
            actions == 32,
            accept_float[:, 2]
            - np.maximum(accept_float[:, 0], accept_float[:, 1]),
            0.0,
        ),
    )
    significant = np.where(actions == 16, incremental_gain >= min_gain16, incremental_gain >= min_gain32)
    waste = incremental_gain <= 1

    def one(indices: np.ndarray) -> Dict[str, Any]:
        weights = indices_weights(data, indices)
        chosen = actions[indices]
        chosen_accept = accepted[indices]
        chosen_regret = regret[indices]
        return {
            "requests": int(len(np.unique(data.request[indices]))),
            "rounds": int(len(indices)),
            "mean_block": float(np.average(chosen, weights=weights)),
            "mean_accept": float(np.average(chosen_accept, weights=weights)),
            "decode_tpf_logical": float(np.average(chosen_accept / 2.0, weights=weights)),
            "accept_per_block_token": float(np.average(chosen_accept, weights=weights) / np.average(chosen, weights=weights)),
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

    all_weights = indices_weights(data, all_indices)

    def conditional_macro(mask: np.ndarray, values: np.ndarray, empty: float) -> float:
        results = []
        for dataset_id in range(len(data.dataset_names)):
            indices = np.flatnonzero((data.dataset == dataset_id) & mask)
            if not len(indices):
                continue
            # Preserve the unconditional request/round weights and normalize
            # only after selecting the event, matching the formal protocol.
            weights = all_weights[indices]
            results.append(float(np.average(values[indices], weights=weights)))
        return float(np.mean(results)) if results else empty

    macro.update(
        {
            "large_rate": conditional_macro(
                controlled, (actions > 8).astype(float), 0.0
            ),
            "large_precision": conditional_macro(large, significant.astype(float), 1.0),
            "large_waste_rate": conditional_macro(large, waste.astype(float), 0.0),
            "compute_saving_vs_l32": 32.0 - macro["mean_block"],
        }
    )
    return {"datasets": by_dataset, "macro": macro}


def summarize_transitions(data: TraceData, actions: np.ndarray) -> Dict[str, Any]:
    """Request-equal transition/oscillation rates, then dataset-equal macro."""
    by_dataset: Dict[str, Any] = {}
    for dataset_id, dataset in enumerate(data.dataset_names):
        dataset_indices = np.flatnonzero(data.dataset == dataset_id)
        request_rows: Dict[int, List[int]] = defaultdict(list)
        for index in dataset_indices:
            request_rows[int(data.request[index])].append(int(index))
        matrices: List[np.ndarray] = []
        oscillations: List[float] = []
        for rows in request_rows.values():
            ordered = sorted(rows, key=lambda index: int(data.round[index]))
            sequence = actions[ordered]
            if len(sequence) >= 2:
                matrix = np.zeros((3, 3), dtype=np.float64)
                for left, right in zip(sequence[:-1], sequence[1:]):
                    matrix[BLOCKS.index(int(left)), BLOCKS.index(int(right))] += 1
                matrix /= matrix.sum()
                matrices.append(matrix)
            if len(sequence) >= 3:
                pattern = (
                    (sequence[:-2] == sequence[2:])
                    & (sequence[:-2] != sequence[1:-1])
                )
                oscillations.append(float(pattern.mean()))
        if not matrices:
            continue
        mean_matrix = np.mean(matrices, axis=0)
        by_dataset[dataset] = {
            "matrix": mean_matrix.tolist(),
            "switch_rate": float(1.0 - np.trace(mean_matrix)),
            "oscillation_rate": float(np.mean(oscillations)) if oscillations else 0.0,
            "requests_with_transitions": len(matrices),
        }
    if by_dataset:
        macro_matrix = np.mean(
            [np.asarray(row["matrix"]) for row in by_dataset.values()], axis=0
        )
        macro = {
            "matrix": macro_matrix.tolist(),
            "switch_rate": float(np.mean([row["switch_rate"] for row in by_dataset.values()])),
            "oscillation_rate": float(np.mean([row["oscillation_rate"] for row in by_dataset.values()])),
        }
    else:
        macro = None
    return {"datasets": by_dataset, "macro": macro}


def candidate_oof(
    data: TraceData,
    spec: Mapping[str, Any],
    gate: Optional[int],
    folds: int,
    bins: int,
) -> np.ndarray:
    signal = data.features[spec["feature"]].astype(np.float64) * float(spec["orientation"])
    historical = data.round > 0
    output = np.zeros((data.size, 3), dtype=np.float64)
    for fold in range(folds):
        train = np.flatnonzero(historical & (data.fold != fold))
        heldout = np.flatnonzero(historical & (data.fold == fold))
        if not len(train) or not len(heldout):
            raise RuntimeError(f"empty OOF fold {fold}")
        tables = fit_tables(data, train, signal, gate, bins)
        output[heldout] = predict_tables(data, heldout, signal, tables, gate)
    return output


def integrated_regret(
    data: TraceData, values: np.ndarray, lambdas: Sequence[float]
) -> Tuple[float, float]:
    first = data.round == 0
    weights = indices_weights(data, np.arange(data.size))
    regrets = []
    accuracies = []
    for lam in lambdas:
        actions = actions_from_values(values, lam, first)
        accepted = selected_accept(data, actions)
        oracle, oracle_utility = oracle_actions(data, lam)
        regrets.append(float(np.average(oracle_utility - (accepted - lam * actions), weights=weights)))
        accuracies.append(float(np.average(actions == oracle, weights=weights)))
    return float(np.mean(regrets)), float(np.mean(accuracies))


def publish_progress(run_dir: Optional[Path], output_dir: Path, payload: Mapping[str, Any]) -> None:
    atomic_json(output_dir / "progress.json", payload)
    if run_dir is not None:
        try:
            from reporting import render

            render(run_dir)
        except Exception as exc:
            print(f"[报告更新警告] {exc}", file=sys.stderr, flush=True)


def validate_protocol(
    data: TraceData,
    allow_partial: bool,
    mmlu_submitted_requests: int,
    min_request_coverage: float,
) -> None:
    datasets = set(data.dataset_names)
    if "aime24" in datasets:
        raise RuntimeError("AIME24 must not participate")
    if not allow_partial and datasets != EXPECTED_DATASETS:
        raise RuntimeError(
            f"formal run requires exactly nine datasets; missing={sorted(EXPECTED_DATASETS-datasets)} "
            f"extra={sorted(datasets-EXPECTED_DATASETS)}"
        )
    if not allow_partial:
        traced = data.request_counts.get("mmlu", 0)
        coverage = traced / max(1, mmlu_submitted_requests)
        if traced > mmlu_submitted_requests or coverage < min_request_coverage:
            raise RuntimeError(
                f"MMLU trace-bearing request coverage is invalid: traced={traced}, "
                f"submitted={mmlu_submitted_requests}, coverage={coverage:.2%}, "
                f"required>={min_request_coverage:.2%}. Requests with no decode "
                "round are reported but cannot be fabricated as policy rows."
            )


def search_policy(args: argparse.Namespace, data: TraceData) -> Dict[str, Any]:
    historical = np.flatnonzero(data.round > 0)
    lambdas = args.lambda_grid
    tier1: List[Dict[str, Any]] = []
    best_single_values: Optional[np.ndarray] = None
    best_single_key: Optional[Tuple[Any, ...]] = None
    progress = ProgressBar("Tier1单信号 OOF", len(SIGNALS))
    for index, spec in enumerate(SIGNALS, 1):
        values = candidate_oof(data, spec, None, args.cv_folds, args.signal_bins)
        regret, accuracy = integrated_regret(data, values, lambdas)
        result = {
            "tier": 1, "signal": spec["name"], "feature": spec["feature"],
            "kind": spec["kind"], "zh": spec["zh"], "gate": None,
            "integrated_mean_regret": regret,
            "integrated_oracle_accuracy": accuracy,
        }
        tier1.append(result)
        candidate_key = (regret, -accuracy, spec["name"])
        if best_single_key is None or candidate_key < best_single_key:
            best_single_key = candidate_key
            best_single_values = values
        publish_progress(
            args.run_dir, args.output_dir,
            {"stage": "tier1", "completed": index, "total": len(SIGNALS), "latest": result, "ranking": sorted(tier1, key=lambda x: x["integrated_mean_regret"])},
        )
        progress.update()
    progress.close()
    tier1.sort(key=lambda x: (x["integrated_mean_regret"], -x["integrated_oracle_accuracy"], x["signal"]))
    best_single = tier1[0]

    # Tier 2 is deliberately restricted: one acceptance-derived scalar plus a
    # global full-streak censoring gate.  No confidence/margin feature vector is
    # allowed, and the gate is shared by both upshift boundaries.
    dual_specs = [spec for spec in SIGNALS if spec["kind"] == "accept" and spec["name"] != "full_streak"]
    dual_candidates = [(spec, gate) for spec in dual_specs for gate in args.full_gate_grid]
    tier2: List[Dict[str, Any]] = []
    best_dual_values: Optional[np.ndarray] = None
    best_dual_key: Optional[Tuple[Any, ...]] = None
    progress = ProgressBar("Tier2接收+full OOF", len(dual_candidates))
    for index, (spec, gate) in enumerate(dual_candidates, 1):
        try:
            values = candidate_oof(data, spec, gate, args.cv_folds, args.signal_bins)
            regret, accuracy = integrated_regret(data, values, lambdas)
            result = {
                "tier": 2, "signal": spec["name"], "feature": spec["feature"],
                "kind": spec["kind"], "zh": spec["zh"], "gate": gate,
                "integrated_mean_regret": regret,
                "integrated_oracle_accuracy": accuracy,
            }
            tier2.append(result)
            candidate_key = (regret, -accuracy, spec["name"], gate)
            if best_dual_key is None or candidate_key < best_dual_key:
                best_dual_key = candidate_key
                best_dual_values = values
        except RuntimeError as exc:
            result = {
                "tier": 2, "signal": spec["name"], "gate": gate,
                "status": "invalid", "reason": str(exc),
                "integrated_mean_regret": None,
            }
        publish_progress(
            args.run_dir, args.output_dir,
            {"stage": "tier2", "completed": index, "total": len(dual_candidates), "latest": result, "tier1_best": best_single, "ranking": sorted(tier2, key=lambda x: x["integrated_mean_regret"])},
        )
        progress.update()
    progress.close()
    tier2.sort(key=lambda x: (x["integrated_mean_regret"], -x["integrated_oracle_accuracy"], x["signal"], x["gate"]))
    best_dual = tier2[0] if tier2 else None
    selected = best_single
    dual_improvement = 0.0
    if best_dual is not None:
        dual_improvement = best_single["integrated_mean_regret"] - best_dual["integrated_mean_regret"]
        if dual_improvement >= args.min_dual_improvement:
            selected = best_dual
    if selected["tier"] == 1:
        assert best_single_values is not None
        oof_values = best_single_values
    else:
        assert best_dual_values is not None
        oof_values = best_dual_values

    pareto = []
    default_candidates = []
    for lam in lambdas:
        actions = actions_from_values(oof_values, lam, data.round == 0)
        summary = summarize(data, actions, lam, args.min_gain16, args.min_gain32)
        macro = summary["macro"]
        feasible = (
            macro["large_precision"] >= args.min_large_precision
            and macro["large_waste_rate"] <= args.max_large_waste
            and macro["loss_vs_l32"] <= args.max_loss_vs_l32
        )
        row = {"lambda": lam, "feasible": feasible, **macro}
        pareto.append(row)
        if feasible:
            default_candidates.append(row)
    if args.default_lambda is None:
        if default_candidates:
            default_row = max(
                default_candidates,
                key=lambda row: (row["compute_saving_vs_l32"], -row["mean_regret"], -row["lambda"]),
            )
            default_feasible = True
        else:
            def penalty(row: Mapping[str, Any]) -> float:
                return (
                    10 * max(0.0, args.min_large_precision - row["large_precision"])
                    + 10 * max(0.0, row["large_waste_rate"] - args.max_large_waste)
                    + max(0.0, row["loss_vs_l32"] - args.max_loss_vs_l32)
                )
            default_row = min(pareto, key=lambda row: (penalty(row), row["mean_regret"], row["mean_block"]))
            default_feasible = False
    else:
        nearest = min(pareto, key=lambda row: abs(row["lambda"] - args.default_lambda))
        if abs(nearest["lambda"] - args.default_lambda) > 1e-12:
            raise RuntimeError("--default-lambda must be present in --lambda-grid")
        default_row = nearest
        default_feasible = bool(nearest["feasible"])
    default_lambda = float(default_row["lambda"])
    default_actions = actions_from_values(oof_values, default_lambda, data.round == 0)
    default_summary = summarize(
        data, default_actions, default_lambda, args.min_gain16, args.min_gain32
    )
    default_summary["transitions"] = summarize_transitions(data, default_actions)

    fixed = {}
    for block in BLOCKS:
        actions = np.full(data.size, block, dtype=np.uint8)
        fixed[str(block)] = summarize(
            data, actions, default_lambda, args.min_gain16, args.min_gain32
        )
    oracle, _ = oracle_actions(data, default_lambda)
    oracle_summary = summarize(
        data, oracle, default_lambda, args.min_gain16, args.min_gain32
    )

    spec = SIGNAL_BY_NAME[selected["signal"]]
    signal = data.features[spec["feature"]].astype(np.float64) * float(spec["orientation"])
    final_tables = fit_tables(
        data, historical, signal, selected["gate"], args.signal_bins
    )
    policy = {
        "schema_version": 1,
        "policy_type": "unified_scalar_value",
        "blocks": list(BLOCKS),
        "cold_start_block": 16,
        "signal_count": selected["tier"],
        "signal": {
            "name": spec["name"], "feature": spec["feature"],
            "orientation": spec["orientation"], "kind": spec["kind"], "zh": spec["zh"],
        },
        "full_streak_gate": (
            {"feature": "full_streak", "threshold": selected["gate"], "role": "right_censor_stratum"}
            if selected["tier"] == 2 else None
        ),
        "value_models": final_tables,
        "default_lambda": default_lambda,
        "lambda_semantics": "choose argmax_L E[A_L|signal]-lambda*L; larger lambda prefers smaller blocks",
        "dataset_weighting": "dataset_equal_then_request_equal_then_round_equal",
        "training_rows": int(len(historical)),
    }
    atomic_json(args.output_dir / "policy.json", {"policy": policy})
    payload = {
        "protocol": {
            "datasets": data.dataset_names,
            "dataset_count": len(data.dataset_names),
            "requests_by_dataset": data.request_counts,
            "rows": data.size,
            "historical_rows": int(len(historical)),
            "cv_folds": args.cv_folds,
            "signal_bins": args.signal_bins,
            "lambda_grid": lambdas,
            "weighting": "dataset equal -> request equal -> request rounds equal",
            "trace_quality": data.quality,
        },
        "tier1_ranking": tier1,
        "tier2_ranking": tier2,
        "selection": {
            "selected": selected,
            "best_single": best_single,
            "best_dual": best_dual,
            "dual_improvement": dual_improvement,
            "min_dual_improvement": args.min_dual_improvement,
            "default_lambda": default_lambda,
            "default_feasible": default_feasible,
            "constraints": {
                "min_large_precision": args.min_large_precision,
                "max_large_waste": args.max_large_waste,
                "max_loss_vs_l32": args.max_loss_vs_l32,
                "min_gain16": args.min_gain16,
                "min_gain32": args.min_gain32,
            },
        },
        "pareto": pareto,
        "oof_default": default_summary,
        "fixed_baselines": fixed,
        "same_state_oracle": oracle_summary,
        "policy": policy,
    }
    atomic_json(args.output_dir / "search_results.json", payload)
    publish_progress(
        args.run_dir, args.output_dir,
        {"stage": "completed", "completed": len(SIGNALS) + len(dual_candidates), "total": len(SIGNALS) + len(dual_candidates), "selected": selected, "default_lambda": default_lambda},
    )
    return payload


def collect_official_metrics(eval_root: Optional[Path]) -> Dict[str, Any]:
    if eval_root is None or not eval_root.exists():
        return {"datasets": {}, "macro": None}
    latest: Dict[str, Tuple[float, Dict[str, Any], str]] = {}
    for path in eval_root.rglob("sglang_metrics_summary.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        dataset = str(payload.get("benchmark", ""))
        if not dataset:
            continue
        mtime = path.stat().st_mtime
        if dataset not in latest or mtime > latest[dataset][0]:
            latest[dataset] = (mtime, payload, str(path))
    results = {}
    for dataset, (_mtime, payload, path) in sorted(latest.items()):
        decode = payload.get("decode") or {}
        serving = payload.get("serving") or {}
        results[dataset] = {
            "decode_tpf": decode.get("tokens_per_forward_pass"),
            "mean_tokens_per_block": decode.get("mean_tokens_per_block"),
            "mean_acceptance_rate": decode.get("mean_acceptance_rate"),
            "weighted_acceptance_rate": decode.get("weighted_acceptance_rate"),
            "decode_forward_passes": decode.get("decode_forward_passes"),
            "decode_tokens": decode.get("decode_tokens"),
            "wall_output_tokens_per_s_observation_only": serving.get("wall_output_tokens_per_s"),
            "source": path,
        }
    numeric = ("decode_tpf", "mean_tokens_per_block", "mean_acceptance_rate", "weighted_acceptance_rate")
    macro = {
        key: float(np.mean([row[key] for row in results.values() if row.get(key) is not None]))
        for key in numeric
        if any(row.get(key) is not None for row in results.values())
    } if results else None
    return {"datasets": results, "macro": macro}


def validate_policy(args: argparse.Namespace, data: TraceData) -> Dict[str, Any]:
    policy_payload = json.loads(args.policy.read_text(encoding="utf-8"))
    policy = policy_payload.get("policy", policy_payload)
    lam = float(policy["default_lambda"] if args.policy_lambda is None else args.policy_lambda)
    actions = data.decision.astype(np.uint8)
    summary = summarize(data, actions, lam, args.min_gain16, args.min_gain32)
    summary["transitions"] = summarize_transitions(data, actions)
    fixed = {
        str(block): summarize(
            data, np.full(data.size, block, dtype=np.uint8), lam,
            args.min_gain16, args.min_gain32,
        )
        for block in BLOCKS
    }
    oracle, _ = oracle_actions(data, lam)
    payload = {
        "protocol": {
            "datasets": data.dataset_names,
            "dataset_count": len(data.dataset_names),
            "requests_by_dataset": data.request_counts,
            "rows": data.size,
            "trace_quality": data.quality,
            "policy_path": str(args.policy),
            "policy_lambda": lam,
            "formal_complete": not args.allow_partial_datasets,
        },
        "unified_policy": summary,
        "fixed_baselines_same_state": fixed,
        "same_state_oracle": summarize(
            data, oracle, lam, args.min_gain16, args.min_gain32
        ),
        "official_sglang_metrics": collect_official_metrics(args.eval_root),
    }
    atomic_json(args.output_dir / "validation.json", payload)
    publish_progress(
        args.run_dir, args.output_dir,
        {
            "stage": (
                "validation_completed"
                if not args.allow_partial_datasets
                else "validation_partial"
            ),
            "completed": len(data.dataset_names),
            "total": len(EXPECTED_DATASETS),
            "lambda": lam,
        },
    )
    return payload


def parse_float_list(text: str) -> List[float]:
    values = sorted(set(float(item.strip()) for item in text.split(",") if item.strip()))
    if not values or values[0] < 0:
        raise argparse.ArgumentTypeError("values must be a nonempty nonnegative list")
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
    parser.add_argument("--policy-lambda", type=float)
    parser.add_argument("--eval-root", type=Path)
    parser.add_argument("--split-seed", type=int, default=20260906)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--signal-bins", type=int, default=32)
    parser.add_argument("--lambda-grid", type=parse_float_list, default=parse_float_list("0,0.025,0.05,0.075,0.1,0.125,0.15,0.175,0.2,0.225,0.25,0.275,0.3,0.35,0.4,0.5,0.6,0.75,1"))
    parser.add_argument("--full-gate-grid", type=parse_int_list, default=parse_int_list("1,2,3"))
    parser.add_argument("--default-lambda", type=float)
    parser.add_argument("--min-dual-improvement", type=float, default=0.02)
    parser.add_argument("--min-large-precision", type=float, default=0.80)
    parser.add_argument("--max-large-waste", type=float, default=0.10)
    parser.add_argument("--max-loss-vs-l32", type=float, default=2.5)
    parser.add_argument("--min-gain16", type=float, default=2.0)
    parser.add_argument("--min-gain32", type=float, default=4.0)
    parser.add_argument("--invalid-row-policy", choices=("strict", "exclude"), default="exclude")
    parser.add_argument("--max-invalid-row-rate", type=float, default=0.05)
    parser.add_argument("--mmlu-required-requests", type=int, default=2000)
    parser.add_argument("--min-request-coverage", type=float, default=0.95)
    parser.add_argument("--allow-partial-datasets", action="store_true")
    parser.add_argument(
        "--max-rows-per-dataset",
        type=int,
        default=0,
        help="development smoke only; 0 reads every row",
    )
    args = parser.parse_args()
    if args.cv_folds < 2 or args.signal_bins < 2:
        parser.error("--cv-folds and --signal-bins must be at least 2")
    if not 0 <= args.max_invalid_row_rate <= 1:
        parser.error("--max-invalid-row-rate must be in [0,1]")
    if not 0 < args.min_request_coverage <= 1:
        parser.error("--min-request-coverage must be in (0,1]")
    if args.mode == "validate" and args.policy is None:
        parser.error("--policy is required in validate mode")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = read_trace(
        args.trace_root, args.split_seed, args.cv_folds,
        args.invalid_row_policy, args.max_invalid_row_rate,
        args.max_rows_per_dataset,
    )
    validate_protocol(
        data,
        args.allow_partial_datasets,
        args.mmlu_required_requests,
        args.min_request_coverage,
    )
    if args.mode == "search":
        search_policy(args, data)
    else:
        validate_policy(args, data)


if __name__ == "__main__":
    main()
