#!/usr/bin/env python3
"""Trace reader, verifier-only signals, and small nonparametric policy models."""

from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from array import array
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from sglang_b200_verify_efficiency_policy.latency_costs import BLOCKS


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

# Every entry is available before the current draft and comes only from prior
# causal verification outcomes.  No draft-logit feature is legal here.
_BASE_SIGNALS: Tuple[Dict[str, Any], ...] = (
    {"name": "accept_last", "feature": "prev_accept", "orientation": 1, "kind": "accept", "zh": "上一轮验证前进长度"},
    {"name": "accept_ma2", "feature": "a_ma2", "orientation": 1, "kind": "accept", "zh": "近2轮验证前进长度均值"},
    {"name": "accept_ma4", "feature": "a_ma4", "orientation": 1, "kind": "accept", "zh": "近4轮验证前进长度均值"},
    {"name": "accept_ma8", "feature": "a_ma8", "orientation": 1, "kind": "accept", "zh": "近8轮验证前进长度均值"},
    {"name": "ratio_last", "feature": "prev_accept_ratio", "orientation": 1, "kind": "accept", "zh": "上一轮验证通过比例"},
    {"name": "ratio_ma2", "feature": "ratio_ma2", "orientation": 1, "kind": "accept", "zh": "近2轮验证通过比例均值"},
    {"name": "ratio_ma4", "feature": "ratio_ma4", "orientation": 1, "kind": "accept", "zh": "近4轮验证通过比例均值"},
    {"name": "ratio_ma8", "feature": "ratio_ma8", "orientation": 1, "kind": "accept", "zh": "近8轮验证通过比例均值"},
    {"name": "full_streak", "feature": "full_streak", "orientation": 1, "kind": "accept", "zh": "连续整块通过轮数"},
    {"name": "nonfull_streak", "feature": "nonfull_streak", "orientation": -1, "kind": "accept", "zh": "连续未整块通过轮数的相反数"},
    {"name": "verify_conf_last", "feature": "prev_verify_head_conf", "orientation": 1, "kind": "verify_conf", "zh": "上一轮verify头部top1置信度"},
    {"name": "verify_conf_ma2", "feature": "verify_conf_ma2", "orientation": 1, "kind": "verify_conf", "zh": "近2轮verify头部top1置信度"},
    {"name": "verify_conf_ma4", "feature": "verify_conf_ma4", "orientation": 1, "kind": "verify_conf", "zh": "近4轮verify头部top1置信度"},
    {"name": "verify_conf_ma8", "feature": "verify_conf_ma8", "orientation": 1, "kind": "verify_conf", "zh": "近8轮verify头部top1置信度"},
    {"name": "verify_margin_last", "feature": "prev_verify_head_margin", "orientation": 1, "kind": "verify_margin", "zh": "上一轮verify头部top1-top2 margin"},
    {"name": "verify_margin_ma2", "feature": "verify_margin_ma2", "orientation": 1, "kind": "verify_margin", "zh": "近2轮verify头部margin"},
    {"name": "verify_margin_ma4", "feature": "verify_margin_ma4", "orientation": 1, "kind": "verify_margin", "zh": "近4轮verify头部margin"},
    {"name": "verify_margin_ma8", "feature": "verify_margin_ma8", "orientation": 1, "kind": "verify_margin", "zh": "近8轮verify头部margin"},
    {"name": "verify_entropy_last", "feature": "prev_verify_head_entropy", "orientation": -1, "kind": "verify_entropy", "zh": "上一轮verify头部entropy的相反数"},
    {"name": "verify_entropy_ma2", "feature": "verify_entropy_ma2", "orientation": -1, "kind": "verify_entropy", "zh": "近2轮verify头部entropy的相反数"},
    {"name": "verify_entropy_ma4", "feature": "verify_entropy_ma4", "orientation": -1, "kind": "verify_entropy", "zh": "近4轮verify头部entropy的相反数"},
    {"name": "verify_entropy_ma8", "feature": "verify_entropy_ma8", "orientation": -1, "kind": "verify_entropy", "zh": "近8轮verify头部entropy的相反数"},
    {"name": "verify_accept_conf_last", "feature": "prev_verify_accepted_conf", "orientation": 1, "kind": "verify_conf", "zh": "上一轮verify已通过位置置信度"},
    {"name": "verify_accept_conf_ma2", "feature": "verify_accept_conf_ma2", "orientation": 1, "kind": "verify_conf", "zh": "近2轮verify已通过位置置信度"},
    {"name": "verify_accept_conf_ma4", "feature": "verify_accept_conf_ma4", "orientation": 1, "kind": "verify_conf", "zh": "近4轮verify已通过位置置信度"},
    {"name": "verify_accept_conf_ma8", "feature": "verify_accept_conf_ma8", "orientation": 1, "kind": "verify_conf", "zh": "近8轮verify已通过位置置信度"},
    {"name": "verify_accept_margin_last", "feature": "prev_verify_accepted_margin", "orientation": 1, "kind": "verify_margin", "zh": "上一轮verify已通过位置margin"},
    {"name": "verify_accept_margin_ma2", "feature": "verify_accept_margin_ma2", "orientation": 1, "kind": "verify_margin", "zh": "近2轮verify已通过位置margin"},
    {"name": "verify_accept_margin_ma4", "feature": "verify_accept_margin_ma4", "orientation": 1, "kind": "verify_margin", "zh": "近4轮verify已通过位置margin"},
    {"name": "verify_accept_margin_ma8", "feature": "verify_accept_margin_ma8", "orientation": 1, "kind": "verify_margin", "zh": "近8轮verify已通过位置margin"},
    {"name": "verify_bonus_conf_last", "feature": "prev_verify_bonus_conf", "orientation": 1, "kind": "verify_conf", "zh": "上一轮整块通过时bonus token置信度"},
    {"name": "verify_bonus_conf_ma2", "feature": "verify_bonus_conf_ma2", "orientation": 1, "kind": "verify_conf", "zh": "近2轮可用verify bonus置信度均值"},
    {"name": "verify_bonus_conf_ma4", "feature": "verify_bonus_conf_ma4", "orientation": 1, "kind": "verify_conf", "zh": "近4轮可用verify bonus置信度均值"},
    {"name": "verify_bonus_conf_ma8", "feature": "verify_bonus_conf_ma8", "orientation": 1, "kind": "verify_conf", "zh": "近8轮可用verify bonus置信度均值"},
    {"name": "verify_bonus_margin_last", "feature": "prev_verify_bonus_margin", "orientation": 1, "kind": "verify_margin", "zh": "上一轮整块通过时bonus token margin"},
    {"name": "verify_bonus_margin_ma2", "feature": "verify_bonus_margin_ma2", "orientation": 1, "kind": "verify_margin", "zh": "近2轮可用verify bonus margin均值"},
    {"name": "verify_bonus_margin_ma4", "feature": "verify_bonus_margin_ma4", "orientation": 1, "kind": "verify_margin", "zh": "近4轮可用verify bonus margin均值"},
    {"name": "verify_bonus_margin_ma8", "feature": "verify_bonus_margin_ma8", "orientation": 1, "kind": "verify_margin", "zh": "近8轮可用verify bonus margin均值"},
    {"name": "verify_bonus_entropy_last", "feature": "prev_verify_bonus_entropy", "orientation": -1, "kind": "verify_entropy", "zh": "上一轮整块通过时bonus token entropy的相反数"},
    {"name": "verify_bonus_entropy_ma2", "feature": "verify_bonus_entropy_ma2", "orientation": -1, "kind": "verify_entropy", "zh": "近2轮可用verify bonus entropy的相反数"},
    {"name": "verify_bonus_entropy_ma4", "feature": "verify_bonus_entropy_ma4", "orientation": -1, "kind": "verify_entropy", "zh": "近4轮可用verify bonus entropy的相反数"},
    {"name": "verify_bonus_entropy_ma8", "feature": "verify_bonus_entropy_ma8", "orientation": -1, "kind": "verify_entropy", "zh": "近8轮可用verify bonus entropy的相反数"},
    {"name": "verify_reject_conf", "feature": "prev_verify_rejected_conf", "orientation": 1, "kind": "verify_conf", "zh": "上一轮首个拒绝位verify top1置信度"},
    {"name": "verify_reject_margin", "feature": "prev_verify_rejected_margin", "orientation": 1, "kind": "verify_margin", "zh": "上一轮首个拒绝位verify margin"},
    {"name": "verify_reject_entropy", "feature": "prev_verify_rejected_entropy", "orientation": -1, "kind": "verify_entropy", "zh": "上一轮首个拒绝位verify entropy的相反数"},
    {"name": "verify_reject_gap", "feature": "prev_verify_rejected_gap", "orientation": -1, "kind": "verify_gap", "zh": "上一轮首拒位verify top1与draft token概率差的相反数"},
)
# Acceptance length/proportion has a known positive direction.  The relation
# between a verifier-logit statistic and *next-round* capacity is empirical,
# so test both monotone orientations without adding another online signal.
SIGNALS: Tuple[Dict[str, Any], ...] = _BASE_SIGNALS + tuple(
    {
        **row,
        "name": f"{row['name']}_inv",
        "orientation": -float(row["orientation"]),
        "zh": f"{row['zh']}（反向单调）",
    }
    for row in _BASE_SIGNALS
    if str(row["kind"]).startswith("verify_")
)
SIGNAL_BY_NAME = {row["name"]: row for row in SIGNALS}
FEATURES = tuple(
    dict.fromkeys(
        [row["feature"] for row in SIGNALS]
        + ["prev_block", "prev_full", "history_rounds"]
    )
)


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
        width = 28
        filled = int(width * ratio)
        line = (
            f"[{self.label}] |{'#' * filled}{'-' * (width-filled)}| "
            f"{self.current}/{self.total} ({100*ratio:5.1f}%) ETA {eta_text}"
        )
        if self.tty:
            sys.stderr.write("\r" + line + "\033[K")
            sys.stderr.flush()
        else:
            print(line, flush=True)
        self.last_percent = percent

    def update(self) -> None:
        self.current += 1
        self.render(self.current >= self.total)

    def close(self) -> None:
        self.current = self.total
        self.render(True)
        if self.tty:
            sys.stderr.write("\n")


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
    full: np.ndarray
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
    model_size: str,
    allow_partial: bool = False,
    max_rows_per_dataset: int = 0,
    committed_only: bool = False,
    max_invalid_rate: float = 0.05,
) -> TraceData:
    paths = sorted(
        root.glob("*.jsonl"),
        key=lambda path: (
            EXPECTED_ORDER.index(path.stem) if path.stem in EXPECTED_DATASETS else 999,
            path.stem,
        ),
    )
    if not paths:
        raise RuntimeError(f"no trace jsonl under {root}")
    dataset_names: List[str] = []
    datasets = array("B")
    requests = array("I")
    fold_values = array("B")
    rounds = array("I")
    decisions = array("B")
    accepts = array("B")
    fulls = array("B")
    feature_values = {name: array("f") for name in FEATURES}
    request_ids: Dict[Tuple[str, str], int] = {}
    request_seen: Dict[str, set[str]] = defaultdict(set)
    quality_by_dataset: Dict[str, Any] = {}
    excluded = 0
    original = 0
    for path in paths:
        dataset = path.stem
        if dataset in {"aime24", "mmlu"}:
            raise RuntimeError(f"excluded dataset present in new trace root: {dataset}")
        if dataset not in EXPECTED_DATASETS:
            raise RuntimeError(f"unexpected dataset in trace root: {dataset}")
        dataset_id = len(dataset_names)
        dataset_names.append(dataset)
        row_quality = {
            "rows_original": 0,
            "rows_usable": 0,
            "rows_excluded": 0,
            "missing_verifier_metrics": 0,
            "replay_mismatch": 0,
            "cross_block_mismatch": 0,
        }
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("event") != "sglang_dynamic_block_shadow_round":
                    continue
                if max_rows_per_dataset and row_quality["rows_original"] >= max_rows_per_dataset:
                    break
                original += 1
                row_quality["rows_original"] += 1
                branches = record.get("branches") or {}
                decision = int(record.get("decision_block", 0))
                replay_bad = not bool(record.get("canonical_replay_match", False))
                cross_bad = not bool(record.get("cross_block_common_prefix_match", True))
                verifier_bad = any(
                    (branches.get(str(block)) or {}).get("verifier_metrics_source")
                    != "causal_verify_logits"
                    or int(
                        (branches.get(str(block)) or {}).get(
                            "verifier_metrics_schema", 0
                        )
                    )
                    < 2
                    for block in BLOCKS
                )
                branch_bad = any(str(block) not in branches for block in BLOCKS)
                invalid = replay_bad or branch_bad or decision not in BLOCKS or verifier_bad
                if not committed_only:
                    invalid = invalid or cross_bad
                row_quality["replay_mismatch"] += int(replay_bad)
                row_quality["cross_block_mismatch"] += int(cross_bad)
                row_quality["missing_verifier_metrics"] += int(verifier_bad)
                if invalid:
                    excluded += 1
                    row_quality["rows_excluded"] += 1
                    continue
                branch_models = {
                    str((branches[str(block)]).get("model_size", "")).lower()
                    for block in BLOCKS
                }
                if branch_models != {model_size.lower()}:
                    raise RuntimeError(
                        f"{path}: trace model labels {sorted(branch_models)} do not match {model_size}"
                    )
                fingerprint = str(
                    record.get("prompt_fingerprint") or record.get("request_id")
                )
                request_seen[dataset].add(fingerprint)
                key = (dataset, fingerprint)
                request_id = request_ids.setdefault(key, len(request_ids))
                datasets.append(dataset_id)
                requests.append(request_id)
                fold_values.append(fold_for(fingerprint, split_seed, folds))
                rounds.append(int(record["round_index"]))
                decisions.append(decision)
                for block in BLOCKS:
                    branch = branches[str(block)]
                    accepts.append(int(branch["accept_length"]))
                    fulls.append(int(bool(branch["full"])))
                feature_map = record.get("history_before_round") or {}
                for name in FEATURES:
                    feature_values[name].append(finite(feature_map.get(name)))
                row_quality["rows_usable"] += 1
        quality_by_dataset[dataset] = row_quality
        dataset_invalid_rate = row_quality["rows_excluded"] / max(
            1, row_quality["rows_original"]
        )
        row_quality["excluded_rate"] = dataset_invalid_rate
        if row_quality["rows_usable"] == 0:
            raise RuntimeError(f"{dataset} has no usable verifier shadow rows")
        if dataset_invalid_rate > max_invalid_rate:
            raise RuntimeError(
                f"{dataset} exclusion {dataset_invalid_rate:.4%} exceeds "
                f"{max_invalid_rate:.4%}"
            )
        print(
            f"[trace] {dataset}: 原始={row_quality['rows_original']} "
            f"可用={row_quality['rows_usable']} 排除={row_quality['rows_excluded']}",
            flush=True,
        )
    present = set(dataset_names)
    if not allow_partial and present != EXPECTED_DATASETS:
        raise RuntimeError(
            f"formal run requires eight datasets; missing={sorted(EXPECTED_DATASETS-present)}, "
            f"extra={sorted(present-EXPECTED_DATASETS)}"
        )
    rate = excluded / max(1, original)
    if rate > max_invalid_rate:
        raise RuntimeError(
            f"new verifier trace contains {excluded}/{original} invalid rows ({rate:.4%}); "
            f"limit is {max_invalid_rate:.4%}"
        )
    return TraceData(
        dataset_names=dataset_names,
        dataset=np.asarray(datasets, dtype=np.uint8),
        request=np.asarray(requests, dtype=np.uint32),
        fold=np.asarray(fold_values, dtype=np.uint8),
        round=np.asarray(rounds, dtype=np.uint32),
        decision=np.asarray(decisions, dtype=np.uint8),
        accept=np.asarray(accepts, dtype=np.uint8).reshape(-1, 3),
        full=np.asarray(fulls, dtype=np.uint8).reshape(-1, 3),
        features={
            name: np.asarray(values, dtype=np.float32)
            for name, values in feature_values.items()
        },
        quality={
            "rows_original": original,
            "rows_usable": original - excluded,
            "rows_excluded": excluded,
            "excluded_rate": rate,
            "by_dataset": quality_by_dataset,
            "verifier_metrics_source": "causal_verify_logits",
            "model_size": model_size,
        },
        request_counts={name: len(request_seen[name]) for name in dataset_names},
    )


def row_weights(data: TraceData, indices: np.ndarray) -> np.ndarray:
    """Each dataset, request, then round has equal nested mass."""
    indices = np.asarray(indices, dtype=np.int64)
    result = np.zeros(len(indices), dtype=np.float64)
    dataset_ids = np.unique(data.dataset[indices])
    for dataset_id in dataset_ids:
        local = np.flatnonzero(data.dataset[indices] == dataset_id)
        _requests, inverse, counts = np.unique(
            data.request[indices[local]], return_inverse=True, return_counts=True
        )
        result[local] = 1.0 / len(dataset_ids) / len(_requests) / counts[inverse]
    return result


def weighted_edges(values: np.ndarray, weights: np.ndarray, bins: int) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ordered = values[order]
    cumulative = np.cumsum(weights[order])
    targets = cumulative[-1] * np.arange(1, bins) / bins
    positions = np.searchsorted(cumulative, targets, side="left")
    edges = np.unique(ordered[np.minimum(positions, len(ordered) - 1)])
    return edges[edges < ordered[-1]].astype(np.float64)


def pava(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
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
        raise RuntimeError("signal has no finite training values")
    x = signal[valid].astype(np.float64)
    y = accepted[valid].astype(np.int16)
    w = weights[valid].astype(np.float64)
    edges = weighted_edges(x, w, bins) if shared_edges is None else shared_edges
    bin_index = np.searchsorted(edges, x, side="left")
    bin_count = len(edges) + 1
    totals = np.bincount(bin_index, weights=w, minlength=bin_count)
    histogram = np.bincount(
        bin_index * (block + 1) + np.clip(y, 0, block),
        weights=w,
        minlength=bin_count * (block + 1),
    ).reshape(bin_count, block + 1)
    raw = np.cumsum(histogram[:, ::-1], axis=1)[:, ::-1][:, 1:] / totals[:, None]
    survival = np.empty((bin_count, block), dtype=np.float64)
    for token in range(block):
        survival[:, token] = pava(raw[:, token], totals)
    survival = np.minimum.accumulate(survival, axis=1)
    total_weight = float(weights.sum())
    missing = [
        float(np.sum(weights * (accepted >= token)) / total_weight)
        for token in range(1, block + 1)
    ]
    return {
        "type": "binned_monotone_survival",
        "block": block,
        "edges": edges.tolist(),
        "survival": survival.tolist(),
        "missing_survival": missing,
    }


def predict_survival(model: Mapping[str, Any], signal: np.ndarray) -> np.ndarray:
    result = np.full(
        len(signal), float(np.sum(model["missing_survival"])), dtype=np.float64
    )
    valid = np.isfinite(signal)
    if valid.any():
        bins = np.searchsorted(np.asarray(model["edges"]), signal[valid], side="left")
        result[valid] = np.asarray(model["survival"])[bins].sum(axis=1)
    return result


def stratum_values(data: TraceData, indices: np.ndarray) -> np.ndarray:
    block = np.nan_to_num(data.features["prev_block"][indices], nan=0.0).astype(int)
    full = np.nan_to_num(data.features["prev_full"][indices], nan=0.0) >= 0.5
    return np.asarray(
        [
            f"b{value}_{'full' if passed else 'partial'}"
            if value in BLOCKS
            else "all"
            for value, passed in zip(block, full)
        ],
        dtype=object,
    )


def fit_value_models(
    data: TraceData,
    indices: np.ndarray,
    oriented_signal: np.ndarray,
    use_censor: bool,
    bins: int,
) -> Dict[str, Dict[str, Any]]:
    indices = np.asarray(indices, dtype=np.int64)
    weights = row_weights(data, indices)

    def fit_group(local_indices: np.ndarray, local_weights: np.ndarray) -> Dict[str, Any]:
        signal = oriented_signal[local_indices]
        finite_mask = np.isfinite(signal)
        if int(finite_mask.sum()) < max(24, bins * 2):
            raise RuntimeError("too few finite rows for signal binning")
        edges = weighted_edges(signal[finite_mask], local_weights[finite_mask], bins)
        return {
            str(block): fit_survival(
                signal,
                data.accept[local_indices, column],
                local_weights,
                block,
                bins,
                edges,
            )
            for column, block in enumerate(BLOCKS)
        }

    result = {"all": fit_group(indices, weights)}
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
            local_weights = row_weights(data, local)
            try:
                result[stratum] = fit_group(local, local_weights)
            except RuntimeError:
                continue
    return result


def predict_value_models(
    data: TraceData,
    indices: np.ndarray,
    oriented_signal: np.ndarray,
    models: Mapping[str, Mapping[str, Any]],
    use_censor: bool,
) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    output = np.empty((len(indices), len(BLOCKS)), dtype=np.float64)
    strata = stratum_values(data, indices) if use_censor else np.full(len(indices), "all", dtype=object)
    for stratum in np.unique(strata):
        mask = strata == stratum
        selected = models.get(str(stratum), models["all"])
        for column, block in enumerate(BLOCKS):
            output[mask, column] = predict_survival(
                selected[str(block)], oriented_signal[indices[mask]]
            )
    return output


def oof_values(
    data: TraceData,
    spec: Mapping[str, Any],
    use_censor: bool,
    folds: int,
    bins: int,
) -> np.ndarray:
    signal = data.features[spec["feature"]].astype(np.float64) * float(
        spec["orientation"]
    )
    historical = data.round > 0
    output = np.zeros((data.size, len(BLOCKS)), dtype=np.float64)
    for fold in range(folds):
        train = np.flatnonzero(historical & (data.fold != fold))
        held = np.flatnonzero(historical & (data.fold == fold))
        if not len(train) or not len(held):
            raise RuntimeError(f"empty OOF fold {fold}")
        models = fit_value_models(data, train, signal, use_censor, bins)
        output[held] = predict_value_models(data, held, signal, models, use_censor)
    return output


def selected_accept(data: TraceData, actions: np.ndarray) -> np.ndarray:
    columns = np.where(actions == 8, 0, np.where(actions == 16, 1, 2))
    return data.accept[np.arange(data.size), columns].astype(np.float64)


def sample_summary(
    data: TraceData,
    actions: np.ndarray,
    concurrency: int,
    costs: Mapping[int, Mapping[int, float]],
) -> Dict[str, Any]:
    """Full-C cost for every request; sample mean, then equal dataset mean."""
    if len(actions) != data.size:
        raise ValueError("action length mismatch")
    accepted = selected_accept(data, actions)
    action_columns = np.where(actions == 8, 0, np.where(actions == 16, 1, 2))
    row_cost = np.asarray(
        [costs[concurrency][block] for block in BLOCKS], dtype=np.float64
    )[action_columns]
    request_count = int(data.request.max()) + 1
    rounds = np.bincount(data.request, minlength=request_count).astype(np.float64)
    accepted_sum = np.bincount(
        data.request, weights=accepted, minlength=request_count
    )
    time_sum = np.bincount(data.request, weights=row_cost, minlength=request_count)
    request_dataset = np.full(request_count, -1, dtype=np.int16)
    request_dataset[data.request] = data.dataset.astype(np.int16)
    by_dataset: Dict[str, Any] = {}
    for dataset_id, name in enumerate(data.dataset_names):
        request_indices = np.flatnonzero(
            (request_dataset == dataset_id) & (rounds > 0)
        )
        local_rounds = rounds[request_indices]
        token_ms = accepted_sum[request_indices] / time_sum[request_indices]
        rates = {
            block: np.bincount(
                data.request,
                weights=(actions == block),
                minlength=request_count,
            )[request_indices]
            / local_rounds
            for block in BLOCKS
        }
        by_dataset[name] = {
            "samples": int(len(request_indices)),
            "rounds": int(local_rounds.sum()),
            "score_token_ms_req": float(token_ms.mean()),
            "pure_forward_token_ms_batch": float(concurrency * token_ms.mean()),
            "decode_tpf": float(
                (accepted_sum[request_indices] / (2.0 * local_rounds)).mean()
            ),
            "mean_accept": float(
                (accepted_sum[request_indices] / local_rounds).mean()
            ),
            "mean_forward_ms": float((time_sum[request_indices] / local_rounds).mean()),
            "l8_rate": float(rates[8].mean()),
            "l16_rate": float(rates[16].mean()),
            "l32_rate": float(rates[32].mean()),
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
            "concurrency": concurrency,
            "aggregation": "sample mean within dataset, then equal dataset mean",
            "cost_scope": "every request uses full nominal C latency",
        }
    )
    return {"datasets": by_dataset, "macro": macro}


def fixed_summaries(
    data: TraceData, concurrency: int, costs: Mapping[int, Mapping[int, float]]
) -> Dict[str, Any]:
    return {
        str(block): sample_summary(
            data, np.full(data.size, block, dtype=np.uint8), concurrency, costs
        )
        for block in BLOCKS
    }


def local_oracle_metrics(
    data: TraceData,
    actions: np.ndarray,
    concurrency: int,
    costs: Mapping[int, Mapping[int, float]],
) -> Dict[str, float]:
    efficiencies = data.accept.astype(np.float64) / np.asarray(
        [costs[concurrency][block] for block in BLOCKS]
    )[None, :]
    oracle_columns = np.argmax(efficiencies + np.asarray([3e-15, 2e-15, 1e-15]), axis=1)
    action_columns = np.where(actions == 8, 0, np.where(actions == 16, 1, 2))
    rows = np.arange(data.size)
    regret = efficiencies[rows, oracle_columns] - efficiencies[rows, action_columns]
    weights = row_weights(data, rows)
    return {
        "oracle_agreement": float(np.sum(weights * (oracle_columns == action_columns))),
        "mean_local_regret": float(np.sum(weights * regret)),
        "p95_local_regret": float(np.quantile(regret, 0.95)),
        "oracle_mean_efficiency": float(
            np.sum(weights * efficiencies[rows, oracle_columns])
        ),
    }


def orientations(data: TraceData, spec: Mapping[str, Any]) -> np.ndarray:
    return data.features[spec["feature"]].astype(np.float64) * float(
        spec["orientation"]
    )
