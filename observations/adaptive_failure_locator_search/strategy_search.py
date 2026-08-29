#!/usr/bin/env python3
"""Search and validate interpretable, training-free first-mismatch rules."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np


EXCLUDED_MACRO = {"aime24"}


def now() -> str:
    return datetime.now().astimezone().isoformat()


def safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in value)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def finite_or_none(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def normalized_dataset(value: str) -> str:
    return value.strip().lower().replace("_", "-")


def stable_hash(*values: str) -> int:
    digest = hashlib.sha256("\x1f".join(values).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    family: str
    decision: str
    threshold: float
    history_window: int = 0
    aggregation: str = "none"
    position_weight: float = 0.0
    original_baseline: bool = False

    @property
    def uses_history(self) -> bool:
        return self.history_window > 0


def candidate_id(
    family: str,
    decision: str,
    threshold: float,
    window: int = 0,
    aggregation: str = "none",
    position_weight: float = 0.0,
) -> str:
    return (
        f"{family}|{decision}|t={threshold:g}|h={window}|a={aggregation}|pw={position_weight:g}"
    )


def candidate_grid(
    windows: Iterable[int], aggregations: Iterable[str], grid: str
) -> list[Candidate]:
    if grid == "compact":
        ratio_thresholds = (-0.1, 0.0, 0.1, 0.15, 0.25, 0.4, 0.6)
        risk_thresholds = (0.2, 0.4, 0.6, 0.8, 0.9)
        entropy_thresholds = (1.0, 2.0, 4.0, 6.0, 8.0)
        separator_thresholds = (-0.5, 0.0, 0.5, 1.0, 1.5)
        position_weights = (0.0, 0.25)
    elif grid == "extended":
        ratio_thresholds = (-0.5, -0.25, -0.1, 0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.35, 0.5, 0.75, 1.0)
        risk_thresholds = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
        entropy_thresholds = (0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0)
        separator_thresholds = (-1.0, -0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
        position_weights = (0.0, 0.15, 0.35, 0.6)
    else:
        ratio_thresholds = (-0.25, -0.1, 0.0, 0.05, 0.1, 0.15, 0.25, 0.4, 0.6, 0.85)
        risk_thresholds = (0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9)
        entropy_thresholds = (1.0, 2.0, 3.5, 5.0, 7.0, 9.0)
        separator_thresholds = (-0.75, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5)
        position_weights = (0.0, 0.2, 0.4)

    candidates = [
        Candidate(
            candidate_id("prefix_drop", "first", 0.15),
            "prefix_drop",
            "first",
            0.15,
            original_baseline=True,
        )
    ]
    current_thresholds = {
        "prefix_drop": ratio_thresholds,
        "prefix_median_drop": ratio_thresholds,
        "local_drop": ratio_thresholds,
        "abs_risk": risk_thresholds,
        "margin_risk": risk_thresholds,
        "entropy": entropy_thresholds,
    }
    for family, thresholds in current_thresholds.items():
        for decision in ("first", "max"):
            for threshold in thresholds:
                item = Candidate(
                    candidate_id(family, decision, threshold),
                    family,
                    decision,
                    float(threshold),
                )
                if item.candidate_id != candidates[0].candidate_id:
                    candidates.append(item)

    history_thresholds = {
        "hist_good_drop": ratio_thresholds,
        "hist_error_drop": ratio_thresholds,
        "hist_separator": separator_thresholds,
    }
    for window in sorted({int(value) for value in windows}):
        for aggregation in aggregations:
            for family, thresholds in history_thresholds.items():
                for position_weight in position_weights:
                    for decision in ("first", "max"):
                        for threshold in thresholds:
                            candidates.append(
                                Candidate(
                                    candidate_id(
                                        family,
                                        decision,
                                        threshold,
                                        window,
                                        aggregation,
                                        position_weight,
                                    ),
                                    family,
                                    decision,
                                    float(threshold),
                                    history_window=window,
                                    aggregation=aggregation,
                                    position_weight=float(position_weight),
                                )
                            )
    unique = {candidate.candidate_id: candidate for candidate in candidates}
    return list(unique.values())


def aggregate(values_by_round: list[list[float]], aggregation: str) -> float:
    nonempty = [values for values in values_by_round if values]
    if not nonempty:
        return float("nan")
    if aggregation == "mean":
        return float(statistics.fmean(value for values in nonempty for value in values))
    if aggregation == "median":
        return float(statistics.median(value for values in nonempty for value in values))
    if aggregation == "ewma":
        round_values = [statistics.fmean(values) for values in nonempty]
        result = float(round_values[0])
        for value in round_values[1:]:
            result = 0.5 * float(value) + 0.5 * result
        return result
    raise ValueError(f"unknown aggregation: {aggregation}")


def split_requests(
    request_keys: dict[str, set[str]],
    *,
    seed: int,
    search_ratio: float,
    selection_ratio: float,
) -> dict[tuple[str, str], str]:
    assignments: dict[tuple[str, str], str] = {}
    for dataset, request_ids in request_keys.items():
        ordered = sorted(
            request_ids,
            key=lambda request_id: stable_hash(str(seed), dataset, request_id),
        )
        count = len(ordered)
        if count == 1:
            sizes = (1, 0, 0)
        elif count == 2:
            sizes = (1, 0, 1)
        else:
            n_search = max(1, int(math.floor(count * search_ratio)))
            n_selection = max(1, int(math.floor(count * selection_ratio)))
            if n_search + n_selection >= count:
                overflow = n_search + n_selection - (count - 1)
                reduce_search = min(overflow, max(0, n_search - 1))
                n_search -= reduce_search
                overflow -= reduce_search
                n_selection = max(1, n_selection - overflow)
            sizes = (n_search, n_selection, count - n_search - n_selection)
        boundaries = (sizes[0], sizes[0] + sizes[1])
        for index, request_id in enumerate(ordered):
            split = "search" if index < boundaries[0] else "selection" if index < boundaries[1] else "test"
            assignments[(dataset, request_id)] = split
    return assignments


def to_float_array(values: list[Any], positions: int) -> np.ndarray:
    if len(values) != positions:
        raise ValueError(f"expected {positions} positions, found {len(values)}")
    return np.asarray(
        [float(value) if finite_or_none(value) is not None else np.nan for value in values],
        dtype=np.float32,
    )


def iter_trace_records(trace_dir: Path, block_size: int):
    """Yield validated records without retaining raw JSON dictionaries."""

    for path in sorted(trace_dir.glob("failure_locator_*.jsonl")):
        fallback_dataset = path.stem.removeprefix("failure_locator_")
        with path.open("r", encoding="utf-8", errors="replace") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON {path}:{line_number}: {exc}") from exc
                if record.get("event") != "linearspec_failure_locator_round":
                    continue
                if int(record.get("block_size", -1)) != block_size:
                    raise ValueError(f"trace block size differs from configured L={block_size}: {path}")
                dataset = str(record.get("benchmark") or fallback_dataset)
                request_id = str(record.get("request_id", ""))
                if not request_id:
                    raise ValueError(f"missing request_id in {path}:{line_number}")
                record["benchmark"] = dataset
                yield record


def scan_trace_inputs(
    trace_dir: Path,
    block_size: int,
    *,
    include_boundary_rounds: bool,
) -> tuple[dict[str, Any], dict[str, set[str]], int]:
    """First pass: counts and request identities only."""

    raw: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "request_ids": set(),
            "raw_rounds": 0,
            "default_valid_rounds": 0,
            "eos_rounds": 0,
            "budget_boundary_rounds": 0,
        }
    )
    request_keys: dict[str, set[str]] = defaultdict(set)
    analysis_count = 0
    for record in iter_trace_records(trace_dir, block_size):
        dataset = str(record["benchmark"])
        request_id = str(record["request_id"])
        item = raw[dataset]
        item["request_ids"].add(request_id)
        item["raw_rounds"] += 1
        item["default_valid_rounds"] += int(bool(record.get("analysis_valid")))
        item["eos_rounds"] += int(bool(record.get("eos_hit")))
        item["budget_boundary_rounds"] += int(bool(record.get("budget_boundary")))
        request_keys[dataset].add(request_id)
        analysis_count += int(include_boundary_rounds or bool(record.get("analysis_valid")))
    return raw, request_keys, analysis_count


def build_arrays_streaming(
    trace_dir: Path,
    *,
    block_size: int,
    windows: list[int],
    aggregations: list[str],
    include_boundary_rounds: bool,
    assignments: dict[tuple[str, str], str],
    analysis_count: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Second pass: preallocate compact arrays and keep at most max(history) per request."""

    positions = block_size - 1
    arrays: dict[str, Any] = {
        "dataset": np.empty(analysis_count, dtype=object),
        "request_id": np.empty(analysis_count, dtype=object),
        "split": np.empty(analysis_count, dtype=object),
        "round_index": np.empty(analysis_count, dtype=np.int32),
        "q": np.empty(analysis_count, dtype=np.int16),
        "history_rounds": np.empty(analysis_count, dtype=np.int16),
    }
    for field in ("confidence", "margin", "entropy", "prefix_drop", "local_drop", "prefix_median_drop"):
        arrays[field] = np.empty((analysis_count, positions), dtype=np.float32)
    history_arrays: dict[tuple[int, str], dict[str, np.ndarray]] = {
        (window, aggregation): {
            name: np.full(analysis_count, np.nan, dtype=np.float32)
            for name in ("good", "error", "q")
        }
        for window in windows
        for aggregation in aggregations
    }
    max_window = max(windows)
    past_by_request: dict[tuple[str, str], deque[dict[str, Any]]] = defaultdict(
        lambda: deque(maxlen=max_window)
    )
    summaries: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "requests": set(),
            "analysis_rounds": 0,
            "failure_rounds": 0,
            "position1_failures": 0,
            "accept_lengths": [],
        }
    )
    index = 0
    for record in iter_trace_records(trace_dir, block_size):
        dataset = str(record["benchmark"])
        request_id = str(record["request_id"])
        key = (dataset, request_id)
        past = past_by_request[key]
        position = record.get("position", {})
        confidence = to_float_array(position.get("selected_confidence", []), positions)
        q = int(record["mismatch_position"]) if record.get("mismatch_position") is not None else 0
        valid = include_boundary_rounds or bool(record.get("analysis_valid"))
        if valid:
            arrays["dataset"][index] = dataset
            arrays["request_id"][index] = request_id
            arrays["split"][index] = assignments[key]
            arrays["round_index"][index] = int(record.get("round_index", 0))
            arrays["q"][index] = q
            arrays["history_rounds"][index] = len(past)
            arrays["confidence"][index] = confidence
            arrays["margin"][index] = to_float_array(position.get("top1_top2_margin", []), positions)
            arrays["entropy"][index] = to_float_array(position.get("entropy", []), positions)
            arrays["prefix_drop"][index] = to_float_array(position.get("prefix_drop_pct", []), positions)
            arrays["local_drop"][index] = to_float_array(position.get("local_drop_pct", []), positions)
            prefix_median = to_float_array(position.get("prefix_median_before", []), positions)
            with np.errstate(divide="ignore", invalid="ignore"):
                median_drop = 1.0 - confidence / prefix_median
            median_drop[~np.isfinite(median_drop)] = np.nan
            arrays["prefix_median_drop"][index] = median_drop

            recent_all = list(past)
            for window in windows:
                recent = recent_all[-window:]
                good_by_round = [item["good"] for item in recent]
                error_by_round = [[item["error"]] if item["error"] is not None else [] for item in recent]
                position_by_round = [[float(item["q"])] if item["q"] > 0 else [] for item in recent]
                for aggregation in aggregations:
                    target = history_arrays[(window, aggregation)]
                    target["good"][index] = aggregate(good_by_round, aggregation)
                    target["error"][index] = aggregate(error_by_round, aggregation)
                    target["q"][index] = aggregate(position_by_round, aggregation)

            summary = summaries[dataset]
            summary["requests"].add(request_id)
            summary["analysis_rounds"] += 1
            summary["failure_rounds"] += int(q > 0)
            summary["position1_failures"] += int(q == 1)
            summary["accept_lengths"].append(int(record.get("accept_length", q or block_size)))
            index += 1

        past.append(
            {
                "q": q,
                "good": [
                    float(value)
                    for value in confidence[: (q - 1 if q > 0 else positions)]
                    if np.isfinite(value)
                ],
                "error": float(confidence[q - 1]) if q > 0 and np.isfinite(confidence[q - 1]) else None,
            }
        )
    if index != analysis_count:
        raise AssertionError(f"streaming count changed between passes: expected {analysis_count}, found {index}")
    arrays["abs_risk"] = 1.0 - arrays["confidence"]
    arrays["margin_risk"] = 1.0 - arrays["margin"]
    arrays["history"] = history_arrays

    result_summaries: dict[str, Any] = {}
    for dataset, summary in summaries.items():
        rounds = int(summary["analysis_rounds"])
        failures = int(summary["failure_rounds"])
        result_summaries[dataset] = {
            "requests": len(summary["requests"]),
            "analysis_rounds": rounds,
            "failure_rounds": failures,
            "full_accept_rounds": rounds - failures,
            "failure_rate": failures / rounds if rounds else None,
            "position1_failures": int(summary["position1_failures"]),
            "position1_share_of_failures": summary["position1_failures"] / failures if failures else None,
            "accept_length_mean": statistics.fmean(summary["accept_lengths"]) if summary["accept_lengths"] else None,
        }
    return arrays, result_summaries


def build_arrays(
    records: list[dict[str, Any]],
    *,
    block_size: int,
    windows: list[int],
    aggregations: list[str],
    include_boundary_rounds: bool,
    split_seed: int,
    search_ratio: float,
    selection_ratio: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    request_keys: dict[str, set[str]] = defaultdict(set)
    for record in records:
        key = (str(record["benchmark"]), str(record["request_id"]))
        grouped[key].append(record)
        request_keys[key[0]].add(key[1])
    assignments = split_requests(
        request_keys,
        seed=split_seed,
        search_ratio=search_ratio,
        selection_ratio=selection_ratio,
    )

    positions = block_size - 1
    rows: dict[str, list[Any]] = defaultdict(list)
    history_rows: dict[tuple[int, str], dict[str, list[float]]] = {
        (window, aggregation): defaultdict(list)
        for window in windows
        for aggregation in aggregations
    }
    raw_summary: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"requests": set(), "analysis_rounds": 0, "failures": 0, "position1": 0, "accept_lengths": []}
    )
    for (dataset, request_id), request_records in sorted(grouped.items()):
        request_records.sort(key=lambda item: int(item.get("round_index", 0)))
        past: list[dict[str, Any]] = []
        for record in request_records:
            valid = include_boundary_rounds or bool(record.get("analysis_valid"))
            position = record.get("position", {})
            confidences = to_float_array(position.get("selected_confidence", []), positions)
            q = int(record["mismatch_position"]) if record.get("mismatch_position") is not None else 0
            current_past = past[:]
            if valid:
                rows["dataset"].append(dataset)
                rows["request_id"].append(request_id)
                rows["split"].append(assignments[(dataset, request_id)])
                rows["round_index"].append(int(record.get("round_index", 0)))
                rows["q"].append(q)
                rows["confidence"].append(confidences)
                rows["margin"].append(to_float_array(position.get("top1_top2_margin", []), positions))
                rows["entropy"].append(to_float_array(position.get("entropy", []), positions))
                rows["prefix_drop"].append(to_float_array(position.get("prefix_drop_pct", []), positions))
                rows["local_drop"].append(to_float_array(position.get("local_drop_pct", []), positions))
                prefix_median = to_float_array(position.get("prefix_median_before", []), positions)
                with np.errstate(divide="ignore", invalid="ignore"):
                    median_drop = 1.0 - confidences / prefix_median
                median_drop[~np.isfinite(median_drop)] = np.nan
                rows["prefix_median_drop"].append(median_drop)
                rows["history_rounds"].append(len(current_past))
                summary = raw_summary[dataset]
                summary["requests"].add(request_id)
                summary["analysis_rounds"] += 1
                summary["failures"] += int(q > 0)
                summary["position1"] += int(q == 1)
                summary["accept_lengths"].append(int(record.get("accept_length", q or block_size)))

                for window in windows:
                    recent = current_past[-window:]
                    good_by_round = [item["good"] for item in recent]
                    error_by_round = [[item["error"]] if item["error"] is not None else [] for item in recent]
                    position_by_round = [[float(item["q"])] if item["q"] > 0 else [] for item in recent]
                    for aggregation in aggregations:
                        target = history_rows[(window, aggregation)]
                        target["good"].append(aggregate(good_by_round, aggregation))
                        target["error"].append(aggregate(error_by_round, aggregation))
                        target["q"].append(aggregate(position_by_round, aggregation))
            past.append(
                {
                    "q": q,
                    "good": [float(value) for value in confidences[: (q - 1 if q > 0 else positions)] if np.isfinite(value)],
                    "error": float(confidences[q - 1]) if q > 0 and np.isfinite(confidences[q - 1]) else None,
                }
            )

    count = len(rows["q"])
    arrays: dict[str, Any] = {
        "dataset": np.asarray(rows["dataset"], dtype=object),
        "request_id": np.asarray(rows["request_id"], dtype=object),
        "split": np.asarray(rows["split"], dtype=object),
        "round_index": np.asarray(rows["round_index"], dtype=np.int32),
        "q": np.asarray(rows["q"], dtype=np.int16),
        "history_rounds": np.asarray(rows["history_rounds"], dtype=np.int16),
    }
    for field in ("confidence", "margin", "entropy", "prefix_drop", "local_drop", "prefix_median_drop"):
        arrays[field] = np.stack(rows[field]).astype(np.float32) if count else np.empty((0, positions), dtype=np.float32)
    arrays["abs_risk"] = 1.0 - arrays["confidence"]
    arrays["margin_risk"] = 1.0 - arrays["margin"]
    arrays["history"] = {
        key: {name: np.asarray(values, dtype=np.float32) for name, values in payload.items()}
        for key, payload in history_rows.items()
    }
    split_counts = {
        dataset: {
            split: len({request_id for (name, request_id), value in assignments.items() if name == dataset and value == split})
            for split in ("search", "selection", "test")
        }
        for dataset in request_keys
    }
    summaries = {}
    for dataset, summary in raw_summary.items():
        rounds = int(summary["analysis_rounds"])
        failures = int(summary["failures"])
        summaries[dataset] = {
            "requests": len(summary["requests"]),
            "analysis_rounds": rounds,
            "failure_rounds": failures,
            "full_accept_rounds": rounds - failures,
            "failure_rate": failures / rounds if rounds else None,
            "position1_failures": int(summary["position1"]),
            "position1_share_of_failures": summary["position1"] / failures if failures else None,
            "accept_length_mean": statistics.fmean(summary["accept_lengths"]) if summary["accept_lengths"] else None,
            "request_split_counts": split_counts.get(dataset, {}),
        }
    return arrays, summaries


def original_predictions(arrays: dict[str, Any]) -> np.ndarray:
    return predictions_from_score(arrays["prefix_drop"], threshold=0.15, decision="first")


def score_for_candidate(candidate: Candidate, arrays: dict[str, Any]) -> np.ndarray:
    if not candidate.uses_history:
        return arrays[candidate.family]
    history = arrays["history"][(candidate.history_window, candidate.aggregation)]
    confidence = arrays["confidence"]
    reference_name = {
        "hist_good_drop": "good",
        "hist_error_drop": "error",
    }.get(candidate.family)
    with np.errstate(divide="ignore", invalid="ignore"):
        if reference_name is not None:
            score = 1.0 - confidence / history[reference_name][:, None]
        elif candidate.family == "hist_separator":
            denominator = history["good"] - history["error"]
            score = (history["good"][:, None] - confidence) / denominator[:, None]
        else:
            raise ValueError(f"unknown history family: {candidate.family}")
    score = score.astype(np.float32)
    score[~np.isfinite(score)] = np.nan
    if candidate.position_weight:
        positions = np.arange(1, confidence.shape[1] + 1, dtype=np.float32)[None, :]
        history_q = history["q"][:, None]
        prior = 1.0 - np.abs(positions - history_q) / max(1, confidence.shape[1])
        prior = np.clip(prior, 0.0, 1.0)
        prior[~np.isfinite(prior)] = 0.0
        score = score + candidate.position_weight * prior
    return score


def predictions_from_score(score: np.ndarray, *, threshold: float, decision: str) -> np.ndarray:
    count = score.shape[0]
    clean = np.where(np.isfinite(score), score, -np.inf)
    if decision == "first":
        crossing = clean > float(threshold)
        any_crossing = crossing.any(axis=1)
        predicted = np.where(any_crossing, crossing.argmax(axis=1) + 1, 0)
    elif decision == "max":
        indices = clean.argmax(axis=1)
        maxima = clean[np.arange(count), indices] if count else np.empty(0)
        predicted = np.where(maxima > float(threshold), indices + 1, 0)
    else:
        raise ValueError(f"unknown decision: {decision}")
    return predicted.astype(np.int16)


def predict(candidate: Candidate, arrays: dict[str, Any], fallback: np.ndarray) -> np.ndarray:
    score = score_for_candidate(candidate, arrays)
    predicted = predictions_from_score(score, threshold=candidate.threshold, decision=candidate.decision)
    if candidate.uses_history:
        unavailable = ~np.isfinite(score).any(axis=1)
        predicted[unavailable] = fallback[unavailable]
    return predicted


METRIC_FIELDS = (
    "coverage", "precision", "recall", "exact_f1", "hits_per_100",
    "full_false_alarm", "miss_rate", "early_rate", "late_rate", "mae",
    "bias", "tol1_recall", "tol2_recall", "position1_recall",
)


def safe_ratio(numerator: int, denominator: int) -> Optional[float]:
    return numerator / denominator if denominator else None


def metrics(predicted: np.ndarray, q: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    p = predicted[mask]
    labels = q[mask]
    rounds = int(labels.size)
    failures_mask = labels > 0
    attempts_mask = p > 0
    exact_mask = attempts_mask & failures_mask & (p == labels)
    failures = int(failures_mask.sum())
    attempts = int(attempts_mask.sum())
    exact = int(exact_mask.sum())
    full = rounds - failures
    precision = safe_ratio(exact, attempts)
    recall = safe_ratio(exact, failures)
    exact_f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall > 0
        else 0.0 if failures > 0 and attempts > 0 else None
    )
    attempted_failures = attempts_mask & failures_mask
    position1_mask = labels == 1
    errors = p[attempted_failures].astype(np.float64) - labels[attempted_failures].astype(np.float64)
    return {
        "rounds": rounds,
        "failures": failures,
        "full_accepts": full,
        "attempts": attempts,
        "exact_hits": exact,
        "coverage": safe_ratio(attempts, rounds),
        "precision": precision,
        "recall": recall,
        "exact_f1": exact_f1,
        "hits_per_100": safe_ratio(100 * exact, rounds),
        "full_false_alarm": safe_ratio(int((attempts_mask & ~failures_mask).sum()), full),
        "miss_rate": safe_ratio(int((~attempts_mask & failures_mask).sum()), failures),
        "early_rate": safe_ratio(int((attempted_failures & (p < labels)).sum()), failures),
        "late_rate": safe_ratio(int((attempted_failures & (p > labels)).sum()), failures),
        "mae": float(np.mean(np.abs(errors))) if errors.size else None,
        "bias": float(np.mean(errors)) if errors.size else None,
        "tol1_recall": safe_ratio(int((attempted_failures & (np.abs(p - labels) <= 1)).sum()), failures),
        "tol2_recall": safe_ratio(int((attempted_failures & (np.abs(p - labels) <= 2)).sum()), failures),
        "position1_failures": int(position1_mask.sum()),
        "position1_recall": safe_ratio(int((position1_mask & (p == 1)).sum()), int(position1_mask.sum())),
    }


def macro_metrics(by_dataset: dict[str, dict[str, Any]]) -> dict[str, Any]:
    included = {
        dataset: payload
        for dataset, payload in by_dataset.items()
        if normalized_dataset(dataset) not in EXCLUDED_MACRO and payload.get("rounds", 0) > 0
    }
    result: dict[str, Any] = {"datasets": len(included)}
    for field in METRIC_FIELDS:
        values = [float(payload[field]) for payload in included.values() if payload.get(field) is not None]
        result[field] = statistics.fmean(values) if values else None
    f1_values = [float(payload["exact_f1"]) for payload in included.values() if payload.get("exact_f1") is not None]
    result["min_exact_f1"] = min(f1_values) if f1_values else None
    result["std_exact_f1"] = statistics.pstdev(f1_values) if len(f1_values) > 1 else 0.0 if f1_values else None
    return result


def evaluate(
    predicted: np.ndarray,
    arrays: dict[str, Any],
    *,
    split: Optional[str] = None,
    extra_mask: Optional[np.ndarray] = None,
) -> dict[str, Any]:
    mask = np.ones(len(arrays["q"]), dtype=bool)
    if split is not None:
        mask &= arrays["split"] == split
    if extra_mask is not None:
        mask &= extra_mask
    by_dataset = {
        str(dataset): metrics(predicted, arrays["q"], mask & (arrays["dataset"] == dataset))
        for dataset in sorted(set(arrays["dataset"].tolist()))
    }
    return {"by_dataset": by_dataset, "macro": macro_metrics(by_dataset)}


def ranking_key(result: dict[str, Any], stage: str) -> tuple[float, ...]:
    metric = result.get(stage, {}).get("macro", {})
    f1 = metric.get("exact_f1")
    minimum_f1 = metric.get("min_exact_f1")
    hits = metric.get("hits_per_100")
    false_alarm = metric.get("full_false_alarm")
    mae = metric.get("mae")
    return (
        float(f1) if f1 is not None else -1.0,
        float(minimum_f1) if minimum_f1 is not None else -1.0,
        float(hits) if hits is not None else -1.0,
        -float(false_alarm) if false_alarm is not None else -1.0,
        -float(mae) if mae is not None else -1e9,
    )


def capped_search_mask(arrays: dict[str, Any], cap_per_dataset: int, seed: int) -> np.ndarray:
    mask = arrays["split"] == "search"
    if cap_per_dataset <= 0:
        return mask
    result = np.zeros(len(mask), dtype=bool)
    for dataset in sorted(set(arrays["dataset"].tolist())):
        indices = np.flatnonzero(mask & (arrays["dataset"] == dataset))
        if len(indices) > cap_per_dataset:
            indices = np.asarray(
                sorted(
                    indices.tolist(),
                    key=lambda index: stable_hash(
                        str(seed), dataset, str(arrays["request_id"][index]), str(arrays["round_index"][index])
                    ),
                )[:cap_per_dataset],
                dtype=np.int64,
            )
        result[indices] = True
    return result


def candidate_result(
    candidate: Candidate,
    predicted: np.ndarray,
    arrays: dict[str, Any],
    *,
    search_mask: np.ndarray,
    with_search_by_dataset: bool = False,
) -> dict[str, Any]:
    search = evaluate(predicted, arrays, extra_mask=search_mask)
    if not with_search_by_dataset:
        search = {"macro": search["macro"]}
    return {"candidate": asdict(candidate), "search": search}


def choose_shortlist(results: list[dict[str, Any]], limit: int) -> list[int]:
    ordered = sorted(range(len(results)), key=lambda index: ranking_key(results[index], "search"), reverse=True)
    selected = ordered[:limit]
    groups: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index in ordered:
        spec = results[index]["candidate"]
        groups[(spec["family"], int(spec["history_window"]))].append(index)
    for indices in groups.values():
        selected.extend(indices[:2])
    original = next(
        (index for index, result in enumerate(results) if result["candidate"].get("original_baseline")),
        None,
    )
    if original is not None:
        selected.append(original)
    return list(dict.fromkeys(selected))


def bootstrap_ci(
    predicted: np.ndarray,
    arrays: dict[str, Any],
    *,
    split: str,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    if replicates <= 0:
        return {}
    rng = np.random.default_rng(seed)
    output: dict[str, Any] = {"replicates": replicates, "by_dataset": {}}
    macro_values: list[float] = []
    datasets = [
        str(dataset)
        for dataset in sorted(set(arrays["dataset"].tolist()))
        if normalized_dataset(str(dataset)) not in EXCLUDED_MACRO
    ]
    per_dataset_samples: dict[str, list[float]] = {dataset: [] for dataset in datasets}
    request_indices: dict[str, dict[str, np.ndarray]] = {}
    for dataset in datasets:
        base = (arrays["dataset"] == dataset) & (arrays["split"] == split)
        grouped_indices: dict[str, list[int]] = defaultdict(list)
        for index in np.flatnonzero(base):
            grouped_indices[str(arrays["request_id"][index])].append(int(index))
        request_indices[dataset] = {
            request_id: np.asarray(indices, dtype=np.int64)
            for request_id, indices in sorted(grouped_indices.items())
        }
    for _ in range(replicates):
        replicate_f1 = []
        for dataset in datasets:
            groups = request_indices[dataset]
            keys = list(groups)
            if not keys:
                continue
            sampled = rng.choice(keys, size=len(keys), replace=True)
            indices = np.concatenate([groups[str(key)] for key in sampled])
            selected_pred = predicted[indices]
            selected_q = arrays["q"][indices]
            value = metrics(selected_pred, selected_q, np.ones(len(indices), dtype=bool))["exact_f1"]
            if value is not None:
                per_dataset_samples[dataset].append(float(value))
                replicate_f1.append(float(value))
        if replicate_f1:
            macro_values.append(float(statistics.fmean(replicate_f1)))
    for dataset, values in per_dataset_samples.items():
        output["by_dataset"][dataset] = {
            "exact_f1_ci95": [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]
            if values else None
        }
    output["macro_exact_f1_ci95"] = (
        [float(np.percentile(macro_values, 2.5)), float(np.percentile(macro_values, 97.5))]
        if macro_values else None
    )
    return output


def run_search(args: argparse.Namespace) -> dict[str, Any]:
    windows = sorted({int(value) for value in args.history_windows.split(",") if value.strip()})
    aggregations = [value.strip() for value in args.aggregations.split(",") if value.strip()]
    raw, request_keys, analysis_count = scan_trace_inputs(
        args.trace_dir,
        args.block_size,
        include_boundary_rounds=args.include_boundary_rounds,
    )
    assignments = split_requests(
        request_keys,
        seed=args.split_seed,
        search_ratio=args.search_ratio,
        selection_ratio=args.selection_ratio,
    )
    arrays, summaries = build_arrays_streaming(
        args.trace_dir,
        block_size=args.block_size,
        windows=windows,
        aggregations=aggregations,
        include_boundary_rounds=args.include_boundary_rounds,
        assignments=assignments,
        analysis_count=analysis_count,
    )
    for dataset, item in raw.items():
        summary = summaries.setdefault(dataset, {})
        summary["request_split_counts"] = {
            split: len(
                {
                    request_id
                    for (name, request_id), assigned in assignments.items()
                    if name == dataset and assigned == split
                }
            )
            for split in ("search", "selection", "test")
        }
        summary.update(
            {
                "raw_requests": len(item["request_ids"]),
                "raw_rounds": item["raw_rounds"],
                "default_valid_rounds": item["default_valid_rounds"],
                "eos_rounds": item["eos_rounds"],
                "budget_boundary_rounds": item["budget_boundary_rounds"],
                "include_boundary_rounds": args.include_boundary_rounds,
            }
        )
        atomic_json(args.run_dir / "summaries" / f"failure_locator_{safe_name(dataset)}.json", summary)

    payload: dict[str, Any] = {
        "schema_version": 1,
        "updated_at": now(),
        "status": "empty" if len(arrays["q"]) == 0 else "ok",
        "block_size": args.block_size,
        "excluded_macro_datasets": sorted(EXCLUDED_MACRO),
        "split": {
            "seed": args.split_seed,
            "search_ratio": args.search_ratio,
            "selection_ratio": args.selection_ratio,
            "test_ratio": 1.0 - args.search_ratio - args.selection_ratio,
            "unit": "request",
        },
        "grid": args.grid,
        "candidate_count": 0,
        "search_round_cap_per_dataset": args.search_max_rounds_per_dataset,
        "dataset_summaries": summaries,
    }
    if len(arrays["q"]) == 0:
        atomic_json(args.output_json, payload)
        return payload

    candidates = candidate_grid(windows, aggregations, args.grid)
    payload["candidate_count"] = len(candidates)
    fallback = original_predictions(arrays)
    search_mask = capped_search_mask(
        arrays, args.search_max_rounds_per_dataset, args.split_seed
    )
    all_results: list[dict[str, Any]] = []
    for candidate in candidates:
        predicted = predict(candidate, arrays, fallback)
        all_results.append(
            candidate_result(candidate, predicted, arrays, search_mask=search_mask)
        )

    shortlist_indices = choose_shortlist(all_results, args.shortlist)
    shortlist: list[dict[str, Any]] = []
    prediction_cache: dict[str, np.ndarray] = {}
    for index in shortlist_indices:
        result = all_results[index]
        candidate = candidates[index]
        predicted = predict(candidate, arrays, fallback)
        prediction_cache[candidate.candidate_id] = predicted
        result["selection"] = evaluate(predicted, arrays, split="selection")
        shortlist.append(result)
    has_selection = any(
        item["selection"]["macro"].get("datasets", 0) > 0
        and item["selection"]["macro"].get("exact_f1") is not None
        for item in shortlist
    )
    selection_stage = "selection" if has_selection else "search"
    winner_result = max(shortlist, key=lambda item: ranking_key(item, selection_stage))
    winner_spec = Candidate(**winner_result["candidate"])
    winner_predictions = prediction_cache[winner_spec.candidate_id]
    winner_result["test"] = evaluate(winner_predictions, arrays, split="test")
    winner_result["all_descriptive"] = evaluate(winner_predictions, arrays)
    winner_result["cold_start"] = evaluate(
        winner_predictions, arrays, extra_mask=arrays["history_rounds"] < max(1, winner_spec.history_window)
    )
    winner_result["history_ready"] = evaluate(
        winner_predictions, arrays, extra_mask=arrays["history_rounds"] >= max(1, winner_spec.history_window)
    )

    original_index = next(index for index, candidate in enumerate(candidates) if candidate.original_baseline)
    original_result = all_results[original_index]
    original_result["selection"] = evaluate(fallback, arrays, split="selection")
    original_result["test"] = evaluate(fallback, arrays, split="test")
    original_result["all_descriptive"] = evaluate(fallback, arrays)

    # Evaluate only the leakage-free shortlist on test; this is diagnostic and
    # cannot change the already frozen winner.
    for result in shortlist:
        spec = Candidate(**result["candidate"])
        if "test" not in result:
            result["test"] = evaluate(prediction_cache[spec.candidate_id], arrays, split="test")

    ablations: list[dict[str, Any]] = []
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in shortlist:
        spec = result["candidate"]
        groups["history" if int(spec["history_window"]) > 0 else "current_only"].append(result)
        groups[f"family:{spec['family']}"].append(result)
        if int(spec["history_window"]) > 0:
            groups[f"window:{spec['history_window']}"].append(result)
    for group, results in sorted(groups.items()):
        best = max(results, key=lambda item: ranking_key(item, selection_stage))
        ablations.append(
            {
                "group": group,
                "candidate": best["candidate"],
                "search_macro": best["search"]["macro"],
                "selection_macro": best["selection"]["macro"],
                "test_macro": best["test"]["macro"],
            }
        )

    baseline_coverage = original_result[selection_stage]["macro"].get("coverage")
    matched = None
    if baseline_coverage is not None:
        within_one_point = [
            item
            for item in shortlist
            if item[selection_stage]["macro"].get("coverage") is not None
            and abs(item[selection_stage]["macro"]["coverage"] - baseline_coverage) <= 0.01
        ]
        if within_one_point:
            matched = max(within_one_point, key=lambda item: ranking_key(item, selection_stage))
        else:
            matched = min(
                shortlist,
                key=lambda item: abs(
                    (item[selection_stage]["macro"].get("coverage") or 0.0) - baseline_coverage
                ),
            )

    per_dataset_oracle: dict[str, Any] = {}
    for dataset in sorted(set(arrays["dataset"].tolist())):
        if normalized_dataset(str(dataset)) in EXCLUDED_MACRO:
            continue
        best = max(
            shortlist,
            key=lambda item: (
                item["test"]["by_dataset"].get(str(dataset), {}).get("exact_f1")
                if item["test"]["by_dataset"].get(str(dataset), {}).get("exact_f1") is not None
                else -1.0
            ),
        )
        oracle_f1 = best["test"]["by_dataset"].get(str(dataset), {}).get("exact_f1")
        global_f1 = winner_result["test"]["by_dataset"].get(str(dataset), {}).get("exact_f1")
        per_dataset_oracle[str(dataset)] = {
            "candidate_id": best["candidate"]["candidate_id"],
            "oracle_exact_f1": oracle_f1,
            "global_exact_f1": global_f1,
            "regret": oracle_f1 - global_f1 if oracle_f1 is not None and global_f1 is not None else None,
            "diagnostic_only_test_selected": True,
        }

    comparison = {"wins": 0, "ties": 0, "losses": 0, "by_dataset": {}}
    for dataset, winner_metrics in winner_result["test"]["by_dataset"].items():
        if normalized_dataset(dataset) in EXCLUDED_MACRO:
            continue
        winner_f1 = winner_metrics.get("exact_f1")
        baseline_f1 = original_result["test"]["by_dataset"].get(dataset, {}).get("exact_f1")
        delta = winner_f1 - baseline_f1 if winner_f1 is not None and baseline_f1 is not None else None
        outcome = "NA"
        if delta is not None:
            outcome = "win" if delta > 1e-12 else "loss" if delta < -1e-12 else "tie"
            comparison[{"win": "wins", "loss": "losses", "tie": "ties"}[outcome]] += 1
        comparison["by_dataset"][dataset] = {
            "winner_exact_f1": winner_f1,
            "baseline_exact_f1": baseline_f1,
            "delta": delta,
            "outcome": outcome,
        }

    ordered_all = sorted(all_results, key=lambda item: ranking_key(item, "search"), reverse=True)
    ordered_shortlist = sorted(shortlist, key=lambda item: ranking_key(item, selection_stage), reverse=True)
    payload.update(
        {
            "status": "ok",
            "selection_stage_used": selection_stage,
            "analysis_rounds": len(arrays["q"]),
            "split_round_counts": {
                split: int((arrays["split"] == split).sum()) for split in ("search", "selection", "test")
            },
            "search_rounds_after_cap": int(search_mask.sum()),
            "winner": winner_result,
            "original_baseline": original_result,
            "matched_coverage": (
                {
                    "candidate": matched["candidate"],
                    "selection_macro": matched["selection"]["macro"],
                    "test_macro": matched["test"]["macro"],
                }
                if matched is not None
                else None
            ),
            "ablations": ablations,
            "per_dataset_oracle": per_dataset_oracle,
            "winner_vs_original": comparison,
            "top_search": ordered_all[: args.report_top],
            "top_shortlist": ordered_shortlist[: args.report_top],
            "all_search_candidates": [
                {
                    "candidate": result["candidate"],
                    "search_macro": result["search"]["macro"],
                }
                for result in ordered_all
            ],
            "bootstrap": bootstrap_ci(
                winner_predictions,
                arrays,
                split="test",
                replicates=args.bootstrap_replicates,
                seed=args.split_seed + 991,
            ),
            "selection_contract": {
                "primary": "equal-dataset macro Exact-F1",
                "tie_breakers": ["higher minimum-dataset Exact-F1", "hits_per_100", "lower full_false_alarm", "lower mae"],
                "test_used_for_selection": False,
                "aime24_used_for_selection": False,
                "cold_start_fallback": "original strict prefix_drop > 0.15",
                "per_dataset_parameters": False,
                "oracle_is_diagnostic_only": True,
            },
        }
    )
    atomic_json(args.output_json, payload)
    return payload


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--trace-dir", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--history-windows", default="1,2,4")
    parser.add_argument("--aggregations", default="mean,median,ewma")
    parser.add_argument("--grid", choices=["compact", "standard", "extended"], default="standard")
    parser.add_argument("--split-seed", type=int, default=20260828)
    parser.add_argument("--search-ratio", type=float, default=0.6)
    parser.add_argument("--selection-ratio", type=float, default=0.2)
    parser.add_argument("--shortlist", type=int, default=80)
    parser.add_argument("--report-top", type=int, default=20)
    parser.add_argument("--search-max-rounds-per-dataset", type=int, default=50000)
    parser.add_argument("--bootstrap-replicates", type=int, default=200)
    parser.add_argument("--include-boundary-rounds", action="store_true")
    return parser


def main() -> int:
    args = create_parser().parse_args()
    args.run_dir = args.run_dir.resolve()
    args.trace_dir = (args.trace_dir or args.run_dir / "traces").resolve()
    args.output_json = (args.output_json or args.run_dir / "analysis" / "strategy_search.json").resolve()
    if args.block_size < 2:
        raise ValueError("--block-size must be >=2")
    if not 0 < args.search_ratio < 1 or not 0 <= args.selection_ratio < 1:
        raise ValueError("invalid split ratios")
    if args.search_ratio + args.selection_ratio >= 1:
        raise ValueError("search_ratio + selection_ratio must be <1")
    if args.shortlist <= 0 or args.report_top <= 0:
        raise ValueError("shortlist and report-top must be positive")
    aggregations = {value.strip() for value in args.aggregations.split(",") if value.strip()}
    if not aggregations or not aggregations <= {"mean", "median", "ewma"}:
        raise ValueError("aggregations must be a nonempty subset of mean,median,ewma")
    payload = run_search(args)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "candidate_count": payload["candidate_count"],
                "analysis_rounds": payload.get("analysis_rounds", 0),
                "winner": payload.get("winner", {}).get("candidate", {}).get("candidate_id"),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
