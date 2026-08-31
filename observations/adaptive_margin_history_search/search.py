#!/usr/bin/env python3
"""Global training-free search for history-conditioned margin-risk thresholds.

The label is the verifier's first mismatch q.  A strategy reports the first
position whose current margin risk, ``1 - (top1 - top2)``, strictly exceeds a
causally computed per-round threshold.  History is reconstructed only from
earlier rounds of the same request.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from reporting import atomic_json, render_report


EXCLUDED_MACRO = {"aime24"}
FEATURE_NAMES = ("good_mean", "good_q75", "good_q90", "error", "sep", "accept", "full")
MACRO_FIELDS = (
    "report_rate",
    "precision",
    "recall",
    "exact_f1",
    "hits_per_100",
    "correct_fp_round",
    "correct_fp_report",
    "position_fpr",
    "early_rate",
    "full_false_alarm_round",
    "full_false_alarm_given_full",
    "miss_rate",
    "late_rate",
    "mae",
    "bias",
    "tol1_recall",
    "tol2_recall",
    "position1_recall",
)


def format_duration(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds < 0:
        return "--:--"
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:d}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


class ProgressBar:
    """Dependency-free progress reporter for terminals and redirected logs."""

    def __init__(
        self,
        label: str,
        total: int,
        *,
        enabled: bool = True,
        unit: str = "item",
        scale: float = 1.0,
        stream: Any | None = None,
    ) -> None:
        self.label = label
        self.total = max(0, int(total))
        self.enabled = bool(enabled)
        self.unit = unit
        self.scale = float(scale)
        self.stream = stream if stream is not None else sys.stderr
        self.is_tty = bool(getattr(self.stream, "isatty", lambda: False)())
        self.started = time.monotonic()
        self.completed = 0
        self.detail = ""
        self.last_emit = 0.0
        self.next_log_fraction = 0.0
        self.last_width = 0
        self.complete_emitted = False
        if self.enabled:
            self.update(0, force=True)

    def _quantity(self, value: int) -> str:
        scaled = value / self.scale
        return f"{scaled:,.1f}" if self.scale != 1.0 else f"{int(scaled):,d}"

    def _line(self, now: float) -> str:
        fraction = min(1.0, self.completed / self.total) if self.total else 1.0
        elapsed = max(now - self.started, 1e-9)
        rate = self.completed / self.scale / elapsed
        remaining = (self.total - self.completed) / self.scale
        eta = remaining / rate if rate > 0 else float("inf")
        width = 24
        filled = min(width, int(fraction * width))
        bar = "#" * filled + "-" * (width - filled)
        suffix = f" | {self.detail[:48]}" if self.detail else ""
        return (
            f"[{self.label}] [{bar}] {fraction * 100:6.2f}% "
            f"{self._quantity(self.completed)}/{self._quantity(self.total)} {self.unit} "
            f"| {rate:,.1f} {self.unit}/s | elapsed {format_duration(elapsed)} "
            f"| ETA {format_duration(eta)}{suffix}"
        )

    def update(self, completed: int, *, detail: str | None = None, force: bool = False) -> None:
        if not self.enabled:
            return
        self.completed = min(self.total, max(self.completed, int(completed)))
        if detail is not None:
            self.detail = " ".join(str(detail).split())
        now = time.monotonic()
        fraction = min(1.0, self.completed / self.total) if self.total else 1.0
        complete = self.completed >= self.total
        if self.is_tty:
            should_emit = force or complete or now - self.last_emit >= 0.2
        else:
            should_emit = force or complete or fraction + 1e-12 >= self.next_log_fraction
        if not should_emit or (complete and self.complete_emitted):
            return
        line = self._line(now)
        if self.is_tty:
            print(
                "\r" + line.ljust(self.last_width),
                end="\n" if complete else "",
                file=self.stream,
                flush=True,
            )
            self.last_width = len(line)
        else:
            print(line, file=self.stream, flush=True)
            self.next_log_fraction = min(1.0, (math.floor(fraction / 0.02) + 1) * 0.02)
        self.last_emit = now
        self.complete_emitted = complete

    def finish(self, *, detail: str | None = None) -> None:
        self.update(self.total, detail=detail, force=True)

    def close(self) -> None:
        if not self.enabled or self.complete_emitted:
            return
        if self.is_tty:
            print(file=self.stream, flush=True)
        else:
            self.update(self.completed, force=True)


def normalized_dataset(value: str) -> str:
    return value.strip().lower().replace("_", "-")


def safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in value)


def stable_hash(*values: str) -> int:
    digest = hashlib.sha256("\x1f".join(values).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def safe_ratio(numerator: int | float, denominator: int | float) -> float | None:
    return float(numerator) / float(denominator) if denominator else None


def finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def strict_first(score: np.ndarray, threshold: np.ndarray | float) -> np.ndarray:
    clean = np.where(np.isfinite(score), score, -np.inf)
    crossing = clean > np.asarray(threshold, dtype=np.float32).reshape(-1, 1) if np.ndim(threshold) else clean > float(threshold)
    any_crossing = crossing.any(axis=1)
    return np.where(any_crossing, crossing.argmax(axis=1) + 1, 0).astype(np.int16)


def aggregate_scalar(values: list[float], aggregation: str) -> float:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    if not clean:
        return float("nan")
    if aggregation == "mean":
        return float(statistics.fmean(clean))
    if aggregation == "median":
        return float(statistics.median(clean))
    if aggregation == "ewma":
        result = clean[0]
        for value in clean[1:]:
            result = 0.5 * value + 0.5 * result
        return float(result)
    raise ValueError(f"unknown aggregation: {aggregation}")


def trace_paths(trace_dir: Path, datasets: set[str] | None) -> list[Path]:
    paths = sorted(trace_dir.glob("failure_locator_*.jsonl"))
    if datasets is None:
        return paths
    selected = []
    for path in paths:
        fallback = path.stem.removeprefix("failure_locator_")
        if normalized_dataset(fallback) in datasets:
            selected.append(path)
    return selected


def iter_records(
    trace_dir: Path,
    datasets: set[str] | None = None,
    *,
    progress: ProgressBar | None = None,
):
    paths = trace_paths(trace_dir, datasets)
    if not paths:
        raise FileNotFoundError(f"no matching failure_locator_*.jsonl in {trace_dir}")
    completed_bytes = 0
    for path in paths:
        fallback = path.stem.removeprefix("failure_locator_")
        file_bytes = path.stat().st_size
        with path.open("r", encoding="utf-8", errors="replace") as stream:
            for line_number, line in enumerate(stream, 1):
                if progress is not None and line_number % 4096 == 0:
                    # The text wrapper can read ahead by one buffer; cap it so
                    # the byte percentage remains monotonic and bounded.
                    progress.update(
                        completed_bytes + min(file_bytes, stream.buffer.tell()),
                        detail=path.name,
                    )
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON {path}:{line_number}: {exc}") from exc
                if record.get("event") != "linearspec_failure_locator_round":
                    continue
                record["benchmark"] = str(record.get("benchmark") or fallback)
                yield record
        completed_bytes += file_bytes
        if progress is not None:
            progress.update(completed_bytes, detail=path.name)


def scan_inputs(
    trace_dir: Path,
    *,
    datasets: set[str] | None,
    configured_block_size: int | None,
    include_boundary_rounds: bool,
    show_progress: bool = True,
) -> tuple[int, dict[str, set[str]], dict[str, dict[str, Any]], int]:
    block_size = configured_block_size
    requests: dict[str, set[str]] = defaultdict(set)
    raw: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"raw_rounds": 0, "analysis_rounds": 0, "eos_rounds": 0, "boundary_rounds": 0}
    )
    count = 0
    paths = trace_paths(trace_dir, datasets)
    if not paths:
        raise FileNotFoundError(f"no matching failure_locator_*.jsonl in {trace_dir}")
    progress = ProgressBar(
        "trace scan",
        sum(path.stat().st_size for path in paths),
        enabled=show_progress,
        unit="MiB",
        scale=1024 * 1024,
    )
    try:
        for record in iter_records(trace_dir, datasets, progress=progress):
            current_block = int(record.get("block_size", -1))
            if current_block < 2:
                raise ValueError("trace record has invalid block_size")
            if block_size is None:
                block_size = current_block
            if current_block != block_size:
                raise ValueError(f"mixed/configuration-mismatched block size: {current_block} vs {block_size}")
            dataset = str(record["benchmark"])
            request_id = str(record.get("request_id", ""))
            if not request_id:
                raise ValueError("trace record lacks request_id")
            requests[dataset].add(request_id)
            item = raw[dataset]
            item["raw_rounds"] += 1
            item["eos_rounds"] += int(bool(record.get("eos_hit")))
            item["boundary_rounds"] += int(bool(record.get("budget_boundary")))
            valid = include_boundary_rounds or bool(record.get("analysis_valid"))
            item["analysis_rounds"] += int(valid)
            count += int(valid)
        progress.finish(detail=f"{len(paths)} trace files")
    finally:
        progress.close()
    if block_size is None:
        raise ValueError("no locator rounds were found")
    return block_size, requests, raw, count


def split_requests(
    requests: dict[str, set[str]], *, seed: int, search_ratio: float, selection_ratio: float
) -> dict[tuple[str, str], str]:
    assignments: dict[tuple[str, str], str] = {}
    for dataset, request_ids in requests.items():
        ordered = sorted(request_ids, key=lambda request_id: stable_hash(str(seed), dataset, request_id))
        count = len(ordered)
        if count == 1:
            n_search, n_selection = 1, 0
        elif count == 2:
            n_search, n_selection = 1, 0
        else:
            n_search = max(1, int(math.floor(count * search_ratio)))
            n_selection = max(1, int(math.floor(count * selection_ratio)))
            if n_search + n_selection >= count:
                n_search = max(1, count - n_selection - 1)
            if n_search + n_selection >= count:
                n_selection = max(0, count - n_search - 1)
        for index, request_id in enumerate(ordered):
            split = "search" if index < n_search else "selection" if index < n_search + n_selection else "test"
            assignments[(dataset, request_id)] = split
    return assignments


def position_array(position: dict[str, Any], field_name: str, positions: int) -> np.ndarray:
    values = position.get(field_name, [])
    if len(values) != positions:
        raise ValueError(f"{field_name}: expected {positions} values, found {len(values)}")
    return np.asarray(
        [float(value) if finite_float(value) is not None else np.nan for value in values],
        dtype=np.float32,
    )


def build_arrays(
    trace_dir: Path,
    *,
    datasets: set[str] | None,
    block_size: int,
    windows: list[int],
    aggregations: list[str],
    include_boundary_rounds: bool,
    assignments: dict[tuple[str, str], str],
    analysis_count: int,
    show_progress: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    positions = block_size - 1
    arrays: dict[str, Any] = {
        "dataset": np.empty(analysis_count, dtype=object),
        "request_id": np.empty(analysis_count, dtype=object),
        "split": np.empty(analysis_count, dtype=object),
        "round_index": np.empty(analysis_count, dtype=np.int32),
        "q": np.empty(analysis_count, dtype=np.int16),
        "history_rounds": np.empty(analysis_count, dtype=np.int16),
        "risk": np.empty((analysis_count, positions), dtype=np.float32),
        "original_p": np.empty(analysis_count, dtype=np.int16),
    }
    features: dict[tuple[int, str, str], np.ndarray] = {
        (window, aggregation, name): np.full(analysis_count, np.nan, dtype=np.float32)
        for window in windows
        for aggregation in aggregations
        for name in FEATURE_NAMES
    }
    histories: dict[tuple[str, str], deque[dict[str, float]]] = defaultdict(
        lambda: deque(maxlen=max(windows))
    )
    summaries: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"requests": set(), "analysis_rounds": 0, "failure_rounds": 0, "position1_failures": 0}
    )
    index = 0
    progress = ProgressBar(
        "history features", analysis_count, enabled=show_progress, unit="round"
    )
    update_interval = max(1, analysis_count // 2000)
    for record in iter_records(trace_dir, datasets):
        dataset = str(record["benchmark"])
        request_id = str(record["request_id"])
        key = (dataset, request_id)
        history = histories[key]
        position = record.get("position") or {}
        margin = position_array(position, "top1_top2_margin", positions)
        risk = 1.0 - margin
        q = int(record["mismatch_position"]) if record.get("mismatch_position") is not None else 0
        if not 0 <= q <= positions:
            raise ValueError(f"invalid mismatch_position={q} for L={block_size}")
        prefix_drop = position_array(position, "prefix_drop_pct", positions)
        valid = include_boundary_rounds or bool(record.get("analysis_valid"))
        if valid:
            arrays["dataset"][index] = dataset
            arrays["request_id"][index] = request_id
            arrays["split"][index] = assignments[key]
            arrays["round_index"][index] = int(record.get("round_index", 0))
            arrays["q"][index] = q
            arrays["history_rounds"][index] = len(history)
            arrays["risk"][index] = risk
            arrays["original_p"][index] = strict_first(prefix_drop[None, :], 0.15)[0]
            recent_all = list(history)
            for window in windows:
                recent = recent_all[-window:]
                for aggregation in aggregations:
                    for name in FEATURE_NAMES:
                        features[(window, aggregation, name)][index] = aggregate_scalar(
                            [item[name] for item in recent], aggregation
                        )
            summary = summaries[dataset]
            summary["requests"].add(request_id)
            summary["analysis_rounds"] += 1
            summary["failure_rounds"] += int(q > 0)
            summary["position1_failures"] += int(q == 1)
            index += 1
            if index % update_interval == 0:
                progress.update(index, detail=dataset)

        correct_risk = risk[: q - 1] if q > 0 else risk
        clean_good = correct_risk[np.isfinite(correct_risk)]
        good_mean = float(np.mean(clean_good)) if clean_good.size else float("nan")
        good_q75 = float(np.percentile(clean_good, 75)) if clean_good.size else float("nan")
        good_q90 = float(np.percentile(clean_good, 90)) if clean_good.size else float("nan")
        error = float(risk[q - 1]) if q > 0 and np.isfinite(risk[q - 1]) else float("nan")
        history.append(
            {
                "good_mean": good_mean,
                "good_q75": good_q75,
                "good_q90": good_q90,
                "error": error,
                "sep": error - good_mean if math.isfinite(error) and math.isfinite(good_mean) else float("nan"),
                "accept": float(q - 1) / positions if q > 0 else 1.0,
                "full": float(q == 0),
            }
        )
    progress.finish(detail="causal history complete")
    if index != analysis_count:
        raise AssertionError(f"trace changed during two-pass read: expected {analysis_count}, found {index}")
    arrays["features"] = features
    output_summaries: dict[str, Any] = {}
    for dataset, item in summaries.items():
        rounds = int(item["analysis_rounds"])
        failures = int(item["failure_rounds"])
        output_summaries[dataset] = {
            "requests": len(item["requests"]),
            "analysis_rounds": rounds,
            "failure_rounds": failures,
            "full_accept_rounds": rounds - failures,
            "failure_rate": safe_ratio(failures, rounds),
            "position1_failures": int(item["position1_failures"]),
            "position1_share": safe_ratio(item["position1_failures"], failures),
            "request_split_counts": {
                split: len(
                    {
                        request_id
                        for (name, request_id), assigned in assignments.items()
                        if name == dataset and assigned == split
                    }
                )
                for split in ("search", "selection", "test", "full_data")
            },
        }
    return arrays, output_summaries


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    family: str
    history_window: int = 0
    aggregation: str = "none"
    feature: str = "none"
    params: dict[str, float] = field(default_factory=dict, compare=False, hash=False)

    @property
    def uses_history(self) -> bool:
        return self.history_window > 0


def candidate(
    family: str,
    window: int = 0,
    aggregation: str = "none",
    feature: str = "none",
    **params: float,
) -> Candidate:
    suffix = "|".join(f"{name}={value:g}" for name, value in sorted(params.items()))
    identity = f"{family}|h={window}|a={aggregation}|f={feature}"
    if suffix:
        identity += "|" + suffix
    return Candidate(identity, family, window, aggregation, feature, {key: float(value) for key, value in params.items()})


def candidate_grid(windows: Iterable[int], aggregations: Iterable[str], grid: str) -> list[Candidate]:
    items = [candidate("fixed_margin", threshold=0.5)]
    if grid == "compact":
        alphas, offsets = (0.5, 1.0), (-0.04, 0.0, 0.04)
        shrinks, deltas = (0.5, 1.0), (0.05, 0.12)
        lambdas = (0.35, 0.65)
        betas, gammas = (0.15, 0.35), (0.1, 0.25)
    elif grid == "extended":
        alphas, offsets = (0.2, 0.4, 0.6, 0.8, 1.0, 1.3, 1.6), (-0.1, -0.06, -0.03, 0.0, 0.03, 0.06, 0.1)
        shrinks, deltas = (0.2, 0.4, 0.6, 0.8, 1.0), (0.0, 0.04, 0.08, 0.12, 0.18, 0.25)
        lambdas = (0.15, 0.3, 0.45, 0.6, 0.75, 0.9)
        betas, gammas = (0.05, 0.1, 0.2, 0.35, 0.55), (0.05, 0.1, 0.2, 0.35)
    else:
        alphas, offsets = (0.35, 0.7, 1.1), (-0.05, 0.0, 0.05)
        shrinks, deltas = (0.35, 0.65, 1.0), (0.05, 0.1, 0.17)
        lambdas = (0.25, 0.5, 0.75)
        betas, gammas = (0.1, 0.25, 0.45), (0.08, 0.2)

    for window in windows:
        for aggregation in aggregations:
            for feature_name in ("good_mean", "good_q90"):
                for alpha in alphas:
                    for offset in offsets:
                        items.append(candidate("good_center", window, aggregation, feature_name, alpha=alpha, offset=offset))
                for shrink in shrinks:
                    for delta in deltas:
                        items.append(candidate("good_direct", window, aggregation, feature_name, shrink=shrink, delta=delta))
            for lam in lambdas:
                for shrink in shrinks:
                    items.append(candidate("separator", window, aggregation, "good_mean", lam=lam, shrink=shrink))
            for beta in betas:
                for gamma in gammas:
                    items.append(candidate("accept_center", window, aggregation, "accept", beta=beta, gamma=gamma))
            for alpha in alphas[:2]:
                for beta in betas[:2]:
                    for gamma in gammas[:2]:
                        items.append(candidate("joint_center", window, aggregation, "good_q90", alpha=alpha, beta=beta, gamma=gamma))
            for alpha in alphas[:2]:
                for gamma in gammas:
                    items.append(candidate("gap_center", window, aggregation, "good_mean", alpha=alpha, gamma=gamma))
            for band in ((0.1, 0.2) if grid != "compact" else (0.15,)):
                for low in (-0.08, -0.04):
                    for high in (0.04, 0.08):
                        items.append(candidate("accept_gate", window, aggregation, "accept", band=band, low=low, high=high))
    unique = {item.candidate_id: item for item in items}
    return list(unique.values())


def equal_dataset_reference(values: np.ndarray, arrays: dict[str, Any], mask: np.ndarray) -> float:
    dataset_means = []
    for dataset in sorted(set(arrays["dataset"].tolist())):
        if normalized_dataset(str(dataset)) in EXCLUDED_MACRO:
            continue
        selected = values[mask & (arrays["dataset"] == dataset)]
        selected = selected[np.isfinite(selected)]
        if selected.size:
            dataset_means.append(float(np.mean(selected)))
    return float(statistics.fmean(dataset_means)) if dataset_means else float("nan")


def fit_references(arrays: dict[str, Any], search_mask: np.ndarray) -> dict[str, float]:
    references: dict[str, float] = {}
    for (window, aggregation, name), values in arrays["features"].items():
        references[f"h{window}|{aggregation}|{name}"] = equal_dataset_reference(values, arrays, search_mask)
    return references


def ref_key(item: Candidate, feature_name: str) -> str:
    return f"h{item.history_window}|{item.aggregation}|{feature_name}"


def dynamic_threshold(
    item: Candidate,
    arrays: dict[str, Any],
    references: dict[str, float],
    indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    count = len(indices)
    if item.family == "fixed_margin":
        return np.full(count, item.params.get("threshold", 0.5), dtype=np.float32), np.ones(count, dtype=bool)
    feature = arrays["features"][(item.history_window, item.aggregation, item.feature)][indices]
    ready = np.isfinite(feature)
    p = item.params
    if item.family == "good_center":
        reference = references[ref_key(item, item.feature)]
        threshold = 0.5 + p["alpha"] * (feature - reference) + p["offset"]
        ready &= math.isfinite(reference)
    elif item.family == "good_direct":
        threshold = (1.0 - p["shrink"]) * 0.5 + p["shrink"] * (feature + p["delta"])
    elif item.family == "separator":
        good = feature
        error = arrays["features"][(item.history_window, item.aggregation, "error")][indices]
        history_cut = good + p["lam"] * (error - good)
        threshold = (1.0 - p["shrink"]) * 0.5 + p["shrink"] * history_cut
        ready &= np.isfinite(error)
    elif item.family == "accept_center":
        full = arrays["features"][(item.history_window, item.aggregation, "full")][indices]
        ref_accept = references[ref_key(item, "accept")]
        ref_full = references[ref_key(item, "full")]
        threshold = 0.5 + p["beta"] * (feature - ref_accept) + p["gamma"] * (full - ref_full)
        ready &= np.isfinite(full) & math.isfinite(ref_accept) & math.isfinite(ref_full)
    elif item.family == "joint_center":
        accept = arrays["features"][(item.history_window, item.aggregation, "accept")][indices]
        full = arrays["features"][(item.history_window, item.aggregation, "full")][indices]
        ref_good = references[ref_key(item, item.feature)]
        ref_accept = references[ref_key(item, "accept")]
        ref_full = references[ref_key(item, "full")]
        threshold = (
            0.5
            + p["alpha"] * (feature - ref_good)
            + p["beta"] * (accept - ref_accept)
            + p["gamma"] * (full - ref_full)
        )
        ready &= np.isfinite(accept) & np.isfinite(full) & all(math.isfinite(x) for x in (ref_good, ref_accept, ref_full))
    elif item.family == "gap_center":
        sep = arrays["features"][(item.history_window, item.aggregation, "sep")][indices]
        ref_good = references[ref_key(item, item.feature)]
        ref_sep = references[ref_key(item, "sep")]
        threshold = 0.5 + p["alpha"] * (feature - ref_good) + p["gamma"] * (sep - ref_sep)
        ready &= np.isfinite(sep) & math.isfinite(ref_good) & math.isfinite(ref_sep)
    elif item.family == "accept_gate":
        ref_accept = references[ref_key(item, "accept")]
        threshold = np.full(count, 0.5, dtype=np.float32)
        threshold[feature < ref_accept - p["band"]] += p["low"]
        threshold[feature > ref_accept + p["band"]] += p["high"]
        ready &= math.isfinite(ref_accept)
    else:
        raise ValueError(f"unknown candidate family: {item.family}")
    threshold = np.asarray(threshold, dtype=np.float32)
    ready &= np.isfinite(threshold)
    threshold = np.clip(threshold, 0.05, 0.95)
    # The mandated cold-start/missing-history behavior is fixed margin-risk 0.5.
    threshold[~ready] = 0.5
    return threshold, ready


def predict_indices(
    item: Candidate,
    arrays: dict[str, Any],
    references: dict[str, float],
    indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    threshold, ready = dynamic_threshold(item, arrays, references, indices)
    predicted = strict_first(arrays["risk"][indices], threshold)
    return predicted, threshold, ready


def count_metrics(predicted: np.ndarray, q: np.ndarray, positions: int) -> dict[str, float]:
    rounds = int(q.size)
    failure = q > 0
    report = predicted > 0
    exact = report & failure & (predicted == q)
    early = report & failure & (predicted < q)
    late = report & failure & (predicted > q)
    full_false = report & ~failure
    correct_fp = early | full_false
    attempted_failure = report & failure
    errors = predicted[attempted_failure].astype(np.float64) - q[attempted_failure].astype(np.float64)
    correct_positions = int(np.where(failure, np.maximum(q - 1, 0), positions).sum())
    return {
        "rounds": rounds,
        "failures": int(failure.sum()),
        "full_accepts": int((~failure).sum()),
        "reports": int(report.sum()),
        "exact_hits": int(exact.sum()),
        "early_correct_reports": int(early.sum()),
        "full_false_reports": int(full_false.sum()),
        "correct_false_reports": int(correct_fp.sum()),
        "late_reports": int(late.sum()),
        "misses": int((~report & failure).sum()),
        "correct_positions": correct_positions,
        "position1_failures": int((q == 1).sum()),
        "position1_hits": int(((q == 1) & (predicted == 1)).sum()),
        "tol1_hits": int((attempted_failure & (np.abs(predicted - q) <= 1)).sum()),
        "tol2_hits": int((attempted_failure & (np.abs(predicted - q) <= 2)).sum()),
        "attempted_failures": int(attempted_failure.sum()),
        "abs_error_sum": float(np.abs(errors).sum()),
        "error_sum": float(errors.sum()),
    }


def metrics_from_counts(c: dict[str, float]) -> dict[str, Any]:
    precision = safe_ratio(c["exact_hits"], c["reports"])
    recall = safe_ratio(c["exact_hits"], c["failures"])
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall > 0
        else 0.0 if c["reports"] and c["failures"] else None
    )
    return {
        **{key: int(value) if float(value).is_integer() else float(value) for key, value in c.items()},
        "report_rate": safe_ratio(c["reports"], c["rounds"]),
        "coverage": safe_ratio(c["reports"], c["rounds"]),
        "precision": precision,
        "recall": recall,
        "exact_f1": f1,
        "hits_per_100": safe_ratio(100 * c["exact_hits"], c["rounds"]),
        "correct_fp_round": safe_ratio(c["correct_false_reports"], c["rounds"]),
        "correct_fp_report": safe_ratio(c["correct_false_reports"], c["reports"]),
        "position_fpr": safe_ratio(c["correct_false_reports"], c["correct_positions"]),
        "early_rate": safe_ratio(c["early_correct_reports"], c["failures"]),
        "full_false_alarm_round": safe_ratio(c["full_false_reports"], c["rounds"]),
        "full_false_alarm_given_full": safe_ratio(c["full_false_reports"], c["full_accepts"]),
        "miss_rate": safe_ratio(c["misses"], c["failures"]),
        "late_rate": safe_ratio(c["late_reports"], c["failures"]),
        "mae": safe_ratio(c["abs_error_sum"], c["attempted_failures"]),
        "bias": safe_ratio(c["error_sum"], c["attempted_failures"]),
        "tol1_recall": safe_ratio(c["tol1_hits"], c["failures"]),
        "tol2_recall": safe_ratio(c["tol2_hits"], c["failures"]),
        "position1_recall": safe_ratio(c["position1_hits"], c["position1_failures"]),
    }


def macro_metrics(by_dataset: dict[str, dict[str, Any]]) -> dict[str, Any]:
    included = {
        dataset: payload
        for dataset, payload in by_dataset.items()
        if normalized_dataset(dataset) not in EXCLUDED_MACRO and payload.get("rounds", 0) > 0
    }
    output: dict[str, Any] = {"datasets": len(included)}
    for field_name in MACRO_FIELDS:
        values = [float(payload[field_name]) for payload in included.values() if payload.get(field_name) is not None]
        output[field_name] = float(statistics.fmean(values)) if values else None
    return output


def evaluate_subset(
    predicted: np.ndarray,
    q: np.ndarray,
    datasets: np.ndarray,
    positions: int,
) -> dict[str, Any]:
    by_dataset = {}
    for dataset in sorted(set(datasets.tolist())):
        mask = datasets == dataset
        by_dataset[str(dataset)] = metrics_from_counts(count_metrics(predicted[mask], q[mask], positions))
    return {"by_dataset": by_dataset, "macro": macro_metrics(by_dataset)}


def evaluate_indices(predicted: np.ndarray, arrays: dict[str, Any], indices: np.ndarray, positions: int) -> dict[str, Any]:
    return evaluate_subset(predicted, arrays["q"][indices], arrays["dataset"][indices], positions)


def capped_search_indices(arrays: dict[str, Any], cap: int, seed: int) -> np.ndarray:
    selected: list[int] = []
    for dataset in sorted(set(arrays["dataset"].tolist())):
        if normalized_dataset(str(dataset)) in EXCLUDED_MACRO:
            continue
        indices = np.flatnonzero((arrays["split"] == "search") & (arrays["dataset"] == dataset))
        if cap > 0 and len(indices) > cap:
            indices = np.asarray(
                sorted(
                    indices.tolist(),
                    key=lambda index: stable_hash(
                        str(seed), str(dataset), str(arrays["request_id"][index]), str(arrays["round_index"][index])
                    ),
                )[:cap],
                dtype=np.int64,
            )
        selected.extend(indices.tolist())
    return np.asarray(sorted(selected), dtype=np.int64)


def ranking(metric: dict[str, Any], baseline: dict[str, Any]) -> tuple[float, ...]:
    recall = float(metric.get("recall") or -1.0)
    cfp = float(metric.get("correct_fp_round") if metric.get("correct_fp_round") is not None else 1.0)
    f1 = float(metric.get("exact_f1") or -1.0)
    dominates = recall > float(baseline.get("recall") or 0.0) and cfp < float(baseline.get("correct_fp_round") or 1.0)
    admissible = cfp <= float(baseline.get("correct_fp_round") or 1.0)
    return (float(dominates), float(admissible), recall, -cfp, f1)


def pareto_indices(results: list[dict[str, Any]], stage: str) -> list[int]:
    valid = []
    for index, result in enumerate(results):
        metric = result.get(stage, {}).get("macro", {})
        if metric.get("recall") is not None and metric.get("correct_fp_round") is not None:
            valid.append(index)
    ordered = sorted(
        valid,
        key=lambda index: (
            -float(results[index][stage]["macro"]["recall"]),
            float(results[index][stage]["macro"]["correct_fp_round"]),
        ),
    )
    frontier = []
    best_cfp = float("inf")
    for index in ordered:
        cfp = float(results[index][stage]["macro"]["correct_fp_round"])
        if cfp < best_cfp - 1e-15:
            frontier.append(index)
            best_cfp = cfp
    return frontier


def select_shortlist(results: list[dict[str, Any]], fixed_index: int, limit: int) -> list[int]:
    baseline = results[fixed_index]["search"]["macro"]
    pareto = pareto_indices(results, "search")
    ordered = sorted(
        range(len(results)),
        key=lambda index: ranking(results[index]["search"]["macro"], baseline),
        reverse=True,
    )
    selected = pareto[: max(1, limit // 2)] + ordered[:limit] + [fixed_index]
    by_group: dict[tuple[str, int, str], list[int]] = defaultdict(list)
    for index in ordered:
        spec = results[index]["candidate"]
        by_group[(spec["family"], int(spec["history_window"]), spec["aggregation"])].append(index)
    for group in by_group.values():
        selected.extend(group[:1])
    return list(dict.fromkeys(selected))


def threshold_stats(threshold: np.ndarray, ready: np.ndarray) -> dict[str, Any]:
    if threshold.size == 0:
        return {}
    return {
        "rounds": int(threshold.size),
        "mean": float(np.mean(threshold)),
        "std": float(np.std(threshold)),
        "p10": float(np.percentile(threshold, 10)),
        "p50": float(np.percentile(threshold, 50)),
        "p90": float(np.percentile(threshold, 90)),
        "raised": float(np.mean(threshold > 0.5000001)),
        "lowered": float(np.mean(threshold < 0.4999999)),
        "equal": float(np.mean(np.abs(threshold - 0.5) <= 1e-7)),
        "cold": float(np.mean(~ready)),
        "ready": float(np.mean(ready)),
    }


def threshold_stats_evaluation(
    threshold: np.ndarray, ready: np.ndarray, datasets: np.ndarray
) -> dict[str, Any]:
    by_dataset = {
        str(dataset): threshold_stats(threshold[datasets == dataset], ready[datasets == dataset])
        for dataset in sorted(set(datasets.tolist()))
        if normalized_dataset(str(dataset)) not in EXCLUDED_MACRO
    }
    fields = ("mean", "std", "p10", "p50", "p90", "raised", "lowered", "equal", "cold", "ready")
    macro = {
        name: (
            float(statistics.fmean(item[name] for item in by_dataset.values() if item.get(name) is not None))
            if by_dataset else None
        )
        for name in fields
    }
    macro["datasets"] = len(by_dataset)
    return {"by_dataset": by_dataset, "macro": macro, "pooled": threshold_stats(threshold, ready)}


def sum_counts(items: list[dict[str, float]], sampled: np.ndarray) -> dict[str, float]:
    keys = items[0].keys()
    return {key: float(sum(items[int(index)][key] for index in sampled)) for key in keys}


def paired_bootstrap(
    winner_p: np.ndarray,
    reference_p: np.ndarray,
    arrays: dict[str, Any],
    indices: np.ndarray,
    *,
    positions: int,
    replicates: int,
    seed: int,
    show_progress: bool = False,
    progress_label: str = "bootstrap",
) -> dict[str, Any]:
    winner_full = evaluate_indices(winner_p, arrays, indices, positions)
    reference_full = evaluate_indices(reference_p, arrays, indices, positions)
    winner_eval = winner_full["macro"]
    reference_eval = reference_full["macro"]
    fields = ("recall", "correct_fp_round", "exact_f1")
    point = {
        name: (winner_eval[name] - reference_eval[name])
        if winner_eval.get(name) is not None and reference_eval.get(name) is not None
        else None
        for name in fields
    }
    dataset_point = {
        dataset: {
            name: winner_full["by_dataset"][dataset][name] - reference_full["by_dataset"][dataset][name]
            for name in fields
        }
        for dataset in winner_full["by_dataset"]
        if normalized_dataset(dataset) not in EXCLUDED_MACRO
    }
    if replicates <= 0:
        return {"replicates": 0, "point": point, "ci95": {}, "by_dataset": dataset_point}
    by_dataset: dict[str, tuple[list[dict[str, float]], list[dict[str, float]]]] = {}
    for dataset in sorted(set(arrays["dataset"][indices].tolist())):
        if normalized_dataset(str(dataset)) in EXCLUDED_MACRO:
            continue
        dataset_indices = indices[arrays["dataset"][indices] == dataset]
        by_request: dict[str, list[int]] = defaultdict(list)
        for local_index, global_index in enumerate(dataset_indices):
            by_request[str(arrays["request_id"][global_index])].append(local_index)
        winner_items, reference_items = [], []
        q = arrays["q"][dataset_indices]
        # winner_p/reference_p are aligned with indices, so locate dataset-local predictions.
        prediction_mask = arrays["dataset"][indices] == dataset
        wp = winner_p[prediction_mask]
        rp = reference_p[prediction_mask]
        for request_local in by_request.values():
            request_local_array = np.asarray(request_local, dtype=np.int64)
            winner_items.append(count_metrics(wp[request_local_array], q[request_local_array], positions))
            reference_items.append(count_metrics(rp[request_local_array], q[request_local_array], positions))
        if winner_items:
            by_dataset[str(dataset)] = (winner_items, reference_items)
    rng = np.random.default_rng(seed)
    samples = {name: [] for name in fields}
    dataset_samples = {dataset: {name: [] for name in fields} for dataset in by_dataset}
    progress = ProgressBar(
        progress_label, replicates, enabled=show_progress, unit="replicate"
    )
    for replicate in range(replicates):
        macro_w = {name: [] for name in fields}
        macro_r = {name: [] for name in fields}
        for dataset, (winner_items, reference_items) in by_dataset.items():
            sample = rng.integers(0, len(winner_items), size=len(winner_items))
            wm = metrics_from_counts(sum_counts(winner_items, sample))
            rm = metrics_from_counts(sum_counts(reference_items, sample))
            for name in fields:
                if wm.get(name) is not None and rm.get(name) is not None:
                    macro_w[name].append(float(wm[name]))
                    macro_r[name].append(float(rm[name]))
                    dataset_samples[dataset][name].append(float(wm[name]) - float(rm[name]))
        for name in fields:
            if macro_w[name]:
                samples[name].append(statistics.fmean(macro_w[name]) - statistics.fmean(macro_r[name]))
        progress.update(replicate + 1)
    progress.finish()
    ci95 = {
        name: [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))] if values else None
        for name, values in samples.items()
    }
    dataset_output = {
        dataset: {
            "point": dataset_point[dataset],
            "ci95": {
                name: [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]
                if values else None
                for name, values in fields_payload.items()
            },
        }
        for dataset, fields_payload in dataset_samples.items()
    }
    return {
        "replicates": replicates,
        "unit": "request",
        "point": point,
        "ci95": ci95,
        "by_dataset": dataset_output,
    }


def run_full_data_global_search(
    args: argparse.Namespace,
    *,
    arrays: dict[str, Any],
    payload: dict[str, Any],
    positions: int,
    windows: list[int],
    aggregations: list[str],
) -> dict[str, Any]:
    """Evaluate every declared candidate on every non-AIME24 valid round.

    This protocol intentionally has no held-out split: its output is the finite
    grid's descriptive all-data global optimum.  Dataset metrics are computed
    from all of that dataset's valid rounds and only then macro-averaged, so a
    large dataset cannot receive more weight than a small dataset.
    """

    included_mask = np.ones(len(arrays["q"]), dtype=bool)
    for dataset in sorted(set(arrays["dataset"].tolist())):
        if normalized_dataset(str(dataset)) in EXCLUDED_MACRO:
            included_mask &= arrays["dataset"] != dataset
    full_indices = np.flatnonzero(included_mask)
    if not full_indices.size:
        raise ValueError("full_data has no non-AIME24 valid rounds")

    references = fit_references(arrays, included_mask)
    candidates = candidate_grid(windows, aggregations, args.grid)
    all_results: list[dict[str, Any]] = []
    progress = ProgressBar(
        "full-data search",
        len(candidates),
        enabled=not getattr(args, "no_progress", False),
        unit="candidate",
    )
    for candidate_index, item in enumerate(candidates, 1):
        predicted, _, _ = predict_indices(item, arrays, references, full_indices)
        all_results.append(
            {
                "candidate": asdict(item),
                "full_data": evaluate_indices(predicted, arrays, full_indices, positions),
            }
        )
        progress.update(candidate_index, detail=item.family)
    progress.finish(detail="all candidates evaluated")

    fixed_index = next(index for index, item in enumerate(candidates) if item.family == "fixed_margin")
    fixed_result = all_results[fixed_index]
    fixed_macro = fixed_result["full_data"]["macro"]
    strict_candidates = [
        result
        for result in all_results
        if result["candidate"]["family"] != "fixed_margin"
        and result["full_data"]["macro"].get("recall") is not None
        and fixed_macro.get("recall") is not None
        and result["full_data"]["macro"]["recall"] > fixed_macro["recall"]
        and result["full_data"]["macro"].get("correct_fp_round") is not None
        and fixed_macro.get("correct_fp_round") is not None
        and result["full_data"]["macro"]["correct_fp_round"] < fixed_macro["correct_fp_round"]
    ]
    if strict_candidates:
        winner = max(
            strict_candidates,
            key=lambda result: (
                float(result["full_data"]["macro"]["recall"]),
                -float(result["full_data"]["macro"]["correct_fp_round"]),
                float(result["full_data"]["macro"].get("exact_f1") or -1.0),
            ),
        )
        dominates = True
    else:
        winner = fixed_result
        dominates = False
    winner_item = Candidate(**winner["candidate"])
    fixed_item = candidates[fixed_index]

    winner_p, winner_tau, winner_ready = predict_indices(
        winner_item, arrays, references, full_indices
    )
    fixed_p, fixed_tau, fixed_ready = predict_indices(
        fixed_item, arrays, references, full_indices
    )
    # Re-evaluate the two retained strategies as an internal consistency guard.
    winner["full_data"] = evaluate_indices(winner_p, arrays, full_indices, positions)
    fixed_result["full_data"] = evaluate_indices(fixed_p, arrays, full_indices, positions)
    winner["threshold_stats"] = {
        "full_data": threshold_stats_evaluation(
            winner_tau, winner_ready, arrays["dataset"][full_indices]
        )
    }
    fixed_result["threshold_stats"] = {
        "full_data": threshold_stats_evaluation(
            fixed_tau, fixed_ready, arrays["dataset"][full_indices]
        )
    }

    original_p = arrays["original_p"][full_indices]
    original_result = {
        "candidate": {
            "candidate_id": "prefix_drop|strict|t=0.15",
            "family": "prefix_drop",
            "history_window": 0,
            "aggregation": "none",
            "feature": "prefix_drop_pct",
            "params": {"threshold": 0.15},
        },
        "full_data": evaluate_indices(original_p, arrays, full_indices, positions),
    }

    frontier_indices = pareto_indices(all_results, "full_data")
    frontier = []
    for index in frontier_indices:
        result = all_results[index]
        macro = result["full_data"]["macro"]
        frontier.append(
            {
                "candidate": result["candidate"],
                "macro": macro,
                "delta_recall": macro["recall"] - fixed_macro["recall"],
                "delta_correct_fp_round": (
                    macro["correct_fp_round"] - fixed_macro["correct_fp_round"]
                ),
            }
        )
    frontier.sort(
        key=lambda item: (-item["macro"]["recall"], item["macro"]["correct_fp_round"])
    )

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in all_results:
        spec = result["candidate"]
        groups[f"Fam:{spec['family']}"].append(result)
        groups[f"H:{spec['history_window']}"].append(result)
        groups[f"Agg:{spec['aggregation']}"].append(result)
    ablations = []
    for group_name, group_results in sorted(groups.items()):
        best = max(
            group_results,
            key=lambda result: ranking(result["full_data"]["macro"], fixed_macro),
        )
        ablations.append(
            {
                "group": group_name,
                "candidate": best["candidate"],
                "full_data": best["full_data"],
                "full_data_macro": best["full_data"]["macro"],
            }
        )

    bootstrap = {
        "vs_margin.5": paired_bootstrap(
            winner_p,
            fixed_p,
            arrays,
            full_indices,
            positions=positions,
            replicates=args.bootstrap_replicates,
            seed=args.split_seed + 101,
            show_progress=not getattr(args, "no_progress", False),
            progress_label="bootstrap vs margin.5",
        ),
        "vs_drop.15": paired_bootstrap(
            winner_p,
            original_p,
            arrays,
            full_indices,
            positions=positions,
            replicates=args.bootstrap_replicates,
            seed=args.split_seed + 211,
            show_progress=not getattr(args, "no_progress", False),
            progress_label="bootstrap vs drop.15",
        ),
    }
    winner_macro = winner["full_data"]["macro"]
    fixed_macro = fixed_result["full_data"]["macro"]
    delta = {
        name: winner_macro[name] - fixed_macro[name]
        for name in ("recall", "correct_fp_round", "report_rate", "exact_f1")
        if winner_macro.get(name) is not None and fixed_macro.get(name) is not None
    }
    ordered = sorted(
        all_results,
        key=lambda result: ranking(result["full_data"]["macro"], fixed_macro),
        reverse=True,
    )
    included_datasets = [
        dataset
        for dataset in sorted(set(arrays["dataset"][full_indices].tolist()))
        if normalized_dataset(str(dataset)) not in EXCLUDED_MACRO
    ]
    payload.update(
        {
            "phase": f"完成：全部候选已在{len(included_datasets)}个非AIME24数据集全部有效轮上做等数据集权重全局搜索。",
            "primary_result_stage": "full_data",
            "candidate_count": len(candidates),
            "shortlist_count": None,
            "analysis_rounds": int(len(arrays["q"])),
            "full_data_rounds": int(full_indices.size),
            "full_data_requests": int(
                sum(payload["dataset_summaries"][dataset]["raw_requests"] for dataset in included_datasets)
            ),
            "full_data_evaluable_requests": int(
                sum(payload["dataset_summaries"][dataset]["valid_requests"] for dataset in included_datasets)
            ),
            "full_data_zero_valid_requests": int(
                sum(payload["dataset_summaries"][dataset]["zero_valid_requests"] for dataset in included_datasets)
            ),
            "full_data_datasets": included_datasets,
            "split_round_counts": {"full_data": int(full_indices.size)},
            "search_rounds_after_cap": None,
            "history_references_fitted_on_full_data": references,
            "selection_stage_used": "full_data",
            "selection_dominates_fixed": dominates,
            "full_data_dominates_fixed": dominates,
            "test_dominates_fixed": None,
            "full_data_delta_vs_fixed": delta,
            "deployment_recommendation": (
                winner["candidate"] if dominates else fixed_result["candidate"]
            ),
            "winner": winner,
            "fixed_margin_baseline": fixed_result,
            "original_drop_baseline": original_result,
            "selection_pareto": frontier,
            "global_pareto": frontier,
            "ablations": ablations,
            "paired_bootstrap": bootstrap,
            "top_full_data": [
                {"candidate": result["candidate"], "full_data_macro": result["full_data"]["macro"]}
                for result in ordered[: args.report_top]
            ],
            "all_full_data_candidates": [
                {"candidate": result["candidate"], "full_data_macro": result["full_data"]["macro"]}
                for result in ordered
            ],
            "full_data_contract": {
                "all_candidates_use_all_included_valid_rounds": True,
                "all_source_trace_requests_scanned": True,
                "zero_valid_requests_have_no_eligible_rounds": True,
                "round_cap": None,
                "shortlist": False,
                "held_out_test": False,
                "within_dataset": "pool counts over every valid round from every request/sample",
                "across_datasets": "arithmetic mean of dataset metrics; every non-AIME24 dataset has weight 1/N",
                "aime24_used": False,
                "descriptive_optimum_scope": "declared finite candidate grid",
            },
        }
    )
    atomic_json(args.output_json, payload)
    render_report(args.run_dir, payload)
    return payload


def run_search(args: argparse.Namespace) -> dict[str, Any]:
    windows = sorted({int(value) for value in args.history_windows.split(",") if value.strip()})
    aggregations = [value.strip() for value in args.aggregations.split(",") if value.strip()]
    dataset_filter = {
        normalized_dataset(item.split(":", 1)[0])
        for item in args.benchmarks.split(",")
        if item.strip()
    } if args.benchmarks else None
    block_size, requests, raw, analysis_count = scan_inputs(
        args.trace_dir,
        datasets=dataset_filter,
        configured_block_size=args.block_size,
        include_boundary_rounds=args.include_boundary_rounds,
        show_progress=not getattr(args, "no_progress", False),
    )
    observed_datasets = {normalized_dataset(dataset) for dataset in requests}
    if dataset_filter is not None:
        missing_datasets = sorted(dataset_filter - observed_datasets)
        if missing_datasets:
            raise ValueError(
                "requested datasets have no matching trace records: " + ",".join(missing_datasets)
            )
    if args.selection_protocol == "full_data":
        empty_datasets = sorted(
            dataset
            for dataset, item in raw.items()
            if normalized_dataset(dataset) not in EXCLUDED_MACRO
            and not int(item.get("analysis_rounds", 0))
        )
        if empty_datasets:
            raise ValueError(
                "full_data requires at least one valid analysis round in every non-AIME24 dataset: "
                + ",".join(empty_datasets)
            )
    positions = block_size - 1
    settings_path = args.run_dir / "Settings.json"
    settings_payload = json.loads(settings_path.read_text(encoding="utf-8"))
    settings_payload["block_size"] = block_size
    settings_payload["trace_dir"] = str(args.trace_dir)
    settings_payload["trace_files"] = [
        {"name": path.name, "bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
        for path in trace_paths(args.trace_dir, dataset_filter)
    ]
    atomic_json(settings_path, settings_payload)
    if args.selection_protocol == "full_data":
        assignments = {
            (dataset, request_id): "full_data"
            for dataset, request_ids in requests.items()
            for request_id in request_ids
        }
    else:
        assignments = split_requests(
            requests,
            seed=args.split_seed,
            search_ratio=args.search_ratio,
            selection_ratio=args.selection_ratio,
        )
    render_report(args.run_dir, None, phase=f"trace 已校验：{analysis_count} 个有效轮；正在构造因果历史特征。")
    arrays, summaries = build_arrays(
        args.trace_dir,
        datasets=dataset_filter,
        block_size=block_size,
        windows=windows,
        aggregations=aggregations,
        include_boundary_rounds=args.include_boundary_rounds,
        assignments=assignments,
        analysis_count=analysis_count,
        show_progress=not getattr(args, "no_progress", False),
    )
    for dataset, item in raw.items():
        summary = summaries.setdefault(
            dataset,
            {
                "requests": 0,
                "analysis_rounds": 0,
                "failure_rounds": 0,
                "full_accept_rounds": 0,
                "failure_rate": None,
                "position1_failures": 0,
                "position1_share": None,
                "request_split_counts": {
                    split: 0 for split in ("search", "selection", "test", "full_data")
                },
            },
        )
        trace_requests = len(requests[dataset])
        valid_requests = int(summary.get("requests", 0))
        summary.update(item)
        summary.update(
            {
                "raw_requests": trace_requests,
                "valid_requests": valid_requests,
                "zero_valid_requests": trace_requests - valid_requests,
                "zero_valid_request_rate": safe_ratio(
                    trace_requests - valid_requests, trace_requests
                ),
                "request_split_counts": {
                    split: sum(
                        1
                        for (name, _request_id), assigned in assignments.items()
                        if name == dataset and assigned == split
                    )
                    for split in ("search", "selection", "test", "full_data")
                },
            }
        )
        atomic_json(args.run_dir / "summaries" / f"trace_{safe_name(dataset)}.json", summary)

    payload: dict[str, Any] = {
        "schema_version": 1,
        "phase": (
            "因果历史特征已完成；开始全数据全候选检索。"
            if args.selection_protocol == "full_data"
            else "因果历史特征已完成；开始 search 参数检索。"
        ),
        "selection_protocol": args.selection_protocol,
        "block_size": block_size,
        "positions": positions,
        "dataset_summaries": summaries,
        "split": {
            "seed": args.split_seed,
            "unit": "request",
            "search_ratio": args.search_ratio,
            "selection_ratio": args.selection_ratio,
            "test_ratio": 1.0 - args.search_ratio - args.selection_ratio,
        },
        "excluded_macro_datasets": sorted(EXCLUDED_MACRO),
        "selection_contract": {
            "primary": "maximize equal-dataset Exact Recall subject to lower correct_fp_round than fixed margin_risk=0.5",
            "strict_dominance": "recall > fixed and correct_fp_round < fixed",
            "cold_start_fallback": "strict margin_risk > 0.5",
            "test_used_for_selection": False,
            "per_dataset_parameters": False,
            "all_datasets_equal_weight": True,
            "all_valid_rounds_used_for_every_candidate": args.selection_protocol == "full_data",
            "held_out_test": args.selection_protocol == "split",
        },
    }
    atomic_json(args.output_json, payload)
    render_report(args.run_dir, payload)

    if args.selection_protocol == "full_data":
        return run_full_data_global_search(
            args,
            arrays=arrays,
            payload=payload,
            positions=positions,
            windows=windows,
            aggregations=aggregations,
        )

    search_indices = capped_search_indices(arrays, args.search_max_rounds_per_dataset, args.split_seed)
    full_search_mask = arrays["split"] == "search"
    references = fit_references(arrays, full_search_mask)
    candidates = candidate_grid(windows, aggregations, args.grid)
    all_results: list[dict[str, Any]] = []
    progress = ProgressBar(
        "search candidates",
        len(candidates),
        enabled=not getattr(args, "no_progress", False),
        unit="candidate",
    )
    for candidate_index, item in enumerate(candidates, 1):
        predicted, _, _ = predict_indices(item, arrays, references, search_indices)
        all_results.append(
            {"candidate": asdict(item), "search": evaluate_indices(predicted, arrays, search_indices, positions)}
        )
        progress.update(candidate_index, detail=item.family)
    progress.finish(detail="search stage complete")
    fixed_index = next(index for index, item in enumerate(candidates) if item.family == "fixed_margin")
    shortlist_indices = select_shortlist(all_results, fixed_index, args.shortlist)
    selection_indices = np.flatnonzero(arrays["split"] == "selection")
    selection_stage = "selection" if selection_indices.size else "search"
    shortlist: list[dict[str, Any]] = []
    progress = ProgressBar(
        "selection shortlist",
        len(shortlist_indices),
        enabled=not getattr(args, "no_progress", False),
        unit="candidate",
    )
    for shortlist_index, index in enumerate(shortlist_indices, 1):
        result = all_results[index]
        item = candidates[index]
        if selection_indices.size:
            predicted, threshold, ready = predict_indices(item, arrays, references, selection_indices)
            result["selection"] = evaluate_indices(predicted, arrays, selection_indices, positions)
            result["selection_threshold_stats"] = threshold_stats(threshold, ready)
        else:
            result["selection"] = result["search"]
        shortlist.append(result)
        progress.update(shortlist_index, detail=item.family)
    progress.finish(detail="selection stage complete")

    fixed_result = all_results[fixed_index]
    fixed_macro = fixed_result[selection_stage]["macro"]
    strict_candidates = [
        result
        for result in shortlist
        if result["candidate"]["family"] != "fixed_margin"
        and (result[selection_stage]["macro"].get("recall") or -1.0) > (fixed_macro.get("recall") or -1.0)
        and (result[selection_stage]["macro"].get("correct_fp_round") or 1.0)
        < (fixed_macro.get("correct_fp_round") or 1.0)
    ]
    if strict_candidates:
        winner = max(
            strict_candidates,
            key=lambda result: (
                float(result[selection_stage]["macro"].get("recall") or -1.0),
                -float(result[selection_stage]["macro"].get("correct_fp_round") or 1.0),
                float(result[selection_stage]["macro"].get("exact_f1") or -1.0),
            ),
        )
        dominates = True
    else:
        winner = fixed_result
        dominates = False
    winner_item = Candidate(**winner["candidate"])

    # Freeze above.  Only now evaluate held-out test.
    test_indices = np.flatnonzero(arrays["split"] == "test")
    winner_p, winner_tau, winner_ready = predict_indices(winner_item, arrays, references, test_indices)
    winner["test"] = evaluate_indices(winner_p, arrays, test_indices, positions)
    winner["threshold_stats"] = {
        "test": threshold_stats_evaluation(
            winner_tau, winner_ready, arrays["dataset"][test_indices]
        )
    }
    fixed_item = candidates[fixed_index]
    fixed_p, fixed_tau, fixed_ready = predict_indices(fixed_item, arrays, references, test_indices)
    fixed_result["test"] = evaluate_indices(fixed_p, arrays, test_indices, positions)
    fixed_result["threshold_stats"] = {
        "test": threshold_stats_evaluation(
            fixed_tau, fixed_ready, arrays["dataset"][test_indices]
        )
    }
    original_p = arrays["original_p"][test_indices]
    original_result = {
        "candidate": {
            "candidate_id": "prefix_drop|strict|t=0.15",
            "family": "prefix_drop",
            "history_window": 0,
            "aggregation": "none",
            "feature": "prefix_drop_pct",
            "params": {"threshold": 0.15},
        },
        "test": evaluate_indices(original_p, arrays, test_indices, positions),
    }
    # Baseline selection metrics are needed in the report but do not affect the adaptive search.
    fixed_sel_p, _, _ = predict_indices(fixed_item, arrays, references, selection_indices)
    original_result["selection"] = evaluate_indices(
        arrays["original_p"][selection_indices], arrays, selection_indices, positions
    ) if selection_indices.size else original_result["test"]
    fixed_result["selection"] = evaluate_indices(fixed_sel_p, arrays, selection_indices, positions) if selection_indices.size else fixed_result["search"]

    selection_frontier_indices = pareto_indices(shortlist, selection_stage)
    selection_frontier = []
    for index in selection_frontier_indices:
        result = shortlist[index]
        macro = result[selection_stage]["macro"]
        selection_frontier.append(
            {
                "candidate": result["candidate"],
                "macro": macro,
                "delta_recall": (macro.get("recall") - fixed_macro.get("recall"))
                if macro.get("recall") is not None and fixed_macro.get("recall") is not None else None,
                "delta_correct_fp_round": (macro.get("correct_fp_round") - fixed_macro.get("correct_fp_round"))
                if macro.get("correct_fp_round") is not None and fixed_macro.get("correct_fp_round") is not None else None,
            }
        )
    selection_frontier.sort(
        key=lambda item: (-(item["macro"].get("recall") or -1.0), item["macro"].get("correct_fp_round") or 1.0)
    )

    # Select ablation representatives on selection, then evaluate those frozen representatives on test.
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in shortlist:
        spec = result["candidate"]
        groups[f"Fam:{spec['family']}"].append(result)
        groups[f"H:{spec['history_window']}"].append(result)
        groups[f"Agg:{spec['aggregation']}"].append(result)
    ablations = []
    for group_name, group_results in sorted(groups.items()):
        best = max(group_results, key=lambda result: ranking(result[selection_stage]["macro"], fixed_macro))
        item = Candidate(**best["candidate"])
        predicted, _, _ = predict_indices(item, arrays, references, test_indices)
        test_evaluation = evaluate_indices(predicted, arrays, test_indices, positions)
        ablations.append(
            {
                "group": group_name,
                "candidate": best["candidate"],
                "selection_macro": best[selection_stage]["macro"],
                "test": test_evaluation,
                "test_macro": test_evaluation["macro"],
            }
        )

    bootstrap = {
        "vs_margin.5": paired_bootstrap(
            winner_p, fixed_p, arrays, test_indices, positions=positions,
            replicates=args.bootstrap_replicates, seed=args.split_seed + 101,
            show_progress=not getattr(args, "no_progress", False),
            progress_label="bootstrap vs margin.5",
        ),
        "vs_drop.15": paired_bootstrap(
            winner_p, original_p, arrays, test_indices, positions=positions,
            replicates=args.bootstrap_replicates, seed=args.split_seed + 211,
            show_progress=not getattr(args, "no_progress", False),
            progress_label="bootstrap vs drop.15",
        ),
    }
    ordered_search = sorted(
        all_results,
        key=lambda result: ranking(result["search"]["macro"], all_results[fixed_index]["search"]["macro"]),
        reverse=True,
    )
    winner_test_macro = winner["test"]["macro"]
    fixed_test_macro = fixed_result["test"]["macro"]
    test_dominates = (
        winner_item.family != "fixed_margin"
        and winner_test_macro.get("recall") is not None
        and fixed_test_macro.get("recall") is not None
        and winner_test_macro["recall"] > fixed_test_macro["recall"]
        and winner_test_macro.get("correct_fp_round") is not None
        and fixed_test_macro.get("correct_fp_round") is not None
        and winner_test_macro["correct_fp_round"] < fixed_test_macro["correct_fp_round"]
    )
    test_delta = {
        name: winner_test_macro[name] - fixed_test_macro[name]
        for name in ("recall", "correct_fp_round", "report_rate", "exact_f1")
        if winner_test_macro.get(name) is not None and fixed_test_macro.get(name) is not None
    }
    payload.update(
        {
            "phase": "完成：策略已冻结并在全部非 AIME24 数据集 test 上评估。",
            "candidate_count": len(candidates),
            "shortlist_count": len(shortlist),
            "analysis_rounds": int(len(arrays["q"])),
            "search_rounds_after_cap": int(search_indices.size),
            "split_round_counts": {
                split: int((arrays["split"] == split).sum()) for split in ("search", "selection", "test")
            },
            "history_references_fitted_on_search": references,
            "selection_stage_used": selection_stage,
            "selection_dominates_fixed": dominates,
            "test_dominates_fixed": test_dominates,
            "test_delta_vs_fixed": test_delta,
            "deployment_recommendation": (
                winner["candidate"] if test_dominates else fixed_result["candidate"]
            ),
            "winner": winner,
            "fixed_margin_baseline": fixed_result,
            "original_drop_baseline": original_result,
            "selection_pareto": selection_frontier,
            "ablations": ablations,
            "paired_bootstrap": bootstrap,
            "top_search": [
                {"candidate": result["candidate"], "search_macro": result["search"]["macro"]}
                for result in ordered_search[: args.report_top]
            ],
            "all_search_candidates": [
                {"candidate": result["candidate"], "search_macro": result["search"]["macro"]}
                for result in ordered_search
            ],
        }
    )
    atomic_json(args.output_json, payload)
    render_report(args.run_dir, payload)
    return payload


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--run-dir", required=True, type=Path)
    result.add_argument("--trace-dir", required=True, type=Path)
    result.add_argument("--output-json", type=Path)
    result.add_argument("--benchmarks", default="")
    result.add_argument("--block-size", type=int)
    result.add_argument("--history-windows", default="1,2,4")
    result.add_argument("--aggregations", default="mean,median,ewma")
    result.add_argument("--grid", choices=("compact", "standard", "extended"), default="standard")
    result.add_argument("--selection-protocol", choices=("split", "full_data"), default="split")
    result.add_argument("--split-seed", type=int, default=20260830)
    result.add_argument("--search-ratio", type=float, default=0.6)
    result.add_argument("--selection-ratio", type=float, default=0.2)
    result.add_argument("--shortlist", type=int, default=120)
    result.add_argument("--report-top", type=int, default=30)
    result.add_argument("--search-max-rounds-per-dataset", type=int, default=30000)
    result.add_argument("--bootstrap-replicates", type=int, default=500)
    result.add_argument("--include-boundary-rounds", action="store_true")
    result.add_argument("--no-progress", action="store_true")
    return result


def validate(args: argparse.Namespace) -> None:
    windows = [int(value) for value in args.history_windows.split(",") if value.strip()]
    if not windows or len(windows) != len(set(windows)) or min(windows) < 1:
        raise ValueError("history windows must be unique positive integers")
    aggregations = [value.strip() for value in args.aggregations.split(",") if value.strip()]
    if not aggregations or len(aggregations) != len(set(aggregations)) or not set(aggregations) <= {"mean", "median", "ewma"}:
        raise ValueError("aggregations must be a unique subset of mean,median,ewma")
    if not 0 < args.search_ratio < 1 or not 0 <= args.selection_ratio < 1 or args.search_ratio + args.selection_ratio >= 1:
        raise ValueError("invalid request split ratios")
    if args.block_size is not None and args.block_size < 2:
        raise ValueError("block size must be >=2")
    if args.shortlist <= 0 or args.report_top <= 0 or args.search_max_rounds_per_dataset < 0 or args.bootstrap_replicates < 0:
        raise ValueError("invalid search/report/bootstrap count")
    if args.selection_protocol == "full_data" and args.search_max_rounds_per_dataset != 0:
        raise ValueError(
            "full_data requires --search-max-rounds-per-dataset 0 so every valid round is evaluated"
        )


def main() -> int:
    args = parser().parse_args()
    args.run_dir = args.run_dir.resolve()
    args.trace_dir = args.trace_dir.resolve()
    args.output_json = (args.output_json or args.run_dir / "analysis" / "strategy_search.json").resolve()
    validate(args)
    payload = run_search(args)
    print(json.dumps({
        "phase": payload["phase"],
        "candidate_count": payload["candidate_count"],
        "winner": payload["winner"]["candidate"]["candidate_id"],
        "selection_dominates_fixed": payload["selection_dominates_fixed"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
