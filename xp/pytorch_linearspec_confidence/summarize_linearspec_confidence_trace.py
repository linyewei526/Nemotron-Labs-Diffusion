#!/usr/bin/env python3
"""Summarize native PyTorch LinearSpec confidence/rank trace JSONL."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Optional


QUANTILES = [0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]


def as_float(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def as_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def percentile(values: list[float], q: float) -> Optional[float]:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    position = q * (len(values) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return values[low]
    weight = position - low
    return values[low] * (1 - weight) + values[high] * weight


def histogram(
    values: list[float], bins: int, fixed_range: Optional[tuple[float, float]]
) -> dict[str, Any]:
    if bins <= 0:
        raise ValueError("--bins must be positive")
    if fixed_range is not None:
        low, high = fixed_range
    elif values:
        low, high = min(values), max(values)
        if low == high:
            delta = abs(low) * 0.05 + 1e-9
            low, high = low - delta, high + delta
    else:
        return {
            "range": [None, None],
            "bins": bins,
            "bin_edges": [],
            "counts": [],
            "underflow": 0,
            "overflow": 0,
        }
    width = (high - low) / bins
    counts = [0] * bins
    underflow = 0
    overflow = 0
    for value in values:
        if value < low:
            underflow += 1
        elif value > high:
            overflow += 1
        else:
            index = bins - 1 if value == high else int((value - low) / width)
            counts[index] += 1
    return {
        "range": [low, high],
        "bins": bins,
        "bin_edges": [low + i * width for i in range(bins + 1)],
        "counts": counts,
        "underflow": underflow,
        "overflow": overflow,
    }


def numeric_distribution(
    values: list[float],
    *,
    bins: int,
    fixed_range: Optional[tuple[float, float]],
    include_values: bool,
) -> dict[str, Any]:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    ordered = sorted(clean)
    if clean:
        mean = sum(clean) / len(clean)
        variance = sum((value - mean) ** 2 for value in clean) / len(clean)
        result = {
            "count": len(clean),
            "mean": mean,
            "std": math.sqrt(variance),
            "min": ordered[0],
            "max": ordered[-1],
            "quantiles": {str(q): percentile(ordered, q) for q in QUANTILES},
        }
    else:
        result = {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "quantiles": {str(q): None for q in QUANTILES},
        }
    result["histogram"] = histogram(clean, bins, fixed_range)
    if include_values:
        result["values"] = clean
    return result


def rank_distribution(values: list[int], include_values: bool) -> dict[str, Any]:
    result = numeric_distribution(
        [float(value) for value in values],
        bins=50,
        fixed_range=None,
        include_values=include_values,
    )
    counts = Counter(values)
    result["exact_rank_counts"] = {
        str(rank): counts[rank] for rank in sorted(counts)
    }
    return result


def load_records(path: Path) -> Iterable[dict[str, Any]]:
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                yield {
                    "event": "_decode_error",
                    "line_number": line_number,
                    "error": str(exc),
                }


def summarize(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    accepted_confidence: list[float] = []
    rejected_confidence: list[float] = []
    rejected_rank: list[int] = []
    drop_abs: list[float] = []
    drop_pct: list[float] = []
    decode_errors: list[dict[str, Any]] = []
    backends: set[str] = set()
    modes: set[str] = set()
    total_rounds = 0
    with_rejection = 0
    with_accepted = 0
    with_drop = 0
    emitted_tokens = 0
    accepted_tokens = 0
    draft_forward_passes = 0
    mask_selected_rounds = 0

    for row in load_records(path) or []:
        if row.get("event") == "_decode_error":
            decode_errors.append(row)
            continue
        if row.get("event") != "linearspec_confidence_round":
            continue
        total_rounds += 1
        if row.get("backend"):
            backends.add(str(row["backend"]))
        if row.get("mode"):
            modes.add(str(row["mode"]))
        emitted_tokens += int(row.get("emitted_tokens") or 0)
        accepted_tokens += int(row.get("accepted_draft_tokens") or 0)
        draft_forward_passes += int(row.get("draft_forward_passes") or 0)
        if row.get("mask_selected_positions"):
            mask_selected_rounds += 1
        values = [
            value
            for value in (
                as_float(item)
                for item in (row.get("accepted_draft_confidences") or [])
            )
            if value is not None
        ]
        accepted_confidence.extend(values)
        if values:
            with_accepted += 1
        if row.get("has_rejection"):
            with_rejection += 1
        value = as_float(row.get("rejected_draft_confidence"))
        if value is not None:
            rejected_confidence.append(value)
        rank = as_int(row.get("rejected_correct_token_rank"))
        if rank is not None:
            rejected_rank.append(rank)
        value = as_float(row.get("confidence_drop_abs"))
        if value is not None:
            drop_abs.append(value)
        value = as_float(row.get("confidence_drop_pct"))
        if value is not None:
            drop_pct.append(value)
            with_drop += 1

    include_values = bool(args.include_values and not args.no_values)
    distributions = {
        "accepted_draft_confidence": numeric_distribution(
            accepted_confidence,
            bins=args.bins,
            fixed_range=(0.0, 1.0),
            include_values=include_values,
        ),
        "rejected_draft_confidence": numeric_distribution(
            rejected_confidence,
            bins=args.bins,
            fixed_range=(0.0, 1.0),
            include_values=include_values,
        ),
        "rejected_correct_token_rank": rank_distribution(
            rejected_rank, include_values
        ),
        "confidence_drop_abs": numeric_distribution(
            drop_abs,
            bins=args.bins,
            fixed_range=None,
            include_values=include_values,
        ),
        "confidence_drop_pct": numeric_distribution(
            drop_pct,
            bins=args.bins,
            fixed_range=None,
            include_values=include_values,
        ),
    }
    checks = {
        "accepted_value_count_matches_tokens": len(accepted_confidence)
        == accepted_tokens,
        "rejected_confidence_count_matches_rank_count": len(rejected_confidence)
        == len(rejected_rank),
        "drop_count_not_greater_than_rejections": len(drop_abs)
        <= len(rejected_confidence),
        "confidence_values_in_unit_interval": all(
            0.0 <= value <= 1.0
            for value in accepted_confidence + rejected_confidence
        ),
        "rank_values_positive": all(value >= 1 for value in rejected_rank),
    }
    return {
        "schema_version": 2,
        "status": "ok" if total_rounds > 0 and all(checks.values()) else "invalid",
        "backend": sorted(backends),
        "modes": sorted(modes),
        "benchmark": args.benchmark,
        "benchmark_spec": args.benchmark_spec,
        "trace_file": str(path),
        "rounds": {
            "total": total_rounds,
            "with_rejection": with_rejection,
            "with_accepted_draft_tokens": with_accepted,
            "with_confidence_drop": with_drop,
            "emitted_tokens": emitted_tokens,
            "accepted_draft_tokens": accepted_tokens,
            "draft_forward_passes": draft_forward_passes,
            "mask_selected_rounds": mask_selected_rounds,
        },
        "distributions": distributions,
        "validation": checks,
        "decode_errors": decode_errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-file", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--benchmark", default="")
    parser.add_argument("--benchmark-spec", default="")
    parser.add_argument("--bins", type=int, default=100)
    parser.add_argument("--include-values", action="store_true")
    parser.add_argument("--no-values", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    payload = summarize(args.trace_file, args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {args.output_json}")
    if payload["status"] != "ok":
        print("Trace summary validation failed or no trace rounds were found")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
