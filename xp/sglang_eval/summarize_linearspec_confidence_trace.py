#!/usr/bin/env python3
"""Summarize LinearSpec confidence trace JSONL files.

By default the summary keeps distribution statistics, histograms, quantiles, and
exact rank counts only. Raw value arrays are included only with --include-values.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


QUANTILES = [0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def percentile(sorted_values: list[float], q: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = q * (len(sorted_values) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_values[lo]
    weight = pos - lo
    return sorted_values[lo] * (1 - weight) + sorted_values[hi] * weight


def fixed_histogram(values: list[float], bins: int, lo: float, hi: float) -> dict[str, Any]:
    if bins <= 0:
        raise ValueError("bins must be positive")
    counts = [0] * bins
    underflow = 0
    overflow = 0
    width = (hi - lo) / bins
    if width <= 0:
        return {"range": [lo, hi], "bins": bins, "counts": counts, "underflow": len(values), "overflow": 0}

    for value in values:
        if value < lo:
            underflow += 1
        elif value > hi:
            overflow += 1
        else:
            idx = bins - 1 if value == hi else int((value - lo) / width)
            counts[idx] += 1
    return {
        "range": [lo, hi],
        "bins": bins,
        "bin_edges": [lo + i * width for i in range(bins + 1)],
        "counts": counts,
        "underflow": underflow,
        "overflow": overflow,
    }


def auto_histogram(values: list[float], bins: int) -> dict[str, Any]:
    if not values:
        return {"range": [None, None], "bins": bins, "bin_edges": [], "counts": [], "underflow": 0, "overflow": 0}
    lo = min(values)
    hi = max(values)
    if lo == hi:
        delta = abs(lo) * 0.05 + 1e-9
        lo -= delta
        hi += delta
    return fixed_histogram(values, bins, lo, hi)


def numeric_distribution(values: list[float], *, bins: int, fixed_range: tuple[float, float] | None, include_values: bool) -> dict[str, Any]:
    values = [float(v) for v in values if math.isfinite(float(v))]
    sorted_values = sorted(values)
    count = len(values)
    if count:
        mean = sum(values) / count
        var = sum((v - mean) ** 2 for v in values) / count
        summary = {
            "count": count,
            "mean": mean,
            "std": math.sqrt(var),
            "min": sorted_values[0],
            "max": sorted_values[-1],
            "quantiles": {str(q): percentile(sorted_values, q) for q in QUANTILES},
        }
    else:
        summary = {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "quantiles": {str(q): None for q in QUANTILES},
        }
    summary["histogram"] = (
        fixed_histogram(values, bins, fixed_range[0], fixed_range[1])
        if fixed_range is not None
        else auto_histogram(values, bins)
    )
    if include_values:
        summary["values"] = values
    return summary


def rank_distribution(values: list[int], include_values: bool) -> dict[str, Any]:
    float_values = [float(v) for v in values]
    out = numeric_distribution(float_values, bins=50, fixed_range=None, include_values=include_values)
    counts = Counter(values)
    out["exact_rank_counts"] = {str(rank): counts[rank] for rank in sorted(counts)}
    return out


def load_records(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                yield {"event": "_decode_error", "line_no": line_no, "error": str(exc)}
                continue
            yield row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-file", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--benchmark", default="")
    parser.add_argument("--benchmark-spec", default="")
    parser.add_argument("--bins", type=int, default=100)
    parser.add_argument("--include-values", action="store_true", help="Include raw value arrays in the summary JSON.")
    parser.add_argument("--no-values", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    accepted_conf: list[float] = []
    rejected_conf: list[float] = []
    rejected_rank: list[int] = []
    drop_abs: list[float] = []
    drop_pct: list[float] = []
    decode_errors: list[dict[str, Any]] = []
    rounds = 0
    rounds_with_rejection = 0
    rounds_with_accepted_draft = 0
    rounds_with_drop = 0
    emitted_tokens = 0
    accepted_draft_tokens = 0

    for row in load_records(args.trace_file) or []:
        if row.get("event") == "_decode_error":
            decode_errors.append(row)
            continue
        if row.get("event") != "linearspec_confidence_round":
            continue
        rounds += 1
        emitted_tokens += int(row.get("emitted_tokens") or 0)
        accepted_draft_tokens += int(row.get("accepted_draft_tokens") or 0)
        vals = row.get("accepted_draft_confidences") or []
        clean_vals = [v for v in (as_float(x) for x in vals) if v is not None]
        accepted_conf.extend(clean_vals)
        if clean_vals:
            rounds_with_accepted_draft += 1
        if row.get("has_rejection"):
            rounds_with_rejection += 1
        v = as_float(row.get("rejected_draft_confidence"))
        if v is not None:
            rejected_conf.append(v)
        r = as_int(row.get("rejected_correct_token_rank"))
        if r is not None:
            rejected_rank.append(r)
        v = as_float(row.get("confidence_drop_abs"))
        if v is not None:
            drop_abs.append(v)
        v = as_float(row.get("confidence_drop_pct"))
        if v is not None:
            drop_pct.append(v)
            rounds_with_drop += 1

    include_values = bool(args.include_values and not args.no_values)
    payload = {
        "schema_version": 1,
        "benchmark": args.benchmark,
        "benchmark_spec": args.benchmark_spec,
        "trace_file": str(args.trace_file),
        "rounds": {
            "total": rounds,
            "with_rejection": rounds_with_rejection,
            "with_accepted_draft_tokens": rounds_with_accepted_draft,
            "with_confidence_drop": rounds_with_drop,
            "emitted_tokens": emitted_tokens,
            "accepted_draft_tokens": accepted_draft_tokens,
        },
        "distributions": {
            "accepted_draft_confidence": numeric_distribution(
                accepted_conf, bins=args.bins, fixed_range=(0.0, 1.0), include_values=include_values
            ),
            "rejected_draft_confidence": numeric_distribution(
                rejected_conf, bins=args.bins, fixed_range=(0.0, 1.0), include_values=include_values
            ),
            "rejected_correct_token_rank": rank_distribution(
                rejected_rank, include_values=include_values
            ),
            "confidence_drop_abs": numeric_distribution(
                drop_abs, bins=args.bins, fixed_range=None, include_values=include_values
            ),
            "confidence_drop_pct": numeric_distribution(
                drop_pct, bins=args.bins, fixed_range=None, include_values=include_values
            ),
        },
        "decode_errors": decode_errors,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, sort_keys=True)
        f.write("\n")


if __name__ == "__main__":
    main()
