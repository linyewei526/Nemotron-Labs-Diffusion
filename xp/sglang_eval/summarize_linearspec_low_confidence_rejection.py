#!/usr/bin/env python3
"""Summarize LinearSpec low-confidence token rejection traces."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable


COUNTED_OUTCOMES = {"accepted", "rejected"}


def decimal_range(start: str, end: str, step: str) -> list[Decimal]:
    current = Decimal(start)
    end_d = Decimal(end)
    step_d = Decimal(step)
    if step_d <= 0:
        raise ValueError("step must be positive")
    values: list[Decimal] = []
    while current <= end_d:
        values.append(current)
        current += step_d
    if values and values[-1] != end_d and values[-1] < end_d:
        values.append(end_d)
    return values


def ratio(num: int, den: int) -> float | None:
    return (num / den) if den else None


def threshold_width(thresholds: list[Decimal]) -> int:
    return max(-min(t.as_tuple().exponent for t in thresholds), 0) if thresholds else 0


def threshold_key(threshold: Decimal, thresholds: list[Decimal], suffix: str) -> str:
    return f"token_{float(threshold):.{threshold_width(thresholds)}f}_{suffix}"


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def load_records(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                yield {"event": "_decode_error", "line_no": line_no, "error": str(exc)}


def empty_threshold_table(
    thresholds: list[Decimal],
    *,
    key_suffix: str,
) -> dict[str, dict[str, Any]]:
    table: dict[str, dict[str, Any]] = {}
    for threshold in thresholds:
        key = threshold_key(threshold, thresholds, key_suffix)
        table[key] = {
            "accepted_count": 0,
            "rejected_count": 0,
            "accepted_ratio_within_flagged": None,
            "rejected_ratio_within_flagged": None,
            "accepted_coverage_of_all_countable_accepted_tokens": None,
            "rejected_coverage_of_all_countable_rejected_tokens": None,
        }
    return table


def finalize_threshold_table(
    table: dict[str, dict[str, Any]],
    *,
    countable_tokens: int,
    countable_accepted: int,
    countable_rejected: int,
) -> None:
    for item in table.values():
        accepted = int(item["accepted_count"])
        rejected = int(item["rejected_count"])
        total = accepted + rejected
        item["accepted_ratio_within_flagged"] = ratio(accepted, total)
        item["rejected_ratio_within_flagged"] = ratio(rejected, total)
        item["accepted_coverage_of_all_countable_accepted_tokens"] = ratio(
            accepted, countable_accepted
        )
        item["rejected_coverage_of_all_countable_rejected_tokens"] = ratio(
            rejected, countable_rejected
        )


def summarize_trace(
    trace_file: Path,
    *,
    abs_start: str,
    abs_end: str,
    abs_step: str,
    pct_start: str,
    pct_end: str,
    pct_step: str,
) -> dict[str, Any]:
    abs_thresholds = decimal_range(abs_start, abs_end, abs_step)
    pct_thresholds = decimal_range(pct_start, pct_end, pct_step)
    abs_table = empty_threshold_table(
        abs_thresholds, key_suffix="drop_abs"
    )
    pct_table = empty_threshold_table(
        pct_thresholds, key_suffix="drop_pct"
    )

    rounds = 0
    decode_errors: list[dict[str, Any]] = []
    outcome_counts: Counter[str] = Counter()
    countable_tokens = 0
    countable_accepted = 0
    countable_rejected = 0
    skipped_no_prefix = 0
    skipped_invalid_confidence = 0
    skipped_invalid_prefix_mean = 0

    for row in load_records(trace_file) or []:
        if row.get("event") == "_decode_error":
            decode_errors.append(row)
            continue
        if row.get("event") != "linearspec_low_confidence_round":
            continue
        rounds += 1
        confidences_raw = row.get("draft_confidences") or []
        outcomes = row.get("outcomes") or []
        n = min(len(confidences_raw), len(outcomes))
        confidences = [as_float(x) for x in confidences_raw[:n]]

        prefix_sum = 0.0
        prefix_count = 0
        for idx in range(n):
            outcome = str(outcomes[idx])
            confidence = confidences[idx]
            outcome_counts[outcome] += 1

            if confidence is None:
                skipped_invalid_confidence += 1
                continue

            if outcome in COUNTED_OUTCOMES:
                if prefix_count == 0:
                    skipped_no_prefix += 1
                else:
                    prefix_mean = prefix_sum / prefix_count
                    drop_abs = prefix_mean - confidence
                    drop_pct = None
                    if prefix_mean > 0:
                        drop_pct = 1.0 - confidence / prefix_mean
                    else:
                        skipped_invalid_prefix_mean += 1

                    countable_tokens += 1
                    if outcome == "accepted":
                        countable_accepted += 1
                    elif outcome == "rejected":
                        countable_rejected += 1

                    for threshold in abs_thresholds:
                        if drop_abs >= float(threshold):
                            key = threshold_key(
                                threshold, abs_thresholds, "drop_abs"
                            )
                            abs_table[key][f"{outcome}_count"] += 1
                    if drop_pct is not None:
                        for threshold in pct_thresholds:
                            if drop_pct >= float(threshold):
                                key = threshold_key(
                                    threshold, pct_thresholds, "drop_pct"
                                )
                                pct_table[key][f"{outcome}_count"] += 1

            prefix_sum += confidence
            prefix_count += 1

    finalize_threshold_table(
        abs_table,
        countable_tokens=countable_tokens,
        countable_accepted=countable_accepted,
        countable_rejected=countable_rejected,
    )
    finalize_threshold_table(
        pct_table,
        countable_tokens=countable_tokens,
        countable_accepted=countable_accepted,
        countable_rejected=countable_rejected,
    )

    return {
        "schema_version": 1,
        "trace_file": str(trace_file),
        "threshold_definition": {
            "drop_abs": {
                "formula": "C_imean - C_i",
                "start": float(Decimal(abs_start)),
                "end": float(Decimal(abs_end)),
                "step": float(Decimal(abs_step)),
                "inclusive": True,
            },
            "drop_pct": {
                "formula": "1 - C_i / C_imean",
                "start": float(Decimal(pct_start)),
                "end": float(Decimal(pct_end)),
                "step": float(Decimal(pct_step)),
                "inclusive": True,
            },
            "C_imean": "mean confidence of all draft candidates before token i in the same round",
            "confidence": "softmax probability of the drafted token after excluding MASK from the vocabulary",
            "counted_outcomes": sorted(COUNTED_OUTCOMES),
            "ignored_outcomes": [
                "unverified_after_rejection",
                "unverified_after_eos",
            ],
        },
        "rounds": {
            "total": rounds,
        },
        "tokens": {
            "candidate_tokens_by_outcome": dict(sorted(outcome_counts.items())),
            "countable_tokens_with_prefix": countable_tokens,
            "countable_accepted_tokens_with_prefix": countable_accepted,
            "countable_rejected_tokens_with_prefix": countable_rejected,
            "skipped_no_prefix": skipped_no_prefix,
            "skipped_invalid_confidence": skipped_invalid_confidence,
            "skipped_invalid_prefix_mean": skipped_invalid_prefix_mean,
        },
        "token_x_drop_abs": abs_table,
        "token_y_drop_pct": pct_table,
        "decode_errors": decode_errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-file", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--benchmark", default="")
    parser.add_argument("--benchmark-spec", default="")
    parser.add_argument("--abs-start", default="0.300")
    parser.add_argument("--abs-end", default="0.400")
    parser.add_argument("--abs-step", default="0.005")
    parser.add_argument("--pct-start", default="0.40")
    parser.add_argument("--pct-end", default="0.60")
    parser.add_argument("--pct-step", default="0.01")
    args = parser.parse_args()

    payload = summarize_trace(
        args.trace_file,
        abs_start=args.abs_start,
        abs_end=args.abs_end,
        abs_step=args.abs_step,
        pct_start=args.pct_start,
        pct_end=args.pct_end,
        pct_step=args.pct_step,
    )
    payload["benchmark"] = args.benchmark
    payload["benchmark_spec"] = args.benchmark_spec

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, sort_keys=True)
        f.write("\n")


if __name__ == "__main__":
    main()
