#!/usr/bin/env python3
"""Summarize paired block-size JSONL traces into complete Q1/Q2/Q3 metrics."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Optional


QUANTILES = (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)


def finite_float(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def percentile(ordered: list[float], q: float) -> Optional[float]:
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    position = q * (len(ordered) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    weight = position - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def distribution(values: Iterable[Any], *, exact: bool = False) -> dict[str, Any]:
    clean = [value for item in values if (value := finite_float(item)) is not None]
    ordered = sorted(clean)
    if clean:
        mean = sum(clean) / len(clean)
        variance = sum((value - mean) ** 2 for value in clean) / len(clean)
        payload: dict[str, Any] = {
            "count": len(clean),
            "mean": mean,
            "std": math.sqrt(variance),
            "min": ordered[0],
            "max": ordered[-1],
            "quantiles": {str(q): percentile(ordered, q) for q in QUANTILES},
        }
    else:
        payload = {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "quantiles": {str(q): None for q in QUANTILES},
        }
    if exact:
        counts = Counter(clean)
        payload["exact_counts"] = {
            str(int(key) if float(key).is_integer() else key): counts[key]
            for key in sorted(counts)
        }
    return payload


def pearson(left: list[float], right: list[float]) -> Optional[float]:
    if len(left) != len(right) or len(left) < 2:
        return None
    mean_left = sum(left) / len(left)
    mean_right = sum(right) / len(right)
    numerator = sum(
        (x - mean_left) * (y - mean_right) for x, y in zip(left, right)
    )
    left_scale = math.sqrt(sum((x - mean_left) ** 2 for x in left))
    right_scale = math.sqrt(sum((y - mean_right) ** 2 for y in right))
    if left_scale == 0 or right_scale == 0:
        return None
    return numerator / (left_scale * right_scale)


def ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    output = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average_rank = (start + 1 + end) / 2
        for offset in range(start, end):
            output[order[offset]] = average_rank
        start = end
    return output


def correlation(left: Iterable[Any], right: Iterable[Any]) -> dict[str, Any]:
    pairs = []
    for x_value, y_value in zip(left, right):
        x = finite_float(x_value)
        y = finite_float(y_value)
        if x is not None and y is not None:
            pairs.append((x, y))
    x_values = [item[0] for item in pairs]
    y_values = [item[1] for item in pairs]
    return {
        "count": len(pairs),
        "pearson": pearson(x_values, y_values),
        "spearman": pearson(ranks(x_values), ranks(y_values)) if pairs else None,
    }


def load_records(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    if not path.is_file():
        return records, [{"error": "trace file missing", "path": str(path)}]
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append({"line": line_number, "error": str(exc)})
                continue
            if payload.get("event") == "linearspec_block_size_shadow_round":
                records.append(payload)
    return records, errors


def _rate(flags: Iterable[Any]) -> Optional[float]:
    values = [bool(value) for value in flags]
    return sum(values) / len(values) if values else None


def _nested_mean(values: list[Optional[float]]) -> Optional[float]:
    clean = [float(value) for value in values if value is not None]
    return sum(clean) / len(clean) if clean else None


def summarize_trace(
    trace_path: Path,
    *,
    benchmark: str,
    benchmark_spec: str,
    include_boundary_rounds: bool = False,
) -> dict[str, Any]:
    raw_records, decode_errors = load_records(trace_path)
    records = (
        raw_records
        if include_boundary_rounds
        else [record for record in raw_records if record.get("analysis_valid")]
    )
    block_sizes = sorted(
        {
            int(size)
            for record in raw_records
            for size in (record.get("block_sizes") or [])
        }
    )
    pair_names = [
        f"{small}_{large}"
        for index, small in enumerate(block_sizes)
        for large in block_sizes[index + 1 :]
    ]
    block_metrics: dict[str, Any] = {}
    for size in block_sizes:
        key = str(size)
        branches = [
            record.get("branches", {}).get(key)
            for record in records
            if isinstance(record.get("branches", {}).get(key), dict)
        ]
        accept = [branch.get("accept_length") for branch in branches]
        matched = [branch.get("matched_draft_tokens") for branch in branches]
        position_buckets: dict[int, dict[str, list[Any]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for branch in branches:
            position = branch.get("position") or {}
            fields = (
                "selected_confidence",
                "top1_top2_margin",
                "entropy",
                "selected_is_top1",
                "draft_accepted",
            )
            for field in fields:
                for index, value in enumerate(position.get(field) or [], 1):
                    position_buckets[index][field].append(value)
        per_position = []
        for position in sorted(position_buckets):
            bucket = position_buckets[position]
            per_position.append(
                {
                    "draft_position": position,
                    "count": len(bucket.get("selected_confidence", [])),
                    "accepted_rate": _rate(bucket.get("draft_accepted", [])),
                    "selected_top1_rate": _rate(bucket.get("selected_is_top1", [])),
                    "selected_confidence": distribution(
                        bucket.get("selected_confidence", [])
                    ),
                    "top1_top2_margin": distribution(
                        bucket.get("top1_top2_margin", [])
                    ),
                    "entropy": distribution(bucket.get("entropy", [])),
                }
            )
        block_metrics[key] = {
            "rounds": len(branches),
            "accept_length": distribution(accept, exact=True),
            "matched_draft_tokens": distribution(matched, exact=True),
            "accept_rate_mean": _nested_mean(
                [finite_float(branch.get("accept_rate")) for branch in branches]
            ),
            "full_accept_rate": _rate(
                branch.get("full_accept") for branch in branches
            ),
            "zero_draft_match_rate": _rate(
                branch.get("zero_draft_match") for branch in branches
            ),
            "draft_forward_passes": distribution(
                branch.get("draft_forward_passes") for branch in branches
            ),
            "accepted_confidence_mean": distribution(
                branch.get("accepted_confidence_mean") for branch in branches
            ),
            "accepted_confidence_min": distribution(
                branch.get("accepted_confidence_min") for branch in branches
            ),
            "accepted_confidence_last": distribution(
                branch.get("accepted_confidence_last") for branch in branches
            ),
            "rejected_confidence": distribution(
                branch.get("rejected_confidence") for branch in branches
            ),
            "rejected_margin": distribution(
                branch.get("rejected_margin") for branch in branches
            ),
            "rejected_entropy": distribution(
                branch.get("rejected_entropy") for branch in branches
            ),
            "per_position": per_position,
        }

    pair_metrics: dict[str, Any] = {}
    survival: list[dict[str, Any]] = []
    identity_errors = 0
    bounds_errors = 0
    for pair_name in pair_names:
        small_size, large_size = (int(value) for value in pair_name.split("_"))
        rows = [
            record.get("pairs", {}).get(pair_name)
            for record in records
            if isinstance(record.get("pairs", {}).get(pair_name), dict)
        ]
        small_a = [
            record["branches"][str(small_size)]["accept_length"]
            for record in records
            if pair_name in record.get("pairs", {})
        ]
        large_a = [
            record["branches"][str(large_size)]["accept_length"]
            for record in records
            if pair_name in record.get("pairs", {})
        ]
        for row, a_small, a_large in zip(rows, small_a, large_a):
            if int(row["delta_a"]) != int(row["tail"]) + int(row["decay"]):
                identity_errors += 1
            if not (1 <= int(a_small) <= small_size and 1 <= int(a_large) <= large_size):
                bounds_errors += 1
        conditional: dict[str, Counter[int]] = defaultdict(Counter)
        for a_small, a_large in zip(small_a, large_a):
            conditional[str(int(a_large))][int(a_small)] += 1
        pair_metrics[pair_name] = {
            "small_size": small_size,
            "large_size": large_size,
            "rounds": len(rows),
            "delta_a": distribution((row.get("delta_a") for row in rows), exact=True),
            "tail": distribution((row.get("tail") for row in rows), exact=True),
            "decay": distribution((row.get("decay") for row in rows), exact=True),
            "large_better_rate": _rate(row.get("delta_a", 0) > 0 for row in rows),
            "equal_rate": _rate(row.get("delta_a", 0) == 0 for row in rows),
            "small_better_rate": _rate(row.get("delta_a", 0) < 0 for row in rows),
            "monotonic_violation_rate": _rate(
                row.get("small_gt_large") for row in rows
            ),
            "large_exceeds_small_capacity_rate": _rate(
                int(value) > small_size for value in large_a
            ),
            "accept_length_correlation": correlation(small_a, large_a),
            "draft_agreement_rate": distribution(
                (row.get("draft_agreement") or {}).get("agreement_rate")
                for row in rows
            ),
            "draft_all_common_equal_rate": _rate(
                (row.get("draft_agreement") or {}).get("all_common_equal")
                for row in rows
            ),
            "first_draft_divergence_position": distribution(
                (row.get("draft_agreement") or {}).get("first_divergence_position")
                for row in rows
            ),
            "verifier_agreement_rate": distribution(
                (row.get("verifier_agreement") or {}).get("agreement_rate")
                for row in rows
            ),
            "conditional_a_small_given_a_large_counts": {
                large: {str(small): counter[small] for small in sorted(counter)}
                for large, counter in sorted(
                    conditional.items(), key=lambda item: int(item[0])
                )
            },
        }
        for k in range(1, small_size + 2):
            denominator = sum(1 for value in large_a if int(value) >= k)
            numerator = sum(
                1
                for a_small, a_large in zip(small_a, large_a)
                if int(a_large) >= k and int(a_small) >= k
            )
            survival.append(
                {
                    "pair": pair_name,
                    "small_size": small_size,
                    "large_size": large_size,
                    "k": k,
                    "denominator_n2": denominator,
                    "numerator_n12": numerator,
                    "survival": numerator / denominator if denominator else None,
                    "structural_endpoint": k == small_size + 1,
                }
            )

    feature_names = sorted(
        {
            key
            for record in records
            for key in (record.get("history_before_round") or {})
        }
    )
    targets: dict[str, list[Any]] = {}
    for size in block_sizes:
        targets[f"a{size}"] = [
            record.get("branches", {}).get(str(size), {}).get("accept_length")
            for record in records
        ]
    for pair_name in pair_names:
        for field in ("delta_a", "tail", "decay"):
            targets[f"{field}_{pair_name}"] = [
                record.get("pairs", {}).get(pair_name, {}).get(field)
                for record in records
            ]
    history_correlations: dict[str, Any] = {}
    for feature in feature_names:
        feature_values = [
            (record.get("history_before_round") or {}).get(feature)
            for record in records
        ]
        history_correlations[feature] = {
            target: correlation(feature_values, values)
            for target, values in targets.items()
        }
    predictor_mae: dict[str, Any] = {}
    for feature in [name for name in feature_names if name.startswith("a_") or name == "prev_anchor_a"]:
        predictions = [
            (record.get("history_before_round") or {}).get(feature)
            for record in records
        ]
        predictor_mae[feature] = {}
        for size in block_sizes:
            errors = []
            for prediction, target in zip(predictions, targets[f"a{size}"]):
                value = finite_float(prediction)
                actual = finite_float(target)
                if value is not None and actual is not None:
                    errors.append(abs(min(max(value, 1.0), float(size)) - actual))
            predictor_mae[feature][f"a{size}"] = distribution(errors)

    expected_pairs = len(block_sizes) * (len(block_sizes) - 1) // 2
    checks = {
        "has_rounds": bool(raw_records),
        "has_analysis_rounds": bool(records),
        "all_records_have_all_blocks": all(
            len(record.get("branches", {})) == len(block_sizes)
            for record in raw_records
        ),
        "all_records_have_all_pairs": all(
            len(record.get("pairs", {})) == expected_pairs for record in raw_records
        ),
        "pair_decomposition_identity": identity_errors == 0,
        "accept_length_bounds": bounds_errors == 0,
        "survival_has_all_k_including_endpoint": len(survival)
        == sum(small + 1 for index, small in enumerate(block_sizes) for _ in block_sizes[index + 1 :]),
        "no_decode_errors": not decode_errors,
    }
    return {
        "schema_version": 1,
        "status": "ok" if all(checks.values()) else "invalid",
        "benchmark": benchmark,
        "benchmark_spec": benchmark_spec,
        "trace_file": str(trace_path.resolve()),
        "analysis_scope": (
            "all_rounds" if include_boundary_rounds else "valid_non_eos_non_budget_boundary_rounds"
        ),
        "block_sizes": block_sizes,
        "rounds": {
            "raw": len(raw_records),
            "analysis": len(records),
            "excluded": len(raw_records) - len(records),
            "budget_boundary": sum(
                bool(record.get("budget_boundary")) for record in raw_records
            ),
            "any_branch_eos": sum(
                bool(record.get("any_branch_eos")) for record in raw_records
            ),
            "requests": len({record.get("request_id") for record in raw_records}),
        },
        "blocks": block_metrics,
        "pairs": pair_metrics,
        "survival": survival,
        "history_reference": {
            "feature_target_correlation": history_correlations,
            "clipped_history_predictor_mae": predictor_mae,
        },
        "validation": checks,
        "decode_errors": decode_errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-file", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--benchmark-spec", required=True)
    parser.add_argument("--include-boundary-rounds", action="store_true")
    args = parser.parse_args()
    payload = summarize_trace(
        args.trace_file,
        benchmark=args.benchmark,
        benchmark_spec=args.benchmark_spec,
        include_boundary_rounds=args.include_boundary_rounds,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output_json.with_suffix(args.output_json.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output_json)
    print(f"Wrote {args.output_json} ({payload['status']})")
    return 0 if payload["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
