#!/usr/bin/env python3
"""Merge fixed-margin-risk P1/P2 plus always-New stats into NeMo metrics."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable, Optional


OUTCOME_STATES = (
    "miss_no_candidate_error",
    "miss_before_first",
    "miss_between_candidates",
    "miss_after_last",
    "candidate_1_fixed",
    "candidate_1_wrong",
    "candidate_2_fixed",
    "candidate_2_wrong",
    "full_continuation_hit",
    "full_continuation_miss",
    "full_continuation_absent",
)
FORWARD_KINDS = ("prefill", "normal_draft", "normal_verify", "multi_fused")
OUTCOME_SUM_FIELDS = (
    "count",
    "current_accept_sum",
    "next_count",
    "paired_current_accept_sum",
    "next_accept_sum",
    "next_minus_current_sum",
)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, sort_keys=True)
        f.write("\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def as_int(value: Any) -> Optional[int]:
    result = as_float(value)
    if result is None or not result.is_integer():
        return None
    return int(result)


def rounded(value: Optional[float], digits: int = 6) -> Optional[float]:
    return None if value is None else round(float(value), digits)


def percentile(values: list[float], quantile: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - position) + ordered[high] * (position - low)


def distribution(values: Iterable[Optional[float]]) -> dict[str, Optional[float]]:
    clean = [float(value) for value in values if value is not None]
    return {
        "mean": rounded(sum(clean) / len(clean)) if clean else None,
        "p50": rounded(percentile(clean, 0.50)),
        "p90": rounded(percentile(clean, 0.90)),
        "p95": rounded(percentile(clean, 0.95)),
        "p99": rounded(percentile(clean, 0.99)),
    }


def histogram_percentile(histogram: dict[str, int], quantile: float) -> Optional[float]:
    total = sum(histogram.values())
    if total <= 0:
        return None
    target = (total - 1) * quantile
    low_rank = math.floor(target)
    high_rank = math.ceil(target)

    def value_at(rank: int) -> int:
        seen = 0
        for value, count in sorted(histogram.items(), key=lambda item: int(item[0])):
            seen += count
            if rank < seen:
                return int(value)
        raise RuntimeError("invalid forward histogram")

    low = value_at(low_rank)
    high = value_at(high_rank)
    return low * (high_rank - target) + high * (target - low_rank) if low_rank != high_rank else float(low)


def aggregate_forward_kinds(
    rows: list[dict[str, Any]], selected_kinds: tuple[str, ...]
) -> dict[str, Any]:
    sums = {
        "count": 0,
        "computed_token_sum": 0,
        "valid_token_sum": 0,
        "padding_token_sum": 0,
        "row_sum": 0,
        "query_length_sum": 0,
    }
    histogram: dict[str, int] = {}
    for row in rows:
        kinds = ((row.get("overlap") or {}).get("forward_kinds") or {})
        for kind in selected_kinds:
            item = kinds.get(kind) or {}
            for key in sums:
                sums[key] += int(as_float(item.get(key)) or 0)
            for value, count in (item.get("computed_token_histogram") or {}).items():
                histogram[str(value)] = histogram.get(str(value), 0) + int(count)
    count = sums["count"]
    computed = sums["computed_token_sum"]
    values = [int(value) for value in histogram]
    return {
        **sums,
        "computed_token_histogram": histogram,
        "computed_token_avg": rounded(sums["computed_token_sum"] / count if count else None),
        "valid_token_avg": rounded(sums["valid_token_sum"] / count if count else None),
        "padding_token_avg": rounded(sums["padding_token_sum"] / count if count else None),
        "padding_ratio": rounded(sums["padding_token_sum"] / computed if computed else None),
        "rows_avg": rounded(sums["row_sum"] / count if count else None),
        "query_length_avg": rounded(sums["query_length_sum"] / count if count else None),
        "computed_token_min": min(values) if values else None,
        "computed_token_p50": rounded(histogram_percentile(histogram, 0.50)),
        "computed_token_p90": rounded(histogram_percentile(histogram, 0.90)),
        "computed_token_p95": rounded(histogram_percentile(histogram, 0.95)),
        "computed_token_p99": rounded(histogram_percentile(histogram, 0.99)),
        "computed_token_max": max(values) if values else None,
    }


def forward_pass_breakdown(row: dict[str, Any]) -> tuple[float, float, float]:
    """Return decode, prefill and total NFE; legacy method rows included prefill."""
    decode_nfe = as_float(row.get("decode_nfe"))
    prefill_nfe = as_float(row.get("prefill_nfe"))
    total_nfe = as_float(row.get("total_nfe"))
    if decode_nfe is not None:
        decode_nfe = max(decode_nfe, 0.0)
        prefill_nfe = max(prefill_nfe or 0.0, 0.0)
        total_nfe = max(total_nfe if total_nfe is not None else decode_nfe + prefill_nfe, 0.0)
        return decode_nfe, prefill_nfe, total_nfe
    total_nfe = max(total_nfe or as_float(row.get("nfe")) or 0.0, 0.0)
    prefill_nfe = max(prefill_nfe, 0.0) if prefill_nfe is not None else (1.0 if total_nfe > 0 else 0.0)
    return max(total_nfe - prefill_nfe, 0.0), prefill_nfe, total_nfe


def summarize(rows: list[dict[str, Any]], wall_time_s: Optional[float]) -> dict[str, Any]:
    ok_rows = [row for row in rows if row.get("ok") is True]
    failed_rows = [row for row in rows if row.get("ok") is not True]
    prompt_tokens = sum(as_float(row.get("prompt_tokens")) or 0.0 for row in ok_rows)
    completion_tokens = sum(as_float(row.get("completion_tokens")) or 0.0 for row in ok_rows)
    raw_generated_tokens = sum(as_float(row.get("raw_generated_tokens")) or 0.0 for row in ok_rows)
    nfe_breakdowns = [forward_pass_breakdown(row) for row in ok_rows]
    decode_nfe = sum(item[0] for item in nfe_breakdowns)
    prefill_nfe = sum(item[1] for item in nfe_breakdowns)
    total_nfe = sum(item[2] for item in nfe_breakdowns)
    model_time_s = sum(as_float(row.get("model_time_s")) or 0.0 for row in ok_rows)
    sum_request_time_s = sum(as_float(row.get("request_time_s")) or 0.0 for row in ok_rows)
    tpf = completion_tokens / decode_nfe if decode_nfe > 0 else None
    end_to_end_tpf = completion_tokens / total_nfe if total_nfe > 0 else None

    summary = {
        "attempted_request_count": len(rows),
        "request_count": len(ok_rows),
        "failed_request_count": len(failed_rows),
        "successful_request_rate": rounded(
            len(ok_rows) / len(rows) if rows else None
        ),
        "oom_skipped_request_count": sum(
            1
            for row in failed_rows
            if row.get("oom_skipped_for_efficiency") is True
            or "OutOfMemory" in str(row.get("error_type") or "")
        ),
        "prompt_tokens": round(prompt_tokens, 4),
        "completion_tokens": round(completion_tokens, 4),
        "raw_generated_tokens": round(raw_generated_tokens, 4),
        "forward_passes": round(decode_nfe, 4),
        "decode_forward_passes": round(decode_nfe, 4),
        "prefill_forward_passes": round(prefill_nfe, 4),
        "total_forward_passes": round(total_nfe, 4),
        "tokens_per_forward_pass": rounded(tpf, 4),
        "end_to_end_tokens_per_forward_pass": rounded(end_to_end_tpf, 4),
        "average_forward_passes_per_sample": rounded(
            decode_nfe / len(ok_rows) if ok_rows else None, 4
        ),
        "average_total_forward_passes_per_sample": rounded(
            total_nfe / len(ok_rows) if ok_rows else None, 4
        ),
        "average_completion_tokens": rounded(
            completion_tokens / len(ok_rows) if ok_rows else None, 4
        ),
        "model_generation_time_s": rounded(model_time_s),
        "sum_request_time_s": rounded(sum_request_time_s),
        "benchmark_wall_time_s": rounded(wall_time_s),
        "model_output_tokens_per_s": rounded(
            completion_tokens / model_time_s if model_time_s > 0 else None, 4
        ),
        "benchmark_wall_output_tokens_per_s": rounded(
            completion_tokens / wall_time_s
            if wall_time_s is not None and wall_time_s > 0
            else None,
            4,
        ),
        "benchmark_wall_requests_per_s": rounded(
            len(ok_rows) / wall_time_s
            if wall_time_s is not None and wall_time_s > 0
            else None,
            6,
        ),
        "request_time_s": distribution(
            as_float(row.get("request_time_s")) for row in ok_rows
        ),
        "queue_wait_s": distribution(
            as_float(row.get("queue_wait_s")) for row in ok_rows
        ),
        "per_request_model_time_s": distribution(
            as_float(row.get("model_time_s")) for row in ok_rows
        ),
        "per_request_model_output_tokens_per_s": distribution(
            as_float(row.get("model_output_tokens_per_s")) for row in ok_rows
        ),
        "finish_reasons": {},
        "failure_types": {},
    }
    for row in ok_rows:
        reason = str(row.get("finish_reason") or "unknown")
        summary["finish_reasons"][reason] = summary["finish_reasons"].get(reason, 0) + 1
    for row in failed_rows:
        error_type = str(row.get("error_type") or "unknown")
        summary["failure_types"][error_type] = summary["failure_types"].get(error_type, 0) + 1
    modes = sorted({str(row.get("mode")) for row in ok_rows if row.get("mode")})
    summary["modes"] = modes
    summary["top_p_requested"] = sorted(
        {
            value
            for row in ok_rows
            if (value := as_float(row.get("top_p_requested"))) is not None
        }
    )
    summary["top_p_applied"] = all(
        bool(row.get("top_p_applied")) for row in ok_rows
    ) if ok_rows else None
    summary["top_k_requested"] = sorted(
        {
            value
            for row in ok_rows
            if (value := as_int(row.get("top_k_requested"))) is not None
        }
    )
    summary["top_k_applied"] = all(
        bool(row.get("top_k_applied")) for row in ok_rows
    ) if ok_rows else None
    overlap_sum_fields = [
        "physical_nfe", "processed_rows", "processed_query_tokens",
        "valid_query_tokens", "padding_query_tokens", "rounds",
        "normal_draft_forwards", "normal_verify_forwards",
        "fused_verify_draft_forwards", "rounds_without_crossing",
        "rounds_without_speculative_branch", "prefetch_attempts",
        "prefetch_verified_hits", "prefetch_hits",
        "prefetch_saved_draft_forwards", "candidate_branches_executed",
        "continuation_branches_executed", "continuation_verified_hits",
        "candidate_position_sum", "candidate_position_count",
        "prospective_query_tokens", "candidate_skipped_no_future_round",
        "candidate_skipped_context_limit", "candidate_skipped_thinking_budget",
        "continuation_skipped_no_future_round",
        "continuation_skipped_context_limit",
        "continuation_skipped_thinking_budget",
        "risk_candidates_discarded_after_p2", "rounds_with_3plus_crossings",
        "continuation_attempts_3plus_crossings",
        "continuation_verified_hits_3plus_crossings",
        "continuation_prefetch_hits_3plus_crossings",
        "prefetch_discarded_eos",
        "prefetch_discarded_thinking_budget",
    ]
    overlap = {
        key: int(
            sum(
                as_float((row.get("overlap") or {}).get(key)) or 0.0
                for row in ok_rows
            )
        )
        for key in overlap_sum_fields
    }
    attempts = overlap["prefetch_attempts"]
    overlap["prefetch_hit_rate"] = rounded(
        overlap["prefetch_hits"] / attempts if attempts else None
    )
    overlap["prefetch_verified_hit_rate"] = rounded(
        overlap["prefetch_verified_hits"] / attempts if attempts else None
    )
    overlap["average_candidate_position"] = rounded(
        overlap["candidate_position_sum"] / overlap["candidate_position_count"]
        if overlap["candidate_position_count"]
        else None
    )
    overlap["saved_draft_fraction_of_rounds"] = rounded(
        overlap["prefetch_saved_draft_forwards"] / overlap["rounds"]
        if overlap["rounds"]
        else None
    )
    overlap["padding_query_ratio"] = rounded(
        overlap["padding_query_tokens"] / overlap["processed_query_tokens"]
        if overlap["processed_query_tokens"]
        else None
    )
    rounds = overlap["rounds"]
    rounds_3plus = overlap["rounds_with_3plus_crossings"]
    new_3plus_attempts = overlap["continuation_attempts_3plus_crossings"]
    overlap["continuation_attempt_rate"] = rounded(
        overlap["continuation_branches_executed"] / rounds if rounds else None
    )
    overlap["continuation_3plus_attempt_coverage"] = rounded(
        new_3plus_attempts / rounds_3plus if rounds_3plus else None
    )
    overlap["continuation_3plus_verified_hit_rate"] = rounded(
        overlap["continuation_verified_hits_3plus_crossings"] / new_3plus_attempts
        if new_3plus_attempts
        else None
    )
    overlap["continuation_3plus_prefetch_hit_rate"] = rounded(
        overlap["continuation_prefetch_hits_3plus_crossings"] / new_3plus_attempts
        if new_3plus_attempts
        else None
    )
    for field, keys in (
        ("crossing_count_rounds", ("0", "1", "2", "3+")),
        ("fused_row_count", ("2", "3", "4")),
    ):
        overlap[field] = {
            key: int(
                sum(
                    as_float(((row.get("overlap") or {}).get(field) or {}).get(key)) or 0
                    for row in ok_rows
                )
            )
            for key in keys
        }
    overlap["forward_kinds"] = {
        kind: aggregate_forward_kinds(ok_rows, (kind,)) for kind in FORWARD_KINDS
    }
    overlap["forward_distribution_all"] = aggregate_forward_kinds(ok_rows, FORWARD_KINDS)
    overlap["forward_distribution_decode"] = aggregate_forward_kinds(
        ok_rows, ("normal_draft", "normal_verify", "multi_fused")
    )
    outcome_states: dict[str, dict[str, int | float | None]] = {}
    for state in OUTCOME_STATES:
        aggregate: dict[str, int | float | None] = {
            key: int(
                sum(
                    as_float(
                        (((row.get("overlap") or {}).get("outcome_states") or {}).get(state) or {}).get(key)
                    )
                    or 0.0
                    for row in ok_rows
                )
            )
            for key in OUTCOME_SUM_FIELDS
        }
        count = int(aggregate["count"] or 0)
        next_count = int(aggregate["next_count"] or 0)
        aggregate.update(
            {
                "share_of_attempts": rounded(count / attempts if attempts else None),
                "current_accept_avg": rounded(
                    int(aggregate["current_accept_sum"] or 0) / count if count else None
                ),
                "next_coverage": rounded(next_count / count if count else None),
                "paired_current_accept_avg": rounded(
                    int(aggregate["paired_current_accept_sum"] or 0) / next_count
                    if next_count
                    else None
                ),
                "next_accept_avg": rounded(
                    int(aggregate["next_accept_sum"] or 0) / next_count
                    if next_count
                    else None
                ),
                "next_minus_current_avg": rounded(
                    int(aggregate["next_minus_current_sum"] or 0) / next_count
                    if next_count
                    else None
                ),
            }
        )
        outcome_states[state] = aggregate
    overlap["outcome_states"] = outcome_states
    overlap["outcome_state_count_sum"] = sum(
        int(outcome_states[state]["count"] or 0) for state in OUTCOME_STATES
    )
    overlap["outcome_partition_valid"] = (
        overlap["outcome_state_count_sum"] == attempts
    )
    miss_states = (
        "miss_no_candidate_error", "miss_before_first",
        "miss_between_candidates", "miss_after_last",
    )
    overlap["prediction_miss_count"] = sum(
        int(outcome_states[state]["count"] or 0) for state in miss_states
    )
    overlap["prediction_miss_rate"] = rounded(
        overlap["prediction_miss_count"] / attempts if attempts else None
    )
    overlap["candidate_fixed_count"] = sum(
        int(outcome_states[f"candidate_{rank}_fixed"]["count"] or 0)
        for rank in (1, 2)
    )
    overlap["candidate_wrong_count"] = sum(
        int(outcome_states[f"candidate_{rank}_wrong"]["count"] or 0)
        for rank in (1, 2)
    )
    summary["overlap"] = overlap
    summary["margin_risk_thresholds"] = sorted(
        {
            value
            for row in ok_rows
            if (value := as_float(row.get("margin_risk_threshold"))) is not None
        }
    )
    summary["peak_gpu_memory_gib"] = distribution(
        as_float(row.get("peak_gpu_memory_gib")) for row in ok_rows
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-json", required=True)
    parser.add_argument("--request-stats-file", required=True)
    parser.add_argument("--benchmark", default="")
    parser.add_argument("--wall-time-s", type=float, default=None)
    parser.add_argument("--summary-json", default="")
    parser.add_argument("--create-metrics-if-missing", action="store_true")
    parser.add_argument("--accuracy-status", default="available")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    metrics_path = Path(args.metrics_json)
    if not metrics_path.is_file() and not args.create_metrics_if_missing:
        print(f"Metrics file not found: {metrics_path}")
        return 1
    rows = read_jsonl(Path(args.request_stats_file))
    if not rows:
        print(f"No request stats found: {args.request_stats_file}")
        return 1
    if not any(row.get("ok") is True for row in rows):
        print(f"Request stats contain no successful generations: {args.request_stats_file}")
        return 1
    native = summarize(rows, args.wall_time_s)
    payload = {
        "backend": "native_pytorch_margin_risk_two_plus_new_overlap",
        "benchmark": args.benchmark,
        "accuracy_status": args.accuracy_status,
        "decode": native,
        "metric_schema_version": 2,
        "metric_notes": {
            "model_output_tokens_per_s": "completion tokens divided by synchronized native generation time; excludes prompt formatting/tokenization and NeMo scoring",
            "benchmark_wall_output_tokens_per_s": "completion tokens divided by the complete NeMo-Skills benchmark command wall time",
            "tokens_per_forward_pass": "returned completion tokens divided by decode-only physical encoder invocations; prompt prefill is excluded to match SGLang",
            "end_to_end_tokens_per_forward_pass": "returned completion tokens divided by all physical encoder invocations including prompt prefill",
            "top_p_applied": "false for current native NLD generation methods; top_p is accepted by the OpenAI API but the model methods expose temperature only",
            "top_k_applied": "false for current native NLD generation methods; top_k is accepted by the OpenAI API but is not forwarded to the model methods",
            "physical_nfe": "one encoder invocation counts as one forward even when it contains verifier and prospective rows",
            "processed_rows": "sum of sequence rows processed across encoder invocations; reported beside physical NFE to expose fused batch work",
            "prefetch_saved_draft_forwards": "number of prefetched full draft blocks actually consumed instead of issuing a normal draft forward",
            "outcome_states": "eleven mutually-exclusive outcomes among actual overlap attempts, with P1/P2 correction and full-block continuation states",
            "continuation_attempt_rate": "rounds with an actually executed New branch divided by all decode rounds; only sequence-boundary guards may suppress New",
            "continuation_3plus_verified_hit_rate": "among rounds with at least three strict risk crossings and an executed New branch, the fraction whose full verify bonus equals New token 0",
            "next_coverage": "attempts with an observed immediate next verifier divided by state count; terminal rounds are not imputed as zero",
            "computed_token_avg": "mean dense query-token slots per physical forward; rows multiplied by common padded query length",
            "padding_ratio": "padding query-token slots divided by all dense query-token slots; attention masks do not remove these dense token computations",
            "request_count": "successful requests included in efficiency aggregates",
            "attempted_request_count": "all server requests, including requests skipped after CUDA OOM",
            "failed_request_count": "requests excluded from efficiency aggregates; inspect failure_types and oom_skipped_request_count",
            "successful_request_rate": "request_count divided by attempted_request_count; required when interpreting partial efficiency coverage",
            "oom_skipped_request_count": "CUDA OOM requests returned as empty placeholders in efficiency-only mode and excluded from all efficiency means",
        },
    }

    if args.dry_run:
        print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))
        return 0

    metrics = read_json(metrics_path) if metrics_path.is_file() else {}
    metrics["pytorch_margin_risk_two_plus_new_overlap"] = payload
    if native["average_forward_passes_per_sample"] is not None:
        metrics["average_nfe"] = native["average_forward_passes_per_sample"]
        metrics["nfe_count"] = native["request_count"]
    if native["tokens_per_forward_pass"] is not None:
        metrics["tokens_per_forward_pass"] = native["tokens_per_forward_pass"]
        metrics["tpf"] = native["tokens_per_forward_pass"]
    if native["model_output_tokens_per_s"] is not None:
        metrics["model_output_tokens_per_s"] = native["model_output_tokens_per_s"]
        metrics["tps"] = native["model_output_tokens_per_s"]

    summary_path = (
        Path(args.summary_json)
        if args.summary_json
        else metrics_path.parent / "pytorch_margin_risk_two_plus_new_overlap_metrics_summary.json"
    )
    write_json(metrics_path, metrics)
    write_json(summary_path, payload)
    print(f"Updated {metrics_path}")
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
