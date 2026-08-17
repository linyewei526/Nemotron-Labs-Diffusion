#!/usr/bin/env python3
"""Merge confidence MASK-redraft request stats into NeMo-Skills metrics."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable, Optional


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


def summarize(rows: list[dict[str, Any]], wall_time_s: Optional[float]) -> dict[str, Any]:
    ok_rows = [row for row in rows if row.get("ok") is True]
    failed_rows = [row for row in rows if row.get("ok") is not True]
    prompt_tokens = sum(as_float(row.get("prompt_tokens")) or 0.0 for row in ok_rows)
    completion_tokens = sum(as_float(row.get("completion_tokens")) or 0.0 for row in ok_rows)
    raw_generated_tokens = sum(as_float(row.get("raw_generated_tokens")) or 0.0 for row in ok_rows)
    total_nfe = sum(as_float(row.get("nfe")) or 0.0 for row in ok_rows)
    model_time_s = sum(as_float(row.get("model_time_s")) or 0.0 for row in ok_rows)
    sum_request_time_s = sum(as_float(row.get("request_time_s")) or 0.0 for row in ok_rows)
    tpf = completion_tokens / total_nfe if total_nfe > 0 else None

    summary = {
        "request_count": len(ok_rows),
        "failed_request_count": len(failed_rows),
        "prompt_tokens": round(prompt_tokens, 4),
        "completion_tokens": round(completion_tokens, 4),
        "raw_generated_tokens": round(raw_generated_tokens, 4),
        "forward_passes": round(total_nfe, 4),
        "tokens_per_forward_pass": rounded(tpf, 4),
        "average_forward_passes_per_sample": rounded(
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
    }
    for row in ok_rows:
        reason = str(row.get("finish_reason") or "unknown")
        summary["finish_reasons"][reason] = summary["finish_reasons"].get(reason, 0) + 1
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
    redraft_sum_fields = [
        "physical_nfe",
        "processed_rows",
        "processed_query_tokens",
        "rounds",
        "draft_length_sum",
        "partial_draft_rounds",
        "normal_draft_forwards",
        "normal_verify_forwards",
        "fused_verify_redraft_forwards",
        "rounds_without_candidate",
        "redraft_attempts",
        "redraft_verified_hits",
        "redraft_reuse_hits",
        "redraft_saved_draft_forwards",
        "redraft_direct_trigger_hits",
        "redraft_downstream_correction_hits",
        "redraft_full_block_bonus_hits",
        "redraft_discarded_before_trigger",
        "redraft_discarded_trigger_token_mismatch",
        "redraft_discarded_before_correction",
        "redraft_discarded_correction_mismatch",
        "redraft_discarded_eos",
        "redraft_discarded_thinking_budget",
        "redraft_discarded_generation_end",
        "redraft_skipped_no_future_round",
        "redraft_skipped_context_limit",
        "candidate_position_sum",
        "retained_draft_tokens_sum",
        "full_length_reuses",
        "partial_length_reuses",
        "prospective_query_tokens",
    ]
    redraft = {
        key: int(
            sum(
                as_float((row.get("mask_redraft") or {}).get(key)) or 0.0
                for row in ok_rows
            )
        )
        for key in redraft_sum_fields
    }
    retained_mins = [
        value
        for row in ok_rows
        if (
            value := as_int(
                (row.get("mask_redraft") or {}).get("retained_draft_tokens_min")
            )
        )
        is not None
    ]
    retained_maxes = [
        value
        for row in ok_rows
        if (
            value := as_int(
                (row.get("mask_redraft") or {}).get("retained_draft_tokens_max")
            )
        )
        is not None
    ]
    redraft["retained_draft_tokens_min"] = min(retained_mins) if retained_mins else None
    redraft["retained_draft_tokens_max"] = max(retained_maxes) if retained_maxes else None
    attempts = redraft["redraft_attempts"]
    reuse_hits = redraft["redraft_reuse_hits"]
    redraft["redraft_hit_rate"] = rounded(
        reuse_hits / attempts if attempts else None
    )
    redraft["redraft_verified_hit_rate"] = rounded(
        redraft["redraft_verified_hits"] / attempts if attempts else None
    )
    redraft["average_candidate_position"] = rounded(
        redraft["candidate_position_sum"] / attempts if attempts else None
    )
    redraft["average_draft_length"] = rounded(
        redraft["draft_length_sum"] / redraft["rounds"]
        if redraft["rounds"]
        else None
    )
    redraft["average_retained_draft_length"] = rounded(
        redraft["retained_draft_tokens_sum"] / reuse_hits if reuse_hits else None
    )
    redraft["saved_draft_fraction_of_rounds"] = rounded(
        redraft["redraft_saved_draft_forwards"] / redraft["rounds"]
        if redraft["rounds"]
        else None
    )
    summary["mask_redraft"] = redraft
    summary["drop_pct_thresholds"] = sorted(
        {
            value
            for row in ok_rows
            if (value := as_float(row.get("drop_pct_threshold"))) is not None
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
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    metrics_path = Path(args.metrics_json)
    if not metrics_path.is_file():
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
        "backend": "native_pytorch_confidence_mask_redraft",
        "benchmark": args.benchmark,
        "decode": native,
        "metric_notes": {
            "model_output_tokens_per_s": "completion tokens divided by synchronized native generation time; excludes prompt formatting/tokenization and NeMo scoring",
            "benchmark_wall_output_tokens_per_s": "completion tokens divided by the complete NeMo-Skills benchmark command wall time",
            "tokens_per_forward_pass": "returned completion tokens divided by physical encoder invocations counted by the isolated MASK-redraft decoder",
            "top_p_applied": "false for current native NLD generation methods; top_p is accepted by the OpenAI API but the model methods expose temperature only",
            "top_k_applied": "false for current native NLD generation methods; top_k is accepted by the OpenAI API but is not forwarded to the model methods",
            "physical_nfe": "one encoder invocation counts as one forward even when it contains verifier and prospective rows",
            "processed_rows": "sum of sequence rows processed across encoder invocations; reported beside physical NFE to expose fused batch work",
            "redraft_saved_draft_forwards": "number of full or partial autonomous redrafts actually consumed instead of issuing a normal draft forward",
            "average_retained_draft_length": "mean variable-length suffix retained after matching the trustworthy verifier segment",
        },
    }

    if args.dry_run:
        print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))
        return 0

    metrics = read_json(metrics_path)
    metrics["pytorch_confidence_mask_redraft"] = payload
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
        else metrics_path.parent / "pytorch_confidence_mask_redraft_metrics_summary.json"
    )
    write_json(metrics_path, metrics)
    write_json(summary_path, payload)
    print(f"Updated {metrics_path}")
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
