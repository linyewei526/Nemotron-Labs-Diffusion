#!/usr/bin/env python3
"""Merge SGLang decode/timing stats into a NeMo-Skills metrics.json file."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Optional


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, sort_keys=True)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path or not path.is_file():
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
        return float(value)
    except (TypeError, ValueError):
        return None


def percentile(values: list[float], q: float) -> Optional[float]:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def rounded(value: Optional[float], digits: int = 6) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), digits)


def summarize_values(values: Iterable[Optional[float]]) -> dict[str, Optional[float]]:
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return {
            "mean": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "p99": None,
        }
    return {
        "mean": rounded(sum(clean) / len(clean)),
        "p50": rounded(percentile(clean, 0.50)),
        "p90": rounded(percentile(clean, 0.90)),
        "p95": rounded(percentile(clean, 0.95)),
        "p99": rounded(percentile(clean, 0.99)),
    }


def extract_tokens_from_outputs(eval_results_dir: Path) -> list[float]:
    tokens: list[float] = []
    if not eval_results_dir.is_dir():
        return tokens
    for output_path in sorted(eval_results_dir.glob("output*.jsonl")):
        for row in read_jsonl(output_path):
            usage = row.get("usage")
            candidates = [
                row.get("num_generated_tokens"),
                row.get("completion_tokens"),
                row.get("generated_tokens"),
            ]
            if isinstance(usage, dict):
                candidates.append(usage.get("completion_tokens"))
            for candidate in candidates:
                value = as_float(candidate)
                if value is not None:
                    tokens.append(value)
                    break
    return tokens


def summarize_decode_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    records = 0
    tokens = 0.0
    forward_passes = 0.0
    block_gen_positions = 0.0
    acceptance_rates: list[float] = []

    for row in rows:
        row_tokens = as_float(row.get("tokens"))
        row_fp = as_float(row.get("forward_passes"))
        if row_tokens is None or row_fp is None:
            continue
        records += 1
        tokens += row_tokens
        forward_passes += row_fp
        block_positions = as_float(row.get("block_gen_positions"))
        if block_positions is not None:
            block_gen_positions += block_positions
        acc = as_float(row.get("acceptance_rate"))
        if acc is not None:
            acceptance_rates.append(acc)

    tpf = tokens / forward_passes if forward_passes > 0 else None
    weighted_acceptance = (
        tokens / block_gen_positions if block_gen_positions > 0 else None
    )
    return {
        "decode_blocks": records,
        "decode_tokens": round(tokens, 4),
        "decode_forward_passes": round(forward_passes, 4),
        "tokens_per_forward_pass": rounded(tpf, 4),
        "mean_tokens_per_block": rounded(tokens / records if records else None, 4),
        "mean_forward_passes_per_block": rounded(
            forward_passes / records if records else None, 4
        ),
        "block_gen_positions": round(block_gen_positions, 4),
        "weighted_acceptance_rate": rounded(weighted_acceptance, 6),
        "mean_acceptance_rate": rounded(
            sum(acceptance_rates) / len(acceptance_rates)
            if acceptance_rates
            else None,
            6,
        ),
    }


def summarize_timing(rows: list[dict[str, Any]], wall_time_s: Optional[float]) -> dict[str, Any]:
    ok_rows = [r for r in rows if r.get("ok", True)]
    completion_tokens = sum(as_float(r.get("completion_tokens")) or 0.0 for r in ok_rows)
    prompt_tokens = sum(as_float(r.get("prompt_tokens")) or 0.0 for r in ok_rows)
    latency_sum = sum(as_float(r.get("latency_s")) or 0.0 for r in ok_rows)
    start_times = [as_float(r.get("start_time_s")) for r in ok_rows]
    end_times = [as_float(r.get("end_time_s")) for r in ok_rows]
    start_times = [v for v in start_times if v is not None]
    end_times = [v for v in end_times if v is not None]
    request_window_time_s = None
    if start_times and end_times:
        request_window_time_s = max(end_times) - min(start_times)

    timing = {
        "request_count": len(ok_rows),
        "failed_request_count": len(rows) - len(ok_rows),
        "prompt_tokens": round(prompt_tokens, 4),
        "completion_tokens": round(completion_tokens, 4),
        "wall_time_s": rounded(wall_time_s),
        "benchmark_wall_time_s": rounded(wall_time_s),
        "request_window_time_s": rounded(request_window_time_s),
        "latency_s": summarize_values(as_float(r.get("latency_s")) for r in ok_rows),
        "ttft_s": summarize_values(as_float(r.get("ttft_s")) for r in ok_rows),
        "tpot_s": summarize_values(as_float(r.get("tpot_s")) for r in ok_rows),
        "per_request_output_tokens_per_s": summarize_values(
            as_float(r.get("output_tokens_per_s")) for r in ok_rows
        ),
    }
    timing["sum_request_latency_s"] = rounded(latency_sum)
    timing["wall_output_tokens_per_s"] = rounded(
        completion_tokens / wall_time_s if wall_time_s and wall_time_s > 0 else None,
        4,
    )
    timing["request_window_output_tokens_per_s"] = rounded(
        completion_tokens / request_window_time_s
        if request_window_time_s and request_window_time_s > 0
        else None,
        4,
    )
    timing["wall_requests_per_s"] = rounded(
        len(ok_rows) / wall_time_s if wall_time_s and wall_time_s > 0 else None,
        4,
    )
    timing["request_window_requests_per_s"] = rounded(
        len(ok_rows) / request_window_time_s
        if request_window_time_s and request_window_time_s > 0
        else None,
        4,
    )
    return timing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-json", required=True)
    parser.add_argument("--eval-results-dir", required=True)
    parser.add_argument("--decode-stats-file", default="")
    parser.add_argument("--timing-log", default="")
    parser.add_argument("--benchmark", default="")
    parser.add_argument("--wall-time-s", type=float, default=None)
    parser.add_argument(
        "--assume-ar-forward-pass",
        action="store_true",
        help="If no decode stats are present, use completion_tokens as forward passes (AR TPF=1).",
    )
    parser.add_argument("--summary-json", default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    metrics_path = Path(args.metrics_json)
    eval_results_dir = Path(args.eval_results_dir)
    if not metrics_path.is_file():
        print(f"Metrics file not found: {metrics_path}")
        return 1

    metrics = load_json(metrics_path)
    decode_rows = read_jsonl(Path(args.decode_stats_file)) if args.decode_stats_file else []
    timing_rows = read_jsonl(Path(args.timing_log)) if args.timing_log else []

    if args.benchmark:
        timing_rows = [
            row
            for row in timing_rows
            if not row.get("benchmark") or str(row.get("benchmark")) == args.benchmark
        ]

    decode = summarize_decode_stats(decode_rows)
    timing = summarize_timing(timing_rows, args.wall_time_s)

    if decode["decode_forward_passes"] == 0 and args.assume_ar_forward_pass:
        completion_tokens = timing["completion_tokens"]
        decode.update(
            {
                "decode_blocks": timing["request_count"],
                "decode_tokens": completion_tokens,
                "decode_forward_passes": completion_tokens,
                "tokens_per_forward_pass": 1.0 if completion_tokens else None,
                "mean_tokens_per_block": None,
                "mean_forward_passes_per_block": None,
                "block_gen_positions": 0.0,
                "weighted_acceptance_rate": None,
                "mean_acceptance_rate": None,
            }
        )

    generated_tokens = extract_tokens_from_outputs(eval_results_dir)
    if generated_tokens:
        avg_tokens = sum(generated_tokens) / len(generated_tokens)
    elif timing["request_count"]:
        avg_tokens = timing["completion_tokens"] / timing["request_count"]
    else:
        avg_tokens = None

    average_nfe = None
    if decode["decode_forward_passes"] and timing["request_count"]:
        average_nfe = decode["decode_forward_passes"] / timing["request_count"]

    sglang_metrics = {
        "benchmark": args.benchmark or eval_results_dir.name,
        "decode": decode,
        "serving": timing,
        "avg_completion_tokens": rounded(avg_tokens, 4),
        "average_forward_passes_per_sample": rounded(average_nfe, 4),
    }

    metrics["sglang"] = sglang_metrics
    if average_nfe is not None:
        metrics["average_nfe"] = rounded(average_nfe, 4)
        metrics["nfe_count"] = timing["request_count"]
    if avg_tokens is not None:
        metrics["avg_tokens_sglang"] = rounded(avg_tokens, 4)
    if decode["tokens_per_forward_pass"] is not None:
        metrics["tokens_per_forward_pass"] = decode["tokens_per_forward_pass"]
        metrics["tpf"] = decode["tokens_per_forward_pass"]

    summary_path = (
        Path(args.summary_json)
        if args.summary_json
        else eval_results_dir / "sglang_metrics_summary.json"
    )

    if args.dry_run:
        print(json.dumps(sglang_metrics, indent=2, ensure_ascii=False, sort_keys=True))
        return 0

    save_json(metrics_path, metrics)
    save_json(summary_path, sglang_metrics)
    print(f"Updated {metrics_path}")
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
