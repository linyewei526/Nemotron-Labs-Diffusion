#!/usr/bin/env python3
"""Merge native PyTorch request stats into a NeMo-Skills metrics.json."""

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


def forward_pass_breakdown(row: dict[str, Any]) -> tuple[float, float, float]:
    """Return decode, prefill and total NFE for new and legacy request rows."""
    explicit_decode = as_float(row.get("decode_nfe"))
    explicit_prefill = as_float(row.get("prefill_nfe"))
    explicit_total = as_float(row.get("total_nfe"))

    if explicit_decode is not None:
        decode_nfe = max(explicit_decode, 0.0)
        prefill_nfe = max(explicit_prefill or 0.0, 0.0)
        total_nfe = max(
            explicit_total
            if explicit_total is not None
            else decode_nfe + prefill_nfe,
            0.0,
        )
        return decode_nfe, prefill_nfe, total_nfe

    # Legacy native rows stored the HF remote-code NFE in ``nfe``.  Only
    # LinearSpec included its prompt prefill in that number.
    total_nfe = max(explicit_total or as_float(row.get("nfe")) or 0.0, 0.0)
    mode = str(row.get("mode") or "")
    prefill_nfe = (
        max(explicit_prefill, 0.0)
        if explicit_prefill is not None
        else (1.0 if total_nfe > 0 and mode in {"linearspec_base", "linearspec_lora"} else 0.0)
    )
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
        "request_count": len(ok_rows),
        "failed_request_count": len(failed_rows),
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
        "backend": "native_pytorch",
        "benchmark": args.benchmark,
        "decode": native,
        "metric_schema_version": 2,
        "metric_notes": {
            "model_output_tokens_per_s": "completion tokens divided by synchronized native generation time; excludes prompt formatting/tokenization and NeMo scoring",
            "benchmark_wall_output_tokens_per_s": "completion tokens divided by the complete NeMo-Skills benchmark command wall time",
            "tokens_per_forward_pass": "returned completion tokens divided by decode-only forward passes; prompt prefill is excluded to match SGLang",
            "end_to_end_tokens_per_forward_pass": "returned completion tokens divided by total model-reported forward passes including LinearSpec prompt prefill",
            "top_p_applied": "false for current native NLD generation methods; top_p is accepted by the OpenAI API but the model methods expose temperature only",
            "top_k_applied": "false for current native NLD generation methods; top_k is accepted by the OpenAI API but is not forwarded to the model methods",
        },
    }

    if args.dry_run:
        print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))
        return 0

    metrics = read_json(metrics_path)
    metrics["pytorch_native"] = payload
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
        else metrics_path.parent / "pytorch_native_metrics_summary.json"
    )
    write_json(metrics_path, metrics)
    write_json(summary_path, payload)
    print(f"Updated {metrics_path}")
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
