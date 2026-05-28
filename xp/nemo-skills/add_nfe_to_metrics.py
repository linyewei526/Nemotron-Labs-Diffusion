#!/usr/bin/env python3
"""
Add average NFE (number of function evaluations) to evaluation metrics.

The LLaDA/Nemotron server returns NFE in the API response under usage.nfe.
This script adds average_nfe to metrics.json so it is recorded per task/benchmark.

Usage:
  # Add a known average NFE to a metrics file
  python add_nfe_to_metrics.py --metrics-json path/to/metrics.json --average-nfe 1234.5

  # Compute average from eval output and update metrics (if output jsonl contains usage.nfe)
  python add_nfe_to_metrics.py --eval-results-dir path/to/eval-results/gsm8k

  # Compute average from a log file (one JSON object per line with "nfe" key)
  python add_nfe_to_metrics.py --metrics-json path/to/metrics.json --nfe-log path/to/nfe_per_sample.jsonl
"""

import argparse
import json
import os
from typing import Optional


def load_metrics(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def save_metrics(path: str, data: dict) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _extract_nfe_from_record(obj: dict) -> Optional[float]:
    """Extract NFE from one JSONL record."""
    u = obj.get("usage") or {}
    nfe = u.get("nfe") if isinstance(u, dict) else None
    if nfe is None:
        nfe = obj.get("nfe")
    if nfe is None:
        return None
    try:
        return float(nfe)
    except (TypeError, ValueError):
        return None


def _extract_generated_tokens_from_record(obj: dict) -> Optional[float]:
    """Extract generated token count from one JSONL record."""
    candidates = [
        obj.get("num_generated_tokens"),
        obj.get("completion_tokens"),
        obj.get("generated_tokens"),
    ]
    usage = obj.get("usage")
    if isinstance(usage, dict):
        candidates.append(usage.get("completion_tokens"))
    for value in candidates:
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def collect_nfe_and_tokens_from_jsonl(
    dir_path: str,
    *,
    exclude_unfinished_nfe: bool = False,
    nfe_cutoff: Optional[float] = None,
) -> tuple[list[float], list[float], int]:
    """Collect NFE and generated tokens from output*.jsonl in dir.

    Returns: (nfes, generated_tokens, skipped_unfinished_count).
    """
    nfes: list[float] = []
    generated_tokens: list[float] = []
    skipped_unfinished = 0
    for name in sorted(os.listdir(dir_path)):
        if not name.startswith("output") or not name.endswith(".jsonl"):
            continue
        path = os.path.join(dir_path, name)
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    nfe = _extract_nfe_from_record(obj)
                    tokens = _extract_generated_tokens_from_record(obj)
                    if (
                        exclude_unfinished_nfe
                        and nfe_cutoff is not None
                        and nfe is not None
                        and nfe >= nfe_cutoff
                    ):
                        skipped_unfinished += 1
                        continue
                    if nfe is not None:
                        nfes.append(nfe)
                    if tokens is not None:
                        generated_tokens.append(tokens)
                except (json.JSONDecodeError, TypeError):
                    continue
    return nfes, generated_tokens, skipped_unfinished


def collect_nfe_from_nfe_log(
    log_path: str,
    filter_benchmark: str = "",
    *,
    exclude_unfinished_nfe: bool = False,
    nfe_cutoff: Optional[float] = None,
) -> list[float]:
    """Collect NFE values from a log file (one JSON object per line with 'nfe' key).

    If *filter_benchmark* is non-empty, only entries whose ``benchmark`` field
    matches (case-insensitive) are included.
    """
    nfes = []
    with open(log_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                # Apply benchmark filter if requested
                if filter_benchmark:
                    entry_bench = obj.get("benchmark", "")
                    if entry_bench.lower() != filter_benchmark.lower():
                        continue
                nfe = obj.get("nfe")
                if nfe is not None:
                    nfe_val = float(nfe)
                    if exclude_unfinished_nfe and nfe_cutoff is not None and nfe_val >= nfe_cutoff:
                        continue
                    nfes.append(nfe_val)
            except (json.JSONDecodeError, TypeError):
                continue
    return nfes


def update_avg_tokens_in_metrics(metrics: dict, benchmark_name: str, avg_tokens: float) -> bool:
    """Update avg_tokens fields for one benchmark in metrics.json."""
    bench_data = metrics.get(benchmark_name)
    if not isinstance(bench_data, dict):
        return False

    updated = False
    for val in bench_data.values():
        if isinstance(val, dict) and "avg_tokens" in val:
            val["avg_tokens"] = round(avg_tokens, 1)
            updated = True

    if "avg_tokens" in bench_data:
        bench_data["avg_tokens"] = round(avg_tokens, 1)
        updated = True
    return updated


def main():
    parser = argparse.ArgumentParser(
        description="Add average NFE to evaluation metrics.json",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--average-nfe",
        type=float,
        metavar="VALUE",
        help="Average NFE value to write into metrics (use with --metrics-json)",
    )
    parser.add_argument(
        "--eval-results-dir",
        type=str,
        metavar="DIR",
        help="Eval results dir (e.g. .../eval-results/gsm8k); used to read output*.jsonl for NFE and avg_tokens.",
    )
    parser.add_argument(
        "--nfe-log",
        type=str,
        metavar="FILE",
        help="Log file with one JSON object per line containing 'nfe'; use with --metrics-json to compute average and update metrics",
    )
    parser.add_argument(
        "--metrics-json",
        type=str,
        metavar="PATH",
        help="Path to metrics.json to update (required when using --average-nfe or --nfe-log)",
    )
    parser.add_argument(
        "--filter-benchmark",
        type=str,
        default="",
        metavar="NAME",
        help="Only count NFE entries whose 'benchmark' field matches this name (used with --nfe-log)",
    )
    parser.add_argument(
        "--exclude-unfinished-nfe",
        action="store_true",
        help="Exclude entries where nfe >= --nfe-cutoff from both NFE and avg_tokens aggregation.",
    )
    parser.add_argument(
        "--nfe-cutoff",
        type=float,
        default=None,
        metavar="VALUE",
        help="NFE cutoff used with --exclude-unfinished-nfe (typically tokens_to_generate).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be written without modifying files",
    )
    args = parser.parse_args()

    if args.average_nfe is None and not args.nfe_log and not args.eval_results_dir:
        parser.error("One of --average-nfe, --nfe-log, or --eval-results-dir is required.")

    if args.exclude_unfinished_nfe and args.nfe_cutoff is None:
        parser.error("--nfe-cutoff is required when --exclude-unfinished-nfe is enabled")

    if args.average_nfe is not None:
        if not args.metrics_json:
            parser.error("--metrics-json is required when using --average-nfe")
        metrics_path = args.metrics_json
        average_nfe = args.average_nfe
        nfe_count = None
    elif args.nfe_log:
        if not args.metrics_json:
            parser.error("--metrics-json is required when using --nfe-log")
        nfes = collect_nfe_from_nfe_log(
            args.nfe_log,
            filter_benchmark=args.filter_benchmark,
            exclude_unfinished_nfe=args.exclude_unfinished_nfe,
            nfe_cutoff=args.nfe_cutoff,
        )
        if not nfes:
            filter_msg = f" (filter: benchmark={args.filter_benchmark})" if args.filter_benchmark else ""
            print(f"No NFE values found in log file{filter_msg}.")
            return 1
        average_nfe = sum(nfes) / len(nfes)
        nfe_count = len(nfes)
        metrics_path = args.metrics_json
    else:
        # --eval-results-dir
        metrics_path = os.path.join(args.eval_results_dir, "metrics.json")
        if not os.path.isfile(metrics_path):
            print(f"Metrics file not found: {metrics_path}")
            return 1
        nfes, _, skipped_unfinished = collect_nfe_and_tokens_from_jsonl(
            args.eval_results_dir,
            exclude_unfinished_nfe=args.exclude_unfinished_nfe,
            nfe_cutoff=args.nfe_cutoff,
        )
        if not nfes:
            print("No NFE values found in output jsonl (ensure API response usage is stored in output).")
            return 1
        average_nfe = sum(nfes) / len(nfes)
        nfe_count = len(nfes)
        if args.exclude_unfinished_nfe:
            print(f"Excluded {skipped_unfinished} unfinished samples from NFE/tokens aggregation.")

    metrics = load_metrics(metrics_path)
    metrics["average_nfe"] = round(average_nfe, 4)
    if nfe_count is not None:
        metrics["nfe_count"] = nfe_count

    if args.exclude_unfinished_nfe and args.eval_results_dir:
        benchmark_name = args.filter_benchmark.strip() or os.path.basename(os.path.abspath(args.eval_results_dir))
        _, token_values, skipped_unfinished = collect_nfe_and_tokens_from_jsonl(
            args.eval_results_dir,
            exclude_unfinished_nfe=True,
            nfe_cutoff=args.nfe_cutoff,
        )
        if token_values:
            avg_tokens = sum(token_values) / len(token_values)
            if update_avg_tokens_in_metrics(metrics, benchmark_name, avg_tokens):
                print(
                    f"Updated avg_tokens for {benchmark_name}: {round(avg_tokens, 1)} "
                    f"(excluded {skipped_unfinished} unfinished samples)"
                )
            else:
                print(f"Note: could not find avg_tokens fields for benchmark '{benchmark_name}' in metrics.json")
        else:
            print("Note: no token counts found after filtering; avg_tokens was not updated.")

    if args.dry_run:
        print(
            f"Would update {metrics_path} with: average_nfe={metrics['average_nfe']}"
            + (f", nfe_count={nfe_count}" if nfe_count else "")
        )
        return 0

    save_metrics(metrics_path, metrics)
    print(f"Updated {metrics_path}: average_nfe={metrics['average_nfe']}" + (f", nfe_count={nfe_count}" if nfe_count else ""))
    return 0


if __name__ == "__main__":
    exit(main())
