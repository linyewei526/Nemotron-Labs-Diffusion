#!/usr/bin/env python3
"""Create an isolated run, optionally collect fresh traces, then search offline."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from reporting import atomic_json, initialize_run, now, render_report


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
DEFAULT_RESULTS = Path("/data/home/wly/dLLM/NLD_results/observations/adaptive_margin_history_search_results")
DEFAULT_SOURCE = Path(
    "/data/home/wly/dLLM/NLD_results/observations/adaptive_failure_locator_search_results/"
    "adaptive_failure_locator_20260828_203527"
)
DEFAULT_MODEL = "/data1/linyewei/models/Nemotron-Labs-Diffusion-8B"
DEFAULT_DATA = "/data1/linyewei/datasets/NLD"
DEFAULT_PYTHON = "/data/home/wly/.conda/envs/nld_sglang/bin/python"
DEFAULT_BENCHMARKS = (
    "gsm8k:1,human-eval:1,mbpp:1,math-500:1,aime25:1,gpqa:1,mmlu:1,ifeval:1,livecodebench-cpp:1"
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Training-free global history-conditioned margin-risk locator search.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    result.add_argument("--trace-mode", choices=("offline", "rerun"), default="offline")
    result.add_argument("--source-run-dir", type=Path, default=DEFAULT_SOURCE)
    result.add_argument("--output-path", "--out-dir", dest="output_path", type=Path, default=DEFAULT_RESULTS)
    result.add_argument("--benchmarks", default=DEFAULT_BENCHMARKS)
    result.add_argument("--block-size", "--block-length", dest="block_size", type=int)
    result.add_argument("--history-windows", default="1,2,4")
    result.add_argument("--aggregations", default="mean,median,ewma")
    result.add_argument("--grid", choices=("compact", "standard", "extended"), default="standard")
    result.add_argument(
        "--selection-protocol",
        choices=("split", "full_data"),
        default="split",
        help="split keeps a held-out test; full_data evaluates every candidate on every valid round",
    )
    result.add_argument("--split-seed", type=int, default=20260830)
    result.add_argument("--search-ratio", type=float, default=0.6)
    result.add_argument("--selection-ratio", type=float, default=0.2)
    result.add_argument("--shortlist", type=int, default=120)
    result.add_argument("--report-top", type=int, default=30)
    result.add_argument("--search-max-rounds-per-dataset", type=int, default=30000)
    result.add_argument("--bootstrap-replicates", type=int, default=500)
    result.add_argument("--include-boundary-rounds", action="store_true")
    result.add_argument(
        "--no-progress",
        action="store_true",
        help="disable live offline scan/search/bootstrap progress output",
    )

    # Fresh trace collection mirrors the established PyTorch/NeMo-Skills interface.
    result.add_argument("--mode", choices=("linearspec_lora", "linearspec_base"), default="linearspec_lora")
    result.add_argument("--model", default=DEFAULT_MODEL)
    result.add_argument("--served-model-name", "--model-name", dest="served_model_name", default="nemotron-labs-diffusion-8b")
    result.add_argument("--lora-path", default="")
    result.add_argument("--dtype", default="bfloat16")
    result.add_argument("--threshold", type=float, default=0.0, help="LinearSpec draft confidence threshold, not locator threshold")
    result.add_argument("--temperature", type=float, default=0.0)
    result.add_argument("--top-p", type=float, default=0.95)
    result.add_argument("--tokens", type=int, default=8192)
    result.add_argument("--context-length", type=int)
    result.add_argument("--gpu-device", "--gpu-devices", dest="gpu_device", default="auto")
    result.add_argument("--gpu-candidates", default="all")
    result.add_argument("--gpu-min-free-gb", type=float, default=24.0)
    result.add_argument("--gpu-wait-timeout-s", type=int, default=0)
    result.add_argument("--gpu-poll-interval-s", type=int, default=30)
    result.add_argument("--gpu-memory-reserve-gb", type=float, default=0.0)
    result.add_argument("--port", type=int)
    result.add_argument("--batch-size", type=int, default=1)
    result.add_argument("--client-concurrency", type=int, default=1)
    result.add_argument("--num-chunks", type=int)
    result.add_argument("--max-samples", type=int)
    result.add_argument("--quick-test", action="store_true")
    thinking = result.add_mutually_exclusive_group()
    thinking.add_argument("--enable-thinking", action="store_true")
    thinking.add_argument("--disable-thinking", action="store_true")
    keep = result.add_mutually_exclusive_group()
    keep.add_argument("--keep-thinking", action="store_true")
    keep.add_argument("--strip-thinking", action="store_true")
    result.add_argument("--max-thinking-tokens", type=int)
    result.add_argument("--math-prompt-config", default="")
    result.add_argument("--trace-detail", choices=("position", "tokens"), default="position")
    result.add_argument("--pytorch-python", default=DEFAULT_PYTHON)
    result.add_argument("--eval-python", default="")
    result.add_argument("--nemo-skills-data-dir", default=DEFAULT_DATA)
    result.add_argument("--google-research-dir", default="")
    result.add_argument("--prepare-missing-data", action="store_true")
    result.add_argument("--dry-run", action="store_true")
    return result


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def validate(args: argparse.Namespace) -> None:
    benchmark_names = [item.split(":", 1)[0] for item in parse_csv(args.benchmarks)]
    if not benchmark_names or len(benchmark_names) != len(set(benchmark_names)):
        raise ValueError("--benchmarks must contain unique nonempty dataset names")
    windows = [int(item) for item in parse_csv(args.history_windows)]
    if not windows or len(windows) != len(set(windows)) or min(windows) < 1:
        raise ValueError("--history-windows must be unique positive integers")
    aggregations = parse_csv(args.aggregations)
    if not aggregations or len(aggregations) != len(set(aggregations)) or not set(aggregations) <= {"mean", "median", "ewma"}:
        raise ValueError("--aggregations must be a unique subset of mean,median,ewma")
    if args.block_size is not None and args.block_size < 2:
        raise ValueError("--block-size must be >=2")
    if not 0 < args.search_ratio < 1 or not 0 <= args.selection_ratio < 1 or args.search_ratio + args.selection_ratio >= 1:
        raise ValueError("invalid search/selection ratios")
    for name in ("shortlist", "report_top", "tokens", "client_concurrency", "batch_size"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    for name in ("search_max_rounds_per_dataset", "bootstrap_replicates", "gpu_wait_timeout_s"):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be nonnegative")
    if args.selection_protocol == "full_data" and args.search_max_rounds_per_dataset != 0:
        raise ValueError(
            "--selection-protocol full_data requires --search-max-rounds-per-dataset 0; "
            "full-data global search may not be capped"
        )
    if args.selection_protocol == "full_data" and args.trace_mode == "rerun":
        if args.max_samples is not None or args.quick_test:
            raise ValueError(
                "full_data rerun requires the complete datasets; --max-samples and --quick-test are forbidden"
            )
    if args.trace_mode == "rerun" and args.block_size is None:
        args.block_size = 16
    if args.trace_mode == "rerun" and args.batch_size != 1:
        raise ValueError("fresh native trace collection currently requires --batch-size 1")
    if args.context_length is None:
        args.context_length = args.tokens + 2048
    if args.num_chunks is None:
        args.num_chunks = args.client_concurrency
    if not args.eval_python:
        args.eval_python = args.pytorch_python
    if not args.lora_path:
        args.lora_path = str(Path(args.model) / "linear_spec_lora")
    if not args.google_research_dir:
        args.google_research_dir = str(Path(args.nemo_skills_data_dir) / "google-research")


def validate_offline_full_source(args: argparse.Namespace) -> None:
    if args.selection_protocol != "full_data" or args.trace_mode != "offline":
        return
    source_run = args.source_run_dir.resolve()
    source_settings_path = source_run / "Settings.json"
    if not source_settings_path.is_file():
        raise FileNotFoundError(
            f"full_data offline mode requires source provenance Settings.json: {source_settings_path}"
        )
    source_settings = json.loads(source_settings_path.read_text(encoding="utf-8"))
    if source_settings.get("max_samples") is not None or bool(source_settings.get("quick_test")):
        raise ValueError(
            "full_data source was collected with max_samples/quick_test and cannot represent all samples"
        )
    requested = {item.split(":", 1)[0] for item in parse_csv(args.benchmarks)}
    source_specs = source_settings.get("benchmark_specs")
    if source_specs is None:
        source_specs = parse_csv(str(source_settings.get("benchmarks", "")))
    available = {str(item).split(":", 1)[0] for item in source_specs}
    missing = sorted(requested - available)
    if missing:
        raise ValueError(
            "full_data source Settings does not declare all requested datasets: " + ",".join(missing)
        )
    trace_dir = source_run / "traces"
    trace_names = {
        path.stem.removeprefix("failure_locator_")
        for path in trace_dir.glob("failure_locator_*.jsonl")
        if path.stat().st_size > 0
    }
    missing_traces = sorted(requested - trace_names)
    if missing_traces:
        raise ValueError(
            "full_data source has missing/empty requested trace files: " + ",".join(missing_traces)
        )


def command_text() -> str:
    return shlex.join(["bash", "observations/adaptive_margin_history_search/eval_adaptive_margin_history.sh", *sys.argv[1:]])


def settings(args: argparse.Namespace, run_dir: Path) -> dict[str, Any]:
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_DIR, text=True, capture_output=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        head = None
    return {
        "schema_version": 1,
        "created_at": now(),
        "purpose": "搜索只依赖同 request 先前轮状态的免训练动态 margin_risk 阈值，在提高当前轮首错精确命中的同时降低 verifier 正确位置误报。",
        "entrypoint": "observations/adaptive_margin_history_search/eval_adaptive_margin_history.sh",
        "command": command_text(),
        "project_git_head_at_start": head,
        "output_dir": str(run_dir),
        "trace_mode": args.trace_mode,
        "source_run_dir": str(args.source_run_dir.resolve()) if args.trace_mode == "offline" else None,
        "benchmarks": args.benchmarks,
        "block_size": args.block_size,
        "history_windows": args.history_windows,
        "aggregations": args.aggregations,
        "grid": args.grid,
        "selection_protocol": args.selection_protocol,
        "full_data_all_candidates_all_valid_rounds": args.selection_protocol == "full_data",
        "within_dataset_weighting": "pooled valid decoding rounds from every request/sample",
        "across_dataset_weighting": "equal arithmetic mean over non-AIME24 datasets",
        "split_seed": args.split_seed,
        "search_ratio": args.search_ratio,
        "selection_ratio": args.selection_ratio,
        "test_ratio": 1.0 - args.search_ratio - args.selection_ratio,
        "shortlist": args.shortlist,
        "report_top": args.report_top,
        "search_max_rounds_per_dataset": args.search_max_rounds_per_dataset,
        "bootstrap_replicates": args.bootstrap_replicates,
        "include_boundary_rounds": args.include_boundary_rounds,
        "progress_enabled": not args.no_progress,
        "strict_comparator": ">",
        "cold_start_fallback": "margin_risk > 0.5",
        "aime24": "excluded from search, selection, test macro, and global strategy",
        "global_strategy": "one formula and one parameter set shared by all included datasets",
        "mode": args.mode,
        "model": args.model,
        "served_model_name": args.served_model_name,
        "lora_path": args.lora_path,
        "dtype": args.dtype,
        "draft_threshold": args.threshold,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "tokens": args.tokens,
        "context_length": args.context_length,
        "gpu_device": args.gpu_device,
        "gpu_candidates": args.gpu_candidates,
        "gpu_min_free_gb": args.gpu_min_free_gb,
        "gpu_wait_timeout_s": args.gpu_wait_timeout_s,
        "gpu_poll_interval_s": args.gpu_poll_interval_s,
        "gpu_memory_reserve_gb": args.gpu_memory_reserve_gb,
        "port": args.port,
        "batch_size": args.batch_size,
        "client_concurrency": args.client_concurrency,
        "num_chunks": args.num_chunks,
        "max_samples": args.max_samples,
        "quick_test": args.quick_test,
        "enable_thinking": args.enable_thinking,
        "disable_thinking": args.disable_thinking,
        "keep_thinking": args.keep_thinking,
        "strip_thinking": args.strip_thinking,
        "max_thinking_tokens": args.max_thinking_tokens,
        "math_prompt_config": args.math_prompt_config,
        "trace_detail": args.trace_detail,
        "pytorch_python": args.pytorch_python,
        "eval_python": args.eval_python,
        "nemo_skills_data_dir": args.nemo_skills_data_dir,
        "google_research_dir": args.google_research_dir,
        "prepare_missing_data": args.prepare_missing_data,
    }


def allocate_run_dir(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    stem = "adaptive_margin_history_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    for suffix in range(1000):
        name = stem if suffix == 0 else f"{stem}_{suffix:02d}"
        target = root / name
        if not target.exists():
            # initialize_run performs the final atomic exclusive mkdir.
            return target
    raise RuntimeError("could not allocate unique result directory")


def append_value(command: list[str], option: str, value: Any) -> None:
    if value is not None and value != "":
        command.extend([option, str(value)])


def collect_fresh_trace(args: argparse.Namespace, run_dir: Path) -> Path:
    source_root = run_dir / "trace_source"
    before = set(source_root.glob("adaptive_failure_locator_*"))
    command = [
        "bash",
        str(PROJECT_DIR / "observations/adaptive_failure_locator_search/eval_adaptive_failure_locator.sh"),
        "--output-path", str(source_root),
        "--mode", args.mode,
        "--benchmarks", args.benchmarks,
        "--model", args.model,
        "--served-model-name", args.served_model_name,
        "--dtype", args.dtype,
        "--block-size", str(args.block_size),
        "--history-windows", "1",
        "--aggregations", "mean",
        "--grid", "compact",
        "--threshold", str(args.threshold),
        "--temperature", str(args.temperature),
        "--top-p", str(args.top_p),
        "--tokens", str(args.tokens),
        "--context-length", str(args.context_length),
        "--gpu-device", str(args.gpu_device),
        "--gpu-candidates", args.gpu_candidates,
        "--gpu-min-free-gb", str(args.gpu_min_free_gb),
        "--gpu-wait-timeout-s", str(args.gpu_wait_timeout_s),
        "--gpu-poll-interval-s", str(args.gpu_poll_interval_s),
        "--gpu-memory-reserve-gb", str(args.gpu_memory_reserve_gb),
        "--batch-size", str(args.batch_size),
        "--client-concurrency", str(args.client_concurrency),
        "--num-chunks", str(args.num_chunks),
        "--shortlist", "1",
        "--report-top", "1",
        "--search-max-rounds-per-dataset", "1",
        "--bootstrap-replicates", "0",
        "--pytorch-python", args.pytorch_python,
        "--eval-python", args.eval_python,
        "--nemo-skills-data-dir", args.nemo_skills_data_dir,
        "--google-research-dir", args.google_research_dir,
        "--trace-detail", args.trace_detail,
    ]
    if args.mode == "linearspec_lora":
        append_value(command, "--lora-path", args.lora_path)
    append_value(command, "--port", args.port)
    append_value(command, "--max-samples", args.max_samples)
    append_value(command, "--max-thinking-tokens", args.max_thinking_tokens)
    append_value(command, "--math-prompt-config", args.math_prompt_config)
    for enabled, option in (
        (args.quick_test, "--quick-test"),
        (args.enable_thinking, "--enable-thinking"),
        (args.disable_thinking, "--disable-thinking"),
        (args.keep_thinking, "--keep-thinking"),
        (args.strip_thinking, "--strip-thinking"),
        (args.prepare_missing_data, "--prepare-missing-data"),
    ):
        if enabled:
            command.append(option)
    subprocess.run(command, cwd=PROJECT_DIR, check=True)
    created = sorted(set(source_root.glob("adaptive_failure_locator_*")) - before)
    if len(created) != 1 or not (created[0] / "traces").is_dir():
        raise RuntimeError(f"could not uniquely resolve fresh trace child under {source_root}")
    return created[0]


def search_command(args: argparse.Namespace, run_dir: Path, trace_dir: Path) -> list[str]:
    command = [
        args.eval_python,
        str(SCRIPT_DIR / "search.py"),
        "--run-dir", str(run_dir),
        "--trace-dir", str(trace_dir),
        "--benchmarks", args.benchmarks,
        "--history-windows", args.history_windows,
        "--aggregations", args.aggregations,
        "--grid", args.grid,
        "--selection-protocol", args.selection_protocol,
        "--split-seed", str(args.split_seed),
        "--search-ratio", str(args.search_ratio),
        "--selection-ratio", str(args.selection_ratio),
        "--shortlist", str(args.shortlist),
        "--report-top", str(args.report_top),
        "--search-max-rounds-per-dataset", str(args.search_max_rounds_per_dataset),
        "--bootstrap-replicates", str(args.bootstrap_replicates),
    ]
    append_value(command, "--block-size", args.block_size)
    if args.include_boundary_rounds:
        command.append("--include-boundary-rounds")
    if args.no_progress:
        command.append("--no-progress")
    return command


def dry_run(args: argparse.Namespace) -> int:
    if args.trace_mode == "offline":
        trace_dir = args.source_run_dir.resolve() / "traces"
        if not trace_dir.is_dir() or not list(trace_dir.glob("failure_locator_*.jsonl")):
            raise FileNotFoundError(f"offline trace directory is missing/empty: {trace_dir}")
    print("trace_mode:", args.trace_mode)
    print("benchmarks:", args.benchmarks)
    print("block_size:", args.block_size or "infer from trace")
    print("history/grid:", args.history_windows, args.aggregations, args.grid)
    print("selection_protocol:", args.selection_protocol)
    print("progress:", "disabled" if args.no_progress else "enabled")
    if args.selection_protocol == "full_data":
        print("scope: every candidate x every non-AIME24 valid round; datasets macro-averaged equally")
    else:
        print("split:", args.search_ratio, args.selection_ratio, 1 - args.search_ratio - args.selection_ratio)
    print("output root:", args.output_path.resolve())
    if args.trace_mode == "rerun":
        print("GPU/reserve:", args.gpu_device, args.gpu_memory_reserve_gb)
        print("port:", args.port or "auto")
    print("[dry-run] validation only; no result directory, model, port, or GPU process was created.")
    return 0


def main() -> int:
    args = parser().parse_args()
    validate(args)
    validate_offline_full_source(args)
    if args.dry_run:
        return dry_run(args)
    run_dir = allocate_run_dir(args.output_path.resolve())
    initialize_run(run_dir, settings(args, run_dir))
    print(f"Result directory initialized: {run_dir}", flush=True)
    try:
        if args.trace_mode == "offline":
            source_run = args.source_run_dir.resolve()
            trace_dir = source_run / "traces"
            if not trace_dir.is_dir():
                raise FileNotFoundError(f"trace directory missing: {trace_dir}")
        else:
            render_report(run_dir, None, phase="正在独立进行真实推理并逐数据集采集 trace。")
            source_run = collect_fresh_trace(args, run_dir)
            trace_dir = source_run / "traces"
            settings_payload = json.loads((run_dir / "Settings.json").read_text(encoding="utf-8"))
            settings_payload["source_run_dir"] = str(source_run)
            settings_payload["fresh_trace_child"] = str(source_run)
            atomic_json(run_dir / "Settings.json", settings_payload)
        render_report(run_dir, None, phase=f"trace 来源已锁定：{trace_dir}；开始离线搜索。")
        subprocess.run(search_command(args, run_dir, trace_dir), cwd=PROJECT_DIR, check=True)
    except BaseException as exc:
        atomic_json(run_dir / "runtime" / "failure.json", {"failed_at": now(), "error": repr(exc)})
        render_report(run_dir, None, phase=f"失败：{type(exc).__name__}: {exc}")
        raise
    print(f"Completed: {run_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
