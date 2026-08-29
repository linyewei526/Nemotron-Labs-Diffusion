#!/usr/bin/env python3
"""Initialize run metadata, append status, and render the live Chinese report."""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional


ACCURACY_FIELDS = {
    "human-eval": "passing_base_tests",
    "mbpp": "passing_base_tests",
    "livecodebench-cpp": "accuracy",
    "ifeval": "average_score",
}
EXCLUDED_MACRO = {"aime24"}


def now() -> str:
    return datetime.now().astimezone().isoformat()


def safe_name(value: str) -> str:
    name = value.split(":", 1)[0].strip()
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in name)


def benchmark_name(spec: str) -> str:
    return spec.split(":", 1)[0].strip()


def benchmark_specs(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def parse_number(value: str, *, integer: bool = False) -> Optional[float | int]:
    if value == "":
        return None
    return int(value) if integer else float(value)


def parse_bool(value: str) -> bool:
    return value.lower() in {"1", "true", "yes", "on"}


def initialize(args: argparse.Namespace) -> None:
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    for child in ("traces", "summaries", "metrics", "eval_runs", "runtime"):
        (run_dir / child).mkdir(parents=True, exist_ok=True)
    settings = {
        "schema_version": 1,
        "created_at": now(),
        "purpose": (
            "在同一 L16 canonical 解码轮上对 L4/L8/L16/L32 执行成对的真实 "
            "draft+verify，分离容量尾部收益与 lookahead 引起的区间内衰减，并为历史自适应 "
            "block size 建模保留参照特征。"
        ),
        "entrypoint": args.entrypoint,
        "command": args.command,
        "backend": "native_pytorch_linearspec_block_size_shadow",
        "mode": args.mode,
        "benchmarks": args.benchmarks,
        "benchmark_specs": benchmark_specs(args.benchmarks),
        "model": args.model,
        "served_model_name": args.served_model_name,
        "lora_path": args.lora_path,
        "dtype": args.dtype,
        "block_sizes": [int(value) for value in args.block_sizes.split(",")],
        "anchor_block_size": args.anchor_block_size,
        "history_windows": [int(value) for value in args.history_windows.split(",")],
        "threshold": args.threshold,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "tokens": args.tokens,
        "context_length": args.context_length,
        "gpu_device": args.gpu_device,
        "gpu_candidates": args.gpu_candidates,
        "gpu_min_free_gb": args.gpu_min_free_gb,
        "gpu_memory_reserve_gb": args.gpu_memory_reserve_gb,
        "port": args.port,
        "batch_size": 1,
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
        "include_boundary_rounds": args.include_boundary_rounds,
        "pytorch_python": args.pytorch_python,
        "eval_python": args.eval_python,
        "nemo_skills_data_dir": args.nemo_skills_data_dir,
        "google_research_dir": args.google_research_dir,
        "output_dir": str(run_dir),
        "analysis_contract": {
            "committed_branch": args.anchor_block_size,
            "shadow_branches_discarded": [
                value
                for value in [int(item) for item in args.block_sizes.split(",")]
                if value != args.anchor_block_size
            ],
            "primary_round_filter": (
                "all_rounds"
                if args.include_boundary_rounds
                else "exclude any-branch-EOS and remaining-budget<max-block"
            ),
            "pair_decomposition": "delta_a=tail+decay",
            "survival_k": "1..L_small+1 inclusive",
            "macro_average": "equal dataset weight; AIME24 excluded",
        },
    }
    atomic_json(run_dir / "Settings.json", settings)
    settings_md = f"""# 本轮实验设置

- 创建时间：`{settings['created_at']}`
- 目的：{settings['purpose']}
- 数据集：`{settings['benchmarks']}`
- 模式/权重：`{settings['mode']}` / `{settings['model']}`
- LoRA：`{settings['lora_path'] or '无（base 模式）'}`
- block：`{settings['block_sizes']}`；唯一提交锚点 `L={settings['anchor_block_size']}`
- history windows：`{settings['history_windows']}`
- threshold/temperature/top-p：`{settings['threshold']}` / `{settings['temperature']}` / `{settings['top_p']}`
- tokens/context：`{settings['tokens']}` / `{settings['context_length']}`
- GPU/预留/端口：`{settings['gpu_device']}` / `{settings['gpu_memory_reserve_gb']} GiB` / `{settings['port']}`
- client concurrency/chunks：`{settings['client_concurrency']}` / `{settings['num_chunks']}`
- trace detail：`{settings['trace_detail']}`
- 主分析边界策略：`{settings['analysis_contract']['primary_round_filter']}`
- 宏平均：AIME24 不参与；其余数据集先各自统计，再做等权平均。
- 命令：`{settings['command']}`

完整机器可读配置见同目录 `Settings.json`。
"""
    atomic_text(run_dir / "Settings.md", settings_md)
    status_path = run_dir / "benchmark_status.jsonl"
    with status_path.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                {
                    "created_at": now(),
                    "event": "run_initialized",
                    "status": "initialized",
                    "benchmarks": settings["benchmark_specs"],
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
        )
    render_report(run_dir)


def append_status(args: argparse.Namespace) -> None:
    payload = {
        "created_at": now(),
        "event": "benchmark_status",
        "benchmark_spec": args.benchmark_spec,
        "benchmark": benchmark_name(args.benchmark_spec),
        "status": args.status,
        "stage": args.stage,
        "exit_code": args.exit_code,
        "message": args.message,
        "trace_file": args.trace_file,
        "summary_file": args.summary_file,
        "metrics_file": args.metrics_file,
    }
    with (args.run_dir / "benchmark_status.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    render_report(args.run_dir)


def load_json(path: Path) -> Optional[dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def load_status(path: Path) -> list[dict[str, Any]]:
    rows = []
    if path.is_file():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def latest_benchmark_status(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    latest = {}
    for row in rows:
        if row.get("event") == "benchmark_status" and row.get("benchmark"):
            latest[str(row["benchmark"])] = row
    return latest


def accuracy(metrics: Optional[dict[str, Any]], name: str) -> Optional[float]:
    if not metrics:
        return None
    pass_one = metrics.get(name, {}).get("pass@1", {})
    field = ACCURACY_FIELDS.get(name, "symbolic_correct")
    value = pass_one.get(field)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "NA"
    if isinstance(value, bool):
        return "是" if value else "否"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "NA"
    if number.is_integer() and abs(number) >= 100:
        return str(int(number))
    return f"{number:.{digits}f}".rstrip("0").rstrip(".")


def table(headers: list[str], rows: Iterable[Iterable[Any]]) -> str:
    output = ["|" + "|".join(headers) + "|", "|" + "|".join(":---:" for _ in headers) + "|"]
    output.extend("|" + "|".join(str(value) for value in row) + "|" for row in rows)
    return "\n".join(output)


def mean(values: Iterable[Any]) -> Optional[float]:
    clean = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            clean.append(number)
    return sum(clean) / len(clean) if clean else None


def _get(payload: Optional[dict[str, Any]], *keys: str) -> Any:
    value: Any = payload
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def macro_datasets(names: list[str], summaries: dict[str, Any]) -> list[str]:
    return [
        name
        for name in names
        if name.lower() not in EXCLUDED_MACRO
        and summaries.get(name)
        and summaries[name].get("status") == "ok"
    ]


def render_report(run_dir: Path) -> None:
    settings = load_json(run_dir / "Settings.json")
    if not settings:
        return
    specs = settings.get("benchmark_specs") or benchmark_specs(settings["benchmarks"])
    names = [benchmark_name(spec) for spec in specs]
    status_rows = load_status(run_dir / "benchmark_status.jsonl")
    latest = latest_benchmark_status(status_rows)
    summaries = {
        name: load_json(run_dir / "summaries" / f"block_size_shadow_{safe_name(name)}.json")
        for name in names
    }
    metrics = {
        name: load_json(run_dir / "metrics" / f"metrics_{safe_name(name)}.json")
        for name in names
    }
    macro_names = macro_datasets(names, summaries)
    block_sizes = [int(value) for value in settings.get("block_sizes", [])]
    pairs = [
        f"{small}_{large}"
        for index, small in enumerate(block_sizes)
        for large in block_sizes[index + 1 :]
    ]

    lines = [
        "# LinearSpec 多 block size 同状态影子实验统计",
        "",
        f"> 创建：{settings.get('created_at','')}；最近更新：{now()}",
        "> 本文从实验启动时即存在；每个数据集完成或失败后原子刷新。`均(已成)` 是当前已完成且非 AIME24 数据集的等权宏平均，最终全量完成后即为正式宏平均。",
        "",
        "## 1. 进度与正确率性质",
        "",
    ]
    progress_rows = []
    for name in names:
        status = latest.get(name, {}).get("status", "待运行")
        summary = summaries.get(name)
        progress_rows.append(
            [
                name,
                status,
                fmt(accuracy(metrics.get(name), name)),
                _get(summary, "rounds", "raw") or 0,
                _get(summary, "rounds", "analysis") or 0,
                "排除" if name.lower() in EXCLUDED_MACRO else "纳入",
            ]
        )
    acc_macro_names = [name for name in macro_names if accuracy(metrics.get(name), name) is not None]
    progress_rows.append(
        [
            "均(已成)",
            f"D={len(macro_names)}",
            fmt(mean(accuracy(metrics[name], name) for name in acc_macro_names)),
            sum(int(_get(summaries[name], "rounds", "raw") or 0) for name in macro_names),
            sum(int(_get(summaries[name], "rounds", "analysis") or 0) for name in macro_names),
            "等权",
        ]
    )
    lines += [table(["集", "状态", "Acc%", "R原", "R有效", "宏均"], progress_rows), ""]

    lines += ["## 2. 各 block 的接收长度与置信度", ""]
    block_rows = []
    for name in names:
        summary = summaries.get(name)
        for size in block_sizes:
            block = _get(summary, "blocks", str(size))
            if not block:
                continue
            block_rows.append(
                [name, size, int(block.get("rounds", 0)), fmt(_get(block, "accept_length", "mean")), fmt(_get(block, "accept_length", "quantiles", "0.5")), fmt(_get(block, "accept_length", "quantiles", "0.9")), fmt(block.get("accept_rate_mean")), fmt(block.get("full_accept_rate")), fmt(block.get("zero_draft_match_rate")), fmt(_get(block, "accepted_confidence_mean", "mean")), fmt(_get(block, "rejected_confidence", "mean"))]
            )
    for size in block_sizes:
        blocks = [_get(summaries[name], "blocks", str(size)) for name in macro_names]
        blocks = [value for value in blocks if value]
        if blocks:
            block_rows.append(["均(已成)", size, f"D={len(blocks)}", fmt(mean(_get(value, "accept_length", "mean") for value in blocks)), fmt(mean(_get(value, "accept_length", "quantiles", "0.5") for value in blocks)), fmt(mean(_get(value, "accept_length", "quantiles", "0.9") for value in blocks)), fmt(mean(value.get("accept_rate_mean") for value in blocks)), fmt(mean(value.get("full_accept_rate") for value in blocks)), fmt(mean(value.get("zero_draft_match_rate") for value in blocks)), fmt(mean(_get(value, "accepted_confidence_mean", "mean") for value in blocks)), fmt(mean(_get(value, "rejected_confidence", "mean") for value in blocks))])
    lines += [table(["集", "L", "R/D", "A均", "A50", "A90", "A/L", "P满", "P零", "C收", "C拒"], block_rows or [["待运行"] * 11]), ""]

    lines += ["## 3. 接收长度精确分布", ""]
    hist_rows = []
    for name in names:
        summary = summaries.get(name)
        for size in block_sizes:
            counts = _get(summary, "blocks", str(size), "accept_length", "exact_counts") or {}
            rounds = int(_get(summary, "blocks", str(size), "rounds") or 0)
            for accept_value, count in sorted(counts.items(), key=lambda item: int(item[0])):
                hist_rows.append([name, size, accept_value, count, fmt(int(count) / rounds if rounds else None)])
    for size in block_sizes:
        for accept_value in range(1, size + 1):
            probabilities = []
            for name in macro_names:
                rounds = int(_get(summaries[name], "blocks", str(size), "rounds") or 0)
                count = int((_get(summaries[name], "blocks", str(size), "accept_length", "exact_counts") or {}).get(str(accept_value), 0))
                if rounds:
                    probabilities.append(count / rounds)
            if probabilities:
                hist_rows.append(["均(已成)", size, accept_value, f"D={len(probabilities)}", fmt(mean(probabilities))])
    lines += [table(["集", "L", "A", "N/D", "P(A)"], hist_rows or [["待运行"] * 5]), ""]

    lines += ["## 4. 两两差异：容量尾部与区间内衰减", ""]
    pair_rows = []
    for name in names:
        summary = summaries.get(name)
        for pair in pairs:
            payload = _get(summary, "pairs", pair)
            if not payload:
                continue
            pair_rows.append([name, pair.replace("_", "→"), payload.get("rounds", 0), fmt(_get(payload, "delta_a", "mean")), fmt(_get(payload, "tail", "mean")), fmt(_get(payload, "decay", "mean")), fmt(payload.get("large_better_rate")), fmt(payload.get("equal_rate")), fmt(payload.get("small_better_rate")), fmt(payload.get("large_exceeds_small_capacity_rate")), fmt(payload.get("monotonic_violation_rate")), fmt(_get(payload, "accept_length_correlation", "spearman")), fmt(_get(payload, "draft_agreement_rate", "mean"))])
    for pair in pairs:
        payloads = [_get(summaries[name], "pairs", pair) for name in macro_names]
        payloads = [value for value in payloads if value]
        if payloads:
            pair_rows.append(["均(已成)", pair.replace("_", "→"), f"D={len(payloads)}", fmt(mean(_get(value, "delta_a", "mean") for value in payloads)), fmt(mean(_get(value, "tail", "mean") for value in payloads)), fmt(mean(_get(value, "decay", "mean") for value in payloads)), fmt(mean(value.get("large_better_rate") for value in payloads)), fmt(mean(value.get("equal_rate") for value in payloads)), fmt(mean(value.get("small_better_rate") for value in payloads)), fmt(mean(value.get("large_exceeds_small_capacity_rate") for value in payloads)), fmt(mean(value.get("monotonic_violation_rate") for value in payloads)), fmt(mean(_get(value, "accept_length_correlation", "spearman") for value in payloads)), fmt(mean(_get(value, "draft_agreement_rate", "mean") for value in payloads))])
    lines += [table(["集", "对", "R/D", "Δ", "尾", "衰", "P+", "P=", "P-", "P越界", "P逆", "ρA", "同草"], pair_rows or [["待运行"] * 13]), ""]

    lines += ["## 5. 条件接收长度矩阵", ""]
    conditional_rows = []
    for name in names:
        summary = summaries.get(name)
        for pair in pairs:
            matrix = _get(summary, "pairs", pair, "conditional_a_small_given_a_large_counts") or {}
            for a_large, counts in sorted(matrix.items(), key=lambda item: int(item[0])):
                denominator = sum(int(value) for value in counts.values())
                for a_small, count in sorted(counts.items(), key=lambda item: int(item[0])):
                    conditional_rows.append([name, pair.replace("_", "→"), a_large, a_small, count, denominator, fmt(int(count) / denominator if denominator else None)])
    for pair in pairs:
        small, large = (int(value) for value in pair.split("_"))
        for a_large in range(1, large + 1):
            for a_small in range(1, small + 1):
                probabilities = []
                for name in macro_names:
                    counts = _get(summaries[name], "pairs", pair, "conditional_a_small_given_a_large_counts", str(a_large)) or {}
                    denominator = sum(int(value) for value in counts.values())
                    if denominator:
                        probabilities.append(int(counts.get(str(a_small), 0)) / denominator)
                if probabilities:
                    conditional_rows.append(["均(已成)", pair.replace("_", "→"), a_large, a_small, f"D={len(probabilities)}", "—", fmt(mean(probabilities))])
    lines += [table(["集", "对", "A2", "A1", "N/D", "N2", "P(A1|A2)"], conditional_rows or [["待运行"] * 7]), ""]

    lines += ["## 6. 条件生存率 S(L1,L2,k)", ""]
    survival_rows = []
    for name in names:
        for row in (summaries.get(name) or {}).get("survival", []):
            survival_rows.append([name, row["pair"].replace("_", "→"), row["k"], row["denominator_n2"], row["numerator_n12"], fmt(row.get("survival")), "端" if row.get("structural_endpoint") else ""])
    for pair in pairs:
        small = int(pair.split("_", 1)[0])
        for k in range(1, small + 2):
            values = []
            for name in macro_names:
                match = next((row for row in (summaries[name] or {}).get("survival", []) if row.get("pair") == pair and int(row.get("k", -1)) == k), None)
                if match and match.get("survival") is not None:
                    values.append(float(match["survival"]))
            if values:
                survival_rows.append(["均(已成)", pair.replace("_", "→"), k, f"D={len(values)}", "—", fmt(mean(values)), "端" if k == small + 1 else ""])
    lines += [table(["集", "对", "k", "N2/D", "N12", "S", "界"], survival_rows or [["待运行"] * 7]), ""]

    lines += ["## 7. 历史信息对当前接收长度的参照", ""]
    history_a_rows = []
    features = ["prev_anchor_a", "a_ma2", "a_ma4", "a_ma8", "a_ewma05", "prev_anchor_conf", "conf_ma2", "conf_ma4", "conf_ma8"]
    for name in names:
        summary = summaries.get(name)
        for feature in features:
            for size in block_sizes:
                corr = _get(summary, "history_reference", "feature_target_correlation", feature, f"a{size}")
                if corr and int(corr.get("count", 0)):
                    history_a_rows.append([name, feature, f"A{size}", corr["count"], fmt(corr.get("pearson")), fmt(corr.get("spearman"))])
    for feature in features:
        for size in block_sizes:
            corrs = [_get(summaries[name], "history_reference", "feature_target_correlation", feature, f"a{size}") for name in macro_names]
            corrs = [value for value in corrs if value and int(value.get("count", 0))]
            if corrs:
                history_a_rows.append(["均(已成)", feature, f"A{size}", f"D={len(corrs)}", fmt(mean(value.get("pearson") for value in corrs)), fmt(mean(value.get("spearman") for value in corrs))])
    lines += [table(["集", "史特", "标", "N/D", "r", "ρ"], history_a_rows or [["待运行"] * 6]), ""]

    lines += ["## 8. 历史信息对 block-size 衰减的参照", ""]
    history_decay_rows = []
    for name in names:
        summary = summaries.get(name)
        for feature in features:
            for pair in pairs:
                corr = _get(summary, "history_reference", "feature_target_correlation", feature, f"decay_{pair}")
                if corr and int(corr.get("count", 0)):
                    history_decay_rows.append([name, feature, pair.replace("_", "→"), corr["count"], fmt(corr.get("pearson")), fmt(corr.get("spearman"))])
    for feature in features:
        for pair in pairs:
            corrs = [_get(summaries[name], "history_reference", "feature_target_correlation", feature, f"decay_{pair}") for name in macro_names]
            corrs = [value for value in corrs if value and int(value.get("count", 0))]
            if corrs:
                history_decay_rows.append(["均(已成)", feature, pair.replace("_", "→"), f"D={len(corrs)}", fmt(mean(value.get("pearson") for value in corrs)), fmt(mean(value.get("spearman") for value in corrs))])
    lines += [table(["集", "史特", "对", "N/D", "r衰", "ρ衰"], history_decay_rows or [["待运行"] * 6]), ""]

    lines += ["## 9. 无训练历史预测基线", ""]
    predictor_rows = []
    predictor_features = ["prev_anchor_a", "a_ma1", "a_ma2", "a_ma4", "a_ma8", "a_ewma05"]
    for name in names:
        summary = summaries.get(name)
        for feature in predictor_features:
            for size in block_sizes:
                payload = _get(summary, "history_reference", "clipped_history_predictor_mae", feature, f"a{size}")
                if payload and int(payload.get("count", 0)):
                    predictor_rows.append([name, feature, f"A{size}", payload["count"], fmt(payload.get("mean")), fmt(_get(payload, "quantiles", "0.5")), fmt(_get(payload, "quantiles", "0.9"))])
    for feature in predictor_features:
        for size in block_sizes:
            payloads = [_get(summaries[name], "history_reference", "clipped_history_predictor_mae", feature, f"a{size}") for name in macro_names]
            payloads = [value for value in payloads if value and int(value.get("count", 0))]
            if payloads:
                predictor_rows.append(["均(已成)", feature, f"A{size}", f"D={len(payloads)}", fmt(mean(value.get("mean") for value in payloads)), fmt(mean(_get(value, "quantiles", "0.5") for value in payloads)), fmt(mean(_get(value, "quantiles", "0.9") for value in payloads))])
    lines += [table(["集", "预测", "标", "N/D", "MAE", "E50", "E90"], predictor_rows or [["待运行"] * 7]), ""]

    lines += ["## 10. 逐位置诊断", ""]
    position_rows = []
    for name in names:
        summary = summaries.get(name)
        for size in block_sizes:
            for row in _get(summary, "blocks", str(size), "per_position") or []:
                position_rows.append([name, size, row.get("draft_position"), row.get("count"), fmt(row.get("accepted_rate")), fmt(_get(row, "selected_confidence", "mean")), fmt(_get(row, "top1_top2_margin", "mean")), fmt(_get(row, "entropy", "mean"))])
    for size in block_sizes:
        for position in range(1, size):
            matched = []
            for name in macro_names:
                row = next((item for item in (_get(summaries[name], "blocks", str(size), "per_position") or []) if int(item.get("draft_position", -1)) == position), None)
                if row:
                    matched.append(row)
            if matched:
                position_rows.append(["均(已成)", size, position, f"D={len(matched)}", fmt(mean(row.get("accepted_rate") for row in matched)), fmt(mean(_get(row, "selected_confidence", "mean") for row in matched)), fmt(mean(_get(row, "top1_top2_margin", "mean") for row in matched)), fmt(mean(_get(row, "entropy", "mean") for row in matched))])
    lines += [table(["集", "L", "p", "N/D", "P收", "C", "Mg", "H"], position_rows or [["待运行"] * 8]), ""]

    completed_events = [row for row in status_rows if row.get("event") == "benchmark_status" and row.get("status") in {"completed", "failed"}]
    lines += ["## 11. 完成记录", "", table(["时间", "集", "状态", "阶段", "码"], [[row.get("created_at", ""), row.get("benchmark", ""), row.get("status", ""), row.get("stage", ""), row.get("exit_code", "")] for row in completed_events] or [["暂无", "", "", "", ""]]), ""]

    lines += [
        "## 12. 变量字典、口径与例子",
        "",
        "- `L`：本轮模拟的 block size。例：`L=8` 的 block 含 1 个 seed、7 个 draft token，并由 verifier 产生最多 8 个输出 token。",
        "- `M_L`：从 draft position 1 开始连续匹配 verifier 的 draft token 数，范围 `0..L-1`。",
        "- `A_L=M_L+1`：本轮可提交长度，包含首次 correction/bonus，范围 `1..L`。例：连续匹配 5 个 draft token 时 `A_L=6`。",
        "- `R原/R有效`：原始 paired round 数/主分析 round 数；默认主分析排除任一分支命中 EOS 以及剩余预算小于 32 的轮。",
        "- `A均/A50/A90`：`A_L` 的均值、中位数和 90 分位；`A/L` 是每轮接收长度除以 block size 后的均值。",
        "- `P满/P零`：`A_L=L` 的比例/一个 draft token 都没匹配即 `M_L=0` 的比例。",
        "- `C收/C拒`：已匹配 draft token 的平均选择置信度/首次不匹配 draft token 的选择置信度；softmax 分母排除 MASK。",
        "- `Δ=A_L2-A_L1`：大 block 相对小 block 的总接收长度变化。",
        "- `尾=max(A_L2-L1,0)`：只有大 block 才有容量容纳的超出 `L1` 部分。例：`L1=8,A_32=11` 时尾部为 3。",
        "- `衰=min(A_L2,L1)-A_L1`：都落在小 block 容量内时，lookahead 改变带来的差异；恒有 `Δ=尾+衰`。例：`A_8=5,A_32=7` 时尾=0、衰=2。负衰表示小 block 反而更好。",
        "- `P+/P=/P-`：`Δ` 大于/等于/小于 0 的轮次比例；`P越界=P(A_L2>L1)`；`P逆=P(A_L1>A_L2)`。",
        "- `ρA`：同轮 `A_L1` 与 `A_L2` 的 Spearman 相关；`同草` 是两个 block 在共同 draft positions 上 token 一致率。",
        "- `S=P(A_L1≥k|A_L2≥k)=N12/N2`。`N2` 是满足大 block 阈值的轮数，`N12` 是大小 block 同时满足的轮数。`k=L1+1` 也报告：若 `N2>0` 则结构上 `S=0`，若 `N2=0` 则为 `NA`。",
        "- `史特`：历史特征；`prev_anchor_a` 是上一轮 L16 接收长度，`a_maN/conf_maN` 是此前 N 轮 L16 接收长度/已接收 draft 置信度均值，`a_ewma05` 是 alpha=0.5 的指数滑动平均。",
        "- `r/ρ`：历史特征与当前目标的 Pearson/Spearman 相关；`r衰/ρ衰` 的目标是相应 pair 的区间内衰减。它们是第二、三阶段建模参考，不等于因果结论。",
        "- `P(A1|A2)`：在某个大 block 精确接收长度 `A2` 条件下，小 block 精确接收 `A1` 的频率；例：`A32=6` 的 100 轮中 70 轮 `A8=6`，对应格为 0.7。",
        "- `N/D`：数据集行表示样本/轮次数 `N`，宏平均行表示参与数据集数 `D`；条件矩阵中的 `N2` 是固定 `A2` 后的行总数。",
        "- `预测/MAE/E50/E90`：直接把历史接收特征裁剪到 `[1,L]` 后预测当前 `A_L`；三列是绝对误差均值、中位数和 90 分位。例：历史预测 5、当前 `A8=7`，误差为 2。",
        "- `p/P收/C/Mg/H`：draft 位置（seed 后从 1 起）、该位置被连续接收的比例、选择置信度、top1-top2 概率 margin、预测分布 entropy。",
        "- `Acc%`：NeMo-Skills 任务正确率性质；数学/知识题用 `symbolic_correct`，HumanEval/MBPP 用 `passing_base_tests`，IFEval 用 `average_score`，LiveCodeBench 用 `accuracy`。它只验证 L16 输出质量，不是 shadow 效率。",
        "- `均(已成)`：先在每个非 AIME24 数据集内部算统计量，再对数据集等权平均；`D` 是参与该格宏平均的数据集数。AIME24 原始行始终保留，但不参与任何宏平均。",
        "",
        "完整的精确计数、分位数、条件矩阵、所有 feature-target 相关与 clipped-history MAE 位于 `summaries/*.json`；逐轮原始记录位于 `traces/*.jsonl`。本文不把 wall time、吞吐或显存作为本阶段结论指标。",
        "",
    ]
    atomic_text(run_dir / "report.md", "\n".join(lines))


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command_name", required=True)
    init = subparsers.add_parser("init")
    init.add_argument("--run-dir", required=True, type=Path)
    init.add_argument("--entrypoint", required=True)
    init.add_argument("--command", required=True)
    init.add_argument("--mode", required=True)
    init.add_argument("--benchmarks", required=True)
    init.add_argument("--model", required=True)
    init.add_argument("--served-model-name", required=True)
    init.add_argument("--lora-path", default="")
    init.add_argument("--dtype", required=True)
    init.add_argument("--block-sizes", required=True)
    init.add_argument("--anchor-block-size", type=int, required=True)
    init.add_argument("--history-windows", required=True)
    init.add_argument("--threshold", type=float, required=True)
    init.add_argument("--temperature", type=float, required=True)
    init.add_argument("--top-p", type=float, required=True)
    init.add_argument("--tokens", type=int, required=True)
    init.add_argument("--context-length", type=int, required=True)
    init.add_argument("--gpu-device", type=int, required=True)
    init.add_argument("--gpu-candidates", required=True)
    init.add_argument("--gpu-min-free-gb", type=float, required=True)
    init.add_argument("--gpu-memory-reserve-gb", type=float, required=True)
    init.add_argument("--port", type=int, required=True)
    init.add_argument("--client-concurrency", type=int, required=True)
    init.add_argument("--num-chunks", type=int, required=True)
    init.add_argument("--max-samples", type=int)
    init.add_argument("--quick-test", action="store_true")
    init.add_argument("--enable-thinking", action="store_true")
    init.add_argument("--disable-thinking", action="store_true")
    init.add_argument("--keep-thinking", action="store_true")
    init.add_argument("--strip-thinking", action="store_true")
    init.add_argument("--max-thinking-tokens", type=int)
    init.add_argument("--math-prompt-config", default="")
    init.add_argument("--trace-detail", required=True)
    init.add_argument("--include-boundary-rounds", action="store_true")
    init.add_argument("--pytorch-python", required=True)
    init.add_argument("--eval-python", required=True)
    init.add_argument("--nemo-skills-data-dir", required=True)
    init.add_argument("--google-research-dir", required=True)

    status = subparsers.add_parser("status")
    status.add_argument("--run-dir", required=True, type=Path)
    status.add_argument("--benchmark-spec", required=True)
    status.add_argument("--status", required=True, choices=["running", "completed", "failed"])
    status.add_argument("--stage", required=True)
    status.add_argument("--exit-code", type=int, default=0)
    status.add_argument("--message", default="")
    status.add_argument("--trace-file", default="")
    status.add_argument("--summary-file", default="")
    status.add_argument("--metrics-file", default="")

    report = subparsers.add_parser("report")
    report.add_argument("--run-dir", required=True, type=Path)
    return parser


def main() -> int:
    args = create_parser().parse_args()
    if args.command_name == "init":
        initialize(args)
    elif args.command_name == "status":
        append_status(args)
    else:
        render_report(args.run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
