#!/usr/bin/env python3
"""Initialize run metadata, append progress, and render the live Chinese report."""

from __future__ import annotations

import argparse
import json
import math
import os
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


def benchmark_specs(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def benchmark_name(spec: str) -> str:
    return spec.split(":", 1)[0].strip()


def safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in value)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def load_json(path: Path) -> Optional[dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if path.is_file():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


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


def pct(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    return f"{100 * number:.2f}"


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


def table(headers: list[str], rows: Iterable[Iterable[Any]]) -> str:
    def cell(value: Any) -> str:
        return str(value).replace("|", "/").replace("\n", "<br>")

    lines = ["|" + "|".join(headers) + "|", "|" + "|".join(":---:" for _ in headers) + "|"]
    lines.extend("|" + "|".join(cell(value) for value in row) + "|" for row in rows)
    return "\n".join(lines)


def accuracy(metrics: Optional[dict[str, Any]], name: str) -> Optional[float]:
    if not metrics:
        return None
    field = ACCURACY_FIELDS.get(name, "symbolic_correct")
    try:
        return float(metrics[name]["pass@1"][field])
    except (KeyError, TypeError, ValueError):
        return None


def initialize(args: argparse.Namespace) -> None:
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    for child in ("traces", "summaries", "analysis", "metrics", "eval_runs", "runtime"):
        (run_dir / child).mkdir(parents=True, exist_ok=True)
    specs = benchmark_specs(args.benchmarks)
    settings = {
        "schema_version": 1,
        "created_at": now(),
        "purpose": (
            "用独立真实 LinearSpec 推理采集逐轮首错标签与仅在当前 verify 前可得的 confidence/"
            "历史特征；离线枚举免训练可解释规则，在 request 级无泄漏切分上选择唯一的跨数据集全局策略。"
        ),
        "entrypoint": args.entrypoint,
        "command": args.command,
        "backend": "native_pytorch_linearspec_failure_locator",
        "mode": args.mode,
        "benchmarks": args.benchmarks,
        "benchmark_specs": specs,
        "model": args.model,
        "served_model_name": args.served_model_name,
        "lora_path": args.lora_path,
        "dtype": args.dtype,
        "block_size": args.block_size,
        "history_windows": [int(value) for value in args.history_windows.split(",")],
        "aggregations": [value for value in args.aggregations.split(",") if value],
        "grid": args.grid,
        "draft_threshold": args.threshold,
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
        "split_seed": args.split_seed,
        "search_ratio": args.search_ratio,
        "selection_ratio": args.selection_ratio,
        "test_ratio": 1.0 - args.search_ratio - args.selection_ratio,
        "shortlist": args.shortlist,
        "report_top": args.report_top,
        "search_max_rounds_per_dataset": args.search_max_rounds_per_dataset,
        "bootstrap_replicates": args.bootstrap_replicates,
        "pytorch_python": args.pytorch_python,
        "eval_python": args.eval_python,
        "nemo_skills_data_dir": args.nemo_skills_data_dir,
        "google_research_dir": args.google_research_dir,
        "output_dir": str(run_dir),
        "analysis_contract": {
            "label": "q=first verifier mismatch position in 1..L-1; NONE for full accept",
            "causal_features_only": True,
            "decoding_intervention": False,
            "split_unit": "request",
            "selection_metric": "equal-dataset macro Exact-F1",
            "strict_comparator": ">",
            "cold_start_fallback": "original prefix_drop > 0.15",
            "aime24": "may be collected when explicitly requested, never used in macro/search/selection",
            "global_only": "one formula and one parameter set shared by every dataset",
        },
    }
    atomic_json(run_dir / "Settings.json", settings)
    settings_md = f"""# 本轮实验设置

- 创建时间：`{settings['created_at']}`
- 实验目的：{settings['purpose']}
- 数据集：`{settings['benchmarks']}`
- 模式/模型：`{settings['mode']}` / `{settings['model']}`
- LoRA：`{settings['lora_path'] or '无（base 模式）'}`
- block/history/聚合：`L={settings['block_size']}` / `{settings['history_windows']}` / `{settings['aggregations']}`
- 搜索网格：`{settings['grid']}`；shortlist=`{settings['shortlist']}`；每数据集搜索轮上限=`{settings['search_max_rounds_per_dataset']}`
- request 切分：search/selection/test=`{settings['search_ratio']}/{settings['selection_ratio']}/{settings['test_ratio']}`；seed=`{settings['split_seed']}`
- 推理 threshold/temperature/top-p：`{settings['draft_threshold']}` / `{settings['temperature']}` / `{settings['top_p']}`
- tokens/context：`{settings['tokens']}` / `{settings['context_length']}`
- GPU/显存预留/端口：`{settings['gpu_device']}` / `{settings['gpu_memory_reserve_gb']} GiB` / `{settings['port']}`
- 并发/chunks：`{settings['client_concurrency']}` / `{settings['num_chunks']}`（模型调用仍由原生 server 串行保护）
- 主分析边界：`{'纳入边界轮' if settings['include_boundary_rounds'] else '排除 EOS 与不足一个完整 block 的末轮'}`
- bootstrap：request 级 `{settings['bootstrap_replicates']}` 次
- 命令：`{settings['command']}`

完整机器可读设置见 `Settings.json`。本实验不读取任何此前 observation 的结果文件。
"""
    atomic_text(run_dir / "Settings.md", settings_md)
    with (run_dir / "benchmark_status.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                {"created_at": now(), "event": "run_initialized", "status": "initialized", "benchmarks": specs},
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


def candidate_short(spec: Optional[dict[str, Any]]) -> str:
    if not spec:
        return "NA"
    family = str(spec.get("family", "?"))
    decision = "首" if spec.get("decision") == "first" else "峰"
    window = int(spec.get("history_window", 0))
    history = f"H{window}/{spec.get('aggregation')}" if window else "H0"
    return f"{family}/{decision}/{history}/t{fmt(spec.get('threshold'))}/w{fmt(spec.get('position_weight'))}"


def latest_status(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result = {}
    for row in rows:
        if row.get("event") == "benchmark_status" and row.get("benchmark"):
            result[str(row["benchmark"])] = row
    return result


def metric_row(dataset: str, strategy: str, payload: dict[str, Any], ci: str = "NA") -> list[Any]:
    return [
        dataset,
        strategy,
        payload.get("rounds", "NA"),
        pct(payload.get("coverage")),
        pct(payload.get("precision")),
        pct(payload.get("recall")),
        pct(payload.get("exact_f1")),
        fmt(payload.get("hits_per_100"), 2),
        pct(payload.get("full_false_alarm")),
        pct(payload.get("miss_rate")),
        fmt(payload.get("mae"), 3),
        pct(payload.get("tol1_recall")),
        pct(payload.get("position1_recall")),
        ci,
    ]


def render_report(run_dir: Path) -> None:
    settings = load_json(run_dir / "Settings.json") or {}
    specs = settings.get("benchmark_specs", [])
    statuses = latest_status(load_jsonl(run_dir / "benchmark_status.jsonl"))
    analysis = load_json(run_dir / "analysis" / "strategy_search.json")
    completed = sum(statuses.get(benchmark_name(spec), {}).get("status") == "completed" for spec in specs)
    failed = sum(statuses.get(benchmark_name(spec), {}).get("status") == "failed" for spec in specs)
    all_done = bool(specs) and completed + failed == len(specs)
    test_rounds = (analysis or {}).get("split_round_counts", {}).get("test", 0)
    if all_done and not test_rounds:
        phase = "链路完成（无test，仅smoke）"
    else:
        phase = "最终（所有项已终止）" if all_done else "进行中（策略排名会随新数据集刷新）"

    lines = [
        "# LinearSpec 自适应首错位置免训练策略实验报告",
        "",
        f"> 报告状态：**{phase}**；更新时间：`{now()}`。本文件在结果目录建立时即生成，每项完成后原子刷新。",
        "> `test` 在策略冻结前不参与选择；AIME24 即使显式运行也不参与搜索、选择或宏平均。进行中的 winner 只是当前已完成数据上的临时全局策略。",
        "",
        "## 1. 配置与进度",
        "",
        f"- 真实推理：`{settings.get('mode','NA')}`，`L={settings.get('block_size','NA')}`，temperature=`{settings.get('temperature','NA')}`，draft threshold=`{settings.get('draft_threshold','NA')}`。",
        f"- 历史窗口：`{settings.get('history_windows','NA')}`；聚合：`{settings.get('aggregations','NA')}`；规则网格：`{settings.get('grid','NA')}`。",
        f"- request 切分：`{settings.get('search_ratio','NA')}/{settings.get('selection_ratio','NA')}/{settings.get('test_ratio','NA')}`；seed=`{settings.get('split_seed','NA')}`。",
        f"- 已完成/失败/总计：`{completed}/{failed}/{len(specs)}`；结果目录：`{run_dir}`。",
        "",
    ]
    progress_rows = []
    for spec in specs:
        name = benchmark_name(spec)
        status = statuses.get(name, {})
        progress_rows.append(
            [name, status.get("status", "待运行"), status.get("stage", "pending"), status.get("message", "")]
        )
    lines.extend([table(["集", "状态", "阶段", "说明"], progress_rows), ""])

    lines.extend(["## 2. 已采集数据", ""])
    if not analysis or not analysis.get("dataset_summaries"):
        lines.extend(["尚无完成数据集；以下统计表将在第一项完成后出现。", ""])
    else:
        summary_rows = []
        for dataset, summary in sorted(analysis["dataset_summaries"].items()):
            splits = summary.get("request_split_counts", {})
            summary_rows.append(
                [
                    dataset,
                    summary.get("raw_requests", summary.get("requests", "NA")),
                    summary.get("raw_rounds", "NA"),
                    summary.get("analysis_rounds", "NA"),
                    pct(summary.get("failure_rate")),
                    pct(summary.get("position1_share_of_failures")),
                    fmt(summary.get("accept_length_mean"), 3),
                    f"{splits.get('search',0)}/{splits.get('selection',0)}/{splits.get('test',0)}",
                ]
            )
        macro_summaries = [
            summary
            for dataset, summary in analysis["dataset_summaries"].items()
            if dataset.lower().replace("_", "-") not in EXCLUDED_MACRO
            and summary.get("analysis_rounds", 0) > 0
        ]
        if macro_summaries:
            summary_rows.append(
                [
                    "均(等权)",
                    f"D={len(macro_summaries)}",
                    fmt(mean(item.get("raw_rounds") for item in macro_summaries), 2),
                    fmt(mean(item.get("analysis_rounds") for item in macro_summaries), 2),
                    pct(mean(item.get("failure_rate") for item in macro_summaries)),
                    pct(mean(item.get("position1_share_of_failures") for item in macro_summaries)),
                    fmt(mean(item.get("accept_length_mean") for item in macro_summaries), 3),
                    "各集内切",
                ]
            )
        lines.extend(
            [
                table(["集", "Req", "R原", "R有效", "P错", "P首1", "A均", "Req切"], summary_rows),
                "",
                "`P首1` 单列显示真实首错位于 position 1 的比例；原始 prefix-drop 规则在该位置没有前缀均值，结构上不能触发。",
                "",
            ]
        )

    lines.extend(["## 3. 当前全局策略", ""])
    if not analysis or analysis.get("status") != "ok" or not analysis.get("winner"):
        lines.extend(["尚未形成候选排名。", ""])
    else:
        winner = analysis["winner"]
        baseline = analysis["original_baseline"]
        lines.extend(
            [
                f"- 当前 winner：`{candidate_short(winner.get('candidate'))}`。",
                f"- 搜索空间：`{analysis.get('candidate_count')}` 个免训练公式/参数组合；先按 search 宏 Exact-F1 建 shortlist，再按 `{analysis.get('selection_stage_used')}` 冻结唯一策略。",
                "- 所有数据集共享同一公式和参数；cold-start 回退为原始严格 `prefix_drop > 0.15`。",
                f"- winner 对原0.15的test逐数据集胜/平/负：`{analysis.get('winner_vs_original',{}).get('wins',0)}/{analysis.get('winner_vs_original',{}).get('ties',0)}/{analysis.get('winner_vs_original',{}).get('losses',0)}`。",
                "",
            ]
        )
        ci_by_dataset = analysis.get("bootstrap", {}).get("by_dataset", {})
        rows = []
        datasets = sorted(
            set(winner.get("test", {}).get("by_dataset", {}))
            | set(baseline.get("test", {}).get("by_dataset", {}))
        )
        for dataset in datasets:
            base_payload = baseline.get("test", {}).get("by_dataset", {}).get(dataset, {})
            win_payload = winner.get("test", {}).get("by_dataset", {}).get(dataset, {})
            rows.append(metric_row(dataset, "原0.15", base_payload))
            ci = ci_by_dataset.get(dataset, {}).get("exact_f1_ci95")
            ci_text = f"[{pct(ci[0])},{pct(ci[1])}]" if ci else "NA"
            rows.append(metric_row(dataset, "全局优", win_payload, ci_text))
        rows.append(metric_row("均(等权)", "原0.15", baseline.get("test", {}).get("macro", {})))
        macro_ci = analysis.get("bootstrap", {}).get("macro_exact_f1_ci95")
        macro_ci_text = f"[{pct(macro_ci[0])},{pct(macro_ci[1])}]" if macro_ci else "NA"
        rows.append(metric_row("均(等权)", "全局优", winner.get("test", {}).get("macro", {}), macro_ci_text))
        lines.extend(
            [
                "### 3.1 未见 test：原阈值与唯一全局策略",
                "",
                table(["集", "策", "R", "Cov%", "Pre%", "Rec%", "F1%", "H100", "FA%", "Miss%", "MAE", "±1%", "P1R%", "F1CI%"], rows),
                "",
            ]
        )

        cold = winner.get("cold_start", {}).get("macro", {})
        ready = winner.get("history_ready", {}).get("macro", {})
        matched = analysis.get("matched_coverage") or {}
        diagnostic_rows = [
            ["winner-cold", pct(cold.get("coverage")), pct(cold.get("precision")), pct(cold.get("recall")), pct(cold.get("exact_f1")), pct(cold.get("full_false_alarm"))],
            ["winner-ready", pct(ready.get("coverage")), pct(ready.get("precision")), pct(ready.get("recall")), pct(ready.get("exact_f1")), pct(ready.get("full_false_alarm"))],
        ]
        if matched:
            matched_test = matched.get("test_macro", {})
            diagnostic_rows.append(
                ["等Cov参照", pct(matched_test.get("coverage")), pct(matched_test.get("precision")), pct(matched_test.get("recall")), pct(matched_test.get("exact_f1")), pct(matched_test.get("full_false_alarm"))]
            )
        lines.extend(
            [
                "### 3.2 冷启动、历史充分与等尝试率诊断",
                "",
                table(["切片", "Cov%", "Pre%", "Rec%", "F1%", "FA%"], diagnostic_rows),
                "",
            ]
        )

        top_rows = []
        for rank, item in enumerate(analysis.get("top_shortlist", []), 1):
            top_rows.append(
                [
                    rank,
                    candidate_short(item.get("candidate")),
                    pct(item.get("search", {}).get("macro", {}).get("exact_f1")),
                    pct(item.get("selection", {}).get("macro", {}).get("exact_f1")),
                    pct(item.get("test", {}).get("macro", {}).get("exact_f1")),
                    pct(item.get("test", {}).get("macro", {}).get("min_exact_f1")),
                    pct(item.get("test", {}).get("macro", {}).get("coverage")),
                    pct(item.get("test", {}).get("macro", {}).get("full_false_alarm")),
                ]
            )
        lines.extend(
            [
                "### 3.3 候选短名单",
                "",
                table(["#", "规则", "F1搜%", "F1选%", "F1测%", "F1低%", "Cov测%", "FA测%"], top_rows),
                "",
                "`F1测` 仅在 winner 已由 search/selection 冻结后展示，不参与重新排序；其余候选 test 列只用于诊断策略空间稳定性。",
                "",
            ]
        )

        ablation_rows = []
        for item in analysis.get("ablations", []):
            ablation_rows.append(
                [
                    item.get("group"),
                    candidate_short(item.get("candidate")),
                    pct(item.get("search_macro", {}).get("exact_f1")),
                    pct(item.get("selection_macro", {}).get("exact_f1")),
                    pct(item.get("test_macro", {}).get("exact_f1")),
                ]
            )
        lines.extend(
            [
                "### 3.4 规则族与历史窗口消融",
                "",
                table(["组", "组内优", "F1搜%", "F1选%", "F1测%"], ablation_rows),
                "",
            ]
        )

        oracle_rows = []
        for dataset, item in sorted(analysis.get("per_dataset_oracle", {}).items()):
            oracle_rows.append(
                [dataset, pct(item.get("global_exact_f1")), pct(item.get("oracle_exact_f1")), pct(item.get("regret")), item.get("candidate_id", "NA")]
            )
        oracle_values = list(analysis.get("per_dataset_oracle", {}).values())
        if oracle_values:
            oracle_rows.append(
                [
                    "均(等权)",
                    pct(mean(item.get("global_exact_f1") for item in oracle_values)),
                    pct(mean(item.get("oracle_exact_f1") for item in oracle_values)),
                    pct(mean(item.get("regret") for item in oracle_values)),
                    "D=" + str(len(oracle_values)),
                ]
            )
        lines.extend(
            [
                "### 3.5 数据集专属 oracle（只诊断，不部署）",
                "",
                table(["集", "F1全%", "F1神%", "Reg%", "神规则"], oracle_rows),
                "",
                "oracle 在该数据集 test 上事后选择，因此是乐观上界，不能用于策略选择；`Reg` 只衡量一套全局规则距离数据集专属最优还有多远。",
                "",
            ]
        )

    lines.extend(["## 4. 生成质量审计", ""])
    quality_rows = []
    quality_values = []
    for spec in specs:
        name = benchmark_name(spec)
        metric = load_json(run_dir / "metrics" / f"metrics_{safe_name(name)}.json")
        value = accuracy(metric, name)
        quality_rows.append([name, fmt(value, 3), "排除宏平均" if name.lower() in EXCLUDED_MACRO else "纳入"])
        if name.lower() not in EXCLUDED_MACRO and value is not None:
            quality_values.append(value)
    quality_rows.append(["均(等权)", fmt(mean(quality_values), 3), f"D={len(quality_values)}"])
    lines.extend(
        [
            table(["集", "Acc%", "宏口径"], quality_rows),
            "",
            "Acc 只审计新观察链路仍产生正常任务输出，不作为定位规则的选择目标。",
            "",
            "## 5. 变量、口径与例子",
            "",
            "- `q`：当前 verifier 的真实首个不匹配 draft position；范围 `1..L-1`，全部 draft 通过时为 `NONE`。例：position 1、2 通过而 position 3 首次失败，则 `q=3`。",
            "- `p`：免训练规则在当前 verify 前给出的预测位置；未越阈值为 `NONE`。只有 `p=q` 才算精确命中。",
            "- `prefix_drop=1-C_i/mean(C_1..C_{i-1})`：原始指标；严格使用 `>`。position 1 没有历史前缀，因此原0.15不能预测它。",
            "- `hist_good_drop/hist_error_drop`：当前 confidence 相对过去窗口内已验证通过/首错位置 confidence 的下降比例。例：历史 good 均值0.8、当前0.6，则 good-drop=0.25。",
            "- `hist_separator=(C_good-C_i)/(C_good-C_err)`：把当前 confidence 放到历史 good/error 分隔轴；例：good=0.8、error=0.4、当前=0.6，则 separator=0.5。",
            "- `首/峰`：从左向右第一个严格越阈值的位置/全 block 风险最大且越阈值的位置。`H0/H1/H2/H4` 表示不用历史或使用过去1/2/4轮。",
            "- `w`：历史错误位置软先验权重；`w=0` 表示只看 confidence 风险。历史缺失时不填零，回退原0.15。",
            "- `R原/R有效`：trace 原始轮数/主分析轮数。默认排除 EOS 轮和剩余生成预算不足一个完整 block 的末轮。",
            "- `P错/P首1/A均`：首错存在的轮次比例/所有错误轮中 `q=1` 的比例/接收长度均值。",
            "- `Req切`：该数据集 request 数的 search/selection/test 分配；同一 request 的所有轮只属于一个 split。",
            "- `Cov=attempts/rounds`；`Pre=exact/attempts`；`Rec=exact/failure rounds`；`F1` 是精确位置 Precision 与 Recall 的调和平均。错位预测同时构成一次 FP 和一次 FN。",
            "- `H100`：每100个有效轮的严格精确命中数。例：500轮精确命中75次，则H100=15。",
            "- `FA`：全接收轮却预测了错误位置的比例；`Miss`：有首错但预测NONE的比例。",
            "- `MAE`：在“有首错且作出预测”的轮上计算 `|p-q|` 均值；`±1` 是所有错误轮中预测落在 `q±1` 的比例。",
            "- `P1R`：真实 `q=1` 的错误轮中精确预测 position 1 的比例；原0.15在该项结构上应为0。",
            "- `F1CI`：以 request 为聚类单位 bootstrap 得到的95%区间，不把同一回答的相邻轮误当独立样本。",
            "- `F1搜/F1选/F1测`：同一规则在search/selection/test上的数据集等权宏Exact-F1；只有前两段能参与冻结策略。`F1低` 是test各数据集F1最小值。",
            "- `均(等权)`：先分别计算每个非 AIME24 数据集指标，再对数据集等权平均；不池化所有轮，避免大数据集覆盖小数据集。",
            "- `F1神/Reg`：test 上事后数据集专属 oracle 的F1/它与唯一全局策略的F1差，只作可迁移性诊断。",
            "",
            "完整逐轮事实位于 `traces/*.jsonl`，每数据集摘要位于 `summaries/*.json`，候选规则、三段切分结果和冻结契约位于 `analysis/strategy_search.json`。",
            "",
        ]
    )
    atomic_text(run_dir / "report.md", "\n".join(lines))


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command_name", required=True)
    init = commands.add_parser("init")
    init.add_argument("--run-dir", required=True, type=Path)
    init.add_argument("--entrypoint", required=True)
    init.add_argument("--command", required=True)
    init.add_argument("--mode", required=True)
    init.add_argument("--benchmarks", required=True)
    init.add_argument("--model", required=True)
    init.add_argument("--served-model-name", required=True)
    init.add_argument("--lora-path", default="")
    init.add_argument("--dtype", required=True)
    init.add_argument("--block-size", type=int, required=True)
    init.add_argument("--history-windows", required=True)
    init.add_argument("--aggregations", required=True)
    init.add_argument("--grid", required=True)
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
    init.add_argument("--split-seed", type=int, required=True)
    init.add_argument("--search-ratio", type=float, required=True)
    init.add_argument("--selection-ratio", type=float, required=True)
    init.add_argument("--shortlist", type=int, required=True)
    init.add_argument("--report-top", type=int, required=True)
    init.add_argument("--search-max-rounds-per-dataset", type=int, required=True)
    init.add_argument("--bootstrap-replicates", type=int, required=True)
    init.add_argument("--pytorch-python", required=True)
    init.add_argument("--eval-python", required=True)
    init.add_argument("--nemo-skills-data-dir", required=True)
    init.add_argument("--google-research-dir", required=True)

    status = commands.add_parser("status")
    status.add_argument("--run-dir", required=True, type=Path)
    status.add_argument("--benchmark-spec", required=True)
    status.add_argument("--status", required=True, choices=["running", "completed", "failed"])
    status.add_argument("--stage", required=True)
    status.add_argument("--exit-code", type=int, default=0)
    status.add_argument("--message", default="")
    status.add_argument("--trace-file", default="")
    status.add_argument("--summary-file", default="")
    status.add_argument("--metrics-file", default="")

    report = commands.add_parser("report")
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
