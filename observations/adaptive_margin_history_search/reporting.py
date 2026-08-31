#!/usr/bin/env python3
"""Atomic settings and incremental Chinese report rendering."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any


def now() -> str:
    return datetime.now().astimezone().isoformat()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, payload: Any) -> None:
    atomic_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )


def fmt(value: Any, *, percent: bool = False, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "是" if value else "否"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if percent:
        return f"{100 * number:.{digits}f}%"
    return f"{number:.{digits}f}"


def centered_table(headers: list[str], rows: list[list[Any]]) -> str:
    # Markdown source deliberately contains no padding spaces.  This keeps every
    # column as narrow as its longest cell while :---: centers rendered columns.
    escape = lambda value: str(value).replace("|", "&#124;").replace("\n", "<br>")
    output = ["|" + "|".join(escape(item) for item in headers) + "|", "|" + "|".join(":---:" for _ in headers) + "|"]
    output.extend("|" + "|".join(escape(cell) for cell in row) + "|" for row in rows)
    return "\n".join(output)


def ci_text(values: list[Any] | None) -> str:
    return "[" + ",".join(fmt(value, percent=True) for value in (values or [])) + "]"


def initialize_run(run_dir: Path, settings: dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=False)
    for name in ("analysis", "runtime", "summaries", "trace_source"):
        (run_dir / name).mkdir()
    atomic_json(run_dir / "Settings.json", settings)
    protocol = settings.get("selection_protocol", "split")
    protocol_lines = (
        [
            "- 选择协议：`full_data`；扫描全部请求数据集全部request，每个候选使用其中全部有效轮，不切分、不截断、不设shortlist；零有效轮request单独审计。",
            "- 权重：先在每个数据集内汇总全部有效轮，再对所有请求的非AIME24数据集等权平均。",
            "- 结果性质：声明的有限候选网格上的全数据描述性全局最优；没有held-out test。",
        ]
        if protocol == "full_data"
        else [
            f"- request 切分：search/selection/test = `{settings['search_ratio']}/{settings['selection_ratio']}/{settings['test_ratio']}`",
            "- 选择协议：`split`；selection冻结后才读取held-out test。",
        ]
    )
    settings_md = [
        "# 实验设置：历史自适应 margin_risk 首错定位",
        "",
        f"- 创建时间：`{settings['created_at']}`",
        f"- trace 模式：`{settings['trace_mode']}`",
        f"- 结果目录：`{run_dir}`",
        f"- 数据集：`{settings['benchmarks']}`",
        f"- block size：`{settings.get('block_size') or '从 trace 自动读取'}`",
        f"- 历史窗口：`{settings['history_windows']}`",
        f"- 聚合：`{settings['aggregations']}`",
        f"- 参数网格：`{settings['grid']}`",
        *protocol_lines,
        "- 冷启动回退：严格使用 `margin_risk > 0.5`，不回退到旧 `drop_pct > 0.15`。",
        "- 全局约束：所有非 AIME24 数据集共用同一策略与同一组参数；AIME24 永不参与搜索、选择和宏平均。",
        ("- 选择约束：全数据等权宏指标上同时满足相对固定 `margin_risk=0.5` 的Exact Recall更高且正确位置误报/轮更低；若不存在则保留并报告Pareto前沿。"
         if protocol == "full_data" else "- 选择约束：selection上同时满足相对固定`margin_risk=0.5`的Exact Recall更高且正确位置误报/轮更低；若不存在则保留并报告Pareto前沿。"),
        "- 解码干预：无；该实验只重放 trace 做离线反事实定位。",
        "",
        "完整机器可读设置见 `Settings.json`。",
        "",
    ]
    atomic_text(run_dir / "Settings.md", "\n".join(settings_md))
    render_report(run_dir, None, phase="已初始化；等待 trace 校验。")


def strategy_label(spec: dict[str, Any] | None) -> str:
    if not spec:
        return "—"
    return str(spec.get("candidate_id") or spec.get("family") or "—")


def metric_rows(results: list[tuple[str, dict[str, Any]]], split: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for label, result in results:
        metric = result.get(split, {}).get("macro", {})
        rows.append(
            [
                label,
                fmt(metric.get("report_rate"), percent=True),
                fmt(metric.get("precision"), percent=True),
                fmt(metric.get("recall"), percent=True),
                fmt(metric.get("exact_f1"), percent=True),
                fmt(metric.get("correct_fp_round"), percent=True),
                fmt(metric.get("correct_fp_report"), percent=True),
                fmt(metric.get("position_fpr"), percent=True),
                fmt(metric.get("miss_rate"), percent=True),
                fmt(metric.get("late_rate"), percent=True),
            ]
        )
    return rows


def per_dataset_metric_rows(
    results: list[tuple[str, dict[str, Any]]],
    split: str,
    fields: list[tuple[str, bool]],
) -> list[list[str]]:
    datasets: set[str] = set()
    for _, result in results:
        datasets.update(result.get(split, {}).get("by_dataset", {}))
    rows: list[list[str]] = []
    for dataset in sorted(datasets):
        if dataset.lower().replace("_", "-") == "aime24":
            continue
        for label, result in results:
            metric = result.get(split, {}).get("by_dataset", {}).get(dataset, {})
            rows.append([
                dataset,
                label,
                *[
                    str(int(metric[name]))
                    if name in {"rounds", "failures", "reports"} and metric.get(name) is not None
                    else fmt(metric.get(name), percent=percent)
                    for name, percent in fields
                ],
            ])
    if results:
        for label, result in results:
            metric = result.get(split, {}).get("macro", {})
            rows.append(["等权平均", label, *[fmt(metric.get(name), percent=percent) for name, percent in fields]])
    return rows


def threshold_rows(evaluation: dict[str, Any]) -> list[list[str]]:
    rows = []
    for dataset, stats in sorted(evaluation.get("by_dataset", {}).items()):
        rows.append([
            dataset,
            fmt(stats.get("mean")), fmt(stats.get("std")), fmt(stats.get("p10")),
            fmt(stats.get("p50")), fmt(stats.get("p90")),
            fmt(stats.get("raised"), percent=True), fmt(stats.get("lowered"), percent=True),
            fmt(stats.get("equal"), percent=True), fmt(stats.get("cold"), percent=True),
            fmt(stats.get("ready"), percent=True),
        ])
    stats = evaluation.get("macro", {})
    if stats:
        rows.append([
            "等权平均",
            fmt(stats.get("mean")), fmt(stats.get("std")), fmt(stats.get("p10")),
            fmt(stats.get("p50")), fmt(stats.get("p90")),
            fmt(stats.get("raised"), percent=True), fmt(stats.get("lowered"), percent=True),
            fmt(stats.get("equal"), percent=True), fmt(stats.get("cold"), percent=True),
            fmt(stats.get("ready"), percent=True),
        ])
    return rows


def render_report(run_dir: Path, payload: dict[str, Any] | None, *, phase: str | None = None) -> None:
    settings = json.loads((run_dir / "Settings.json").read_text(encoding="utf-8"))
    protocol = (payload or {}).get("selection_protocol", settings.get("selection_protocol", "split"))
    full_data = protocol == "full_data"
    primary_stage = (payload or {}).get("primary_result_stage", "full_data" if full_data else "test")
    phase_text = phase or (payload or {}).get("phase", "处理中")
    lines = [
        "# 历史自适应 margin_risk 首错定位实验报告",
        "",
        f"> 更新时间：`{now()}`；状态：**{phase_text}**",
        "",
        "本报告在实验启动时建立，并在 trace 校验、特征构造、搜索、冻结策略及最终评估后原子刷新。策略只使用当前 draft verify 前可见的历史轮信息；不训练模型，也不改变解码输出。",
        "",
        "## 固定评估口径",
        "",
        "- `q`：当前轮从左向右第一个 verifier 不通过的位置；full accept 时为 0。`p`：策略首次报告的位置；不报告时为 0。",
        "- `Rec`（Exact Recall）：`p=q>0` 占所有 `q>0` 轮的比例，衡量首错精确命中。",
        "- `CFP/R`（正确位置误报/轮）：`0<p<q` 与 `q=0,p>0` 的总数除以所有轮数；这是本实验的误报主指标。",
        "- `CFP/P`（正确位置误报/报告）：上述误报除以所有发生报告的轮数。",
        "- `PosFPR`：上述误报数除以 verifier 实际通过的全部候选位置数，按位置衡量误报。",
        "- `Rpt`：`p>0` 的总报告率；`Pre/F1/Miss/Late` 分别是精确报告率、Exact-F1、漏报率和晚于首错的报告率。",
        "- 所有“平均”均先对每个数据集计算，再对非 AIME24 数据集等权平均，避免大数据集覆盖小数据集。",
        "",
        "## 当前进度",
        "",
        centered_table(
            ["项", "值"],
            [
                ["trace 模式", settings["trace_mode"]],
                ["源 trace", settings.get("source_run_dir") or "本轮真实重跑"],
                ["数据集", settings["benchmarks"]],
                ["历史窗口", settings["history_windows"]],
                ["聚合", settings["aggregations"]],
                ["参数网格", settings["grid"]],
                ["选择协议", protocol],
                ["阶段", phase_text],
            ],
        ),
        "",
    ]
    if not payload:
        lines.extend(["后续表格将在每一处理阶段自动补充。", ""])
        atomic_text(run_dir / "report.md", "\n".join(lines))
        return

    summaries = payload.get("dataset_summaries", {})
    if summaries:
        if full_data:
            trace_title = "## Trace 与全数据覆盖"
            trace_explanation = (
                "`Req/EvalReq/ZeroReq`分别是源trace全部request数、至少有一轮analysis_valid=true的request数、"
                "只有EOS或不足完整block边界轮因而没有有效轮的request数；必须满足EvalReq+ZeroReq=Req。"
                "`Rnd/Fail`是全部有效轮数和含首错轮数；`Full`是无切分、无抽样地分配给full_data协议的源request数，"
                "必须等于Req。ZeroReq的原始边界轮已被扫描和审计，但按预声明主口径不进入任何候选的定位指标。"
            )
            trace_headers = ["数据集", "Req", "EvalReq", "ZeroReq", "Rnd", "Fail", "Full"]
            trace_rows = [
                [
                    dataset,
                    item.get("raw_requests", item.get("requests", 0)),
                    item.get("valid_requests", item.get("requests", 0)),
                    item.get("zero_valid_requests", 0),
                    item.get("analysis_rounds", 0),
                    item.get("failure_rounds", 0),
                    item.get("request_split_counts", {}).get("full_data", 0),
                ]
                for dataset, item in sorted(summaries.items())
                if dataset.lower().replace("_", "-") != "aime24"
            ]
        else:
            trace_title = "## Trace 与无泄漏切分"
            trace_explanation = (
                "`Req/EvalReq/ZeroReq`分别是源trace request数、有至少一轮有效分析的request数和零有效轮request数；"
                "`Rnd/Fail`是有效轮数和含首错轮数。`S/V/T`以全部源request为单位分配到search/selection/test，"
                "同一request的轮不会跨集合；ZeroReq没有指标轮，但仍保留在来源与切分审计中。"
            )
            trace_headers = ["数据集", "Req", "EvalReq", "ZeroReq", "Rnd", "Fail", "S", "V", "T"]
            trace_rows = [
                [
                    dataset,
                    item.get("raw_requests", item.get("requests", 0)),
                    item.get("valid_requests", item.get("requests", 0)),
                    item.get("zero_valid_requests", 0),
                    item.get("analysis_rounds", 0),
                    item.get("failure_rounds", 0),
                    item.get("request_split_counts", {}).get("search", 0),
                    item.get("request_split_counts", {}).get("selection", 0),
                    item.get("request_split_counts", {}).get("test", 0),
                ]
                for dataset, item in sorted(summaries.items())
                if dataset.lower().replace("_", "-") != "aime24"
            ]
        lines.extend(
            [trace_title, "", trace_explanation, "", centered_table(trace_headers, trace_rows), ""]
        )

    fixed = payload.get("fixed_margin_baseline")
    original = payload.get("original_drop_baseline")
    winner = payload.get("winner")
    result_rows: list[tuple[str, dict[str, Any]]] = []
    if original:
        result_rows.append(("drop.15", original))
    if fixed:
        result_rows.append(("margin.5", fixed))
    if winner:
        result_rows.append(("adaptive", winner))

    if result_rows:
        stage = payload.get("selection_stage_used", "selection")
        if full_data:
            dataset_count = payload.get("winner", {}).get("full_data", {}).get("macro", {}).get("datasets", 0)
            result_title = "## 全数据全局最优结果"
            result_explanation = (
                f"全数据候选：`{strategy_label((winner or {}).get('candidate'))}`。每个候选都使用{dataset_count}个非AIME24数据集全部有效轮；"
                "表内指标先在数据集内汇总，再做等数据集权重宏平均。这里没有search/selection/test切分。"
            )
            dominance_lines = [
                f"- 全数据严格支配固定 margin.5：**{fmt(payload.get('full_data_dominates_fixed'))}**。",
                "- AIME24参与：**否**；按数据集调参：**否**；轮数截断：**否**；shortlist：**否**。",
            ]
        else:
            result_title = "## 冻结策略与选择集结果"
            result_explanation = (
                f"冻结策略：`{strategy_label((winner or {}).get('candidate'))}`。`Dom`表示它在selection上是否严格同时做到"
                "Rec高于、CFP/R低于固定margin.5；test从未参与冻结。"
            )
            dominance_lines = [
                f"- selection严格支配固定margin.5：**{fmt(payload.get('selection_dominates_fixed'))}**。",
                f"- 冻结后test严格支配固定margin.5：**{fmt(payload.get('test_dominates_fixed'))}**。",
                f"- selection阶段：`{stage}`；AIME24参与选择：**否**；按数据集调参：**否**。",
            ]
        lines.extend(
            [
                result_title,
                "",
                result_explanation,
                "",
                centered_table(
                    ["策略", "Rpt", "Pre", "Rec", "F1", "CFP/R", "CFP/P", "PosFPR", "Miss", "Late"],
                    metric_rows(result_rows, stage),
                ),
                "",
                *dominance_lines,
                "",
            ]
        )

    if winner and fixed and (full_data or payload.get("test_dominates_fixed") is not None):
        if full_data:
            delta = payload.get("full_data_delta_vs_fixed", {})
            dominates_value = bool(payload.get("full_data_dominates_fixed"))
            conclusion = (
                f"该策略是声明的有限公式与参数网格中，在{dataset_count}个非AIME24数据集全体有效轮、等数据集权重目标上的描述性全局最优。"
                "同一全数据既用于选择又用于统计，因此它不是held-out泛化估计；后续可另做独立trace验证。"
            )
            comparison_label = "全数据adaptive-margin.5"
            conclusion_title = "## 全数据最优口径说明"
        else:
            delta = payload.get("test_delta_vs_fixed", {})
            dominates_value = bool(payload.get("test_dominates_fixed"))
            conclusion = (
                "冻结候选在未见test上保持了Rec上升且CFP/R下降，可作为待真实解码干预验证的自适应候选。"
                if dominates_value
                else "冻结候选未在未见test上同时保持Rec上升和CFP/R下降，因此当前部署建议仍为固定margin.5；历史策略只作为Pareto候选，不宣称全面更优。"
            )
            comparison_label = "test adaptive-margin.5"
            conclusion_title = "## 严格双目标验证结论"
        lines.extend([
            conclusion_title,
            "",
            conclusion,
            "",
            centered_table(
                ["测试", "ΔRec", "ΔCFP/R", "ΔRpt", "ΔF1", "建议"],
                [[
                    comparison_label,
                    fmt(delta.get("recall"), percent=True),
                    fmt(delta.get("correct_fp_round"), percent=True),
                    fmt(delta.get("report_rate"), percent=True),
                    fmt(delta.get("exact_f1"), percent=True),
                    "adaptive" if dominates_value else "margin.5",
                ]],
            ),
            "",
        ])

    if winner and winner.get("threshold_stats"):
        stats = winner["threshold_stats"].get(primary_stage, {})
        lines.extend(
            [
                "## 自适应阈值行为",
                "",
                "`τ均/标/10/50/90` 是各数据集逐轮动态阈值的均值、标准差和分位数；`升/降/等` 是相对固定0.5被提高、降低或保持的轮占比；`冷/就绪` 是历史不足与历史可用占比。例如τ=0.58比0.5更保守，通常少报。最后一行是各数据集等权平均。",
                "",
                centered_table(
                    ["数据集", "τ均", "τ标", "τ10", "τ50", "τ90", "升", "降", "等", "冷", "就绪"],
                    threshold_rows(stats),
                ),
                "",
            ]
        )

    if winner and winner.get(primary_stage):
        final_title = "## 全数据：各数据集与等权平均" if full_data else "## 最终 test：各数据集与等权平均"
        scope_word = "全数据" if full_data else "冻结后才读取的test"
        lines.extend(
            [
                final_title,
                "",
                f"第一张表使用{scope_word}衡量报告量和精确命中：`Rnd/Rpt/Pre/Rec/F1`分别是有效轮数、总报告率、Exact Precision、Exact Recall和Exact-F1。等权平均不按Rnd加权。",
                "",
                centered_table(
                    ["数据集", "策略", "Rnd", "Rpt", "Pre", "Rec", "F1"],
                    per_dataset_metric_rows(
                        result_rows, primary_stage,
                        [("rounds", False), ("report_rate", True), ("precision", True), ("recall", True), ("exact_f1", True)],
                    ),
                ),
                "",
                "第二张表专门拆解正确位置误报：`CFP/R`按全部轮、`CFP/P`按发生报告的轮、`PosFPR`按全部验证正确位置；`Early`是错误轮上过早报告/错误轮，`FullFP`是full-accept误报/全部轮。例如CFP/R=30%表示每100轮约30轮先报告了一个实际正确位置。",
                "",
                centered_table(
                    ["数据集", "策略", "CFP/R", "CFP/P", "PosFPR", "Early", "FullFP"],
                    per_dataset_metric_rows(
                        result_rows, primary_stage,
                        [("correct_fp_round", True), ("correct_fp_report", True), ("position_fpr", True), ("early_rate", True), ("full_false_alarm_round", True)],
                    ),
                ),
                "",
                "第三张表描述未精确命中的性质：`Miss/Late`是漏报/晚报占错误轮比例；`MAE/Bias`是已报告错误轮的位置绝对误差和有符号误差；`±1/±2`是所有错误轮内的容差召回；`P1`是首错恰在位置1时的召回。",
                "",
                centered_table(
                    ["数据集", "策略", "Miss", "Late", "MAE", "Bias", "±1", "±2", "P1"],
                    per_dataset_metric_rows(
                        result_rows, primary_stage,
                        [("miss_rate", True), ("late_rate", True), ("mae", False), ("bias", False), ("tol1_recall", True), ("tol2_recall", True), ("position1_recall", True)],
                    ),
                ),
                "",
            ]
        )

    frontier = payload.get("selection_pareto", [])
    if frontier:
        frontier_title = "## 全数据 Pareto 前沿" if full_data else "## Selection Pareto 前沿"
        lines.extend(
            [
                frontier_title,
                "",
                "前沿中的策略不存在另一策略能同时给出不低的 Rec 和不高的 CFP/R（至少一项严格更优）。`ΔRec/ΔCFP` 均相对固定 margin.5；理想方向分别为正和负。",
                "",
                centered_table(
                    ["策略", "Rec", "CFP/R", "F1", "ΔRec", "ΔCFP"],
                    [
                        [
                            strategy_label(item.get("candidate")),
                            fmt(item.get("macro", {}).get("recall"), percent=True),
                            fmt(item.get("macro", {}).get("correct_fp_round"), percent=True),
                            fmt(item.get("macro", {}).get("exact_f1"), percent=True),
                            fmt(item.get("delta_recall"), percent=True),
                            fmt(item.get("delta_correct_fp_round"), percent=True),
                        ]
                        for item in frontier[:20]
                    ],
                ),
                "",
            ]
        )

    ablations = payload.get("ablations", [])
    if ablations:
        ablation_scope = "全数据目标" if full_data else "selection合同"
        ablation_result_word = "全数据" if full_data else "冻结test"
        lines.extend(
            [
                "## 历史特征与窗口消融",
                "",
                f"每组只展示按同一{ablation_scope}选出的最好策略。`Fam/H/Agg`分别是公式族、历史轮数和聚合方式。",
                "",
                centered_table(
                    ["组", "Fam", "H", "Agg", "Rec", "CFP/R", "F1"],
                    [
                        [
                            item.get("group"),
                            item.get("candidate", {}).get("family"),
                            item.get("candidate", {}).get("history_window"),
                            item.get("candidate", {}).get("aggregation"),
                            fmt(item.get(f"{primary_stage}_macro", {}).get("recall"), percent=True),
                            fmt(item.get(f"{primary_stage}_macro", {}).get("correct_fp_round"), percent=True),
                            fmt(item.get(f"{primary_stage}_macro", {}).get("exact_f1"), percent=True),
                        ]
                        for item in ablations
                    ],
                ),
                "",
                f"下面把每个消融代表在{ablation_result_word}上的`Rec/CFP/R/F1`展开到所有非AIME24数据集；`等权平均`行再次确认小数据集与大数据集权重相同。一个组例如`H:2`表示全局选出的最佳两轮历史策略，不代表按数据集重新调参。",
                "",
                centered_table(
                    ["数据集", "组", "Rec", "CFP/R", "F1"],
                    [
                        [
                            dataset,
                            item.get("group"),
                            fmt(metric.get("recall"), percent=True),
                            fmt(metric.get("correct_fp_round"), percent=True),
                            fmt(metric.get("exact_f1"), percent=True),
                        ]
                        for item in ablations
                        for dataset, metric in [
                            *sorted(item.get(primary_stage, {}).get("by_dataset", {}).items()),
                            ("等权平均", item.get(primary_stage, {}).get("macro", item.get(f"{primary_stage}_macro", {}))),
                        ]
                        if dataset.lower().replace("_", "-") != "aime24"
                    ],
                ),
                "",
            ]
        )

    bootstrap = payload.get("paired_bootstrap", {})
    if bootstrap:
        lines.extend(
            [
                "## 配对 request bootstrap",
                "",
                ("同一bootstrap replicate对两策略使用相同request重采样；全数据模式下这是全体request分布上的内部不确定性描述，不是held-out泛化区间。"
                 if full_data else "同一bootstrap replicate对两策略使用相同request重采样；区间为数据集等权宏平均差值的95% percentile CI。")
                + "`ΔRec>0`、`ΔCFP<0`才是期望方向。",
                "",
                centered_table(
                    ["数据集", "参照", "ΔRec", "95%CI", "ΔCFP", "95%CI", "ΔF1", "95%CI"],
                    [
                        [
                            dataset,
                            name,
                            fmt(values.get("point", {}).get("recall"), percent=True),
                            ci_text(values.get("ci95", {}).get("recall")),
                            fmt(values.get("point", {}).get("correct_fp_round"), percent=True),
                            ci_text(values.get("ci95", {}).get("correct_fp_round")),
                            fmt(values.get("point", {}).get("exact_f1"), percent=True),
                            ci_text(values.get("ci95", {}).get("exact_f1")),
                        ]
                        for name, item in bootstrap.items()
                        for dataset, values in [
                            *sorted(item.get("by_dataset", {}).items()),
                            ("等权平均", {"point": item.get("point", {}), "ci95": item.get("ci95", {})}),
                        ]
                    ],
                ),
                "",
            ]
        )

    lines.extend(
        [
            "## 机器可读产物",
            "",
            ("- `analysis/strategy_search.json`：全部候选的全数据宏指标、Pareto、全局最优、逐数据集指标和bootstrap。"
             if full_data else "- `analysis/strategy_search.json`：候选、切分、Pareto、冻结策略、逐数据集指标和bootstrap。"),
            "- `Settings.json`：完整参数、命令与 trace 来源。",
            "- `summaries/*.json`：各数据集 trace 和切分摘要。",
            "",
        ]
    )
    atomic_text(run_dir / "report.md", "\n".join(lines))
