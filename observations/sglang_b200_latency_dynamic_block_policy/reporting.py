#!/usr/bin/env python3
"""Initialize state and atomically render the B200 latency experiment report."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def load(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    body = [[str(value) for value in row] for row in rows]
    return "\n".join(
        [
            "|" + "|".join(headers) + "|",
            "|" + "|".join(":---:" for _ in headers) + "|",
        ]
        + ["|" + "|".join(row) + "|" for row in body]
    )


def fmt(value: Any, percent: bool = False, digits: int = 4) -> str:
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{100*number:.2f}%" if percent else f"{number:.{digits}f}"


def init_run(run_dir: Path, settings: dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=False)
    for child in (
        "search",
        "traces/validate",
        "eval_runs/validate",
        "runtime/attempt_traces",
        "runtime/gpu_memory_guard",
    ):
        (run_dir / child).mkdir(parents=True, exist_ok=True)
    settings["created_at"] = datetime.now().astimezone().isoformat()
    atomic_text(
        run_dir / "settings.json",
        json.dumps(settings, ensure_ascii=False, indent=2) + "\n",
    )
    lines = [
        "# SGLang + B200 延迟感知动态 block 实验设置",
        "",
        f"- 创建时间：`{settings['created_at']}`；阶段：`{settings['stage']}`。",
        f"- 正式数据集：`{settings['benchmarks']}`；固定排除 `AIME24,MMLU`。",
        f"- 探索 trace：`{settings['trace_root']}`；不会重新采集探索数据。",
        f"- B200 成本文档：`{settings['cost_document']}`；虚拟并发度：`{settings['concurrencies']}`。",
        "- 每个 request 独立维护历史、逐轮独立选择 B8/B16/B32；并发度只决定延迟表与 λ，不要求并发 request 动作相同。",
        "- 目标：sample 内总接收/总 B200 forward ms；数据集内 sample 等权，再对八个数据集等权。",
        "- 首轮映射：C2/4→B32，C8/16/32→B16，C64/128→B8；后续使用一套公共信号和价值表，仅 λ_C 随并发度变化。",
        f"- 搜索：{settings['cv_folds']} 折 request OOF；bin=`{settings['signal_bins']}`；λ 上限=`{settings['lambda_max']}`；full 门候选=`{settings['full_gate_grid']}`。",
        f"- GPU：`{settings['gpu_devices']}`；CPU 搜索预占=`{settings['search_gpu_hold_gb']}` GiB/卡；chunk=`{settings['search_gpu_hold_chunk_gb']}` GiB。",
        f"- 模型：`{settings['model']}`；mode=`{settings['mode']}`；context=`{settings['context_length']}`；tokens=`{settings['tokens']}`。",
        f"- 模型期 mem fraction=`{settings['mem_fraction']}`；额外预留=`{settings['gpu_memory_reserve_gb']}` GiB；dtype=`{settings['dtype']}`。",
        f"- SGLang 验证并发：每个 C 默认同时设 max-running-requests=C、client-concurrency=C；端口 `{settings['port'] or '自动'}/{settings['proxy_port'] or '自动'}`。",
        f"- 原始单行命令：`{settings['command']}`。",
    ]
    atomic_text(run_dir / "settings.md", "\n".join(lines) + "\n")
    atomic_text(
        run_dir / "run_state.json",
        json.dumps({"events": []}, ensure_ascii=False, indent=2) + "\n",
    )
    render(run_dir)


def add_event(run_dir: Path, event: dict[str, Any]) -> None:
    state = load(run_dir / "run_state.json") or {"events": []}
    event["updated_at"] = datetime.now().astimezone().isoformat()
    state["events"].append(event)
    atomic_text(
        run_dir / "run_state.json",
        json.dumps(state, ensure_ascii=False, indent=2) + "\n",
    )
    render(run_dir)


def progress_documents(
    run_dir: Path, settings: Mapping[str, Any], state: Mapping[str, Any]
) -> tuple[str, Mapping[str, Any]]:
    events = state.get("events") or []
    latest: dict[tuple[str, str], Mapping[str, Any]] = {}
    for event in events:
        latest[(str(event.get("phase")), str(event.get("dataset")))] = event
    search = load(run_dir / "search/progress.json") or {}
    concurrencies = [
        int(value) for value in str(settings.get("concurrencies", "")).split(",") if value
    ]
    benchmarks = [
        value.split(":", 1)[0]
        for value in str(settings.get("benchmarks", "")).split(",")
        if value
    ]
    total_validation = len(concurrencies) * len(benchmarks)
    completed_validation = sum(
        1
        for concurrency in concurrencies
        for benchmark in benchmarks
        if latest.get(("validate", f"C{concurrency}/{benchmark}"), {}).get("status")
        in {"completed", "skipped"}
    )
    search_result = load(run_dir / "search/search_results.json")
    search_done = search_result is not None
    candidate_total = int(
        ((search_result or {}).get("protocol") or {})
        .get("candidate_attempts", {})
        .get("total", 0)
    ) or (
        len((search_result or {}).get("tier1_ranking") or [])
        + len((search_result or {}).get("tier2_ranking") or [])
    )
    rows = [
        ["初始化/成本审计", "1", "1", "完成"],
        [
            "离线信号搜索",
            candidate_total if search_done else search.get("completed", 0),
            candidate_total if search_done else search.get("total", 1),
            "search_completed" if search_done else search.get("stage", "等待"),
        ],
        [
            "并发度×数据集冻结验证",
            completed_validation,
            total_validation,
            "完成" if total_validation and completed_validation == total_validation else "进行/等待",
        ],
        ["最终汇总", sum((run_dir / "search" / f"validation_c{c}.json").exists() for c in concurrencies), len(concurrencies), "随验证增量更新"],
    ]
    lines = [
        "# 实验实时进度",
        "",
        f"更新时间：`{datetime.now().astimezone().isoformat()}`。本文件和 `report.md` 均由原子替换实时更新。",
        "",
        "`阶段`表示初始化、CPU 搜索或 GPU 验证；`已完成/总数`是可恢复任务计数，`状态`说明是否仍在等待。例如 `3/56` 表示七个并发度乘八个数据集的 56 项验证中已有 3 项提交成功。",
        "",
        table(["阶段", "已完成", "总数", "状态"], rows),
        "",
        "最近事件表中，`对象`是 GPU、全局搜索或 `C/数据集`，`记录`是已提交 trace 行数，`说明`给出重试/完成原因。例如对象 `C8/gsm8k`、状态 `completed`、记录 `1200` 表示该项已原子提交 1200 个 decode 轮记录。",
        "",
        table(
            ["阶段", "对象", "状态", "记录", "说明", "时间"],
            [
                [
                    row.get("phase", "—"),
                    row.get("dataset", "—"),
                    row.get("status", "—"),
                    row.get("records", "—"),
                    row.get("message", "—"),
                    row.get("updated_at", "—"),
                ]
                for row in events[-30:]
            ]
            or [["初始化", "—", "等待", "0", "尚无事件", "—"]],
        ),
        "",
    ]
    payload = {
        "updated_at": datetime.now().astimezone().isoformat(),
        "phases": rows,
        "search": search,
        "validation_completed": completed_validation,
        "validation_total": total_validation,
        "latest_events": events[-30:],
    }
    return "\n".join(lines), payload


def quality_table(quality: Mapping[str, Any]) -> str:
    rows = []
    for dataset, metric in (quality.get("by_dataset") or {}).items():
        rows.append(
            [
                dataset,
                metric.get("rows_original", 0),
                metric.get("rows_usable", 0),
                metric.get("rows_excluded", 0),
                metric.get("replay_mismatch", 0),
                metric.get("cross_block_mismatch", 0),
            ]
        )
    return table(["集", "原始轮", "可用轮", "排除轮", "提交异", "跨B异"], rows)


def summary_rows(summary: Mapping[str, Any]) -> list[list[Any]]:
    rows = []
    for dataset, metric in (summary.get("datasets") or {}).items():
        rows.append(
            [
                dataset,
                metric.get("samples", 0),
                metric.get("rounds", 0),
                fmt(metric.get("score_token_ms_req")),
                fmt(metric.get("pure_forward_token_ms_batch")),
                fmt(metric.get("decode_tpf")),
                fmt(metric.get("mean_forward_ms")),
                fmt(metric.get("l8_rate"), True),
                fmt(metric.get("l16_rate"), True),
                fmt(metric.get("l32_rate"), True),
                metric.get("tail_samples", 0),
            ]
        )
    macro = summary.get("macro") or {}
    if macro:
        rows.append(
            [
                "均(集等权)",
                macro.get("samples", 0),
                macro.get("rounds", 0),
                fmt(macro.get("score_token_ms_req")),
                fmt(macro.get("pure_forward_token_ms_batch")),
                fmt(macro.get("decode_tpf")),
                fmt(macro.get("mean_forward_ms")),
                fmt(macro.get("l8_rate"), True),
                fmt(macro.get("l16_rate"), True),
                fmt(macro.get("l32_rate"), True),
                "—",
            ]
        )
    return rows


def comparison_rows(
    summary: Mapping[str, Any], baselines: Mapping[str, Any]
) -> list[list[Any]]:
    rows = []
    for dataset, metric in (summary.get("datasets") or {}).items():
        fixed = {
            block: ((baselines.get(block) or {}).get("datasets") or {}).get(dataset, {}).get("score_token_ms_req")
            for block in ("8", "16", "32")
        }
        usable = {block: value for block, value in fixed.items() if value is not None}
        best_block, best_score = max(usable.items(), key=lambda item: (item[1], -int(item[0])))
        score = metric.get("score_token_ms_req")
        rows.append(
            [
                dataset,
                metric.get("samples", 0),
                fmt(score),
                fmt(fixed["8"]),
                fmt(fixed["16"]),
                fmt(fixed["32"]),
                f"B{best_block}",
                fmt(score / best_score - 1.0, True),
            ]
        )
    macro = summary.get("macro") or {}
    fixed_macro = {
        block: (baselines.get(block) or {}).get("macro", {}).get("score_token_ms_req")
        for block in ("8", "16", "32")
    }
    usable = {block: value for block, value in fixed_macro.items() if value is not None}
    if macro and usable:
        best_block, best_score = max(usable.items(), key=lambda item: (item[1], -int(item[0])))
        score = macro.get("score_token_ms_req")
        rows.append(
            [
                "均(集等权)",
                macro.get("samples", 0),
                fmt(score),
                fmt(fixed_macro["8"]),
                fmt(fixed_macro["16"]),
                fmt(fixed_macro["32"]),
                f"B{best_block}",
                fmt(score / best_score - 1.0, True),
            ]
        )
    return rows


def offline_sections(result: Mapping[str, Any]) -> list[str]:
    selection = result.get("selection") or {}
    selected = selection.get("selected") or {}
    lines = [
        "## 离线搜索结果",
        "",
        "离线搜索完成即生成本节，不等待 GPU 验证。`策略`区分最终选择及两个复杂度各自最优项；`信号数`是历史标量数，`信号`是变量名，`full门`是可选连续整块通过门槛；`均增`是当前全部指定并发度下、相对各自最佳固定 B8/B16/B32 的提升再等权平均，`最差C增`是其中最低值。正式协议包含七个并发度。例如 `信号数=2、full门=2、均增=3%` 表示使用一个接收标量加“连续整块通过至少2轮”，且七个 C 的平均离线提升为 3%。",
        "",
        table(
            ["策略", "信号数", "信号", "full门", "均增", "最差C增"],
            [
                [
                    "最终",
                    selected.get("tier", "—"),
                    selected.get("signal", "—"),
                    selected.get("gate") if selected.get("gate") is not None else "—",
                    fmt(selected.get("mean_relative_gain"), True),
                    fmt(selected.get("min_relative_gain"), True),
                ],
                [
                    "最佳单信号",
                    (selection.get("best_single") or {}).get("tier", "—"),
                    (selection.get("best_single") or {}).get("signal", "—"),
                    "—",
                    fmt((selection.get("best_single") or {}).get("mean_relative_gain"), True),
                    fmt((selection.get("best_single") or {}).get("min_relative_gain"), True),
                ],
                [
                    "最佳双信号",
                    (selection.get("best_dual") or {}).get("tier", "—"),
                    (selection.get("best_dual") or {}).get("signal", "—"),
                    (selection.get("best_dual") or {}).get("gate", "—"),
                    fmt((selection.get("best_dual") or {}).get("mean_relative_gain"), True),
                    fmt((selection.get("best_dual") or {}).get("min_relative_gain"), True),
                ],
            ],
        ),
        "",
        "`C`是虚拟并发度，`首B`是该 C 每个 request 第一轮使用的 block；`λC/λ区间`是该 C 唯一允许变化的超参及保持后续动作不变的区间。首轮遵循 C2/4→B32、C8/16/32→B16、C64/128→B8，之后选择使“预测接收长度 − λC × B200两次forward毫秒数”最大的 block。`分数`为 token/ms/request，`Best/Best分`是最佳固定块及其分数，`增Best`为相对提升；`B8分/B16分/B32分`是固定 baseline，`批吞吐`为按有效 C 放大的纯 forward token/ms，`TPF`为每次模型 forward 产出的 decode token，末三列是 sample 粒度 block 占比。例如 `C=64、首B=B8、Best=B16、增Best=4%` 表示每个 request 从 B8 起步，动态策略最终比全程固定 B16 高 4%。",
        "",
    ]
    profile_rows = []
    offline = result.get("offline_selected") or {}
    fixed = result.get("fixed_baselines") or {}
    for concurrency, row in offline.items():
        macro = (row.get("summary") or {}).get("macro") or {}
        baseline_scores = row.get("baseline_scores") or {}
        profile_rows.append(
            [
                concurrency,
                f"B{row.get('cold_start_block', '—')}",
                fmt(row.get("lambda"), digits=6),
                f"[{fmt(row.get('interval_lo'),digits=6)},{fmt(row.get('interval_hi'),digits=6)}]",
                fmt(row.get("score")),
                f"B{row.get('best_fixed_block','—')}",
                fmt(row.get("best_fixed_score")),
                fmt(row.get("relative_gain_vs_best_fixed"), True),
                fmt(baseline_scores.get("8")),
                fmt(baseline_scores.get("16")),
                fmt(baseline_scores.get("32")),
                fmt(macro.get("pure_forward_token_ms_batch")),
                fmt(macro.get("decode_tpf")),
                fmt(macro.get("l8_rate"), True),
                fmt(macro.get("l16_rate"), True),
                fmt(macro.get("l32_rate"), True),
            ]
        )
    lines += [
        table(
            ["C", "首B", "λC", "λ区间", "分数", "Best", "Best分", "增Best", "B8分", "B16分", "B32分", "批吞吐", "TPF", "B8%", "B16%", "B32%"],
            profile_rows,
        ),
        "",
    ]
    for concurrency, row in offline.items():
        baseline = fixed.get(concurrency) or {}
        lines += [
            f"### 离线 C={concurrency}",
            "",
            "第一张表中 `动态分`和 B8/B16/B32 固定分数都采用完全相同的 sample→数据集两级等权口径；`Best`逐数据集取三个固定块中的最高值，`增Best=动态分/Best分−1`。例如 3% 表示该数据集每 request 每毫秒理论接收量比其最佳固定块高 3%。",
            "",
            table(
                ["集", "样本", "动态分", "B8分", "B16分", "B32分", "Best", "增Best"],
                comparison_rows(row.get("summary") or {}, baseline),
            ),
            "",
            "第二张表描述动态策略本身。`集/样本/轮`给出数据集、有效 request 数和 decode 轮数；`Req吞吐`是 token/ms/request，`批吞吐`按有效虚拟 C 放大，`TPF`是 decode token/forward，`轮时ms`是 sample 内逐轮 B200 两次 forward 理论成本均值；三种占比先在 sample 内算，`尾样`是因样本数不能整除 C 而按 T_r(B) 计费的 request 数。例如 C=8 时样本数 10，最后 2 个 sample 的各轮使用 T2(B)，所以尾样为 2。",
            "",
            table(["集", "样本", "轮", "Req吞吐", "批吞吐", "TPF", "轮时ms", "B8%", "B16%", "B32%", "尾样"], summary_rows(row.get("summary") or {})),
            "",
        ]
    quality = (result.get("protocol") or {}).get("trace_quality") or {}
    if quality:
        lines += [
            "### 离线 trace 完整性",
            "",
            "探索 trace 要求 canonical 回放一致且不同 block 的共同接收前缀一致。`原始轮/可用轮/排除轮`是审计前、后和排除数量；`提交异`是 canonical 回放不一致轮，`跨B异`是三分支共同接收前缀不一致轮。例如原始 100、提交异 1、跨B异 2 且不重叠时，可用轮为 97。",
            "",
            quality_table(quality),
            "",
        ]
    ranking = (result.get("tier1_ranking") or [])[:10] + (result.get("tier2_ranking") or [])[:10]
    lines += [
        "### 候选策略摘要",
        "",
        "这里保留单/双信号各自排名靠前候选，便于复核最终信号不是按某个单一数据集选择。`层`是单/双信号层，`信号/full门/均增/最差C增`含义同上，`C→λ`压缩展示每个并发度冻结的唯一超参。例如 `2:0.18,8:0.09` 表示 C2 使用 λ=0.18、C8 使用 λ=0.09，其余信号和价值表完全相同。",
        "",
        table(
            ["层", "信号", "full门", "均增", "最差C增", "C→λ"],
            [
                [
                    row.get("tier", "—"),
                    row.get("signal", "—"),
                    row.get("gate") if row.get("gate") is not None else "—",
                    fmt(row.get("mean_relative_gain"), True),
                    fmt(row.get("min_relative_gain"), True),
                    ",".join(f"{c}:{fmt(v.get('lambda'),digits=4)}" for c, v in (row.get("profiles") or {}).items()),
                ]
                for row in ranking
            ],
        ),
        "",
    ]
    return lines


def validation_sections(run_dir: Path, settings: Mapping[str, Any]) -> list[str]:
    values = [
        int(value) for value in str(settings.get("concurrencies", "")).split(",") if value
    ]
    available = [
        (concurrency, load(run_dir / "search" / f"validation_c{concurrency}.json"))
        for concurrency in values
        if (run_dir / "search" / f"validation_c{concurrency}.json").exists()
    ]
    lines = [
        "## SGLang 冻结动态轨迹验证",
        "",
        "每个 request 根据自身历史逐轮独立决策；首轮按 C2/4→B32、C8/16/32→B16、C64/128→B8，后续 C 仅选择 λC 和 B200 成本表。这里的真实 SGLang TPF 不含 prefill，shadow 额外分支不计入 canonical decode forward；理论吞吐仍只使用 B200 文档里的纯 forward 时间，不是本机实测 TPS。",
        "",
    ]
    if not available:
        return lines + ["尚未完成任何并发度的八集冻结验证。", ""]
    summary_rows_all = []
    for concurrency, result in available:
        macro = ((result or {}).get("dynamic_policy") or {}).get("macro") or {}
        official = ((result or {}).get("official_sglang_metrics") or {}).get("macro") or {}
        summary_rows_all.append(
            [
                concurrency,
                f"B{((result or {}).get('protocol') or {}).get('cold_start_block', '—')}",
                fmt(macro.get("score_token_ms_req")),
                fmt(macro.get("pure_forward_token_ms_batch")),
                fmt(macro.get("decode_tpf")),
                fmt(official.get("decode_tpf")),
                fmt(macro.get("relative_gain_vs_best_same_state_fixed"), True),
                fmt(macro.get("l8_rate"), True),
                fmt(macro.get("l16_rate"), True),
                fmt(macro.get("l32_rate"), True),
                ((result or {}).get("protocol") or {}).get("policy_replay", {}).get("mismatches", "—"),
            ]
        )
    lines += [
        "`C`是验证并发度，`首B`是该 C 的首轮 block；`Req吞吐/批吞吐`是 B200 理论单 request/整批纯 forward 吞吐；`TraceTPF`由 committed trace 的接收长度/两次 decode forward 得到，`官方TPF`读取 pipeline 的 decode-only 汇总；`同态增`只是在动态轨迹相同状态上读取三个 shadow 分支形成的诊断，不等同于独立固定块 baseline；三种百分比为 sample 粒度动作占比，`回放异`是冻结策略与实际动作不一致数。例如 `C=2、首B=B32、回放异=0` 表示每个 request 首轮都采用 B32，之后也全部按冻结策略执行。",
        "",
        table(["C", "首B", "Req吞吐", "批吞吐", "TraceTPF", "官方TPF", "同态增", "B8%", "B16%", "B32%", "回放异"], summary_rows_all),
        "",
    ]
    for concurrency, result in available:
        dynamic = (result or {}).get("dynamic_policy") or {}
        baseline = (result or {}).get("fixed_baselines_same_dynamic_state") or {}
        lines += [
            f"### 在线 C={concurrency}",
            "",
            "第一张表比较动态策略与同一动态状态上的三个 shadow 固定动作；它用于定位动作质量，但不是三条独立固定块完整轨迹。`集/样本`是数据集及 request 数，`动态分`和三个固定分都是 token/ms/request，`Best`是同态最佳固定动作，`增Best`为相对值。例如动态分 0.52、Best分 0.50 时增Best为 4%。所有分数仍按 sample→数据集两级聚合。",
            "",
            table(["集", "样本", "动态分", "B8分", "B16分", "B32分", "Best", "增Best"], comparison_rows(dynamic, baseline)),
            "",
            "第二张表给出在线动态轨迹的 TPF、理论吞吐与 block 分布；各列与离线动态明细表相同。例如 C=16、样本数 18 时最后 2 个 sample 使用 T2(B)，尾样列为 2。在线轨迹会改变后续历史分布，因此这是判断离线策略是否外推成功的主要依据。",
            "",
            table(["集", "样本", "轮", "Req吞吐", "批吞吐", "TPF", "轮时ms", "B8%", "B16%", "B32%", "尾样"], summary_rows(dynamic)),
            "",
        ]
    return lines


def render(run_dir: Path) -> None:
    settings = load(run_dir / "settings.json") or {}
    state = load(run_dir / "run_state.json") or {"events": []}
    progress_text, progress_json = progress_documents(run_dir, settings, state)
    atomic_text(run_dir / "progress.md", progress_text)
    atomic_text(
        run_dir / "progress.json",
        json.dumps(progress_json, ensure_ascii=False, indent=2) + "\n",
    )
    lines = [
        "# SGLang + B200 延迟感知动态 block 策略实验报告",
        "",
        f"更新时间：`{datetime.now().astimezone().isoformat()}`。报告从运行目录创建时即存在，并随搜索和每个数据集完成而原子更新。",
        "",
        "## 口径",
        "",
        "正式范围固定为 GSM8K、HumanEval、MBPP、MATH-500、AIME25、GPQA、IFEval、LiveCodeBench-C++，排除 AIME24/MMLU。每个 sample 独立决策；八个数据集严格等权。并发度是成本和策略条件，不是分桶调度模拟；首轮 block 映射为 C2/4→B32、C8/16/32→B16、C64/128→B8。",
        "",
    ]
    costs = load(run_dir / "search/b200_latency_costs.json")
    if costs:
        lines += [
            "## B200 成本快照",
            "",
            "`C`是虚拟满并发 request 数；`T8/T16/T32`分别是该 C 下 B8/B16/B32 一次 draft 加一次 causal verify 的 CUDA Graph replay 两次之和，单位 ms。例如 C=2、T8=10.955210 表示两个 request 采用 B8 时一轮两次 forward 的理论成本为 10.955210 ms。该成本不含 prefill、调度和分桶等待；源文档哈希随结果冻结，避免搜索与验证偷换成本。",
            "",
            table(
                ["C", "T8", "T16", "T32"],
                [
                    [c, fmt(row.get("8"), digits=6), fmt(row.get("16"), digits=6), fmt(row.get("32"), digits=6)]
                    for c, row in costs.get("selected_costs_ms", {}).items()
                ],
            ),
            "",
        ]
    search = load(run_dir / "search/search_results.json")
    if search:
        lines += offline_sections(search)
    else:
        current = load(run_dir / "search/progress.json") or {}
        lines += [
            "## 离线搜索结果",
            "",
            f"尚未完成。当前阶段 `{current.get('stage','等待')}`，进度 `{current.get('completed',0)}/{current.get('total','—')}`。完成后本节会立即给出相对固定 B8/B16/B32 的结果，不等待 GPU。",
            "",
        ]
    lines += validation_sections(run_dir, settings)
    lines += [
        "## 解释边界与后续校准",
        "",
        "离线搜索来自既有三分支探索轨迹；在线验证会改变 block 历史与信号分布。若在线结果明显偏离离线结果，可复用在线 trace 的三个 shadow 分支，在保持同一信号形式和七个 λC 结构不变的前提下重估价值表/λ；调参数据与最终报告数据必须再次分开。真实 serving 的分桶等待与桶利用率仍需另做系统实验。",
        "",
    ]
    atomic_text(run_dir / "report.md", "\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init")
    init.add_argument("--run-dir", type=Path, required=True)
    init.add_argument("--settings-json", required=True)
    event = sub.add_parser("event")
    event.add_argument("--run-dir", type=Path, required=True)
    event.add_argument("--event-json", required=True)
    render_parser = sub.add_parser("render")
    render_parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "init":
        init_run(args.run_dir, json.loads(args.settings_json))
    elif args.command == "event":
        add_event(args.run_dir, json.loads(args.event_json))
    else:
        render(args.run_dir)


if __name__ == "__main__":
    main()
