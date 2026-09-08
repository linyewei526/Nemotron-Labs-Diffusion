#!/usr/bin/env python3
"""Initialize state and atomically render the dual-profile Chinese report."""

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
        "traces/validate/s8",
        "traces/validate/s16",
        "eval_runs/validate/s8",
        "eval_runs/validate/s16",
        "runtime/attempt_traces/s8",
        "runtime/attempt_traces/s16",
        "runtime/gpu_memory_guard",
    ):
        (run_dir / child).mkdir(parents=True, exist_ok=True)
    settings["created_at"] = datetime.now().astimezone().isoformat()
    atomic_text(
        run_dir / "settings.json",
        json.dumps(settings, ensure_ascii=False, indent=2) + "\n",
    )
    lines = [
        "# SGLang 双动作空间接收/计算比实验设置",
        "",
        f"- 创建时间：`{settings['created_at']}`。",
        f"- 阶段：`{settings['stage']}`；正式数据集：`{settings['benchmarks']}`。",
        "- 排除：`AIME24,MMLU`；正式搜索和验证必须恰好包含其余八集。",
        f"- 探索 trace：`{settings['trace_root']}`；默认直接复用已有 SGLang+NeMo-Skills 三分支 trace。",
        f"- S8：首轮 L8，后续动作 `8,16,32`；验证 λ=`{settings['policy_lambda_s8']}`。",
        f"- S16：首轮 L16，后续动作 `16,32`；验证 λ=`{settings['policy_lambda_s16']}`。",
        f"- 搜索：{settings['cv_folds']} 折 prompt OOF，{settings['signal_bins']} 个 survival bin，精确 λ 范围 `[0,{settings['lambda_max']}]`。",
        f"- 双信号最低共同改善：`{settings['min_dual_improvement']}`；full 门：`{settings['full_gate_grid']}`。",
        "- 目标：每集先算接收/计算比，再对八集算术平均；集内 request 等权、request 内轮次等权。",
        f"- 模型：`{settings['model']}`；模式：`{settings['mode']}`；LoRA mode：`{settings['lora_mode']}`。",
        f"- GPU：`{settings['gpu_devices']}`；TP：`{settings['tp_size']}`；batch：`{settings['batch_size']}`；client concurrency：`{settings['client_concurrency']}`。",
        f"- CPU 搜索显存守护：每卡 `{settings['search_gpu_hold_gb']}` GiB，chunk=`{settings['search_gpu_hold_chunk_gb']}` GiB；验证前释放。",
        f"- SGLang：context=`{settings['context_length']}`，tokens=`{settings['tokens']}`，dtype=`{settings['dtype']}`，mem fraction=`{settings['mem_fraction']}`，模型期额外预留=`{settings['gpu_memory_reserve_gb']}` GiB。",
        f"- 端口：server=`{settings['port'] or '自动搜索'}`，proxy=`{settings['proxy_port'] or '自动搜索'}`。",
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


def quality_rows(quality: Mapping[str, Any]) -> list[list[Any]]:
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
                fmt(metric.get("excluded_rate"), True),
            ]
        )
    rows.append(
        [
            "总计",
            quality.get("rows_original", 0),
            quality.get("rows_usable", 0),
            quality.get("rows_excluded", 0),
            quality.get("replay_mismatch", 0),
            quality.get("cross_block_mismatch", 0),
            fmt(quality.get("excluded_rate"), True),
        ]
    )
    return rows


def metric_rows(summary: Mapping[str, Any]) -> list[list[Any]]:
    rows = []
    for dataset, metric in (summary.get("datasets") or {}).items():
        rows.append(
            [
                dataset,
                metric.get("requests", 0),
                metric.get("rounds", 0),
                fmt(metric.get("mean_block")),
                fmt(metric.get("mean_accept")),
                fmt(metric.get("decode_tpf_logical")),
                fmt(metric.get("accept_per_block_token")),
                fmt(metric.get("loss_vs_l32")),
                fmt(metric.get("mean_regret")),
                fmt(metric.get("oracle_action_accuracy"), True),
                fmt(metric.get("l8_rate"), True),
                fmt(metric.get("l16_rate"), True),
                fmt(metric.get("l32_rate"), True),
            ]
        )
    macro = summary.get("macro")
    if macro:
        rows.append(
            [
                "均(集等权)",
                macro.get("requests", 0),
                macro.get("rounds", 0),
                fmt(macro.get("mean_block")),
                fmt(macro.get("mean_accept")),
                fmt(macro.get("decode_tpf_logical")),
                fmt(macro.get("accept_per_block_token")),
                fmt(macro.get("loss_vs_l32")),
                fmt(macro.get("mean_regret")),
                fmt(macro.get("oracle_action_accuracy"), True),
                fmt(macro.get("l8_rate"), True),
                fmt(macro.get("l16_rate"), True),
                fmt(macro.get("l32_rate"), True),
            ]
        )
    return rows


def risk_rows(summary: Mapping[str, Any]) -> list[list[Any]]:
    macro = summary.get("macro") or {}
    return [
        [
            "均(集等权)",
            fmt(macro.get("large_rate"), True),
            fmt(macro.get("large_precision"), True),
            fmt(macro.get("large_waste_rate"), True),
            fmt(macro.get("trunc_gt1"), True),
            fmt(macro.get("trunc_gt2"), True),
            fmt(macro.get("trunc_gt4"), True),
            fmt(macro.get("p90_regret")),
            fmt(macro.get("p95_regret")),
            fmt(macro.get("compute_delta_vs_baseline")),
        ]
    ]


def transition_rows(transitions: Mapping[str, Any]) -> list[list[Any]]:
    rows = []

    def one(name: str, metric: Mapping[str, Any], requests: Any) -> list[Any]:
        matrix = metric.get("matrix") or [[0.0] * 3 for _ in range(3)]
        return [
            name,
            requests,
            fmt(metric.get("switch_rate"), True),
            fmt(metric.get("oscillation_rate"), True),
            *[fmt(matrix[left][right], True) for left in range(3) for right in range(3)],
        ]

    for dataset, metric in (transitions.get("datasets") or {}).items():
        rows.append(one(dataset, metric, metric.get("requests_with_transitions", 0)))
    if transitions.get("macro"):
        rows.append(one("均(集等权)", transitions["macro"], "—"))
    return rows


def profile_sections(profile: str, result: Mapping[str, Any], title: str) -> list[str]:
    summary = result.get("summary") or result.get("dynamic_policy") or {}
    macro = summary.get("macro") or {}
    allowed = result.get("allowed_blocks") or macro.get("allowed_blocks") or []
    baseline_name = "固定L8" if profile == "s8" else "固定L16"
    lines = [
        f"### {title}",
        "",
        f"该档首轮固定 `L{result.get('cold_start_block', macro.get('cold_start_block', '—'))}`，后续只在 `{allowed}` 中选择。`效率增`是相对 {baseline_name} 的八集等权接/算比相对提升；例如 5% 表示每个 block token 平均多产生约 5% 接收 token。",
        "",
        table(
            ["λ", "稳定区间", "接/算", "基线接/算", "效率增", "块均", "接均", "TPF逻"],
            [[
                fmt(result.get("lambda", (result.get("protocol") or {}).get("policy_lambda"))),
                str(result.get("lambda_interval", "—")),
                fmt(macro.get("accept_per_block_token", result.get("efficiency"))),
                fmt(macro.get("baseline_efficiency", result.get("baseline_efficiency"))),
                fmt(macro.get("relative_efficiency_gain", result.get("relative_efficiency_gain")), True),
                fmt(macro.get("mean_block")),
                fmt(macro.get("mean_accept")),
                fmt(macro.get("decode_tpf_logical")),
            ]],
        ),
        "",
        "逐集表中 `Reg` 和 `准Or` 均相对该档受限动作空间的同状态 oracle；S16 的 oracle 不允许选择 L8。`损32`是相比同轮 L32 少接收的 token。",
        "",
        table(
            ["集", "Req", "轮", "块均", "接均", "TPF逻", "接/算", "损32", "Reg", "准Or", "L8%", "L16%", "L32%"],
            metric_rows(summary),
        ),
        "",
        "风险表中，S8 的大块是 L16/L32，S16 的大块仅为 L32；`大精`分别要求增益达到配置的 2/4 token，`大浪`表示增益不超过 1。`算差`为平均 block 相对本档固定基线的变化，正数表示多算。",
        "",
        table(
            ["范围", "大块%", "大精", "大浪", "截>1", "截>2", "截>4", "Reg90", "Reg95", "算差"],
            risk_rows(summary),
        ),
        "",
    ]
    transitions = summary.get("transitions") or {}
    if transitions:
        lines += [
            "转移表只统计同一 request 的相邻轮次。`切换`表示下一轮 block 与上一轮不同；`振荡`表示三轮形成 A→B→A。`8→16` 等九列是转移占比，例如 8→16=20% 表示有效相邻轮中有 20% 从 L8 升到 L16。S16 不允许 L8，因此含 L8 的格应为 0。",
            "",
            table(
                ["集", "Req转", "切换", "振荡", "8→8", "8→16", "8→32", "16→8", "16→16", "16→32", "32→8", "32→16", "32→32"],
                transition_rows(transitions),
            ),
            "",
        ]
    diagnostics = result.get("diagnostic_grid") or []
    if diagnostics:
        lines += [
            "下表是便于人工控制倾向的诊断 λ 网格，不参与替代精确交点搜索。λ 越大越偏向该档较小的动作。",
            "",
            table(
                ["λ", "块均", "接均", "TPF逻", "接/算", "Reg", "L8%", "L16%", "L32%"],
                [[
                    fmt(row.get("lambda")),
                    fmt(row.get("mean_block")),
                    fmt(row.get("mean_accept")),
                    fmt(row.get("decode_tpf_logical")),
                    fmt(row.get("accept_per_block_token")),
                    fmt(row.get("mean_regret")),
                    fmt(row.get("l8_rate"), True),
                    fmt(row.get("l16_rate"), True),
                    fmt(row.get("l32_rate"), True),
                ] for row in diagnostics],
            ),
            "",
        ]
    return lines


def render(run_dir: Path) -> None:
    settings = load(run_dir / "settings.json") or {}
    state = load(run_dir / "run_state.json") or {"events": []}
    progress = load(run_dir / "search/progress.json")
    search = load(run_dir / "search/search_results.json")
    validation_s8 = load(run_dir / "search/validation_s8.json")
    validation_s16 = load(run_dir / "search/validation_s16.json")
    lines = [
        "# SGLang 双动作空间接收/计算比策略实验报告",
        "",
        "> 报告从时间戳目录建立时即生成，并在搜索候选、显存守护和每个冻结验证数据集完成后原子更新。所有宏结果固定排除 AIME24/MMLU，并让其余八集等权。",
        "",
        "## 1. 实时进度",
        "",
        "`对象`会明确区分 S8/S16；搜索候选进度不是数据集生成进度。",
        "",
        table(
            ["阶段", "对象", "状态", "轮/数", "说明"],
            [[
                event.get("phase", "—"),
                event.get("dataset", "—"),
                event.get("status", "—"),
                event.get("records", "—"),
                event.get("message", "—"),
            ] for event in state.get("events", [])]
            or [["初始化", "—", "等待", "0", "尚未开始"]],
        ),
        "",
    ]
    if progress:
        latest = progress.get("latest") or progress.get("selected") or {}
        lines += [
            "离线进度每完成一个信号候选即更新；候选内部会完成 S8/S16 两套精确 λ 搜索。",
            "",
            table(
                ["阶段", "完成", "总数", "最新/最终"],
                [[
                    progress.get("stage", "—"),
                    progress.get("completed", 0),
                    progress.get("total", 0),
                    json.dumps(latest, ensure_ascii=False, separators=(",", ":")),
                ]],
            ),
            "",
        ]
    lines += [
        "## 2. 方法和变量",
        "",
        "S8 首轮 L8、后续动作 {8,16,32}；S16 首轮 L16、后续动作 {16,32}。两档共享信号和 V 表，仅 λ 不同。",
        "",
        table(
            ["变量", "含义", "例子"],
            [
                ["s", "唯一主历史信号；越大表示能力越强", "-entropy 越接近0表示越确定"],
                ["V8/16/32", "给定s的条件期望接收长度", "V16=10表示预计接收10"],
                ["λ8/λ16", "各动作空间的每block-token惩罚", "λ越大越倾向较小动作"],
                ["U(L)", "V(L)-λ×L，取允许动作中的最大者", "S16只比较U16/U32"],
                ["接/算", "每集平均接收/平均block，再八集等权", "接收8、block10则0.8"],
                ["效率增", "动态接/算相对固定基线的比例", "0.05表示提升5%"],
                ["Reg", "受限同状态oracle效用减策略真实效用", "Reg=0表示该轮效用最优"],
                ["TPF逻", "平均接收/2，仅计canonical draft+verify次数", "接收10则TPF逻=5"],
            ],
        ),
        "",
        "## 3. Trace 完整性",
        "",
    ]
    quality = (search or {}).get("protocol", {}).get("trace_quality")
    if quality:
        lines += [
            "`原始/可用/排除`仅统计八个正式数据集；MMLU 即使存在于复用目录也不会进入拟合、λ 搜索或均值。回放或三分支公共前缀不一致的旧 trace 行会在保护线内显式排除。",
            "",
            table(
                ["集", "原始", "可用", "排除", "回放异", "跨块异", "排除%"],
                quality_rows(quality),
            ),
            "",
        ]
    else:
        lines += ["等待八集 trace 审计。", ""]

    lines += ["## 4. 八集全局离线搜索", ""]
    if not search:
        lines += ["等待公共信号与两套 λ 搜索完成。", ""]
    else:
        selection = search["selection"]
        rankings = search.get("tier1_ranking", []) + search.get("tier2_ranking", [])
        lines += [
            "每行信号先分别求 S8/S16 的最高接/算比。`最差增`是两档相对收益较小者，主排序最大化该值；`平均增`用于打破并列。这样不会选择只对一个动作空间有效的公共信号。",
            "",
            table(
                ["层", "信号", "full门", "最差增", "平均增", "λ8", "增8", "λ16", "增16"],
                [[
                    row.get("tier"),
                    row.get("signal"),
                    row.get("gate") if row.get("gate") is not None else "—",
                    fmt(row.get("joint_min_gain"), True),
                    fmt(row.get("joint_mean_gain"), True),
                    fmt(row.get("s8_lambda")),
                    fmt(row.get("s8_gain"), True),
                    fmt(row.get("s16_lambda")),
                    fmt(row.get("s16_gain"), True),
                ] for row in rankings],
            ),
            "",
            "双信号仅允许一个接收类主信号加 full-streak 分层；必须同时改善两档，且最差收益改善超过门槛才会增加第二信号。",
            "",
            table(
                ["最终信号", "层", "full门", "双改", "须双升", "最小改", "选取规则"],
                [[
                    selection["selected"]["signal"],
                    selection["selected"]["tier"],
                    selection["selected"].get("gate") or "—",
                    fmt(selection.get("dual_joint_min_improvement"), True),
                    "是" if selection.get("dual_improves_both_profiles") else "否",
                    fmt(selection.get("min_dual_improvement"), True),
                    "最大最差增→最大平均增→低Reg",
                ]],
            ),
            "",
        ]
        for profile in ("s8", "s16"):
            lines += profile_sections(
                profile,
                search["profiles"][profile],
                f"{profile.upper()} OOF最优档位",
            )

    lines += ["## 5. 冻结策略真实 SGLang 验证", ""]
    for profile, validation in (("s8", validation_s8), ("s16", validation_s16)):
        if validation:
            result = {
                **validation["dynamic_policy"].get("macro", {}),
                "protocol": validation.get("protocol", {}),
                "summary": validation["dynamic_policy"],
                "allowed_blocks": validation["dynamic_policy"].get("macro", {}).get("allowed_blocks"),
                "cold_start_block": validation["dynamic_policy"].get("macro", {}).get("cold_start_block"),
            }
            lines += profile_sections(
                profile, result, f"{profile.upper()} 冻结验证"
            )
            official = validation.get("official_sglang_metrics") or {}
            lines += [
                "官方 `decode TPF` 来自 SGLang canonical decode stats，不含 prompt prefill；shadow 观察前向会污染墙钟吞吐，不能据此宣称 serving 吞吐收益。",
                "",
                table(
                    ["集", "decode TPF", "token/块", "接收率", "加权接收率", "decode F", "decode token"],
                    [[
                        dataset,
                        fmt(row.get("decode_tpf")),
                        fmt(row.get("mean_tokens_per_block")),
                        fmt(row.get("mean_acceptance_rate")),
                        fmt(row.get("weighted_acceptance_rate")),
                        row.get("decode_forward_passes", "—"),
                        row.get("decode_tokens", "—"),
                    ] for dataset, row in (official.get("datasets") or {}).items()]
                    + ([
                        [
                            "均(集等权)",
                            fmt(official["macro"].get("decode_tpf")),
                            fmt(official["macro"].get("mean_tokens_per_block")),
                            fmt(official["macro"].get("mean_acceptance_rate")),
                            fmt(official["macro"].get("weighted_acceptance_rate")),
                            "—",
                            "—",
                        ]
                    ] if official.get("macro") else []),
                ),
                "",
            ]
        else:
            lines += [f"### {profile.upper()} 冻结验证", "", "等待运行。", ""]

    lines += [
        "## 6. 解释边界",
        "",
        "离线探索和冻结验证为获得同状态 L8/L16/L32 标签会额外执行 shadow forward，因此平均 block、接/算和 canonical decode TPF 可用于策略判断，wall time、显存峰值和墙吞吐不能代表未来分桶 serving。最终还需只执行被选动作的真实分桶基准测试。",
        "",
    ]
    atomic_text(run_dir / "report.md", "\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    init_parser = subparsers.add_parser("init")
    init_parser.add_argument("--run-dir", type=Path, required=True)
    init_parser.add_argument("--settings-json", required=True)
    event_parser = subparsers.add_parser("event")
    event_parser.add_argument("--run-dir", type=Path, required=True)
    event_parser.add_argument("--event-json", required=True)
    render_parser = subparsers.add_parser("render")
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
