#!/usr/bin/env python3
"""Initialize state and atomically render the incremental Chinese report."""

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
        ["|" + "|".join(headers) + "|", "|" + "|".join(":---:" for _ in headers) + "|"]
        + ["|" + "|".join(row) + "|" for row in body]
    )


def fmt(value: Any, percent: bool = False) -> str:
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{100*number:.2f}%" if percent else f"{number:.4f}"


def init_run(run_dir: Path, settings: dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=False)
    for child in (
        "search", "traces/validate", "eval_runs/validate", "runtime/attempt_traces",
        "runtime/gpu_memory_guard",
    ):
        (run_dir / child).mkdir(parents=True, exist_ok=True)
    settings["created_at"] = datetime.now().astimezone().isoformat()
    atomic_text(
        run_dir / "settings.json",
        json.dumps(settings, ensure_ascii=False, indent=2) + "\n",
    )
    lines = [
        "# SGLang 统一标量动态 block 策略实验设置", "",
        f"- 创建时间：`{settings['created_at']}`",
        f"- 阶段：`{settings['stage']}`；数据集：`{settings['benchmarks']}`。",
        f"- 复用探索 trace：`{settings['trace_root']}`；不重新采集探索数据。",
        f"- 模型：`{settings['model']}`；模式：`{settings['mode']}`；LoRA mode：`{settings['lora_mode']}`。",
        f"- 统一动作：`8,16,32`；第一轮固定 L16；策略 λ：`{settings['policy_lambda']}`（auto 使用搜索出的默认值）。",
        f"- 搜索：{settings['cv_folds']} 折 prompt-hash OOF；{settings['signal_bins']} 个单调 survival bin；λ 网格=`{settings['lambda_grid']}`。",
        f"- 权重：数据集等权→数据集内 request 等权→request 内轮次等权；AIME24 排除；MMLU 必须为 `{settings['mmlu_max_samples']}` 条。",
        f"- GPU：`{settings['gpu_devices']}`；TP：`{settings['tp_size']}`；batch：`{settings['batch_size']}`；client concurrency：`{settings['client_concurrency']}`。",
        f"- CPU 搜索显存守护：每张指定 GPU 预占 `{settings['search_gpu_hold_gb']}` GiB，chunk=`{settings['search_gpu_hold_chunk_gb']}` GiB；搜索结束后先释放再启动验证。",
        f"- SGLang：context=`{settings['context_length']}`，tokens=`{settings['tokens']}`，dtype=`{settings['dtype']}`，mem fraction=`{settings['mem_fraction']}`，模型加载前额外预留=`{settings['gpu_memory_reserve_gb']}` GiB。",
        f"- 端口：server=`{settings['port'] or '自动搜索'}`，proxy=`{settings['proxy_port'] or '自动搜索'}`。",
        f"- 原始单行命令：`{settings['command']}`",
    ]
    atomic_text(run_dir / "settings.md", "\n".join(lines) + "\n")
    atomic_text(run_dir / "run_state.json", json.dumps({"events": []}, ensure_ascii=False, indent=2) + "\n")
    render(run_dir)


def add_event(run_dir: Path, event: dict[str, Any]) -> None:
    state = load(run_dir / "run_state.json") or {"events": []}
    event["updated_at"] = datetime.now().astimezone().isoformat()
    state["events"].append(event)
    atomic_text(run_dir / "run_state.json", json.dumps(state, ensure_ascii=False, indent=2) + "\n")
    render(run_dir)


def quality_rows(quality: Mapping[str, Any]) -> list[list[Any]]:
    rows = []
    for dataset, metric in (quality.get("by_dataset") or {}).items():
        rows.append([
            dataset, metric.get("rows_original", 0), metric.get("rows_usable", 0),
            metric.get("rows_excluded", 0), metric.get("replay_mismatch", 0),
            metric.get("cross_block_mismatch", 0), fmt(metric.get("excluded_rate"), True),
        ])
    rows.append([
        "总计", quality.get("rows_original", 0), quality.get("rows_usable", 0),
        quality.get("rows_excluded", 0), quality.get("replay_mismatch", 0),
        quality.get("cross_block_mismatch", 0), fmt(quality.get("excluded_rate"), True),
    ])
    return rows


def metric_rows(summary: Mapping[str, Any]) -> list[list[Any]]:
    rows = []
    for dataset, metric in (summary.get("datasets") or {}).items():
        rows.append([
            dataset, metric.get("requests", 0), metric.get("rounds", 0),
            fmt(metric.get("mean_block")), fmt(metric.get("mean_accept")),
            fmt(metric.get("decode_tpf_logical")), fmt(metric.get("accept_per_block_token")),
            fmt(metric.get("loss_vs_l32")), fmt(metric.get("mean_regret")),
            fmt(metric.get("oracle_action_accuracy"), True),
            fmt(metric.get("l8_rate"), True), fmt(metric.get("l16_rate"), True),
            fmt(metric.get("l32_rate"), True),
        ])
    macro = summary.get("macro")
    if macro:
        rows.append([
            "均(集等权)", macro.get("requests", 0), macro.get("rounds", 0),
            fmt(macro.get("mean_block")), fmt(macro.get("mean_accept")),
            fmt(macro.get("decode_tpf_logical")), fmt(macro.get("accept_per_block_token")),
            fmt(macro.get("loss_vs_l32")), fmt(macro.get("mean_regret")),
            fmt(macro.get("oracle_action_accuracy"), True),
            fmt(macro.get("l8_rate"), True), fmt(macro.get("l16_rate"), True),
            fmt(macro.get("l32_rate"), True),
        ])
    return rows


def risk_rows(summary: Mapping[str, Any]) -> list[list[Any]]:
    macro = summary.get("macro") or {}
    return [[
        "均(集等权)", fmt(macro.get("large_rate"), True),
        fmt(macro.get("large_precision"), True), fmt(macro.get("large_waste_rate"), True),
        fmt(macro.get("trunc_gt1"), True), fmt(macro.get("trunc_gt2"), True),
        fmt(macro.get("trunc_gt4"), True), fmt(macro.get("p90_regret")),
        fmt(macro.get("p95_regret")), fmt(macro.get("compute_saving_vs_l32")),
    ]]


def transition_rows(summary: Mapping[str, Any]) -> list[list[Any]]:
    transition = (summary.get("transitions") or {}).get("macro")
    if not transition:
        return [["均(集等权)"] + ["—"] * 11]
    matrix = transition["matrix"]
    return [[
        "均(集等权)",
        *[fmt(matrix[left][right], True) for left in range(3) for right in range(3)],
        fmt(transition.get("switch_rate"), True),
        fmt(transition.get("oscillation_rate"), True),
    ]]


def render(run_dir: Path) -> None:
    settings = load(run_dir / "settings.json") or {}
    state = load(run_dir / "run_state.json") or {"events": []}
    progress = load(run_dir / "search/progress.json")
    search = load(run_dir / "search/search_results.json")
    validation = load(run_dir / "search/validation.json")
    lines = [
        "# SGLang 统一标量动态 block 策略实验报告", "",
        "> 本模板在时间戳目录创建时即生成，并随显存守护、搜索阶段、每个验证数据集和最终汇总原子更新。所有宏平均始终让九个非 AIME24 数据集等权；MMLU 固定 2000 条。", "",
        "## 1. 实时进度", "",
        "`状态`包括初始化、GPU 显存预占、Tier1 单信号搜索、Tier2 受限双信号搜索、守护释放和九集冻结验证。`说明`给出可直接排查的文件或当前对象。", "",
        table(
            ["阶段", "对象", "状态", "轮/数", "说明"],
            [[event.get("phase", "—"), event.get("dataset", "—"), event.get("status", "—"), event.get("records", "—"), event.get("message", "—")] for event in state.get("events", [])]
            or [["初始化", "—", "等待", "0", "尚未开始搜索"]],
        ), "",
    ]
    if progress:
        latest = progress.get("latest") or progress.get("selected") or {}
        lines += [
            "搜索进度文件会在每完成一个候选后更新。`完成/总数`是当前层候选数，不是数据集或生成 request 数。", "",
            table(["检索阶段", "完成", "总数", "最新/最优"], [[
                progress.get("stage", "—"), progress.get("completed", 0), progress.get("total", 0),
                json.dumps(latest, ensure_ascii=False, separators=(",", ":")),
            ]]), "",
        ]
    lines += [
        "## 2. 方法与变量", "",
        "策略只计算一个主信号 s；只有当单信号的集等权 OOF regret 至少改善配置值时，才允许再用 `full_streak` 将同一 s 分成“整块连续通过/未通过”两个删失层。三个动作始终使用同一套全局表和同一个 λ，不存在 L8/L16 专属分类器或数据集专属参数。", "",
        table(["变量", "含义", "例子"], [
            ["s", "唯一主历史信号；越大表示历史接收能力越强", "accept_ma4=7 表示近4轮平均接收7"],
            ["V8/V16/V32", "由单调 survival 曲线得到的条件期望接收长度", "V16=9.2 表示该 s 下 L16 预计接收9.2"],
            ["λ", "每计算一个 block token 的惩罚，也是唯一部署倾向旋钮", "λ=0.25 时 L16 比 L8 至少需多2个期望接收"],
            ["U(L)", "V(L)-λ×L；取最大值，完全相等时选较小块", "U8=5、U16=6、U32=4 时选L16"],
            ["Reg", "同状态 oracle utility 减策略 utility，越小越好", "Reg=0 表示动作达到该 λ 下同状态最优"],
            ["TPF逻", "逻辑 decode TPF=接收token/2，因每轮 canonical 有 draft+verify 两次 forward", "接收10则TPF逻=5"],
            ["接/算", "平均接收token/平均block token，衡量compute-bound计算利用率", "8 token算出6 token则0.75"],
        ]), "",
        "## 3. Trace 完整性", "",
    ]
    quality = None
    if search:
        quality = (search.get("protocol") or {}).get("trace_quality")
    elif validation:
        quality = (validation.get("protocol") or {}).get("trace_quality")
    if quality:
        lines += [
            "`原始/可用/排除`分别是读入轮数、参与拟合或验证的轮数及歧义轮数；`回放异/跨块异`检查 canonical replay 和三分支公共接收前缀。旧 schema v1 允许在 5% 保护线内显式排除，新冻结验证使用 schema v2 并要求严格无歧义。", "",
            table(["集", "原始", "可用", "排除", "回放异", "跨块异", "排除%"], quality_rows(quality)), "",
        ]
    else:
        lines += ["等待 trace 审计完成。", ""]

    lines += ["## 4. 离线全局检索", ""]
    if not search:
        lines += ["等待九集等权 OOF 搜索完成；运行中排名可查看上一节进度。", ""]
    else:
        selection = search["selection"]
        lines += [
            "`IntReg`是在完整 λ 网格上的集等权 OOF 平均 regret；`IntAcc`是动作与同状态 oracle 完全一致率。Tier1 每行严格只用一个信号。Tier2 只允许一个接收类信号加全局 `full_streak≥门槛` 分层；若改善不足 `双信号最小改善`，最终仍选择单信号。", "",
            table(["层", "信号", "原字段", "类别", "full门", "IntReg", "IntAcc"], [
                [row["tier"], row["signal"], row["feature"], row["kind"], row.get("gate") if row.get("gate") is not None else "—", fmt(row.get("integrated_mean_regret")), fmt(row.get("integrated_oracle_accuracy"), True)]
                for row in search.get("tier1_ranking", []) + search.get("tier2_ranking", [])
            ]), "",
            "下表解释最终模型选择。`双改`是最佳双信号相对最佳单信号减少的 token-utility regret/轮；只有达到 `最小改`才增加第二信号。`默认可行`表示默认 λ 同时满足大块精度、浪费和相对 L32 接收损失约束。", "",
            table(["最终层", "主信号", "full门", "双改", "最小改", "默认λ", "默认可行"], [[
                selection["selected"]["tier"], selection["selected"]["signal"],
                selection["selected"].get("gate") if selection["selected"].get("gate") is not None else "—",
                fmt(selection.get("dual_improvement")), fmt(selection.get("min_dual_improvement")),
                fmt(selection.get("default_lambda")), "是" if selection.get("default_feasible") else "否",
            ]]), "",
            "λ 是唯一供用户改变大/小块倾向的超参。`大精`表示所选 L16 相对 L8 至少增益2、或 L32 相对 L8/L16 中较优者至少增益4的条件比例；`大浪`表示相应增益不超过1；`损32`是相对同状态 L32 少接收token/轮。λ 越大，理论上动作只会保持或变小。", "",
            table(["λ", "可行", "块均", "接均", "TPF逻", "接/算", "损32", "Reg", "大精", "大浪", "L8%", "L16%", "L32%"], [[
                fmt(row["lambda"]), "是" if row["feasible"] else "否", fmt(row["mean_block"]),
                fmt(row["mean_accept"]), fmt(row["decode_tpf_logical"]), fmt(row["accept_per_block_token"]),
                fmt(row["loss_vs_l32"]), fmt(row["mean_regret"]), fmt(row["large_precision"], True),
                fmt(row["large_waste_rate"], True), fmt(row["l8_rate"], True), fmt(row["l16_rate"], True), fmt(row["l32_rate"], True),
            ] for row in search.get("pareto", [])]), "",
            "默认 λ 的 OOF 逐集表使用所有有效样本：每个 prompt 在预测时都来自不含该 prompt 的折模型，之后才用全数据重拟合导出冻结策略。`准Or`是动作恰好等于该 λ 同状态 oracle 的概率。", "",
            table(["集", "Req", "轮", "块均", "接均", "TPF逻", "接/算", "损32", "Reg", "准Or", "L8%", "L16%", "L32%"], metric_rows(search["oof_default"])), "",
            "风险表中的 `截>k` 表示所选小块相对该轮三个块中的最大接收值损失超过 k token；`省算`是相对固定 L32 每轮少计算的 block token。", "",
            table(["范围", "大块%", "大精", "大浪", "截>1", "截>2", "截>4", "Reg90", "Reg95", "省算"], risk_rows(search["oof_default"])), "",
            "转移表按每个 request 内相邻决策统计，再 request 等权、数据集等权。`8→16` 表示上一轮策略动作 L8、下一轮动作 L16；`切换`是相邻动作不同，`摆动`是三轮出现 L8→L16→L8 等 A→B→A。", "",
            table(["范围", "8→8", "8→16", "8→32", "16→8", "16→16", "16→32", "32→8", "32→16", "32→32", "切换", "摆动"], transition_rows(search["oof_default"])), "",
            "固定 L8/L16/L32 和同状态 oracle 使用相同 trace、相同 λ、相同九集等权口径。它们是反事实参照，不是把不同生成轨迹混成训练样本。", "",
            table(["策略", "块均", "接均", "TPF逻", "接/算", "损32", "Reg", "L8%", "L16%", "L32%"], [
                [f"固定L{block}", fmt(search["fixed_baselines"][str(block)]["macro"].get("mean_block")), fmt(search["fixed_baselines"][str(block)]["macro"].get("mean_accept")), fmt(search["fixed_baselines"][str(block)]["macro"].get("decode_tpf_logical")), fmt(search["fixed_baselines"][str(block)]["macro"].get("accept_per_block_token")), fmt(search["fixed_baselines"][str(block)]["macro"].get("loss_vs_l32")), fmt(search["fixed_baselines"][str(block)]["macro"].get("mean_regret")), fmt(search["fixed_baselines"][str(block)]["macro"].get("l8_rate"), True), fmt(search["fixed_baselines"][str(block)]["macro"].get("l16_rate"), True), fmt(search["fixed_baselines"][str(block)]["macro"].get("l32_rate"), True)]
                for block in (8, 16, 32)
            ] + [[
                "同态Oracle", fmt(search["same_state_oracle"]["macro"].get("mean_block")), fmt(search["same_state_oracle"]["macro"].get("mean_accept")), fmt(search["same_state_oracle"]["macro"].get("decode_tpf_logical")), fmt(search["same_state_oracle"]["macro"].get("accept_per_block_token")), fmt(search["same_state_oracle"]["macro"].get("loss_vs_l32")), fmt(search["same_state_oracle"]["macro"].get("mean_regret")), fmt(search["same_state_oracle"]["macro"].get("l8_rate"), True), fmt(search["same_state_oracle"]["macro"].get("l16_rate"), True), fmt(search["same_state_oracle"]["macro"].get("l32_rate"), True),
            ]]), "",
        ]

    lines += ["## 5. 冻结策略真实 SGLang 验证", ""]
    if not validation:
        lines += ["等待释放 CPU 搜索显存守护后，在九个数据集上运行新统一策略。", ""]
    else:
        validation_quality = (validation.get("protocol") or {}).get("trace_quality")
        if validation_quality:
            lines += [
                "冻结验证 trace 使用当前 schema v2 再做一次独立完整性审计；下表与探索审计同义，不会用搜索阶段的排除率替代新策略质量。", "",
                table(["集", "原始", "可用", "排除", "回放异", "跨块异", "排除%"], quality_rows(validation_quality)), "",
            ]
        lines += [
            "该表来自新策略自身产生的动态 block 历史，而不是旧探索行为策略。每轮决策只读取此前 canonical 历史；同轮 L8/L16/L32 shadow 只用于事后评价。", "",
            table(["集", "Req", "轮", "块均", "接均", "TPF逻", "接/算", "损32", "Reg", "准Or", "L8%", "L16%", "L32%"], metric_rows(validation["unified_policy"])), "",
            "下面是冻结验证风险宏表，变量与搜索阶段完全同义，便于检查分布迁移后是否仍满足保守目标。", "",
            table(["范围", "大块%", "大精", "大浪", "截>1", "截>2", "截>4", "Reg90", "Reg95", "省算"], risk_rows(validation["unified_policy"])), "",
            "冻结验证动作转移与摆动沿用搜索阶段定义，用于判断统一策略在真实动态历史下是否频繁跨桶。", "",
            table(["范围", "8→8", "8→16", "8→32", "16→8", "16→16", "16→32", "32→8", "32→16", "32→32", "切换", "摆动"], transition_rows(validation["unified_policy"])), "",
        ]
        official = validation.get("official_sglang_metrics") or {}
        lines += [
            "官方 SGLang `decode TPF` 读取每个 NeMo-Skills 产物的 `sglang_metrics_summary.json`，仅统计 canonical decode forward，不含 prompt prefill，也不把三分支 observation forward 计入 NFE。`墙吞吐`会受到 shadow 观察开销污染，只保留逐集文件，不作为部署收益。", "",
            table(["集", "decode TPF", "token/块", "接收率", "加权接收率", "decode F", "decode token"], [
                [dataset, fmt(row.get("decode_tpf")), fmt(row.get("mean_tokens_per_block")), fmt(row.get("mean_acceptance_rate")), fmt(row.get("weighted_acceptance_rate")), row.get("decode_forward_passes", "—"), row.get("decode_tokens", "—")]
                for dataset, row in (official.get("datasets") or {}).items()
            ] + ([["均(集等权)", fmt(official["macro"].get("decode_tpf")), fmt(official["macro"].get("mean_tokens_per_block")), fmt(official["macro"].get("mean_acceptance_rate")), fmt(official["macro"].get("weighted_acceptance_rate")), "—", "—"]] if official.get("macro") else [])), "",
        ]

    lines += [
        "## 6. 解释边界", "",
        "本实验验证历史信号和逻辑计算选择。为获得同状态 oracle，每轮额外运行 L8/L16/L32 shadow，因此 wall time、显存峰值和墙吞吐不是部署结果。若统一策略通过冻结验证，下一步仍需在连续 serving 中按 L8/L16/L32 分桶，只执行各桶真实动作，测吞吐、排队、延迟和 compute-bound 转折。", "",
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
