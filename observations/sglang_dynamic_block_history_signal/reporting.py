#!/usr/bin/env python3
"""Atomic settings/state and incremental Chinese report rendering."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temp.write_text(content, encoding="utf-8")
    os.replace(temp, path)


def table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    rows = [[str(value) for value in row] for row in rows]
    return "\n".join(
        ["|" + "|".join(headers) + "|", "|" + "|".join(":---:" for _ in headers) + "|"]
        + ["|" + "|".join(row) + "|" for row in rows]
    )


def fmt(value: Any, percent: bool = False) -> str:
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{100*number:.2f}%" if percent else f"{number:.4f}"


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def init_run(run_dir: Path, settings: dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=False)
    for child in ("traces/explore", "traces/validate_s8", "traces/validate_s16", "eval_runs", "search", "runtime"):
        (run_dir / child).mkdir(parents=True, exist_ok=True)
    settings["created_at"] = datetime.now().astimezone().isoformat()
    atomic_text(run_dir / "settings.json", json.dumps(settings, ensure_ascii=False, indent=2))
    settings_lines = [
        "# 动态 block size 历史信号实验设置", "",
        f"- 创建时间：`{settings['created_at']}`",
        f"- 阶段：`{settings['stage']}`",
        f"- 数据集：`{settings['benchmarks']}`（明确排除 AIME24；MMLU 默认最后执行）",
        f"- 模型：`{settings['model']}`",
        f"- 模式：`{settings['mode']}`；候选 block：`{settings['block_sizes']}`；物理 KV block：`{settings['physical_block_size']}`",
        f"- GPU：`{settings['gpu_devices']}`；TP：`{settings['tp_size']}`；max running requests：`{settings['batch_size']}`；client concurrency：`{settings['client_concurrency']}`",
        f"- GPU 自动选择门槛：`{settings.get('auto_gpu_min_free_gb', '48')} GiB`；仅当 GPU 设置为 `auto` 时逐次生效。",
        f"- 生成上限：`{settings['tokens']}`；context：`{settings['context_length']}`；dtype：`{settings['dtype']}`；mem fraction：`{settings['mem_fraction']}`；预留显存：`{settings['gpu_memory_reserve_gb']} GiB`",
        f"- 解码：temperature=`{settings.get('temperature', '0')}`；top-p=`{settings.get('top_p', '0.95')}`；LoRA=`{settings.get('lora_path') or 'pipeline 默认'}`；LoRA mode=`{settings.get('lora_mode', 'draft_only')}`。",
        f"- MMLU 上限：`{settings['mmlu_max_samples']}`；其他数据集上限：`{settings['max_samples'] or '完整'}`",
        f"- 行为策略：探索阶段首轮 L16，之后分层 Markov 探索；验证阶段读取冻结全局策略。seed=`{settings['seed']}`，split seed=`{settings['split_seed']}`。",
        f"- 保守约束：大块精度≥`{settings.get('min_large_precision', '0.80')}`；大块浪费率≤`{settings.get('max_large_waste', '0.10')}`；S16→L8 安全率≥`{settings.get('min_safe8', '0.98')}`；允许部分数据集搜索=`{settings.get('allow_partial_search', 'false')}`。",
        f"- Trace 完整性：逐轮审计 canonical replay 与跨 block 公共前缀；歧义行保守排除上限=`{settings.get('max_invalid_row_rate', '0.05')}`。",
        f"- 数据集容错：每个数据集最多启动 `{settings.get('dataset_max_attempts', '3')}` 个全新 server 尝试；失败后等待 `{settings.get('dataset_retry_delay_s', '10')}` 秒。benchmark 失败视为非零退出并保留内部 runtime。",
        f"- CPU 搜索 GPU 守护：`{settings.get('search_guardian', 'baseline_9datasets_concurrent_eval')}`；CPU 检索期间并行运行九个非 AIME24 数据集的正常 SGLang+NeMoSkills baseline，搜索结束即终止。",
        f"- Shadow 分支隔离：`{settings.get('shadow_branch_barrier', 'cuda_synchronize_after_each_shadow_branch')}`；每个反事实分支结束后完成 CUDA correctness barrier，再重建下一分支的 batch/KV 视图。",
        f"- Trace 提交：`{settings.get('trace_commit_protocol', 'attempt_trace_then_atomic_move')}`；失败尝试保留在 runtime，只有完整成功才原子替换 canonical trace。",
        f"- 目录写保护：`{settings.get('run_lock_protocol', 'exclusive_flock')}`；同一个时间戳目录同一时刻只允许一个入口进程更新。",
        "- 搜索权重：九数据集等权；数据集内 request 等权；request 内轮次等权。dataset id 不进入特征。",
        f"- 原始单行命令：`{settings['command']}`",
    ]
    atomic_text(run_dir / "settings.md", "\n".join(settings_lines) + "\n")
    atomic_text(run_dir / "run_state.json", json.dumps({"events": []}, ensure_ascii=False, indent=2))
    render(run_dir)


def add_event(run_dir: Path, event: dict[str, Any]) -> None:
    state_path = run_dir / "run_state.json"
    state = load(state_path) or {"events": []}
    event["updated_at"] = datetime.now().astimezone().isoformat()
    state["events"].append(event)
    atomic_text(state_path, json.dumps(state, ensure_ascii=False, indent=2))
    render(run_dir)


def _metric_rows(summary: dict[str, Any]) -> list[list[str]]:
    rows = []
    for dataset, metric in (summary.get("datasets") or {}).items():
        rows.append([
            dataset, metric.get("requests", 0), metric.get("rounds", 0),
            fmt(metric.get("mean_block")), fmt(metric.get("mean_accept")), fmt(metric.get("tpf_proxy")),
            fmt(metric.get("loss_vs_l32")), fmt(metric.get("loss_vs_default")),
            fmt(metric.get("l8_rate"), True), fmt(metric.get("l16_rate"), True), fmt(metric.get("l32_rate"), True),
        ])
    macro = summary.get("macro")
    if macro:
        rows.append([
            "均(集等权)", macro.get("requests", 0), macro.get("rounds", 0),
            fmt(macro.get("mean_block")), fmt(macro.get("mean_accept")), fmt(macro.get("tpf_proxy")),
            fmt(macro.get("loss_vs_l32")), fmt(macro.get("loss_vs_default")),
            fmt(macro.get("l8_rate"), True), fmt(macro.get("l16_rate"), True), fmt(macro.get("l32_rate"), True),
        ])
    return rows


def _risk_rows(summary: dict[str, Any]) -> list[list[str]]:
    items = list((summary.get("datasets") or {}).items())
    if summary.get("macro"):
        items.append(("均(集等权)", summary["macro"]))
    return [
        [
            dataset,
            fmt(metric.get("large_rate"), True),
            fmt(metric.get("large_precision"), True),
            fmt(metric.get("large_waste_rate"), True),
            fmt(metric.get("downgrade8_rate"), True),
            fmt(metric.get("downgrade8_safe_rate"), True),
        ]
        for dataset, metric in items
    ]


def _quality_rows(quality: dict[str, Any]) -> list[list[str]]:
    rows = []
    for dataset, metric in (quality.get("by_dataset") or {}).items():
        rows.append([
            dataset,
            metric.get("rows_original", 0),
            metric.get("rows_usable", 0),
            metric.get("rows_excluded", 0),
            metric.get("replay_mismatch", 0),
            metric.get("cross_block_mismatch", 0),
            fmt(metric.get("excluded_rate"), True),
        ])
    rows.append([
        "总计",
        quality.get("rows_original", 0),
        quality.get("rows_usable", 0),
        quality.get("rows_excluded", 0),
        quality.get("replay_mismatch", 0),
        quality.get("cross_block_mismatch", 0),
        fmt(quality.get("excluded_rate"), True),
    ])
    return rows


def render(run_dir: Path) -> None:
    settings = load(run_dir / "settings.json") or {}
    state = load(run_dir / "run_state.json") or {"events": []}
    search = load(run_dir / "search/search_results.json")
    validation_s8 = load(run_dir / "search/validation_s8.json")
    validation_s16 = load(run_dir / "search/validation_s16.json")
    lines = [
        "# SGLang 动态 block size 历史信号实验报告", "",
        "> 本文由程序从实验开始即建立，并在每个数据集、搜索阶段和验证阶段结束后原子更新。AIME24 不参与任何搜索或宏平均。", "",
        "## 1. 当前进度", "",
        "下表中的“阶段”区分探索 trace、离线搜索、S8 冻结验证和 S16 冻结验证；例如 `explore/gsm8k/completed` 表示 GSM8K 的真实三分支动态轨迹已完整写出。", "",
    ]
    events = state.get("events") or []
    progress_rows = [[item.get("phase", "—"), item.get("dataset", "—"), item.get("status", "—"), item.get("records", "—"), item.get("message", "—")] for item in events]
    lines += [table(["阶段", "数据集", "状态", "轮数", "说明"], progress_rows or [["初始化", "—", "等待", "0", "尚未完成数据集"]]), ""]
    lines += [
        "## 2. 数据与权重协议", "",
        "`D权` 表示每个数据集先各占 1/D；数据集内部每个 request 等权，再让同一 request 的每轮等权。例如一个 10 轮 request 与一个 100 轮 request 对数据集均值贡献相同。MMLU 即使样本更多，也不会盖过其他数据集。", "",
        table(["项", "值"], [
            ["候选L", settings.get("block_sizes", "8,16,32")],
            ["初轮", "L16"], ["排除", "AIME24"],
            ["MMLU上限", settings.get("mmlu_max_samples", "—")],
            ["宏权重", "集等权→request等权→轮等权"],
            ["切分", "prompt hash:70% train/15% selection/15% test"],
        ]), "",
    ]
    if search:
        lines += ["## 3. 搜索结果", ""]
        search_quality = (search.get("protocol") or {}).get("trace_quality")
        if search_quality:
            lines += [
                "搜索前先执行 trace 完整性审计。`原始`是读取轮数，`可用`是进入拟合/选择/测试的轮数，`排除`是 canonical replay 或跨 block 公共接收前缀存在歧义的并集；`回放异`与`跨块异`可能重叠。例：排除率 1% 表示该集 99% 的轮次参与重新计算后的 request/轮次等权搜索。超过设置上限会直接失败，不会生成策略。", "",
                table(
                    ["集", "原始", "可用", "排除", "回放异", "跨块异", "排除%"],
                    _quality_rows(search_quality),
                ), "",
            ]
        for target in ("s8", "s16"):
            result = search["targets"][target]
            selected = result["selected"]
            policy = result["policy"]
            lines += [
                f"### {target.upper()}：默认小块 L{8 if target == 's8' else 16}", "",
                "`gain16/gain32` 是启用更大块所要求的最小接收增益，`eff` 是每增加一个 block 位置至少换回多少接收 token；`阈值` 是历史模型输出概率超过多少才升级。例：gain32=5 表示预测不足以支持至少多接收 5 token 时，不应冒险选 L32。", "",
                table(["族", "特征组", "检索数", "标签规格", "概率阈值"], [[
                    policy["model_family"], policy["feature_group"], result["searched_candidates"],
                    json.dumps(policy["label_spec"], ensure_ascii=False, separators=(",", ":")),
                    json.dumps(policy["thresholds"], ensure_ascii=False, separators=(",", ":")),
                ]]), "",
                "下表是 selection 上各数据集和等权宏结果。`块均`/`接均`分别是平均选择块长和平均接收；`TPF代`为接收/块长的计算代理；`损32`是相对同状态 L32 少接收的 token/轮；`损默`相对该目标默认小块的损失。例：L8%=70% 表示 70% 的轮保持最小计算块。", "",
                table(["集", "Req", "轮", "块均", "接均", "TPF代", "损32", "损默", "L8%", "L16%", "L32%"], _metric_rows(selected["selection"])), "",
                "下表只看是否保守使用大块。`大精`是在升级轮中，大块确实达到标签所要求显著收益的比例；`大浪`是在升级轮中增益不超过 1 token 的比例；`8安`是 S16 下调到 L8 时 A8 不低于 A16 的比例。这三个条件比例先逐数据集计算，再对实际含该动作的数据集宏平均，升级轮多的数据集不会获得更高权重。", "",
                table(["集", "大块%", "大精", "大浪", "下8%", "8安"], _risk_rows(selected["selection"])), "",
                "下表是从未参与拟合或选阈值的 hash-test request。AUROC/AUPRC 衡量历史信号对“大块是否值得”或“L8 是否安全”的排序能力，不等同于最终策略正确率。", "",
                table(["信号", "阳性率", "AUROC", "AUPRC"], [
                    [name, fmt(metric.get("positive_rate"), True), fmt(metric.get("auroc")), fmt(metric.get("auprc"))]
                    for name, metric in result.get("test_signal", {}).items()
                ]), "",
                "下表是只用于诊断的非线性 GBDT 信号上界，不会导出到 serving 策略。`深`是树深，`特征组`依次增加删失感知和 confidence；例：GBDT 明显高于浅树时说明信号存在但可部署模型容量不足，二者都低时说明现有历史信号本身不足。", "",
                table(["特征组", "深", "信号", "阳性率", "AUROC", "AUPRC"], [
                    [item["feature_group"], item["max_depth"], item["label"], fmt(item.get("positive_rate"), True), fmt(item.get("auroc")), fmt(item.get("auprc"))]
                    for item in result.get("test_signal_upper_bounds", [])
                ] or [["—", "—", "—", "—", "—", "—"]]), "",
                "test 动作指标沿用上表变量定义；这是搜索 trace 的严格 held-out 部分，正式结论仍以随后冻结策略真实重跑的验证表为准。", "",
                table(["集", "Req", "轮", "块均", "接均", "TPF代", "损32", "损默", "L8%", "L16%", "L32%"], _metric_rows(result["test"])), "",
            ]
    else:
        lines += ["## 3. 搜索结果", "", "等待探索 trace 完成并执行离线全局搜索。", ""]

    lines += ["## 4. 冻结策略真实验证", ""]
    for target, payload in (("s8", validation_s8), ("s16", validation_s16)):
        lines += [f"### {target.upper()}", ""]
        if not payload:
            lines += ["等待冻结策略在真实 SGLang 动态 canonical 上重跑。", ""]
            continue
        validation_quality = (payload.get("protocol") or {}).get("trace_quality")
        if validation_quality:
            lines += [
                "本次冻结验证也按相同完整性规则审计每轮三分支；表中变量与搜索质量表相同。验证决策使用真实 canonical，歧义只影响反事实效率标签，超过上限仍会拒绝汇报。", "",
                table(
                    ["集", "原始", "可用", "排除", "回放异", "跨块异", "排除%"],
                    _quality_rows(validation_quality),
                ), "",
            ]
        lines += [
            "第一张表统计冻结策略真实重跑的全部样本，因此正式九集运行时必然逐集列出，并给出严格集等权宏平均。策略决策看不到当轮 shadow 标签；`损默`判断是否破坏默认小块潜力，`损32`量化相对最大块的机会损失。", "",
            table(["集", "Req", "轮", "块均", "接均", "TPF代", "损32", "损默", "L8%", "L16%", "L32%"], _metric_rows(payload["all_data"])), "",
            "下表专门检查全部验证数据上的 compute-bound 大块误用和 S16 下调风险。条件比例先逐数据集计算再做宏平均；例如 `大浪=2%` 表示各个实际发生升级的数据集，其升级浪费率的等权平均为 2%。", "",
            table(["集", "大块%", "大精", "大浪", "下8%", "8安"], _risk_rows(payload["all_data"])), "",
            "最后一张表只统计 prompt-hash test request，是与搜索 trace 切分一致的严格 held-out 视图；小样本数据集若 hash-test 恰好为空会显示为空，但不影响上面的九集全样本表。", "",
            table(["集", "Req", "轮", "块均", "接均", "TPF代", "损32", "损默", "L8%", "L16%", "L32%"], _metric_rows(payload["heldout_test"])), "",
        ]
    lines += [
        "## 5. 解释边界", "",
        "本实验报告的 `TPF代` 与 block token 成本是策略检索指标，不含为获得反事实标签而额外执行的 shadow/replay 前向；后者是观察成本，不能当作部署性能。策略确认后还需在按 L8/L16/L32 分桶的连续 serving 中测真实吞吐、延迟和 padding。", "",
    ]
    atomic_text(run_dir / "report.md", "\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    init_parser = sub.add_parser("init")
    init_parser.add_argument("--run-dir", type=Path, required=True)
    init_parser.add_argument("--settings-json", required=True)
    event_parser = sub.add_parser("event")
    event_parser.add_argument("--run-dir", type=Path, required=True)
    event_parser.add_argument("--event-json", required=True)
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
