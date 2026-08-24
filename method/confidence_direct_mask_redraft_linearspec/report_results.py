#!/usr/bin/env python3
"""Generate a compact Chinese report for strict direct MASK-redraft results."""

from __future__ import annotations

import argparse
import json
import math
import unicodedata
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Optional


ORDER = (
    "human-eval",
    "mbpp",
    "livecodebench-cpp",
    "gsm8k",
    "math-500",
    "aime24",
    "aime25",
    "gpqa",
    "mmlu",
    "ifeval",
)
DISPLAY = {
    "human-eval": "HumanEval",
    "mbpp": "MBPP",
    "livecodebench-cpp": "LiveCodeBench-C++",
    "gsm8k": "GSM8K",
    "math-500": "MATH-500",
    "aime24": "AIME24",
    "aime25": "AIME25",
    "gpqa": "GPQA",
    "mmlu": "MMLU",
    "ifeval": "IFEval",
}
STATE_ORDER = (
    "m_lt_p",
    "direct_hit",
    "repeat_a",
    "wrong_non_a",
    "a_ok_later_reject",
    "full_bonus",
)
STATE_SHORT = {
    "m_lt_p": "m<p",
    "direct_hit": "直中",
    "repeat_a": "重A",
    "wrong_non_a": "改错",
    "a_ok_later_reject": "A对后拒",
    "full_bonus": "Bonus",
}
DEFAULT_B16 = Path(
    "/data/home/wly/dLLM/NLD_results/observations/pytorch_nemo_eval_results/eval_20260804_120138"
)
DEFAULT_B32 = Path(
    "/data/home/wly/dLLM/NLD_results/observations/pytorch_nemo_eval_results/eval_20260804_114935"
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def finite(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def macro(values: list[Optional[float]]) -> Optional[float]:
    clean = [value for value in values if value is not None]
    return mean(clean) if clean else None


def fmt(value: Optional[float], digits: int = 4, suffix: str = "") -> str:
    return "—" if value is None else f"{value:.{digits}f}{suffix}"


def fmt_count(value: Optional[float]) -> str:
    if value is None:
        return "—"
    return f"{value:,.0f}" if float(value).is_integer() else f"{value:,.2f}"


def visible_width(value: str) -> int:
    return sum(
        0
        if unicodedata.combining(char)
        else 2 if unicodedata.east_asian_width(char) in ("W", "F") else 1
        for char in value
    )


def table(headers: list[str], rows: list[list[str]]) -> str:
    values = [headers] + rows
    widths = [
        max(3, max(visible_width(str(row[col])) for row in values))
        for col in range(len(headers))
    ]

    def centered(value: str, width: int) -> str:
        gap = width - visible_width(value)
        return " " * (gap // 2) + value + " " * (gap - gap // 2)

    output = ["|" + "|".join(centered(str(v), widths[i]) for i, v in enumerate(headers)) + "|"]
    output.append("|" + "|".join(":" + "-" * (width - 2) + ":" for width in widths) + "|")
    output.extend(
        "|" + "|".join(centered(str(v), widths[i]) for i, v in enumerate(row)) + "|"
        for row in rows
    )
    return "\n".join(output)


def discover(root: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in root.glob("metrics_*.json"):
        name = path.stem.removeprefix("metrics_")
        result[name] = read_json(path)
    return result


def tpf(metrics: Optional[dict[str, Any]]) -> Optional[float]:
    return finite((metrics or {}).get("tpf") or (metrics or {}).get("tokens_per_forward_pass"))


def accuracy(name: str, metrics: Optional[dict[str, Any]]) -> Optional[float]:
    if not metrics or name not in metrics:
        return None
    pass1 = (metrics.get(name) or {}).get("pass@1") or {}
    key = {
        "human-eval": "passing_base_tests",
        "mbpp": "passing_base_tests",
        "livecodebench-cpp": "accuracy",
        "ifeval": "average_score",
    }.get(name, "symbolic_correct")
    return finite(pass1.get(key))


def decode(metrics: dict[str, Any]) -> dict[str, Any]:
    return ((metrics.get("pytorch_confidence_direct_mask_redraft") or {}).get("decode") or {})


def validate_metrics(name: str, metrics: dict[str, Any]) -> None:
    red = decode(metrics).get("direct_mask_redraft") or {}
    states = red.get("state_stats") or {}
    missing = [state for state in STATE_ORDER if state not in states]
    if missing:
        raise ValueError(f"{name}: missing decision states: {missing}")
    attempts = int(red.get("redraft_attempts", -1))
    rounds = int(red.get("rounds", -1))
    all_counts = sum(int((item or {}).get("count", 0)) for item in states.values())
    attempt_counts = sum(int((states[state] or {}).get("count", 0)) for state in STATE_ORDER)
    if all_counts != rounds:
        raise ValueError(f"{name}: state counts {all_counts} != rounds {rounds}")
    if attempt_counts != attempts:
        raise ValueError(f"{name}: attempt-state counts {attempt_counts} != attempts {attempts}")
    for state, item in states.items():
        if int(item.get("next_observed_count", 0)) + int(item.get("no_next_round_count", 0)) != int(item.get("count", 0)):
            raise ValueError(f"{name}: invalid next-round partition for {state}")
        if int(item.get("current_emitted_sum", 0)) != int(item.get("current_matched_sum", 0)) + int(item.get("count", 0)):
            raise ValueError(f"{name}: current emitted != matched+1 for {state}")
        if int(item.get("next_emitted_sum", 0)) != int(item.get("next_matched_sum", 0)) + int(item.get("next_observed_count", 0)):
            raise ValueError(f"{name}: next emitted != matched+1 for {state}")
    if int(red.get("full_length_reuses", -1)) != int(red.get("redraft_reuse_hits", -2)):
        raise ValueError(f"{name}: a retained draft was not full length")


def validate_baseline(root: Path, expected_block: int) -> str:
    path = root / "Settings.json"
    if not path.is_file():
        return "Settings 缺失，仅按 metrics 比较"
    settings = read_json(path)
    pytorch = settings.get("pytorch") or {}
    benchmark = settings.get("benchmark") or {}
    actual = {
        "mode": pytorch.get("mode"),
        "block": pytorch.get("block_length"),
        "threshold": finite(pytorch.get("threshold")),
        "temperature": finite(benchmark.get("temperature")),
    }
    expected = {
        "mode": "linearspec_lora",
        "block": expected_block,
        "threshold": 0.0,
        "temperature": 0.0,
    }
    if actual != expected:
        raise ValueError(f"Invalid B{expected_block} baseline Settings: {actual}")
    return "已核验 PyTorch+NeMo-Skills+LinearSpec LoRA+greedy"


def validate_current_settings(root: Path) -> str:
    path = root / "Settings.json"
    if not path.is_file():
        return "Settings 缺失（仅报告已有 metrics）"
    settings = read_json(path)
    pytorch = settings.get("pytorch") or {}
    benchmark = settings.get("benchmark") or {}
    if pytorch.get("strict_direct_mask_redraft") is not True:
        raise ValueError("Current Settings is not strict direct MASK-redraft")
    if pytorch.get("fixed_length_reuse") is not True:
        raise ValueError("Current Settings does not enable fixed-length reuse")
    if finite(benchmark.get("temperature")) != 0.0:
        raise ValueError("Current Settings is not greedy")
    return f"已核验 block_size={pytorch.get('block_length')}、greedy、strict-direct/full-L"


def equal_label(names: list[str]) -> str:
    included = [name for name in names if name != "aime24"]
    return "九数据集等权平均" if len(included) == 9 else f"可用集等权平均({len(included)})"


def append_defs(parts: list[str], items: list[str]) -> None:
    parts.append("\n变量说明与示例：\n")
    parts.extend(f"- {item}" for item in items)


def state_macro(
    datasets: list[str],
    data: dict[str, dict[str, Any]],
    state: str,
    getter: Callable[[dict[str, Any]], Optional[float]],
) -> Optional[float]:
    values = []
    for name in datasets:
        if name == "aime24":
            continue
        item = ((decode(data[name]).get("direct_mask_redraft") or {}).get("state_stats") or {}).get(state) or {}
        values.append(getter(item))
    return macro(values)


def generate(result_dir: Path, b16_dir: Path, b32_dir: Path) -> str:
    current = discover(result_dir)
    b16 = discover(b16_dir) if b16_dir.is_dir() else {}
    b32 = discover(b32_dir) if b32_dir.is_dir() else {}
    names = [name for name in ORDER if name in current]
    if not names:
        raise ValueError(f"No metrics_*.json found in {result_dir}")
    for name in names:
        validate_metrics(name, current[name])
    included = [name for name in names if name != "aime24"]
    label = equal_label(names)
    complete = set(ORDER).issubset(current)
    current_status = validate_current_settings(result_dir)
    b16_status = validate_baseline(b16_dir, 16)
    b32_status = validate_baseline(b32_dir, 32)
    parts = [
        f"时间戳：{datetime.now().astimezone().isoformat(timespec='seconds')}",
        "\n# Strict Direct MASK-Redraft 实验报告",
        "\n## 1. 口径与完整性",
        "",
        f"- 结果目录：`{result_dir}`",
        f"- B16 baseline：`{b16_dir}`",
        f"- B32 baseline：`{b32_dir}`",
        f"- 新方法配置：{current_status}。",
        f"- B16 配置：{b16_status}；B32 配置：{b32_status}。",
        f"- 当前状态：{'十数据集完整结果' if complete else '部分数据/smoke 结果'}。",
        "- AIME24 可以保留原始 metrics，但不进入任何平均。其余数据集先各自统计，再按数据集等权平均，不按样本数加权。",
        "- Matched 是草稿中经 verifier 一致性检查通过的 token 数；Emitted=Matched+1，额外包含 verifier 修正或 bonus token。",
    ]
    if not complete:
        parts.append("- 当前是子集/smoke：其 Accuracy 与全量 baseline 的样本集合不一致，只用于检查全链路，不能据此评价正式精度。")

    compare_rows = []
    for name in names:
        new = tpf(current[name])
        old16 = tpf(b16.get(name))
        old32 = tpf(b32.get(name))
        compare_rows.append(
            [DISPLAY[name], fmt(old16), fmt(old32), fmt(new), fmt(None if new is None or old16 is None else new-old16), fmt(None if new is None or old32 is None else new-old32)]
        )
    compare_rows.append(
        [
            label,
            fmt(macro([tpf(b16.get(n)) for n in included])),
            fmt(macro([tpf(b32.get(n)) for n in included])),
            fmt(macro([tpf(current[n]) for n in included])),
            fmt(macro([None if tpf(current[n]) is None or tpf(b16.get(n)) is None else tpf(current[n])-tpf(b16.get(n)) for n in included])),
            fmt(macro([None if tpf(current[n]) is None or tpf(b32.get(n)) is None else tpf(current[n])-tpf(b32.get(n)) for n in included])),
        ]
    )
    parts.extend(["\n## 2. TPF 对比", "", table(["数据集", "B16", "B32", "新法", "Δ16", "Δ32"], compare_rows)])
    append_defs(parts, [
        "B16/B32：PyTorch+NeMo-Skills+greedy 的 block size 16/32 baseline TPF；例如 B16=5 表示每次物理 forward 平均返回 5 token。",
        "新法：本方法 Completion tokens/physical NFE；融合双行仍计一次 NFE。",
        "Δ16/Δ32：新法 TPF 减对应 baseline；正数表示本方法更高。",
    ])

    acc_rows = []
    for name in names:
        new = accuracy(name, current[name])
        old16 = accuracy(name, b16.get(name))
        old32 = accuracy(name, b32.get(name))
        acc_rows.append([DISPLAY[name], fmt(old16, 4, "%"), fmt(old32, 4, "%"), fmt(new, 4, "%"), fmt(None if new is None or old16 is None else new-old16, 4, " pp"), fmt(None if new is None or old32 is None else new-old32, 4, " pp")])
    acc_rows.append([
        label,
        fmt(macro([accuracy(n, b16.get(n)) for n in included]), 4, "%"),
        fmt(macro([accuracy(n, b32.get(n)) for n in included]), 4, "%"),
        fmt(macro([accuracy(n, current[n]) for n in included]), 4, "%"),
        fmt(macro([None if accuracy(n, current[n]) is None or accuracy(n, b16.get(n)) is None else accuracy(n, current[n])-accuracy(n, b16.get(n)) for n in included]), 4, " pp"),
        fmt(macro([None if accuracy(n, current[n]) is None or accuracy(n, b32.get(n)) is None else accuracy(n, current[n])-accuracy(n, b32.get(n)) for n in included]), 4, " pp"),
    ])
    parts.extend(["\n## 3. Accuracy 对比", "", table(["数据集", "B16", "B32", "新法", "Δ16", "Δ32"], acc_rows)])
    append_defs(parts, [
        "Accuracy：代码题用 Base pass@1，LiveCodeBench 用 accuracy，IFEval 用 average_score，其余用 symbolic_correct。",
        "Δ16/Δ32：新法减 baseline，单位是百分点 pp；例如 80%-79%=+1 pp。",
    ])

    work_fields = (
        ("request_count", "请求"),
        ("completion_tokens", "Tok"),
        ("forward_passes", "NFE"),
        ("rounds", "轮数"),
        ("redraft_saved_draft_forwards", "Saved"),
        ("processed_rows", "Rows"),
        ("processed_query_tokens", "QTok"),
    )
    work_rows = []
    for name in names:
        dec = decode(current[name])
        red = dec.get("direct_mask_redraft") or {}
        vals = []
        for key, _ in work_fields:
            vals.append(finite(dec.get(key)) if key in dec else finite(red.get(key)))
        work_rows.append([DISPLAY[name]] + [fmt_count(v) for v in vals] + [fmt(tpf(current[name]))])
    work_rows.append([label] + [fmt_count(macro([finite(decode(current[n]).get(key)) if key in decode(current[n]) else finite((decode(current[n]).get("direct_mask_redraft") or {}).get(key)) for n in included])) for key, _ in work_fields] + [fmt(macro([tpf(current[n]) for n in included]))])
    parts.extend(["\n## 4. 工作量", "", table(["数据集"] + [short for _, short in work_fields] + ["TPF"], work_rows)])
    append_defs(parts, [
        "请求/Tok：成功请求数与返回 completion token 总数；例如请求=2、Tok=400 表示两条请求共返回 400 token。",
        "NFE/轮数：物理 encoder 调用数与逻辑 verify 轮数；每轮必有 verifier，Saved 只省普通 draft。",
        "Saved：下一轮实际消费完整 row 1、因此省掉普通 draft forward 的次数。",
        "Rows/QTok：所有 forward 处理的 batch row 数及 batch×query length，总量用于暴露融合双行的隐藏工作。",
    ])

    perf_rows = []
    for name in names:
        dec = decode(current[name])
        peak = dec.get("peak_gpu_memory_gib") or {}
        perf_rows.append([
            DISPLAY[name],
            fmt(finite(dec.get("model_output_tokens_per_s")), 2),
            fmt(finite(dec.get("benchmark_wall_output_tokens_per_s")), 2),
            fmt(finite(peak.get("mean")), 2),
        ])
    perf_rows.append([
        label,
        fmt(macro([finite(decode(current[n]).get("model_output_tokens_per_s")) for n in included]), 2),
        fmt(macro([finite(decode(current[n]).get("benchmark_wall_output_tokens_per_s")) for n in included]), 2),
        fmt(macro([finite((decode(current[n]).get("peak_gpu_memory_gib") or {}).get("mean")) for n in included]), 2),
    ])
    parts.extend(["\n### 4.1 TPS 与显存参考", "", table(["数据集", "ModelTPS", "WallTPS", "PeakGiB"], perf_rows)])
    append_defs(parts, [
        "ModelTPS：completion token 除以同步后的原生模型生成时间；不含 prompt 格式化、HTTP 和评分，例如 200 表示模型阶段约 200 token/s。",
        "WallTPS：completion token 除以整个 benchmark 命令墙钟时间，包含 NeMo-Skills 调度与评分，仅作端到端参考。",
        "PeakGiB：请求期间 torch 记录的峰值已分配显存；共享 GPU 上不代表整卡总占用。",
    ])

    state_rows = []
    for name in names:
        red = decode(current[name]).get("direct_mask_redraft") or {}
        states = red.get("state_stats") or {}
        row = [DISPLAY[name], fmt_count(finite(red.get("redraft_attempts")))]
        for state in STATE_ORDER:
            item = states.get(state) or {}
            count = finite(item.get("count"))
            share = finite(item.get("attempt_share"))
            row.append("—" if count is None else f"{count:,.0f}({(share or 0)*100:.2f}%)")
        state_rows.append(row)
    avg_row = [label, fmt_count(macro([finite((decode(current[n]).get("direct_mask_redraft") or {}).get("redraft_attempts")) for n in included]))]
    for state in STATE_ORDER:
        avg_count = state_macro(included, current, state, lambda item: finite(item.get("count")))
        avg_share = state_macro(included, current, state, lambda item: finite(item.get("attempt_share")))
        avg_row.append("—" if avg_count is None else f"{avg_count:,.2f}({(avg_share or 0)*100:.2f}%)")
    state_rows.append(avg_row)
    parts.extend(["\n## 5. 六类决策状态", "", table(["数据集", "尝试"] + [STATE_SHORT[s] for s in STATE_ORDER], state_rows)])
    append_defs(parts, [
        "尝试：实际执行 fused verifier+row 1 的轮数；括号内占比均以该数据集尝试数为分母。",
        "m<p：verifier 在触发位置 p 前已拒绝；例如 p=8、m=5。",
        "直中：m=p、C≠A 且 R0=C，是唯一可复用状态；例如 A=x、C=y、R0=y。",
        "重A：m=p、C≠A 且 R0=A，row 1 重复原错误 token。",
        "改错：m=p，但 R0 既不是 A 也不是正确 C。",
        "A对后拒：A 在 p 正确，但 verifier 在更后位置拒绝；新规则无条件丢弃 row 1。",
        "Bonus：当前完整 draft 通过并产生 verifier bonus；新规则无条件丢弃 row 1。",
    ])

    split_rows = []
    split_fields = (
        ("redraft_a_ok_r0_a", "后拒R=A"),
        ("redraft_a_ok_r0_changed", "后拒R≠A"),
        ("redraft_bonus_r0_a", "BonusR=A"),
        ("redraft_bonus_r0_changed", "BonusR≠A"),
        ("redraft_discarded_eos", "EOS"),
        ("redraft_discarded_generation_end", "GenEnd"),
        ("redraft_discarded_thinking_budget", "Think"),
    )
    for name in names:
        red = decode(current[name]).get("direct_mask_redraft") or {}
        split_rows.append([DISPLAY[name]] + [fmt_count(finite(red.get(key))) for key, _ in split_fields])
    split_rows.append([label] + [fmt_count(macro([finite((decode(current[n]).get("direct_mask_redraft") or {}).get(key)) for n in included])) for key, _ in split_fields])
    parts.extend(["\n## 6. A 正确子类与边界", "", table(["数据集"] + [short for _, short in split_fields], split_rows)])
    append_defs(parts, [
        "后拒R=A/后拒R≠A：A 在 p 正确且后面才拒绝时，row 1 在 p 保持 A/错误改写 A 的次数。",
        "BonusR=A/BonusR≠A：整块通过时，row 1 在 p 保持 A/改写 A 的次数。",
        "EOS/GenEnd/Think：直中已经发生，但因 EOS、生成预算结束或 thinking seed 强制替换而没有复用的次数。",
    ])

    trigger_rows = []
    trigger_fields = (
        ("rounds_without_candidate", "NoCand"),
        ("redraft_skipped_no_future_round", "NoFut"),
        ("redraft_skipped_context_limit", "Ctx"),
        ("full_length_reuses", "FullL"),
    )
    for name in names:
        red = decode(current[name]).get("direct_mask_redraft") or {}
        rounds = finite(red.get("rounds"))
        attempts = finite(red.get("redraft_attempts"))
        trigger_rows.append([
            DISPLAY[name],
            fmt(None if not rounds or attempts is None else attempts / rounds * 100, 2, "%"),
            fmt(finite(red.get("average_candidate_position")), 2),
            *[fmt_count(finite(red.get(key))) for key, _ in trigger_fields],
        ])
    trigger_rows.append([
        label,
        fmt(macro([None if not finite((decode(current[n]).get("direct_mask_redraft") or {}).get("rounds")) else finite((decode(current[n]).get("direct_mask_redraft") or {}).get("redraft_attempts")) / finite((decode(current[n]).get("direct_mask_redraft") or {}).get("rounds")) * 100 for n in included]), 2, "%"),
        fmt(macro([finite((decode(current[n]).get("direct_mask_redraft") or {}).get("average_candidate_position")) for n in included]), 2),
        *[fmt_count(macro([finite((decode(current[n]).get("direct_mask_redraft") or {}).get(key)) for n in included])) for key, _ in trigger_fields],
    ])
    parts.extend(["\n### 6.1 触发覆盖与非尝试边界", "", table(["数据集", "Try%", "AvgP"] + [short for _, short in trigger_fields], trigger_rows)])
    append_defs(parts, [
        "Try%：redraft attempts / verifier rounds；例如 80% 表示八成轮次实际执行了双行 fused redraft。",
        "AvgP：有尝试轮次的平均触发位置 p；p 是 row 1 左侧 prefix 长度，也是候选在 draft tensor 中的下标。",
        "NoCand：本轮没有 token_y_drop_pct 严格超过阈值，恢复普通 verify。",
        "NoFut：发现候选但生成预算已不足以形成有价值的下一轮，未执行 row 1。",
        "Ctx：p+L 会超过模型位置上限，未执行 row 1。",
        "FullL：实际保留并在下一轮消费的完整 L-token row 1 数；严格等于 Reuse 与 Saved。",
    ])

    transition_rows = []
    for name in names:
        states = ((decode(current[name]).get("direct_mask_redraft") or {}).get("state_stats") or {})
        for state in STATE_ORDER:
            item = states.get(state) or {}
            transition_rows.append([
                DISPLAY[name], STATE_SHORT[state], fmt_count(finite(item.get("count"))),
                fmt(finite(item.get("current_matched_mean")), 3),
                fmt(finite(item.get("current_emitted_mean")), 3),
                fmt_count(finite(item.get("next_observed_count"))),
                fmt(finite(item.get("next_matched_mean")), 3),
                fmt(finite(item.get("next_emitted_mean")), 3),
                fmt(finite(item.get("next_minus_current_matched_mean")), 3),
                fmt(finite(item.get("next_minus_current_emitted_mean")), 3),
                fmt_count(finite(item.get("no_next_round_count"))),
            ])
    for state in STATE_ORDER:
        transition_rows.append([
            label, STATE_SHORT[state],
            fmt_count(state_macro(included, current, state, lambda x: finite(x.get("count")))),
            fmt(state_macro(included, current, state, lambda x: finite(x.get("current_matched_mean"))), 3),
            fmt(state_macro(included, current, state, lambda x: finite(x.get("current_emitted_mean"))), 3),
            fmt_count(state_macro(included, current, state, lambda x: finite(x.get("next_observed_count")))),
            fmt(state_macro(included, current, state, lambda x: finite(x.get("next_matched_mean"))), 3),
            fmt(state_macro(included, current, state, lambda x: finite(x.get("next_emitted_mean"))), 3),
            fmt(state_macro(included, current, state, lambda x: finite(x.get("next_minus_current_matched_mean"))), 3),
            fmt(state_macro(included, current, state, lambda x: finite(x.get("next_minus_current_emitted_mean"))), 3),
            fmt_count(state_macro(included, current, state, lambda x: finite(x.get("no_next_round_count")))),
        ])
    parts.extend(["\n## 7. 各状态的下一轮验证质量", "", table(["数据集", "状态", "N", "本M", "本E", "下N", "下M", "下E", "ΔM", "ΔE", "无下轮"], transition_rows)])
    append_defs(parts, [
        "N：当前状态次数；本M/本E：当前轮平均 Matched/Emitted。",
        "下N：确实存在下一轮并完成配对统计的次数；下M/下E：这些事件下一轮的平均 Matched/Emitted。",
        "ΔM/ΔE：对同一批有下一轮事件逐事件计算“下一轮−当前轮”后取均值；例如本M=4、下M=7，则 ΔM=+3。",
        "无下轮：因 EOS 或 generation end 等原因不存在下一轮的事件，不会按 0 混入下M/下E。",
        "等权平均：先计算每个数据集自己的均值，再对有定义的数据集等权平均；没有该状态的数据集不参与该状态均值。",
    ])

    parts.extend([
        "\n## 8. 自动一致性检查",
        "",
        "- 六个尝试状态在每个 metrics 文件中应严格划分 redraft attempts。",
        "- 全部 round 状态（含无候选和边界跳过）应严格划分 rounds。",
        "- 新方法只允许 full-length reuse，不生成 partial draft。",
        "- 只有 causal verifier token 和 row 0 KV cache 可以提交。",
        f"- 已对 {len(names)} 个 metrics 文件执行状态划分、下一轮配对与 full-length reuse 一致性校验：全部通过。",
    ])
    return "\n".join(parts).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--baseline-b16", default=str(DEFAULT_B16))
    parser.add_argument("--baseline-b32", default=str(DEFAULT_B32))
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    result_dir = Path(args.result_dir).resolve()
    output = Path(args.output).resolve() if args.output else result_dir / "report.md"
    report = generate(result_dir, Path(args.baseline_b16).resolve(), Path(args.baseline_b32).resolve())
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(report, encoding="utf-8")
    temporary.replace(output)
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
