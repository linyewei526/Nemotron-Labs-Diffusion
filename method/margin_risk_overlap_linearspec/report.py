#!/usr/bin/env python3
"""Incrementally render the fixed-margin-risk overlap experiment report."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable, Optional


DEFAULT_BASELINE_16 = Path(
    "/data/home/wly/dLLM/NLD_results/observations/pytorch_nemo_eval_results/eval_20260804_120138"
)
DEFAULT_BASELINE_32 = Path(
    "/data/home/wly/dLLM/NLD_results/observations/pytorch_nemo_eval_results/eval_20260804_114935"
)
DATASETS = (
    "gsm8k",
    "human-eval",
    "mbpp",
    "math-500",
    "aime25",
    "gpqa",
    "mmlu",
    "ifeval",
    "livecodebench-cpp",
)
OUTCOME_STATES = (
    "before_candidate_error",
    "candidate_fixed_by_alternative",
    "candidate_wrong_alternative",
    "after_candidate_error",
    "full_block_bonus",
)
STATE_ZH = {
    "before_candidate_error": "预测前出错",
    "candidate_fixed_by_alternative": "预测位B修正对",
    "candidate_wrong_alternative": "预测位B仍错",
    "after_candidate_error": "预测后出错",
    "full_block_bonus": "整块通过+bonus",
}
FUNNEL_FIELDS = (
    ("rounds", "轮数"),
    ("rounds_without_candidate", "无候选"),
    ("prefetch_attempts", "实尝试"),
    ("prefetch_verified_hits", "B验中"),
    ("prefetch_hits", "可复用"),
    ("prefetch_saved_draft_forwards", "已复用"),
    ("prefetch_skipped_no_future_round", "跳过终轮"),
    ("prefetch_skipped_context_limit", "跳过上下文"),
    ("prefetch_skipped_thinking_budget", "跳过思考"),
)


def read_json(path: Path) -> Optional[dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def number(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def integer(value: Any) -> int:
    parsed = number(value)
    return int(parsed) if parsed is not None else 0


def mean(values: Iterable[Any]) -> Optional[float]:
    clean = [parsed for value in values if (parsed := number(value)) is not None]
    return fmean(clean) if clean else None


def fmt(value: Any, digits: int = 4) -> str:
    parsed = number(value)
    if parsed is None:
        return "—"
    if abs(parsed - round(parsed)) < 1e-12:
        return str(int(round(parsed)))
    return f"{parsed:.{digits}f}".rstrip("0").rstrip(".")


def pct(value: Any) -> str:
    parsed = number(value)
    return "—" if parsed is None else f"{parsed * 100:.2f}%"


def text_value(value: Any, missing: str = "—") -> str:
    return missing if value is None or value == "" else str(value)


def table(headers: list[str], rows: list[list[Any]]) -> str:
    rendered = [[str(cell) for cell in row] for row in rows]
    lines = ["|" + "|".join(headers) + "|", "|" + "|".join(":---:" for _ in headers) + "|"]
    lines.extend("|" + "|".join(row) + "|" for row in rendered)
    return "\n".join(lines)


def metric_path(root: Path, dataset: str) -> Path:
    return root / f"metrics_{dataset}.json"


def load_metrics(root: Path) -> dict[str, dict[str, Any]]:
    return {
        dataset: payload
        for dataset in DATASETS
        if (payload := read_json(metric_path(root, dataset))) is not None
    }


def decode(payload: dict[str, Any], *, new_method: bool) -> dict[str, Any]:
    key = "pytorch_margin_risk_overlap" if new_method else "pytorch_native"
    wrapper = payload.get(key)
    if not isinstance(wrapper, dict):
        return {}
    result = wrapper.get("decode")
    return result if isinstance(result, dict) else {}


def overlap(payload: dict[str, Any]) -> dict[str, Any]:
    result = decode(payload, new_method=True).get("overlap")
    return result if isinstance(result, dict) else {}


def config_summary(root: Path) -> dict[str, Any]:
    settings = read_json(root / "Settings.json") or {}
    bench = settings.get("benchmark") if isinstance(settings.get("benchmark"), dict) else {}
    pytorch = settings.get("pytorch") if isinstance(settings.get("pytorch"), dict) else {}
    return {
        "path": str(root),
        "status": settings.get("status"),
        "mode": pytorch.get("mode"),
        "block": pytorch.get("block_length"),
        "threshold": pytorch.get("draft_threshold", pytorch.get("threshold")),
        "margin_risk": pytorch.get("margin_risk_threshold"),
        "tokens": bench.get("tokens"),
        "context": pytorch.get("context_length"),
        "temperature": bench.get("temperature"),
        "thinking": pytorch.get("enable_thinking", bench.get("enable_thinking")),
        "dtype": pytorch.get("dtype"),
        "gpu": pytorch.get("gpu_device"),
        "reserve": pytorch.get("gpu_memory_reserve_gb"),
    }


def config_warnings(new: dict[str, Any], b16: dict[str, Any], b32: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    for label, baseline, expected in (("B16", b16, 16), ("B32", b32, 32)):
        if baseline.get("block") != expected:
            warnings.append(f"{label} 的 block size 不是预期的 {expected}。")
        if baseline.get("mode") != "linearspec_lora":
            warnings.append(f"{label} 不是 linearspec_lora baseline。")
        for key, text in (("tokens", "tokens"), ("context", "context"), ("temperature", "temperature"), ("thinking", "thinking"), ("dtype", "dtype")):
            if new.get(key) is not None and baseline.get(key) is not None and new.get(key) != baseline.get(key):
                warnings.append(f"新方法与 {label} 的 {text} 不一致：{new.get(key)} vs {baseline.get(key)}。")
    if b16.get("gpu") != b32.get("gpu") or b16.get("reserve") != b32.get("reserve"):
        warnings.append("B16 与 B32 使用的 GPU 或预留显存不同；TPS 只作参考，不能归因为 block size。")
    return warnings


def render(result_dir: Path, baseline16: Path, baseline32: Path) -> str:
    new_metrics = load_metrics(result_dir)
    b16_metrics = load_metrics(baseline16)
    b32_metrics = load_metrics(baseline32)
    completed = [dataset for dataset in DATASETS if dataset in new_metrics]
    new_cfg = config_summary(result_dir)
    b16_cfg = config_summary(baseline16)
    b32_cfg = config_summary(baseline32)
    settings = read_json(result_dir / "Settings.json") or {}
    status = settings.get("status", "initialized")
    lines = [
        "# 固定 margin-risk=0.5 重起草实验报告",
        "",
        f"> 状态：`{status}`；非 AIME24 数据集完成 `{len(completed)}/9`：`{', '.join(completed) if completed else '尚无'}`。本文件由每个数据集完成后的增量步骤重写。AIME24 即使运行也不进入本报告主表和等权平均。",
        "",
        "## 1. 配置与对照来源",
        "",
        "变量说明：`MR阈` 是 margin-risk 的严格触发阈值；`BS` 是 block size；`Tok/Ctx` 是最大生成/上下文 token；`预留` 是加载模型前占用的显存 GiB。B16/B32 分别是既有 PyTorch+NeMo-Skills、LinearSpec LoRA、greedy、block size 16/32 baseline。示例：`MR阈=0.5` 表示只报告最左侧严格满足 margin-risk 大于 0.5 的位置，等于 0.5 不报告。",
        "",
        table(
            ["方法", "状态", "模式", "BS", "MR阈", "Tok", "Ctx", "温度", "思考", "dtype", "GPU", "预留"],
            [
                ["新方法", text_value(new_cfg["status"]), text_value(new_cfg["mode"]), fmt(new_cfg["block"]), fmt(new_cfg["margin_risk"]), fmt(new_cfg["tokens"]), fmt(new_cfg["context"]), fmt(new_cfg["temperature"]), text_value(new_cfg["thinking"]), text_value(new_cfg["dtype"]), text_value(new_cfg["gpu"]), fmt(new_cfg["reserve"])],
                ["B16 greedy", text_value(b16_cfg["status"], "未记录"), text_value(b16_cfg["mode"]), fmt(b16_cfg["block"]), "—", fmt(b16_cfg["tokens"]), fmt(b16_cfg["context"]), fmt(b16_cfg["temperature"]), text_value(b16_cfg["thinking"]), text_value(b16_cfg["dtype"]), text_value(b16_cfg["gpu"]), fmt(b16_cfg["reserve"])],
                ["B32 greedy", text_value(b32_cfg["status"], "未记录"), text_value(b32_cfg["mode"]), fmt(b32_cfg["block"]), "—", fmt(b32_cfg["tokens"]), fmt(b32_cfg["context"]), fmt(b32_cfg["temperature"]), text_value(b32_cfg["thinking"]), text_value(b32_cfg["dtype"]), text_value(b32_cfg["gpu"]), fmt(b32_cfg["reserve"])],
            ],
        ),
        "",
        f"- B16：`{baseline16}`",
        f"- B32：`{baseline32}`",
    ]
    warnings = config_warnings(new_cfg, b16_cfg, b32_cfg)
    if warnings:
        lines.extend(["", "配置核验提示："] + [f"- {item}" for item in warnings])

    lines.extend([
        "",
        "## 2. 请求覆盖与解码效率对比",
        "",
        "变量说明：`Att/OK/Fail/OOM` 是请求尝试数、纳入效率聚合的成功数、排除的失败数和其中因显存不足跳过的数量；`Cov=OK/Att`。`TPF` 是成功请求返回 token/物理 encoder forward；`NFE` 是成功请求每样本平均物理 forward；`TPS` 是同步模型生成阶段 token/s。示例：Att=164、OK=163、OOM=1、Cov=99.39% 时，效率均值只基于 163 个成功请求，不能误读为 164 个请求全量结果。accuracy 不进入本报告。",
        "",
    ])
    comparison_rows: list[list[Any]] = []
    comparison_values: dict[str, list[float]] = {key: [] for key in ("att", "ok", "fail", "oom", "cov", "nt", "bt16", "bt32", "nn", "bn16", "bn32", "np", "bp16", "bp32")}
    for dataset in completed:
        new = new_metrics[dataset]
        b16 = b16_metrics.get(dataset, {})
        b32 = b32_metrics.get(dataset, {})
        nd = decode(new, new_method=True)
        d16 = decode(b16, new_method=False)
        d32 = decode(b32, new_method=False)
        values = {
            "att": nd.get("attempted_request_count", (number(nd.get("request_count")) or 0) + (number(nd.get("failed_request_count")) or 0)),
            "ok": nd.get("request_count"), "fail": nd.get("failed_request_count"),
            "oom": nd.get("oom_skipped_request_count"), "cov": nd.get("successful_request_rate"),
            "nt": nd.get("tokens_per_forward_pass"), "bt16": d16.get("tokens_per_forward_pass"), "bt32": d32.get("tokens_per_forward_pass"),
            "nn": nd.get("average_forward_passes_per_sample"), "bn16": d16.get("average_forward_passes_per_sample"), "bn32": d32.get("average_forward_passes_per_sample"),
            "np": nd.get("model_output_tokens_per_s"), "bp16": d16.get("model_output_tokens_per_s"), "bp32": d32.get("model_output_tokens_per_s"),
        }
        for key, value in values.items():
            if (parsed := number(value)) is not None:
                comparison_values[key].append(parsed)
        comparison_rows.append([dataset, fmt(values["att"]), fmt(values["ok"]), fmt(values["fail"]), fmt(values["oom"]), pct(values["cov"]), fmt(values["nt"]), fmt(values["bt16"]), fmt(values["bt32"]), fmt(values["nn"]), fmt(values["bn16"]), fmt(values["bn32"]), fmt(values["np"]), fmt(values["bp16"]), fmt(values["bp32"])])
    if completed:
        comparison_rows.append([f"等权均值({len(completed)}/9)", *[fmt(mean(comparison_values[key])) for key in ("att", "ok", "fail", "oom")], pct(mean(comparison_values["cov"])), *[fmt(mean(comparison_values[key])) for key in ("nt", "bt16", "bt32", "nn", "bn16", "bn32", "np", "bp16", "bp32")]])
        lines.append(table(["数据集", "Att", "OK", "Fail", "OOM", "Cov", "新TPF", "B16TPF", "B32TPF", "新NFE", "B16NFE", "B32NFE", "新TPS", "B16TPS", "B32TPS"], comparison_rows))
    else:
        lines.append("尚无已完成的新方法数据集；表格会在首个数据集完成后出现。")

    lines.extend([
        "",
        "## 3. 候选与复用漏斗",
        "",
        "变量说明：`轮数` 是 verify 轮总数；`无候选` 是没有 margin-risk 候选的轮数；`实尝试` 是实际执行双行融合的轮数；`B验中` 是首错恰在预测位且 B 正确；`可复用` 是命中后满足 EOS/思考/预算条件并保存 prospective draft；`已复用` 是该 draft 在下一轮真正替代普通 draft forward。各跳过列表示虽找到候选但因边界条件未发起融合。示例：`B验中=20、可复用=18、已复用=17` 表示 20 次验证命中中有 18 次可保存，其中 17 次确实进入下一轮。",
        "",
    ])
    funnel_rows: list[list[Any]] = []
    for dataset in completed:
        values = overlap(new_metrics[dataset])
        funnel_rows.append([dataset] + [fmt(values.get(key)) for key, _ in FUNNEL_FIELDS])
    if completed:
        funnel_rows.append(["总计"] + [fmt(sum(integer(overlap(new_metrics[d]).get(key)) for d in completed)) for key, _ in FUNNEL_FIELDS])
        funnel_rows.append([f"等权均次({len(completed)}/9)"] + [fmt(mean(integer(overlap(new_metrics[d]).get(key)) for d in completed)) for key, _ in FUNNEL_FIELDS])
        lines.append(table(["数据集"] + [label for _, label in FUNNEL_FIELDS], funnel_rows))
    else:
        lines.append("尚无漏斗统计。")

    lines.extend([
        "",
        "## 4. 五种互斥状态的计数与占比",
        "",
        "变量说明：`预测前` 表示 q<p；`B修正对` 表示 q=p 且替代 token B 正确；`B仍错` 表示 q=p 且 B 也错误；`预测后` 表示 q>p，即预测位置原 token 实际正确；`整块+bonus` 表示没有首错，整块通过并产生 bonus。每个单元格为 `次数/占实尝试比例`。示例：`12/30.00%` 表示该状态出现 12 次，占本数据集所有实际 overlap 尝试的 30%。五类次数之和必须等于实尝试。",
        "",
    ])
    state_count_rows: list[list[Any]] = []
    totals = {state: 0 for state in OUTCOME_STATES}
    counts_by_state = {state: [] for state in OUTCOME_STATES}
    macro_shares = {state: [] for state in OUTCOME_STATES}
    for dataset in completed:
        values = overlap(new_metrics[dataset])
        states = values.get("outcome_states") if isinstance(values.get("outcome_states"), dict) else {}
        row: list[Any] = [dataset]
        for state in OUTCOME_STATES:
            item = states.get(state) if isinstance(states.get(state), dict) else {}
            count = integer(item.get("count"))
            totals[state] += count
            counts_by_state[state].append(count)
            if number(item.get("share_of_attempts")) is not None:
                macro_shares[state].append(float(item["share_of_attempts"]))
            row.append(f"{count}/{pct(item.get('share_of_attempts'))}")
        row.append("是" if values.get("outcome_partition_valid") is True else "否")
        state_count_rows.append(row)
    if completed:
        state_count_rows.append(["九集总计(当前)"] + [str(totals[state]) for state in OUTCOME_STATES] + ["—"])
        state_count_rows.append([f"等权均值({len(completed)}/9)"] + [f"{fmt(mean(counts_by_state[state]))}/{pct(mean(macro_shares[state]))}" for state in OUTCOME_STATES] + ["—"])
        lines.append(table(["数据集", "预测前", "B修正对", "B仍错", "预测后", "整块+bonus", "分区校验"], state_count_rows))
    else:
        lines.append("尚无状态统计。")

    lines.extend([
        "",
        "## 5. 各状态当前轮与下一轮 verify 接收量",
        "",
        "变量说明：verify 接收长度按“连续匹配的 draft token 数+本轮 verifier 产出的 1 个 token”计算；`Cnt` 是状态次数；`占比` 是状态次数/实尝试；`本轮均` 是该状态全部实例的当前 verify 接收长度均值；`NextN` 是确实观察到同一 request 下一轮 verify 的实例数；`NextCov` 是 NextN/Cnt；`配对本轮` 只在有下一轮的配对样本上计算当前均值；`下轮均` 是下一轮 verify 接收长度均值；`差值` 是逐对计算“下轮接收长度−当前轮接收长度”后取平均。示例：当前轮接收 3、下一轮接收 5，则该配对的差值为 +2。因 EOS/预算终止而不存在下一轮的实例不按 0 填充。",
        "",
    ])
    transition_rows: list[list[Any]] = []
    macro_fields = ("share_of_attempts", "current_accept_avg", "next_coverage", "paired_current_accept_avg", "next_accept_avg", "next_minus_current_avg")
    macro_by_state = {state: {field: [] for field in macro_fields} for state in OUTCOME_STATES}
    next_totals = {state: 0 for state in OUTCOME_STATES}
    for dataset in completed:
        states = overlap(new_metrics[dataset]).get("outcome_states") or {}
        for state in OUTCOME_STATES:
            item = states.get(state) if isinstance(states.get(state), dict) else {}
            next_totals[state] += integer(item.get("next_count"))
            for field in macro_fields:
                if (parsed := number(item.get(field))) is not None:
                    macro_by_state[state][field].append(parsed)
            transition_rows.append([dataset, STATE_ZH[state], fmt(item.get("count")), pct(item.get("share_of_attempts")), fmt(item.get("current_accept_avg")), fmt(item.get("next_count")), pct(item.get("next_coverage")), fmt(item.get("paired_current_accept_avg")), fmt(item.get("next_accept_avg")), fmt(item.get("next_minus_current_avg"))])
    if completed:
        for state in OUTCOME_STATES:
            values = macro_by_state[state]
            transition_rows.append([f"等权均值({len(completed)}/9)", STATE_ZH[state], fmt(mean(counts_by_state[state])), pct(mean(values["share_of_attempts"])), fmt(mean(values["current_accept_avg"])), fmt(mean(integer((overlap(new_metrics[d]).get("outcome_states") or {}).get(state, {}).get("next_count")) for d in completed)), pct(mean(values["next_coverage"])), fmt(mean(values["paired_current_accept_avg"])), fmt(mean(values["next_accept_avg"])), fmt(mean(values["next_minus_current_avg"]))])
        lines.append(table(["数据集", "状态", "Cnt", "占比", "本轮均", "NextN", "NextCov", "配对本轮", "下轮均", "差值"], transition_rows))
    else:
        lines.append("尚无跨轮统计。")

    lines.extend([
        "",
        "## 6. 统计口径与完整性",
        "",
        "- 九数据集等权：所有比例、均值先在每个数据集内部计算，再对已完成的非 AIME24 数据集做算术平均；不会按样本数或轮次数加权。最终应显示 `9/9`。",
        "- `九集总计` 仅用于审计绝对事件数，不作为全局比例的分母；因此 MMLU 的大量样本不会覆盖小数据集。",
        "- 五状态只覆盖实际发起融合 overlap 的轮次；未找到候选或因边界跳过的轮次在漏斗表中另列。",
        "- 下一轮统计是同一请求的相邻 verify 描述性转移，不把无下一轮的终止状态补成 0，也不单独构成因果结论。",
        "- 某状态在某数据集没有实例时，其占比按 0 进入等权平均；该状态的接收均值和 NextCov 没有定义，因此只在有定义的数据集间平均。",
        "- 准确率不进入本报告。OOM/失败请求从效率聚合中排除，并通过 Att、OK、Fail、OOM、Cov 显式披露；Cov 不足 100% 时只代表成功请求子集。",
        "- B16/B32 TPS 若 GPU 或预留显存不同，只作运行量级参考；TPF、NFE 和请求覆盖是更直接的算法对照。",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--baseline-block16-dir", default=str(DEFAULT_BASELINE_16))
    parser.add_argument("--baseline-block32-dir", default=str(DEFAULT_BASELINE_32))
    args = parser.parse_args()
    result_dir = Path(args.result_dir).resolve()
    result_dir.mkdir(parents=True, exist_ok=True)
    content = render(result_dir, Path(args.baseline_block16_dir).resolve(), Path(args.baseline_block32_dir).resolve())
    target = result_dir / "report.md"
    temporary = target.with_name(f".{target.name}.tmp.{os.getpid()}")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, target)
    print(f"Updated {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
