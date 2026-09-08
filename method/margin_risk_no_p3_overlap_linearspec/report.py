#!/usr/bin/env python3
"""Incrementally render the fixed-margin-risk no-P3 overlap report."""

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
DEFAULT_REPORT_DATASETS = (
    "gsm8k", "human-eval", "mbpp", "math-500", "aime25", "gpqa",
    "ifeval", "livecodebench-cpp",
)
REPORT_EXCLUDED_DATASETS = frozenset({"aime24", "mmlu"})
OUTCOME_STATES = (
    "miss_no_risk_position",
    "miss_before_first",
    "miss_between_positions",
    "miss_after_last",
    "p1_rank2_fixed",
    "p1_rank2_wrong",
    "p1_no_executable_candidate",
    "p2_rank2_fixed",
    "p2_rank2_wrong",
    "p2_no_executable_candidate",
    "p3_omitted_rank2_would_fix",
    "p3_omitted_rank2_wrong",
    "full_continuation_hit",
    "full_continuation_miss",
    "full_continuation_absent",
)
STATE_ZH = {
    "miss_no_risk_position": "无风险位但出错",
    "miss_before_first": "P1前出错",
    "miss_between_positions": "风险位间出错",
    "miss_after_last": "末风险位后出错",
    "p1_rank2_fixed": "P1二选命中",
    "p1_rank2_wrong": "P1二选错误",
    "p1_no_executable_candidate": "P1候选不可执行",
    "p2_rank2_fixed": "P2二选命中",
    "p2_rank2_wrong": "P2二选错误",
    "p2_no_executable_candidate": "P2候选不可执行",
    "p3_omitted_rank2_would_fix": "P3省略但二选本可修正",
    "p3_omitted_rank2_wrong": "P3省略且二选也会错",
    "full_continuation_hit": "整块new命中",
    "full_continuation_miss": "整块new未中",
    "full_continuation_absent": "整块无new",
}
MISS_STATES = (
    "miss_no_risk_position", "miss_before_first",
    "miss_between_positions", "miss_after_last",
)
CANDIDATE_BRANCHES = (
    ("p1_rank2", "P1二选"),
    ("p2_rank2", "P2二选"),
)
FORWARD_SCOPES = (
    ("all", "全部", "forward_distribution_all"),
    ("decode", "解码", "forward_distribution_decode"),
    ("multi_fused", "融合", None),
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
    for row in rendered:
        if len(row) != len(headers):
            raise ValueError(
                f"report table has {len(headers)} headers but {len(row)} cells"
            )
    lines = [
        "|" + "|".join(headers) + "|",
        "|" + "|".join(":---:" for _ in headers) + "|",
    ]
    lines.extend("|" + "|".join(row) + "|" for row in rendered)
    return "\n".join(lines)


def metric_path(root: Path, dataset: str) -> Path:
    return root / f"metrics_{dataset}.json"


def report_datasets(settings: dict[str, Any]) -> tuple[str, ...]:
    """Use the ordered benchmark scope recorded by the entrypoint."""

    benchmark = settings.get("benchmark")
    if not isinstance(benchmark, dict) or "benchmarks" not in benchmark:
        return DEFAULT_REPORT_DATASETS
    raw = benchmark.get("benchmarks")
    specs = raw.split(",") if isinstance(raw, str) else raw
    if not isinstance(specs, (list, tuple)):
        return DEFAULT_REPORT_DATASETS
    datasets: list[str] = []
    seen: set[str] = set()
    for spec in specs:
        name = str(spec).strip().split(":", 1)[0].strip()
        if not name or name in REPORT_EXCLUDED_DATASETS or name in seen:
            continue
        if any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-" for ch in name):
            continue
        seen.add(name)
        datasets.append(name)
    return tuple(datasets)


def dataset_scope_label(count: int) -> str:
    chinese = {0: "零", 1: "一", 2: "二", 3: "三", 4: "四", 5: "五", 6: "六", 7: "七", 8: "八", 9: "九", 10: "十"}
    return f"{chinese.get(count, count)}数据集"


def load_metrics(root: Path, datasets: Iterable[str]) -> dict[str, dict[str, Any]]:
    return {
        dataset: payload
        for dataset in datasets
        if (payload := read_json(metric_path(root, dataset))) is not None
    }


def decode(payload: dict[str, Any], *, new_method: bool) -> dict[str, Any]:
    key = "pytorch_margin_risk_no_p3_overlap" if new_method else "pytorch_native"
    wrapper = payload.get(key)
    if not isinstance(wrapper, dict):
        return {}
    result = wrapper.get("decode")
    if not isinstance(result, dict):
        return {}
    if new_method or number(result.get("decode_forward_passes")) is not None:
        return result

    # Legacy PyTorch baseline artifacts counted one prompt-prefill encoder call
    # per successful request in ``forward_passes``.  Rebase those immutable
    # artifacts to decode-only TPF/NFE instead of comparing unlike definitions.
    total_forwards = number(result.get("forward_passes"))
    requests = number(result.get("request_count"))
    completion = number(result.get("completion_tokens"))
    if total_forwards is None or requests is None:
        return result
    decode_forwards = max(total_forwards - requests, 0.0)
    normalized = dict(result)
    normalized["legacy_reported_forward_passes"] = total_forwards
    normalized["decode_forward_passes"] = decode_forwards
    normalized["prefill_forward_passes"] = requests
    normalized["total_forward_passes"] = total_forwards
    normalized["average_forward_passes_per_sample"] = (
        decode_forwards / requests if requests > 0 else None
    )
    normalized["tokens_per_forward_pass"] = (
        completion / decode_forwards
        if completion is not None and decode_forwards > 0
        else None
    )
    normalized["tpf_rebased_from_legacy_prefill_inclusive"] = True
    return normalized


def overlap(payload: dict[str, Any]) -> dict[str, Any]:
    result = decode(payload, new_method=True).get("overlap")
    return result if isinstance(result, dict) else {}


def state_item(payload: dict[str, Any], state: str) -> dict[str, Any]:
    states = overlap(payload).get("outcome_states")
    if not isinstance(states, dict):
        return {}
    item = states.get(state)
    return item if isinstance(item, dict) else {}


def correction_item(payload: dict[str, Any], branch: str) -> dict[str, Any]:
    values = overlap(payload).get("candidate_correction")
    if not isinstance(values, dict):
        return {}
    item = values.get(branch)
    return item if isinstance(item, dict) else {}


def forward_scope(payload: dict[str, Any], scope: str, key: Optional[str]) -> dict[str, Any]:
    values = overlap(payload)
    if key is not None:
        item = values.get(key)
    else:
        kinds = values.get("forward_kinds")
        item = kinds.get(scope) if isinstance(kinds, dict) else None
    return item if isinstance(item, dict) else {}


def config_summary(root: Path) -> dict[str, Any]:
    settings = read_json(root / "Settings.json") or {}
    bench = settings.get("benchmark") if isinstance(settings.get("benchmark"), dict) else {}
    pytorch = settings.get("pytorch") if isinstance(settings.get("pytorch"), dict) else {}
    return {
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
        for key, label_zh in (
            ("tokens", "tokens"), ("context", "context"),
            ("temperature", "temperature"), ("thinking", "thinking"), ("dtype", "dtype"),
        ):
            if new.get(key) is not None and baseline.get(key) is not None and new[key] != baseline[key]:
                warnings.append(
                    f"新方法与 {label} 的 {label_zh} 不一致：{new[key]} vs {baseline[key]}。"
                )
    if b16.get("gpu") != b32.get("gpu") or b16.get("reserve") != b32.get("reserve"):
        warnings.append("B16 与 B32 的 GPU 或预留显存不同；TPS 只作参考。")
    return warnings


def render(result_dir: Path, baseline16: Path, baseline32: Path) -> str:
    settings = read_json(result_dir / "Settings.json") or {}
    datasets = report_datasets(settings)
    scope_name = dataset_scope_label(len(datasets))
    new_metrics = load_metrics(result_dir, datasets)
    b16_metrics = load_metrics(baseline16, datasets)
    b32_metrics = load_metrics(baseline32, datasets)
    completed = [dataset for dataset in datasets if dataset in new_metrics]
    new_cfg = config_summary(result_dir)
    b16_cfg = config_summary(baseline16)
    b32_cfg = config_summary(baseline32)
    lines = [
        "# 固定 margin-risk=0.5 去 P3 overlap 实验报告",
        "",
        f"> 状态：`{settings.get('status', 'initialized')}`；报告范围 `{len(completed)}/{len(datasets)}`：`{', '.join(completed) if completed else '尚无'}`。配置范围为 `{', '.join(datasets) if datasets else '空'}`；AIME24/MMLU 即使误传也不进入主表和等权平均。",
        "",
        "## 1. 配置与 baseline",
        "",
        "变量说明：`BS` 是 block size；`MR阈` 是严格触发阈值。分支规则为 C0→new、C1→P1+new、C2→P1+P2+new、C3+→P1+P2；P3 永不执行，只做反事实审计。加 verifier 后最多 4 row。B16/B32 是既有 PyTorch+NeMo-Skills、LinearSpec LoRA、greedy、block size 16/32 baseline。示例：MR阈为 0.5 时，等于 0.5 不触发。",
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
        "## 2. 请求覆盖与解码效率",
        "",
        "变量说明：`Att/OK/Fail/OOM` 是请求尝试数、纳入效率聚合的成功数、排除的失败数和其中因显存不足跳过的数量；`Cov=OK/Att`。`TPF` 是成功请求返回 completion token/decode encoder forward，`NFE` 是成功请求每样本平均 decode forward，二者都排除 prompt prefill 并与 SGLang 对齐；旧 B16/B32 产物没有显式 decode 字段时，报告按“旧 forward 总数−成功 request 数”重算其 decode NFE 和 TPF，不改写 baseline 文件。`TPS` 是同步模型生成阶段 token/s。示例：Att=164、OK=163、OOM=1 时，所有效率均值只来自 163 个成功请求。四行 fused forward 仍只计一次，需结合 FwdTok 和 padding 表。accuracy 不进入本报告。",
        "",
    ])
    compare_rows: list[list[Any]] = []
    macro: dict[str, list[float]] = {key: [] for key in (
        "att", "ok", "fail", "oom", "cov", "nt", "t16", "t32", "nn", "n16", "n32", "np", "p16", "p32"
    )}
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
            "nt": nd.get("tokens_per_forward_pass"), "t16": d16.get("tokens_per_forward_pass"), "t32": d32.get("tokens_per_forward_pass"),
            "nn": nd.get("average_forward_passes_per_sample"), "n16": d16.get("average_forward_passes_per_sample"), "n32": d32.get("average_forward_passes_per_sample"),
            "np": nd.get("model_output_tokens_per_s"), "p16": d16.get("model_output_tokens_per_s"), "p32": d32.get("model_output_tokens_per_s"),
        }
        for key, value in values.items():
            if (parsed := number(value)) is not None:
                macro[key].append(parsed)
        compare_rows.append([
            dataset, fmt(values["att"]), fmt(values["ok"]), fmt(values["fail"]), fmt(values["oom"]), pct(values["cov"]),
            fmt(values["nt"]), fmt(values["t16"]), fmt(values["t32"]),
            fmt(values["nn"]), fmt(values["n16"]), fmt(values["n32"]),
            fmt(values["np"]), fmt(values["p16"]), fmt(values["p32"]),
        ])
    if completed:
        compare_rows.append([
            f"等权均值({len(completed)}/{len(datasets)})",
            *[fmt(mean(macro[key])) for key in (
                "att", "ok", "fail", "oom",
            )], pct(mean(macro["cov"])),
            *[fmt(mean(macro[key])) for key in (
                "nt", "t16", "t32", "nn", "n16", "n32", "np", "p16", "p32",
            )],
        ])
        lines.append(table(
            ["数据集", "Att", "OK", "Fail", "OOM", "Cov", "新TPF", "B16TPF", "B32TPF", "新NFE", "B16NFE", "B32NFE", "新TPS", "B16TPS", "B32TPS"],
            compare_rows,
        ))
    else:
        lines.append("尚无已完成的新方法数据集；首个数据集完成后自动生成表格。")

    lines.extend([
        "",
        "## 3. Crossing、分支与复用漏斗",
        "",
        "变量说明：`C0/C1/C2/C3+` 是当前 draft 中严格 crossing 的数量；同名策略列是实际采用该分支配置的轮数。`R2/R3/R4` 是 fused forward 的总 row 数；`P1B/P2B/NewB` 是实际执行的 P1、P2、continuation 分支数；`验中/可复用/已复用` 依次是 verifier 确认种子命中、通过结束边界后可留用、下一轮实际消费。示例：C2 通常是 verifier+P1+P2+new，共 R4；C3+ 是 verifier+P1+P2，共 R3，不执行 P3。",
        "",
    ])
    funnel_rows: list[list[Any]] = []
    funnel_fields = (
        "rounds", "prefetch_attempts", "candidate_branches_executed",
        "continuation_branches_executed", "prefetch_verified_hits",
        "prefetch_hits", "prefetch_saved_draft_forwards",
    )
    for dataset in completed:
        ov = overlap(new_metrics[dataset])
        crossings = ov.get("crossing_count_rounds") or {}
        rows_hist = ov.get("fused_row_count") or {}
        cases = ov.get("policy_case_rounds") or {}
        branches = ov.get("candidate_branch_executed") or {}
        funnel_rows.append([
            dataset, fmt(ov.get("rounds")),
            fmt(crossings.get("0")), fmt(crossings.get("1")),
            fmt(crossings.get("2")), fmt(crossings.get("3+")),
            fmt(cases.get("c0_new")), fmt(cases.get("c1_p1_new")),
            fmt(cases.get("c2_p1_p2_new")), fmt(cases.get("c3plus_p1_p2")),
            fmt(ov.get("prefetch_attempts")), fmt(rows_hist.get("2")),
            fmt(rows_hist.get("3")), fmt(rows_hist.get("4")),
            fmt(branches.get("p1_rank2")), fmt(branches.get("p2_rank2")),
            fmt(ov.get("candidate_branches_executed")),
            fmt(ov.get("continuation_branches_executed")),
            fmt(ov.get("prefetch_verified_hits")), fmt(ov.get("prefetch_hits")),
            fmt(ov.get("prefetch_saved_draft_forwards")),
        ])
    if completed:
        def total_nested(field: str, key: str) -> int:
            return sum(integer((overlap(new_metrics[d]).get(field) or {}).get(key)) for d in completed)
        funnel_rows.append([
            "总计",
            *[fmt(sum(integer(overlap(new_metrics[d]).get(field)) for d in completed)) for field in ("rounds",)],
            *[fmt(total_nested("crossing_count_rounds", key)) for key in ("0", "1", "2", "3+")],
            *[fmt(total_nested("policy_case_rounds", key)) for key in ("c0_new", "c1_p1_new", "c2_p1_p2_new", "c3plus_p1_p2")],
            fmt(sum(integer(overlap(new_metrics[d]).get("prefetch_attempts")) for d in completed)),
            *[fmt(total_nested("fused_row_count", key)) for key in ("2", "3", "4")],
            *[fmt(total_nested("candidate_branch_executed", key)) for key in ("p1_rank2", "p2_rank2")],
            *[fmt(sum(integer(overlap(new_metrics[d]).get(field)) for d in completed)) for field in funnel_fields[2:]],
        ])
        funnel_rows.append([
            f"等权均次({len(completed)}/{len(datasets)})",
            *[fmt(mean(integer(overlap(new_metrics[d]).get(field)) for d in completed)) for field in ("rounds",)],
            *[fmt(mean(integer((overlap(new_metrics[d]).get("crossing_count_rounds") or {}).get(key)) for d in completed)) for key in ("0", "1", "2", "3+")],
            *[fmt(mean(integer((overlap(new_metrics[d]).get("policy_case_rounds") or {}).get(key)) for d in completed)) for key in ("c0_new", "c1_p1_new", "c2_p1_p2_new", "c3plus_p1_p2")],
            fmt(mean(integer(overlap(new_metrics[d]).get("prefetch_attempts")) for d in completed)),
            *[fmt(mean(integer((overlap(new_metrics[d]).get("fused_row_count") or {}).get(key)) for d in completed)) for key in ("2", "3", "4")],
            *[fmt(mean(integer((overlap(new_metrics[d]).get("candidate_branch_executed") or {}).get(key)) for d in completed)) for key in ("p1_rank2", "p2_rank2")],
            *[fmt(mean(integer(overlap(new_metrics[d]).get(field)) for d in completed)) for field in funnel_fields[2:]],
        ])
        lines.append(table(
            ["数据集", "轮数", "C0", "C1", "C2", "C3+", "策C0", "策C1", "策C2", "策C3+", "融合轮", "R2", "R3", "R4", "P1B", "P2B", "CandB", "NewB", "验中", "可复用", "已复用"],
            funnel_rows,
        ))
    else:
        lines.append("尚无分支漏斗统计。")

    lines.extend([
        "",
        "## 4. 关键状态计数与占比",
        "",
        "变量说明：每格为“次数/占实际融合轮比例”。`漏报` 表示首错不在前三个风险位置；`预测后` 是首错位于最后风险位置之后的子集。`P1中/错`、`P2中/错` 分别表示首错恰在该位置时二选 token 是否修正成功。`P3本可修` 与 `P3也会错` 是不执行 P3 分支的反事实审计：前者表示若保留旧 P3 二选分支本可复用，直接量化去 P3 的潜在 TPF 损失机会；二者都不会实际复用。`边界跳` 是 P1/P2 已定位但候选因 token/context/thinking 边界不可执行。`new未用` 合并整块通过但 new 未命中或 C3+ 没有 new。示例：P3本可修=5/1% 表示 1% 融合轮省掉了一个本可命中的 P3 row。",
        "",
    ])
    key_rows: list[list[Any]] = []
    macro_key: dict[str, list[float]] = {
        key: [] for key in (
            "miss", "after", "p1fixed", "p1wrong", "p2fixed",
            "p2wrong", "p3would", "p3wrong", "boundary", "newhit", "newunused",
        )
    }
    count_key: dict[str, list[int]] = {key: [] for key in macro_key}
    for dataset in completed:
        payload = new_metrics[dataset]
        attempts = integer(overlap(payload).get("prefetch_attempts"))
        counts = {
            "miss": sum(integer(state_item(payload, state).get("count")) for state in MISS_STATES),
            "after": integer(state_item(payload, "miss_after_last").get("count")),
            "p1fixed": integer(state_item(payload, "p1_rank2_fixed").get("count")),
            "p1wrong": integer(state_item(payload, "p1_rank2_wrong").get("count")),
            "p2fixed": integer(state_item(payload, "p2_rank2_fixed").get("count")),
            "p2wrong": integer(state_item(payload, "p2_rank2_wrong").get("count")),
            "p3would": integer(state_item(payload, "p3_omitted_rank2_would_fix").get("count")),
            "p3wrong": integer(state_item(payload, "p3_omitted_rank2_wrong").get("count")),
            "boundary": integer(state_item(payload, "p1_no_executable_candidate").get("count")) + integer(state_item(payload, "p2_no_executable_candidate").get("count")),
            "newhit": integer(state_item(payload, "full_continuation_hit").get("count")),
            "newunused": integer(state_item(payload, "full_continuation_miss").get("count")) + integer(state_item(payload, "full_continuation_absent").get("count")),
        }
        row = [dataset]
        for key, count in counts.items():
            rate = count / attempts if attempts else None
            count_key[key].append(count)
            if rate is not None:
                macro_key[key].append(rate)
            row.append(f"{count}/{pct(rate)}")
        key_rows.append(row)
    if completed:
        key_rows.append([
            "总计",
            *[str(sum(count_key[key])) for key in macro_key],
        ])
        key_rows.append([
            f"等权均值({len(completed)}/{len(datasets)})",
            *[f"{fmt(mean(count_key[key]))}/{pct(mean(macro_key[key]))}" for key in macro_key],
        ])
        lines.append(table(
            ["数据集", "漏报", "预测后", "P1中", "P1错", "P2中", "P2错", "P3本可修", "P3也会错", "边界跳", "new命中", "new未用"],
            key_rows,
        ))
    else:
        lines.append("尚无关键状态统计。")

    lines.extend([
        "",
        "### 4.1 各 confidence-rank 候选的独立正确率",
        "",
        "变量说明：`检查` 表示 verifier 首错恰在实际执行的 P1/P2 分支位置、因而能够判断二选 token 对错的次数；`修对/修错` 是该 token 等于/不等于 verifier token 的次数；`修对率=修对/检查`。某数据集检查为 0 时条件正确率未定义，不用 0 代填。P3 反事实单列在上一表，不混入实际分支正确率。",
        "",
    ])
    correction_rows: list[list[Any]] = []
    correction_macro = {
        branch: {field: [] for field in ("checked", "fixed", "wrong", "fixed_rate")}
        for branch, _label in CANDIDATE_BRANCHES
    }
    for dataset in completed:
        for branch, label in CANDIDATE_BRANCHES:
            item = correction_item(new_metrics[dataset], branch)
            for field in correction_macro[branch]:
                if (parsed := number(item.get(field))) is not None:
                    correction_macro[branch][field].append(parsed)
            correction_rows.append([
                dataset, label, fmt(item.get("checked")), fmt(item.get("fixed")),
                fmt(item.get("wrong")), pct(item.get("fixed_rate")),
            ])
    if completed:
        for branch, label in CANDIDATE_BRANCHES:
            values = correction_macro[branch]
            correction_rows.append([
                f"等权均值({len(completed)}/{len(datasets)})", label,
                fmt(mean(values["checked"])), fmt(mean(values["fixed"])),
                fmt(mean(values["wrong"])), pct(mean(values["fixed_rate"])),
            ])
        lines.append(table(
            ["数据集", "候选", "检查", "修对", "修错", "修对率"],
            correction_rows,
        ))
    else:
        lines.append("尚无候选独立正确率统计。")

    lines.extend([
        "",
        "## 5. 每个 forward 的实际计算 token 与 padding 分布",
        "",
        "变量说明：`总Slot` 是所有 decode forward 实际计算的 dense query token slot 总和；`每FwdSlot=总Slot/decode forward 数`；`每TokSlot=总Slot/completion token 数`，直接衡量每产出一个 token 付出的 dense query-token 工作。三者均排除 prompt prefill。示例：100 次 decode forward 共计算 6400 slot、输出 800 token，则每FwdSlot=64、每TokSlot=8。",
        "",
    ])
    dense_rows: list[list[Any]] = []
    dense_fields = (
        "decode_dense_query_token_slots_total",
        "decode_dense_query_token_slots_per_forward",
        "decode_dense_query_token_slots_per_output_token",
    )
    for dataset in completed:
        ov = overlap(new_metrics[dataset])
        dense_rows.append([dataset, *[fmt(ov.get(field)) for field in dense_fields]])
    if completed:
        dense_rows.append([
            "总计/等权均",
            fmt(sum(integer(overlap(new_metrics[d]).get(dense_fields[0])) for d in completed)),
            fmt(mean(overlap(new_metrics[d]).get(dense_fields[1]) for d in completed)),
            fmt(mean(overlap(new_metrics[d]).get(dense_fields[2]) for d in completed)),
        ])
        lines.append(table(["数据集", "总Slot", "每FwdSlot", "每TokSlot"], dense_rows))
    else:
        lines.append("尚无 dense query token slot 统计。")
    lines.extend([
        "",
        "### 5.1 Forward slot 分布",
        "",
        "变量说明：`FwdTok` 是 dense forward 实际送入模型的 query token slot，即 row数×统一 padding 后 Q；`有效均` 是各 row 有效长度之和的平均；`Pad均` 是 FwdTok−有效 token；`Pad率` 是总 padding slot/总 FwdTok；`Rows均/Q均` 是平均 row 数和平均公共 Q。`全部` 包含 prefill，`解码` 排除 prefill，`融合` 只统计多行 fused forward。示例：4 row、Q=32 时 FwdTok=128，即使部分位置被 attention mask 屏蔽，这些 padding token 仍经过 dense QKV/MLP 计算。",
        "",
    ])
    forward_rows: list[list[Any]] = []
    fields = (
        "computed_token_avg", "computed_token_min", "computed_token_p50",
        "computed_token_p90", "computed_token_p95", "computed_token_p99",
        "computed_token_max", "valid_token_avg", "padding_token_avg",
        "padding_ratio", "rows_avg", "query_length_avg",
    )
    macro_forward = {
        scope: {field: [] for field in fields} for scope, _, _ in FORWARD_SCOPES
    }
    count_forward = {scope: [] for scope, _, _ in FORWARD_SCOPES}
    for dataset in completed:
        for scope, label, key in FORWARD_SCOPES:
            item = forward_scope(new_metrics[dataset], scope, key)
            count_forward[scope].append(integer(item.get("count")))
            for field in fields:
                if (parsed := number(item.get(field))) is not None:
                    macro_forward[scope][field].append(parsed)
            forward_rows.append([
                dataset, label, fmt(item.get("count")),
                *[pct(item.get(field)) if field == "padding_ratio" else fmt(item.get(field)) for field in fields],
            ])
    if completed:
        for scope, label, _ in FORWARD_SCOPES:
            values = macro_forward[scope]
            forward_rows.append([
                f"等权均值({len(completed)}/{len(datasets)})", label, fmt(mean(count_forward[scope])),
                *[pct(mean(values[field])) if field == "padding_ratio" else fmt(mean(values[field])) for field in fields],
            ])
        lines.append(table(
            ["数据集", "范围", "FwdN", "FwdTok均", "Min", "P50", "P90", "P95", "P99", "Max", "有效均", "Pad均", "Pad率", "Rows均", "Q均"],
            forward_rows,
        ))
    else:
        lines.append("尚无 forward token 分布。")

    lines.extend([
        "",
        "## 6. 互斥状态的当前轮与下一轮 verify",
        "",
        "变量说明：下表保留全部 15 个互斥状态，区分 P1/P2 二选命中或错误、P3 省略分支的反事实二选结果、候选不可执行、漏报和 continuation。`Cnt` 是状态次数；`占比` 是 Cnt/融合轮；`本轮均` 是连续匹配 draft token 数加 verifier bonus 后的平均接收长度；`NextN/NextCov` 是确实存在同一 request 下一 verify 的数量/覆盖率；`配对本轮` 只在这些配对样本上计算；`下轮均` 是下一轮接收均值；`差值` 是逐对“下轮−当前轮”的均值。终止轮不补成 0。",
        "",
    ])
    transition_rows: list[list[Any]] = []
    transition_fields = (
        "share_of_attempts", "current_accept_avg", "next_coverage",
        "paired_current_accept_avg", "next_accept_avg", "next_minus_current_avg",
    )
    macro_state = {
        state: {field: [] for field in transition_fields} for state in OUTCOME_STATES
    }
    count_state = {state: [] for state in OUTCOME_STATES}
    next_state = {state: [] for state in OUTCOME_STATES}
    for dataset in completed:
        for state in OUTCOME_STATES:
            item = state_item(new_metrics[dataset], state)
            count_state[state].append(integer(item.get("count")))
            next_state[state].append(integer(item.get("next_count")))
            for field in transition_fields:
                if (parsed := number(item.get(field))) is not None:
                    macro_state[state][field].append(parsed)
            transition_rows.append([
                dataset, STATE_ZH[state], fmt(item.get("count")),
                pct(item.get("share_of_attempts")), fmt(item.get("current_accept_avg")),
                fmt(item.get("next_count")), pct(item.get("next_coverage")),
                fmt(item.get("paired_current_accept_avg")), fmt(item.get("next_accept_avg")),
                fmt(item.get("next_minus_current_avg")),
            ])
    if completed:
        for state in OUTCOME_STATES:
            values = macro_state[state]
            transition_rows.append([
                f"等权均值({len(completed)}/{len(datasets)})", STATE_ZH[state],
                fmt(mean(count_state[state])), pct(mean(values["share_of_attempts"])),
                fmt(mean(values["current_accept_avg"])), fmt(mean(next_state[state])),
                pct(mean(values["next_coverage"])),
                fmt(mean(values["paired_current_accept_avg"])),
                fmt(mean(values["next_accept_avg"])),
                fmt(mean(values["next_minus_current_avg"])),
            ])
        lines.append(table(
            ["数据集", "状态", "Cnt", "占比", "本轮均", "NextN", "NextCov", "配对本轮", "下轮均", "差值"],
            transition_rows,
        ))
    else:
        lines.append("尚无跨轮状态统计。")

    lines.extend([
        "",
        "## 7. 口径与完整性",
        "",
        f"- {scope_name}等权：比例、均值和各分位数先在每个配置内数据集计算，再对数据集做算术平均；本次最终应显示 {len(datasets)}/{len(datasets)}。",
        "- 默认正式范围是排除 AIME24/MMLU 的八个数据集；报告也会按命令传入的数据集数量与名称动态收缩，但 AIME24/MMLU 始终排除。",
        "- 绝对次数另列总计；总计不是主要比例分母，不会让样本多的数据集覆盖小数据集。",
        "- 15 个细状态必须互斥且次数之和等于实际 fused overlap 轮数；报告读取 outcome_partition_valid 校验。",
        "- 某状态在某数据集为 0 次时，占比以 0 参与等权平均；接收均值和 NextCov 未定义，只在有定义的数据集间平均。",
        "- FwdTok/Slot 是 dense query-token slot，不是完整 FLOPs；attention 还受 cache length 影响，但它能直接暴露多 row 和 padding 带来的实际 token 计算。",
        "- 准确率不进入本报告。OOM/失败请求从效率聚合中排除，并通过 Att、OK、Fail、OOM、Cov 显式披露；Cov 不足 100% 时只代表成功请求子集。",
        "- TPS 受 GPU、显存预留和并发环境影响；TPF 也会把四行 fused forward 计为一次，必须和 FwdTok/Pad率共同解读。",
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
    content = render(
        result_dir,
        Path(args.baseline_block16_dir).resolve(),
        Path(args.baseline_block32_dir).resolve(),
    )
    target = result_dir / "report.md"
    temporary = target.with_name(f".{target.name}.tmp.{os.getpid()}")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, target)
    print(f"Updated {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
