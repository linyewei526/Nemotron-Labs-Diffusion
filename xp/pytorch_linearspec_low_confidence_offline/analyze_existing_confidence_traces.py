#!/usr/bin/env python3
"""Reconstruct SGLang-compatible low-confidence curves from PyTorch traces.

This is deliberately a standard-library-only, streaming analysis.  It never loads
the model and never imports torch.  Existing confidence traces are read-only.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import shlex
import sys
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any


def decimal_range(start: str, end: str, step: str) -> list[Decimal]:
    current, stop, stride = Decimal(start), Decimal(end), Decimal(step)
    if stride <= 0 or current > stop:
        raise ValueError("threshold range requires start <= end and step > 0")
    values: list[Decimal] = []
    while current <= stop:
        values.append(current)
        current += stride
    if values[-1] < stop:
        values.append(stop)
    return values


def finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def f1_score(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None or precision + recall == 0:
        return None
    return 2.0 * precision * recall / (precision + recall)


def threshold_width(thresholds: list[Decimal]) -> int:
    return max(-min(value.as_tuple().exponent for value in thresholds), 0)


def threshold_key(value: Decimal, thresholds: list[Decimal], suffix: str) -> str:
    return f"token_{float(value):.{threshold_width(thresholds)}f}_{suffix}"


@dataclass
class ScanCounts:
    """Difference buckets: bucket k contains values matching first k thresholds."""

    thresholds: list[float]
    accepted_buckets: list[int] = field(init=False)
    rejected_buckets: list[int] = field(init=False)

    def __post_init__(self) -> None:
        self.accepted_buckets = [0] * (len(self.thresholds) + 1)
        self.rejected_buckets = [0] * (len(self.thresholds) + 1)

    def add(self, value: float, outcome: str) -> None:
        matched = bisect.bisect_right(self.thresholds, value)
        buckets = self.accepted_buckets if outcome == "accepted" else self.rejected_buckets
        buckets[matched] += 1

    def counts(self) -> tuple[list[int], list[int]]:
        accepted = [0] * len(self.thresholds)
        rejected = [0] * len(self.thresholds)
        running_accepted = running_rejected = 0
        for matched in range(len(self.thresholds), 0, -1):
            running_accepted += self.accepted_buckets[matched]
            running_rejected += self.rejected_buckets[matched]
            accepted[matched - 1] = running_accepted
            rejected[matched - 1] = running_rejected
        return accepted, rejected


def make_table_and_curve(
    decimals: list[Decimal],
    counts: ScanCounts,
    suffix: str,
    total_accepted: int,
    total_rejected: int,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    accepted_counts, rejected_counts = counts.counts()
    table: dict[str, dict[str, Any]] = {}
    curve: list[dict[str, Any]] = []
    for threshold, accepted, rejected in zip(decimals, accepted_counts, rejected_counts):
        flagged = accepted + rejected
        precision = ratio(rejected, flagged)
        recall = ratio(rejected, total_rejected)
        accepted_coverage = ratio(accepted, total_accepted)
        item = {
            "accepted_count": accepted,
            "rejected_count": rejected,
            "accepted_ratio_within_flagged": ratio(accepted, flagged),
            "rejected_ratio_within_flagged": precision,
            "accepted_coverage_of_all_countable_accepted_tokens": accepted_coverage,
            "rejected_coverage_of_all_countable_rejected_tokens": recall,
        }
        table[threshold_key(threshold, decimals, suffix)] = item
        curve.append(
            {
                "threshold": float(threshold),
                "accepted_count": accepted,
                "rejected_count": rejected,
                "flagged_count": flagged,
                "rejected_precision_within_flagged": precision,
                "accepted_false_positive_ratio_within_flagged": ratio(accepted, flagged),
                "rejected_recall": recall,
                "accepted_coverage": accepted_coverage,
                "f1": f1_score(precision, recall),
            }
        )
    return table, curve


def best_row(rows: list[dict[str, Any]], predicate=lambda row: True) -> dict[str, Any] | None:
    choices = [row for row in rows if predicate(row) and row["f1"] is not None]
    if not choices:
        return None
    # Prefer effectiveness, then recall, precision, and finally the lower threshold.
    return dict(
        max(
            choices,
            key=lambda row: (
                row["f1"],
                row["rejected_recall"] or -1.0,
                row["rejected_precision_within_flagged"] or -1.0,
                -row["threshold"],
            ),
        )
    )


def candidates(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any] | None]:
    return {
        "max_f1": best_row(rows),
        "precision_ge_0_90_best_recall": best_row(
            rows, lambda row: (row["rejected_precision_within_flagged"] or 0.0) >= 0.90
        ),
        "accepted_coverage_le_0_01_best_recall": best_row(
            rows, lambda row: (row["accepted_coverage"] or 0.0) <= 0.01
        ),
    }


def threshold_definition(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "drop_abs": {
            "formula": "C_imean - C_i",
            "start": float(Decimal(args.abs_start)),
            "end": float(Decimal(args.abs_end)),
            "step": float(Decimal(args.abs_step)),
            "inclusive": True,
        },
        "drop_pct": {
            "formula": "1 - C_i / C_imean",
            "start": float(Decimal(args.pct_start)),
            "end": float(Decimal(args.pct_end)),
            "step": float(Decimal(args.pct_step)),
            "inclusive": True,
        },
        "C_imean": "mean confidence of all draft candidates before token i in the same round",
        "confidence": "softmax probability of the drafted token after excluding MASK from the vocabulary",
        "counted_outcomes": ["accepted", "rejected"],
        "ignored_outcomes": ["unverified_after_rejection", "unverified_after_eos"],
    }


def analyze_trace(
    trace_path: Path,
    benchmark: str,
    benchmark_spec: str,
    abs_decimals: list[Decimal],
    pct_decimals: list[Decimal],
    args: argparse.Namespace,
) -> dict[str, Any]:
    abs_counts = ScanCounts([float(value) for value in abs_decimals])
    pct_counts = ScanCounts([float(value) for value in pct_decimals])
    rounds = accepted_tokens = rejected_tokens = 0
    countable_accepted = countable_rejected = skipped_no_prefix = 0
    first_position_accepted = first_position_rejected = 0
    invalid_confidence = invalid_prefix = 0
    decode_errors: list[dict[str, Any]] = []
    invariant_errors: list[dict[str, Any]] = []
    block_sizes: set[int] = set()
    request_ids: set[str] = set()
    missing_request_id_rounds = 0
    recorded_drop_checked = recorded_drop_mismatches = 0
    max_abs_error = max_pct_error = 0.0

    def invariant(line_no: int, message: str) -> None:
        if len(invariant_errors) < args.max_recorded_errors:
            invariant_errors.append({"line_no": line_no, "error": message})

    with trace_path.open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                if len(decode_errors) < args.max_recorded_errors:
                    decode_errors.append({"line_no": line_no, "error": str(exc)})
                continue
            if row.get("event") != "linearspec_confidence_round":
                continue
            rounds += 1
            request_id = str(row.get("request_id") or "")
            if request_id:
                request_ids.add(request_id)
            else:
                missing_request_id_rounds += 1
            try:
                block_sizes.add(int(row["block_size"]))
            except (KeyError, TypeError, ValueError):
                invariant(line_no, "missing or invalid block_size")

            raw_accepted = row.get("accepted_draft_confidences")
            if not isinstance(raw_accepted, list):
                invariant(line_no, "accepted_draft_confidences is not a list")
                raw_accepted = []
            confidences = [finite_float(value) for value in raw_accepted]
            declared_accepted = row.get("accepted_draft_tokens")
            if declared_accepted != len(confidences):
                invariant(
                    line_no,
                    f"accepted_draft_tokens={declared_accepted} but list length={len(confidences)}",
                )
            accepted_tokens += len(confidences)

            prefix_sum = 0.0
            prefix_count = 0
            for index, confidence in enumerate(confidences):
                if confidence is None:
                    invalid_confidence += 1
                    continue
                if index == 0 or prefix_count == 0:
                    skipped_no_prefix += 1
                    if index == 0:
                        first_position_accepted += 1
                else:
                    prefix_mean = prefix_sum / prefix_count
                    drop_abs = prefix_mean - confidence
                    countable_accepted += 1
                    abs_counts.add(drop_abs, "accepted")
                    if prefix_mean > 0.0:
                        pct_counts.add(1.0 - confidence / prefix_mean, "accepted")
                    else:
                        invalid_prefix += 1
                prefix_sum += confidence
                prefix_count += 1

            has_rejection = bool(row.get("has_rejection"))
            rejected_confidence = finite_float(row.get("rejected_draft_confidence"))
            if has_rejection:
                rejected_tokens += 1
                if rejected_confidence is None:
                    invalid_confidence += 1
                    invariant(line_no, "has_rejection=true but rejected confidence is invalid")
                elif prefix_count == 0:
                    skipped_no_prefix += 1
                    first_position_rejected += 1
                else:
                    prefix_mean = prefix_sum / prefix_count
                    drop_abs = prefix_mean - rejected_confidence
                    countable_rejected += 1
                    abs_counts.add(drop_abs, "rejected")
                    drop_pct = None
                    if prefix_mean > 0.0:
                        drop_pct = 1.0 - rejected_confidence / prefix_mean
                        pct_counts.add(drop_pct, "rejected")
                    else:
                        invalid_prefix += 1
                    recorded_abs = finite_float(row.get("confidence_drop_abs"))
                    recorded_pct = finite_float(row.get("confidence_drop_pct"))
                    if recorded_abs is not None:
                        recorded_drop_checked += 1
                        difference = abs(recorded_abs - drop_abs)
                        max_abs_error = max(max_abs_error, difference)
                        pct_difference = 0.0
                        if drop_pct is not None and recorded_pct is not None:
                            pct_difference = abs(recorded_pct - drop_pct)
                            max_pct_error = max(max_pct_error, pct_difference)
                        if difference > args.drop_validation_tolerance or pct_difference > args.drop_validation_tolerance:
                            recorded_drop_mismatches += 1
                            invariant(line_no, "reconstructed rejection drop differs from recorded drop")
            elif rejected_confidence is not None:
                invariant(line_no, "has_rejection=false but rejected confidence is present")

    countable_tokens = countable_accepted + countable_rejected
    abs_table, abs_curve = make_table_and_curve(
        abs_decimals, abs_counts, "drop_abs", countable_accepted, countable_rejected
    )
    pct_table, pct_curve = make_table_and_curve(
        pct_decimals, pct_counts, "drop_pct", countable_accepted, countable_rejected
    )
    invalid_block = args.require_block_size not in block_sizes or block_sizes != {args.require_block_size}
    status = "invalid" if decode_errors or invariant_errors or invalid_block else "ok"
    return {
        "schema_version": 2,
        "status": status,
        "benchmark": benchmark,
        "benchmark_spec": benchmark_spec,
        "analysis_mode": "offline_from_existing_pytorch_confidence_trace",
        "backend": "native_pytorch",
        "source_trace_file": str(trace_path.resolve()),
        "source_block_sizes": sorted(block_sizes),
        "required_block_size": args.require_block_size,
        "exact_reconstruction_scope": {
            "core_threshold_tables": True,
            "reason": "accepted prefix confidences plus first rejected confidence exactly cover every outcome counted by the SGLang threshold tables",
            "unverified_outcome_counts_available": False,
            "limitation": "post-rejection and post-EOS unverified candidates were not recorded by the source trace and are not used by these threshold tables",
        },
        "threshold_definition": threshold_definition(args),
        "rounds": {"total": rounds},
        "requests": {
            "unique_request_ids": len(request_ids),
            "rounds_missing_request_id": missing_request_id_rounds,
        },
        "tokens": {
            "reconstructible_candidate_tokens_by_outcome": {
                "accepted": accepted_tokens,
                "rejected": rejected_tokens,
                "unverified_after_rejection": None,
                "unverified_after_eos": None,
            },
            "countable_tokens_with_prefix": countable_tokens,
            "countable_accepted_tokens_with_prefix": countable_accepted,
            "countable_rejected_tokens_with_prefix": countable_rejected,
            "skipped_no_prefix": skipped_no_prefix,
            "unscorable_first_position_accepted_tokens": first_position_accepted,
            "unscorable_first_position_rejections": first_position_rejected,
            "skipped_invalid_confidence": invalid_confidence,
            "skipped_invalid_prefix_mean": invalid_prefix,
        },
        "token_x_drop_abs": abs_table,
        "token_y_drop_pct": pct_table,
        "curves": {"drop_abs": abs_curve, "drop_pct": pct_curve},
        "critical_value_candidates": {
            "drop_abs": candidates(abs_curve),
            "drop_pct": candidates(pct_curve),
        },
        "validation": {
            "recorded_rejection_drops_checked": recorded_drop_checked,
            "recorded_rejection_drop_mismatches": recorded_drop_mismatches,
            "max_reconstructed_abs_error": max_abs_error,
            "max_reconstructed_pct_error": max_pct_error,
            "drop_validation_tolerance": args.drop_validation_tolerance,
            "decode_error_count": len(decode_errors),
            "invariant_error_count": len(invariant_errors),
            "decode_errors_sample": decode_errors,
            "invariant_errors_sample": invariant_errors,
            "block_size_valid": not invalid_block,
        },
    }


def add_results(results: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    first = results[0]
    abs_rows = first["curves"]["drop_abs"]
    pct_rows = first["curves"]["drop_pct"]

    def combined_curve(kind: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for index, source in enumerate(first["curves"][kind]):
            accepted = sum(item["curves"][kind][index]["accepted_count"] for item in results)
            rejected = sum(item["curves"][kind][index]["rejected_count"] for item in results)
            total_accepted = sum(item["tokens"]["countable_accepted_tokens_with_prefix"] for item in results)
            total_rejected = sum(item["tokens"]["countable_rejected_tokens_with_prefix"] for item in results)
            flagged = accepted + rejected
            precision = ratio(rejected, flagged)
            recall = ratio(rejected, total_rejected)
            rows.append(
                {
                    "threshold": source["threshold"],
                    "accepted_count": accepted,
                    "rejected_count": rejected,
                    "flagged_count": flagged,
                    "rejected_precision_within_flagged": precision,
                    "accepted_false_positive_ratio_within_flagged": ratio(accepted, flagged),
                    "rejected_recall": recall,
                    "accepted_coverage": ratio(accepted, total_accepted),
                    "f1": f1_score(precision, recall),
                }
            )
        return rows

    abs_curve = combined_curve("drop_abs") if abs_rows else []
    pct_curve = combined_curve("drop_pct") if pct_rows else []
    abs_decimals = decimal_range(args.abs_start, args.abs_end, args.abs_step)
    pct_decimals = decimal_range(args.pct_start, args.pct_end, args.pct_step)

    def curve_to_table(rows: list[dict[str, Any]], decimals: list[Decimal], suffix: str) -> dict[str, Any]:
        table = {}
        for decimal, row in zip(decimals, rows):
            table[threshold_key(decimal, decimals, suffix)] = {
                "accepted_count": row["accepted_count"],
                "rejected_count": row["rejected_count"],
                "accepted_ratio_within_flagged": row["accepted_false_positive_ratio_within_flagged"],
                "rejected_ratio_within_flagged": row["rejected_precision_within_flagged"],
                "accepted_coverage_of_all_countable_accepted_tokens": row["accepted_coverage"],
                "rejected_coverage_of_all_countable_rejected_tokens": row["rejected_recall"],
            }
        return table

    token_keys = (
        "countable_tokens_with_prefix",
        "countable_accepted_tokens_with_prefix",
        "countable_rejected_tokens_with_prefix",
        "skipped_no_prefix",
        "unscorable_first_position_accepted_tokens",
        "unscorable_first_position_rejections",
        "skipped_invalid_confidence",
        "skipped_invalid_prefix_mean",
    )
    tokens = {key: sum(item["tokens"][key] for item in results) for key in token_keys}
    tokens["reconstructible_candidate_tokens_by_outcome"] = {
        outcome: sum(
            item["tokens"]["reconstructible_candidate_tokens_by_outcome"][outcome]
            for item in results
        )
        for outcome in ("accepted", "rejected")
    }
    tokens["reconstructible_candidate_tokens_by_outcome"].update(
        {"unverified_after_rejection": None, "unverified_after_eos": None}
    )
    return {
        "schema_version": 2,
        "status": "ok" if all(item["status"] == "ok" for item in results) else "invalid",
        "benchmark": "all_benchmarks_micro",
        "aggregation": "micro: raw counts are summed before ratios are computed",
        "benchmarks": [item["benchmark"] for item in results],
        "analysis_mode": "offline_from_existing_pytorch_confidence_trace",
        "backend": "native_pytorch",
        "source_block_sizes": sorted({size for item in results for size in item["source_block_sizes"]}),
        "required_block_size": args.require_block_size,
        "exact_reconstruction_scope": first["exact_reconstruction_scope"],
        "threshold_definition": threshold_definition(args),
        "rounds": {"total": sum(item["rounds"]["total"] for item in results)},
        "requests": {
            "unique_request_ids": sum(item["requests"]["unique_request_ids"] for item in results),
            "rounds_missing_request_id": sum(item["requests"]["rounds_missing_request_id"] for item in results),
        },
        "tokens": tokens,
        "token_x_drop_abs": curve_to_table(abs_curve, abs_decimals, "drop_abs"),
        "token_y_drop_pct": curve_to_table(pct_curve, pct_decimals, "drop_pct"),
        "curves": {"drop_abs": abs_curve, "drop_pct": pct_curve},
        "critical_value_candidates": {
            "drop_abs": candidates(abs_curve),
            "drop_pct": candidates(pct_curve),
        },
        "validation": {
            "all_benchmarks_ok": all(item["status"] == "ok" for item in results),
            "recorded_rejection_drops_checked": sum(item["validation"]["recorded_rejection_drops_checked"] for item in results),
            "recorded_rejection_drop_mismatches": sum(item["validation"]["recorded_rejection_drop_mismatches"] for item in results),
            "decode_error_count": sum(item["validation"]["decode_error_count"] for item in results),
            "invariant_error_count": sum(item["validation"]["invariant_error_count"] for item in results),
        },
    }


CSV_FIELDS = [
    "threshold",
    "accepted_count",
    "rejected_count",
    "flagged_count",
    "rejected_precision_within_flagged",
    "accepted_false_positive_ratio_within_flagged",
    "rejected_recall",
    "accepted_coverage",
    "f1",
]


def write_curve(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def fmt_percent(value: float | None) -> str:
    return "-" if value is None else f"{100.0 * value:.4f}%"


def report_row(label: str, item: dict[str, Any], kind: str) -> str:
    row = item["critical_value_candidates"][kind]["max_f1"]
    if row is None:
        return f"| {label} | - | - | - | - | - |"
    return "| {} | {:.3f} | {} | {} | {} | {:.4f} |".format(
        label,
        row["threshold"],
        fmt_percent(row["rejected_precision_within_flagged"]),
        fmt_percent(row["rejected_recall"]),
        fmt_percent(row["accepted_coverage"]),
        row["f1"],
    )


def detailed_curve_table(item: dict[str, Any], kind: str) -> list[str]:
    rows = item["curves"][kind]
    countable = item["tokens"]["countable_tokens_with_prefix"]
    threshold_label = "x" if kind == "drop_abs" else "y"
    precision = 3 if kind == "drop_abs" else 2
    lines = [
        f"#### token_{threshold_label}_{kind}",
        "",
        f"| {threshold_label} | accepted_count | rejected_count | flagged_evaluable_count | accepted_share | rejected_share / precision | accepted_FPR | rejection_recall | flag_rate | F1 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        flag_rate = ratio(row["flagged_count"], countable)
        lines.append(
            "| {} | {:,} | {:,} | {:,} | {} | {} | {} | {} | {} | {:.4f} |".format(
                f"{row['threshold']:.{precision}f}",
                row["accepted_count"],
                row["rejected_count"],
                row["flagged_count"],
                fmt_percent(row["accepted_false_positive_ratio_within_flagged"]),
                fmt_percent(row["rejected_precision_within_flagged"]),
                fmt_percent(row["accepted_coverage"]),
                fmt_percent(row["rejected_recall"]),
                fmt_percent(flag_rate),
                row["f1"] or 0.0,
            )
        )
    lines.append("")
    return lines


def append_result_section(lines: list[str], title: str, item: dict[str, Any], level: int) -> None:
    tokens = item["tokens"]
    reconstructible = tokens["reconstructible_candidate_tokens_by_outcome"]
    lines.extend(
        [
            f"{'#' * level} {title}",
            "",
            f"- 状态：`{item['status']}`",
            f"- benchmark spec：`{item.get('benchmark_spec', 'all_benchmarks_micro')}`",
            f"- 请求数：{item['requests']['unique_request_ids']:,}",
            f"- verification rounds：{item['rounds']['total']:,}",
            f"- 可评估 accepted/rejected 分母：{tokens['countable_accepted_tokens_with_prefix']:,} / {tokens['countable_rejected_tokens_with_prefix']:,}",
            f"- 原 trace 可复原 accepted/rejected 总数：{reconstructible['accepted']:,} / {reconstructible['rejected']:,}",
            f"- 首位置无前缀的 rejection：{tokens['unscorable_first_position_rejections']:,}",
            f"- 首位置无前缀的 accepted token：{tokens['unscorable_first_position_accepted_tokens']:,}",
            f"- 其他无效 confidence / 无效 prefix mean：{tokens['skipped_invalid_confidence']:,} / {tokens['skipped_invalid_prefix_mean']:,}",
        ]
    )
    if item.get("source_trace_file"):
        lines.append(f"- 源 trace：`{item['source_trace_file']}`")
    lines.append("")
    lines.extend(detailed_curve_table(item, "drop_abs"))
    lines.extend(detailed_curve_table(item, "drop_pct"))


def write_report(path: Path, source_run: Path, results: list[dict[str, Any]], aggregate: dict[str, Any]) -> None:
    abs_definition = aggregate["threshold_definition"]["drop_abs"]
    pct_definition = aggregate["threshold_definition"]["drop_pct"]
    abs_best = aggregate["critical_value_candidates"]["drop_abs"]["max_f1"]
    pct_best = aggregate["critical_value_candidates"]["drop_pct"]["max_f1"]
    lines = [
        "# PyTorch LinearSpec block=16 离线 Probability-Drop Rejection 完整结果",
        "",
        f"- 生成时间：{datetime.now().astimezone().isoformat()}",
        f"- 源 trace：`{source_run}`",
        f"- 分析设置：`{path.parent / 'Settings.json'}`",
        "- 运行方式：仅顺序读取既有 JSONL；未加载模型，未使用 GPU。",
        "- 统计口径：与 SGLang `token_x_drop_abs` / `token_y_drop_pct` 相同，只计 accepted/rejected。",
        "- 下表的比例均以百分比显示；`rejected_share / precision` 表示被阈值标记的 token 中实际 rejected 的比例。",
        "- 每个数据集只有在 trace 成功解析、统计复原和校验通过后才写入本报告。",
        "",
        "## 统计定义与范围",
        "",
        "- `C_i`：排除 MASK 后，位置 i 的 draft token softmax probability。",
        "- `C_imean`：同一轮中位置 i 之前所有 draft candidate confidence 的均值。",
        "- `drop_abs = C_imean - C_i`；当 `drop_abs >= x` 时计入 `token_x_drop_abs`。",
        "- `drop_pct = 1 - C_i / C_imean`；当 `drop_pct >= y` 时计入 `token_y_drop_pct`。",
        "- 第一个 candidate 没有前缀均值，因此不能计算 drop；它不会进入可评估分母。",
        "- 当前 PyTorch trace 没有保存拒绝点后或 accepted EOS 后的未验证 token；这些 outcome 原本也不参与阈值表。",
        f"- `drop_abs`：{abs_definition['start']:.3f}–{abs_definition['end']:.3f}，步长 {abs_definition['step']:.3f}，端点均包含。",
        f"- `drop_pct`：{pct_definition['start']:.2f}–{pct_definition['end']:.2f}，步长 {pct_definition['step']:.2f}，端点均包含。",
        "",
        "### 表格字段",
        "",
        "- `flagged_evaluable_count = accepted_count + rejected_count`。",
        "- `accepted_share = accepted_count / flagged_evaluable_count`。",
        "- `rejected_share / precision = rejected_count / flagged_evaluable_count`。",
        "- `accepted_FPR = accepted_count / 全部可评估 accepted token`。",
        "- `rejection_recall = rejected_count / 全部可评估 rejected token`。",
        "- `flag_rate = flagged_evaluable_count / 全部可评估 accepted+rejected token`。",
        "- `F1` 是 rejected precision 与 rejection recall 的调和平均，仅是 token 级候选选择指标。",
        "",
        "## 结论摘要",
        "",
        f"- 全局 `drop_abs` 最大 token-F1 点：阈值 `{abs_best['threshold']:.3f}`，precision {fmt_percent(abs_best['rejected_precision_within_flagged'])}，recall {fmt_percent(abs_best['rejected_recall'])}，accepted_FPR {fmt_percent(abs_best['accepted_coverage'])}，F1 {abs_best['f1']:.4f}。",
        f"- 全局 `drop_pct` 最大 token-F1 点：阈值 `{pct_best['threshold']:.2f}`，precision {fmt_percent(pct_best['rejected_precision_within_flagged'])}，recall {fmt_percent(pct_best['rejected_recall'])}，accepted_FPR {fmt_percent(pct_best['accepted_coverage'])}，F1 {pct_best['f1']:.4f}。",
        "- 当前范围内 `drop_pct` 的全局最大 F1 高于 `drop_abs`，但最佳点 precision 仍约为 50%；这说明 drop 与 rejection 有相关性，却尚不能视为干净的单阈值分界。",
        "- 本报告复现的是 SGLang 已完成的 token 级累计阈值工作；它尚未计算每轮第一次越阈值是否正好命中第一个 rejected token。",
        "",
        "## 各数据集最大 token-F1 候选",
        "",
        "### drop_abs",
        "",
        "| benchmark | threshold | rejected precision | rejected recall | accepted_FPR | F1 |",
        "|---|---:|---:|---:|---:|---:|",
        report_row("all_benchmarks_micro", aggregate, "drop_abs"),
    ]
    lines.extend(report_row(item["benchmark"], item, "drop_abs") for item in results)
    lines.extend(
        [
            "",
            "### drop_pct",
            "",
            "| benchmark | threshold | rejected precision | rejected recall | accepted_FPR | F1 |",
            "|---|---:|---:|---:|---:|---:|",
            report_row("all_benchmarks_micro", aggregate, "drop_pct"),
        ]
    )
    lines.extend(report_row(item["benchmark"], item, "drop_pct") for item in results)
    lines.extend(
        [
            "",
            "> 这些点只是在当前扫描区间内按 token 级 F1 自动选择的候选，不等于最终在线拒绝策略。后续若研究第一个不接受 token，需要另做轮级 first-crossing 分析。",
            "",
            "## 全局 micro 完整阈值结果",
            "",
            "全局结果先对所有数据集的原始 accepted/rejected count 求和，再计算比例；不是对各数据集百分比做平均。",
            "",
        ]
    )
    append_result_section(lines, "Aggregate: all_benchmarks_micro", aggregate, 3)
    lines.extend(["## 各数据集完整阈值结果", ""])
    for item in results:
        append_result_section(lines, f"Dataset: {item['benchmark']}", item, 3)
    validation = aggregate["validation"]
    lines.extend(
        [
            "## 校验与结果文件",
            "",
            f"- 10 个 benchmark 是否全部成功：`{validation['all_benchmarks_ok']}`",
            f"- JSON decode errors：{validation['decode_error_count']:,}",
            f"- invariant errors：{validation['invariant_error_count']:,}",
            f"- 已重算并核对的 rejection drop：{validation['recorded_rejection_drops_checked']:,}",
            f"- rejection drop 不一致：{validation['recorded_rejection_drop_mismatches']:,}",
            "- 机器可读完整汇总：`summaries/low_confidence_rejection_all_benchmarks.json`。",
            "- 每个 benchmark 的完整 JSON：`summaries/low_confidence_rejection_<benchmark>.json`。",
            "- 全局和每个 benchmark 的 CSV：`curves/*_drop_abs.csv`、`curves/*_drop_pct.csv`。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_benchmarks(value: str, settings: dict[str, Any], trace_dir: Path) -> list[tuple[str, str]]:
    raw = value or str(settings.get("benchmarks") or "")
    if raw:
        parsed = []
        for spec in raw.split(","):
            spec = spec.strip()
            if spec:
                parsed.append((spec.split(":", 1)[0], spec))
        return parsed
    return [
        (path.name[len("raw_trace_") : -len(".jsonl")], path.name[len("raw_trace_") : -len(".jsonl")])
        for path in sorted(trace_dir.glob("raw_trace_*.jsonl"))
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-run", required=True, type=Path, help="completed PyTorch confidence run")
    parser.add_argument("--output-dir", required=True, type=Path, help="new offline-analysis run directory")
    parser.add_argument("--benchmarks", default="", help="comma-separated benchmark specs; default: source Settings.json")
    parser.add_argument("--require-block-size", type=int, default=16)
    parser.add_argument("--abs-start", default="0.140")
    parser.add_argument("--abs-end", default="0.300")
    parser.add_argument("--abs-step", default="0.005")
    parser.add_argument("--pct-start", default="0.15")
    parser.add_argument("--pct-end", default="0.33")
    parser.add_argument("--pct-step", default="0.01")
    parser.add_argument("--drop-validation-tolerance", type=float, default=1e-10)
    parser.add_argument("--max-recorded-errors", type=int, default=20)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    input_run = args.input_run.resolve()
    output_dir = args.output_dir.resolve()
    settings_path = input_run / "Settings.json"
    trace_dir = input_run / "traces"
    if not settings_path.is_file() or not trace_dir.is_dir():
        parser.error(f"input run must contain Settings.json and traces/: {input_run}")
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    source_block = settings.get("block_size")
    try:
        source_block = int(source_block)
    except (TypeError, ValueError):
        parser.error(f"source Settings.json has invalid block_size: {source_block!r}")
    if source_block != args.require_block_size:
        parser.error(f"source block_size={source_block}, required={args.require_block_size}")
    if output_dir == input_run or input_run in output_dir.parents:
        parser.error("output-dir must be independent of the source run")

    abs_decimals = decimal_range(args.abs_start, args.abs_end, args.abs_step)
    pct_decimals = decimal_range(args.pct_start, args.pct_end, args.pct_step)
    benchmark_specs = parse_benchmarks(args.benchmarks, settings, trace_dir)
    if not benchmark_specs:
        parser.error("no benchmarks found")
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "summaries").mkdir()
    (output_dir / "curves").mkdir()

    run_settings = {
        "created_at": datetime.now().astimezone().isoformat(),
        "entrypoint": "observations/analyze_pytorch_linearspec_low_confidence_offline.sh",
        "command": " ".join(shlex.quote(value) for value in sys.argv),
        "source_run": str(input_run),
        "output_dir": str(output_dir),
        "benchmarks": [spec for _, spec in benchmark_specs],
        "require_block_size": args.require_block_size,
        "thresholds": {
            "abs_start": args.abs_start,
            "abs_end": args.abs_end,
            "abs_step": args.abs_step,
            "pct_start": args.pct_start,
            "pct_end": args.pct_end,
            "pct_step": args.pct_step,
        },
        "gpu_used": False,
        "source_settings": settings,
    }
    write_json(output_dir / "Settings.json", run_settings)

    results: list[dict[str, Any]] = []
    status_path = output_dir / "analysis_status.jsonl"
    with status_path.open("w", encoding="utf-8") as status_stream:
        for benchmark, benchmark_spec in benchmark_specs:
            trace_path = trace_dir / f"raw_trace_{benchmark}.jsonl"
            if not trace_path.is_file():
                parser.error(f"trace does not exist: {trace_path}")
            print(f"[offline] analyzing {benchmark}: {trace_path}", flush=True)
            result = analyze_trace(
                trace_path, benchmark, benchmark_spec, abs_decimals, pct_decimals, args
            )
            results.append(result)
            write_json(output_dir / "summaries" / f"low_confidence_rejection_{benchmark}.json", result)
            write_curve(output_dir / "curves" / f"{benchmark}_drop_abs.csv", result["curves"]["drop_abs"])
            write_curve(output_dir / "curves" / f"{benchmark}_drop_pct.csv", result["curves"]["drop_pct"])
            status_stream.write(
                json.dumps(
                    {
                        "benchmark": benchmark,
                        "status": result["status"],
                        "rounds": result["rounds"]["total"],
                        "countable_tokens": result["tokens"]["countable_tokens_with_prefix"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            status_stream.flush()

    aggregate = add_results(results, args)
    write_json(output_dir / "summaries" / "low_confidence_rejection_all_benchmarks.json", aggregate)
    write_curve(output_dir / "curves" / "all_benchmarks_drop_abs.csv", aggregate["curves"]["drop_abs"])
    write_curve(output_dir / "curves" / "all_benchmarks_drop_pct.csv", aggregate["curves"]["drop_pct"])
    write_report(output_dir / "report.md", input_run, results, aggregate)
    print(f"[offline] complete: {output_dir}", flush=True)
    return 0 if aggregate["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
