#!/usr/bin/env python3
"""Atomic live report/progress renderer for the verifier-efficiency experiment."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(text.rstrip() + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    atomic_text(path, json.dumps(payload, ensure_ascii=False, indent=2))


def load_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def pct(value: Any) -> str:
    return "—" if value is None else f"{100*float(value):.2f}%"


def number(value: Any, digits: int = 4) -> str:
    return "—" if value is None else f"{float(value):.{digits}f}"


def markdown_table(headers: Iterable[str], rows: Iterable[Iterable[Any]]) -> List[str]:
    headers = [str(item) for item in headers]
    result = ["|" + "|".join(headers) + "|", "|" + "|".join(":---:" for _ in headers) + "|"]
    result.extend("|" + "|".join(str(item) for item in row) + "|" for row in rows)
    return result


def parse_benchmarks(settings: Mapping[str, Any]) -> List[str]:
    return [
        item.strip().split(":", 1)[0]
        for item in str(settings.get("benchmarks", "")).split(",")
        if item.strip()
    ]


def parse_concurrencies(settings: Mapping[str, Any]) -> List[int]:
    return [
        int(item)
        for item in str(settings.get("concurrencies", "")).split(",")
        if item.strip()
    ]


def write_settings(run_dir: Path, payload: Mapping[str, Any]) -> None:
    atomic_json(run_dir / "settings.json", payload)
    lines = [
        "# 实验设置",
        "",
        f"- 创建时间：{payload['created_at']}",
        f"- 模型规格：{payload.get('model_size')}",
        f"- 模型：`{payload.get('model')}`",
        f"- 数据集：`{payload.get('benchmarks')}`；排除 AIME24/MMLU",
        f"- 并发度：`{payload.get('concurrencies')}`",
        f"- B200延迟表：`{payload.get('cost_document')}`",
        f"- 信号限制：最多两个历史verify信号；第二信号仅为上一轮B与full截断状态",
        f"- 搜索：三类策略离线全量比较，只在线验证唯一全局赢家",
        f"- 成本：每个request均使用满名义并发度C的T_C(B)，不做分桶占用/尾组修正",
        f"- 命令：`{payload.get('command')}`",
    ]
    atomic_text(run_dir / "settings.md", "\n".join(lines))


def init_run(run_dir: Path, settings: Mapping[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=False)
    for child in ("runtime", "traces/explore", "traces/validate", "eval_runs", "search"):
        (run_dir / child).mkdir(parents=True, exist_ok=True)
    payload = dict(settings)
    payload["created_at"] = now()
    payload["run_dir"] = str(run_dir.resolve())
    write_settings(run_dir, payload)
    atomic_json(run_dir / "run_state.json", {"created_at": payload["created_at"], "events": []})
    render(run_dir)


def update_settings(run_dir: Path, updates: Mapping[str, Any]) -> None:
    payload = load_json(run_dir / "settings.json", {})
    if not payload:
        raise RuntimeError(f"missing settings.json in {run_dir}")
    payload.update(updates)
    payload["updated_at"] = now()
    write_settings(run_dir, payload)
    render(run_dir)


def add_event(run_dir: Path, event: Mapping[str, Any]) -> None:
    path = run_dir / "run_state.json"
    state = load_json(path, {"created_at": now(), "events": []})
    row = dict(event)
    row["at"] = now()
    state.setdefault("events", []).append(row)
    state["updated_at"] = row["at"]
    atomic_json(path, state)
    render(run_dir)


def progress_bar(done: int, total: int, width: int = 28) -> str:
    total = max(total, 1)
    done = min(max(done, 0), total)
    filled = int(width * done / total)
    return f"|{'#'*filled}{'-'*(width-filled)}| {done}/{total} ({100*done/total:.1f}%)"


def render_progress(run_dir: Path, settings: Mapping[str, Any], state: Mapping[str, Any]) -> None:
    datasets = parse_benchmarks(settings)
    concurrencies = parse_concurrencies(settings)
    events = state.get("events", [])
    collected = {
        str(row.get("dataset"))
        for row in events
        if row.get("phase") == "collect" and row.get("status") == "completed"
    }
    search_result = load_json(run_dir / "search/search_results.json")
    winner_family = (
        (search_result.get("winner") or {}).get("family")
        if search_result
        else None
    )
    validated = {
        str(row.get("dataset"))
        for row in events
        if row.get("phase") == "validate"
        and row.get("status") == "completed"
        and "/C" in str(row.get("dataset"))
        and (
            winner_family is None
            or str(row.get("dataset")).endswith(f"/{winner_family}")
        )
    }
    search_done = int(search_result is not None)
    total_validation = len(datasets) * len(concurrencies)
    lines = [
        "# 实验实时进度",
        "",
        f"> 更新时间：{now()}",
        "",
        "|阶段|进度|状态|",
        "|:---:|:---:|:---:|",
        f"|1 新trace采集|`{progress_bar(len(collected),len(datasets))}`|{'完成' if len(collected)==len(datasets) else '进行中/待运行'}|",
        f"|2 三策略离线搜索|`{progress_bar(search_done,1)}`|{'完成' if search_done else '待运行或进行中'}|",
        f"|3 唯一赢家在线验证|`{progress_bar(len(validated),total_validation)}`|{'完成' if len(validated)==total_validation else '待运行或进行中'}|",
        "",
    ]
    search_progress = load_json(run_dir / "search/progress.json")
    if search_progress:
        lines.extend(
            [
                "## 当前CPU搜索",
                "",
                f"- 阶段：`{search_progress.get('stage')}`",
                f"- 候选：`{progress_bar(int(search_progress.get('completed',0)),int(search_progress.get('total',1)))}`",
                f"- 当前最佳：`{json.dumps(search_progress.get('best') or search_progress.get('winner'),ensure_ascii=False)}`",
                "",
            ]
        )
    if events:
        latest = events[-1]
        lines.extend(
            [
                "## 最近事件",
                "",
                f"- `{latest.get('at')}`：`{latest.get('phase')}` / `{latest.get('dataset')}` / `{latest.get('status')}`",
                f"- {latest.get('message','')}",
            ]
        )
    atomic_text(run_dir / "progress.md", "\n".join(lines))


def render_offline(lines: List[str], search: Mapping[str, Any]) -> None:
    lines.extend(
        [
            "## 离线搜索结果",
            "",
            "`增益`是相对指定固定基线的八集等权token/ms提升：C2/4对B32，C8/16/32对B16，C64/128对B8。`局遗`是策略相对同一步真实B8/B16/B32局部oracle的token/ms regret；例如oracle效率0.35、策略效率0.33，则局遗为0.02。`截态`表示是否使用上一轮块长与整块通过状态来解释接收长度的截断。",
            "",
        ]
    )
    best = search.get("best_by_family") or {}
    rows = []
    for family in ("local_ratio", "direct_rank", "global_fractional"):
        candidate = best.get(family)
        if not candidate:
            continue
        rows.append(
            [
                candidate.get("family_zh"),
                candidate.get("signal"),
                "是" if candidate.get("use_censor_state") else "否",
                pct(candidate.get("mean_gain_across_c")),
                pct(candidate.get("min_gain_across_c")),
                number(candidate.get("mean_regret_across_c"), 6),
            ]
        )
    lines.extend(markdown_table(["策略族", "信号", "截态", "C均增益", "最差C", "局遗"], rows))
    lines.extend(
        [
            "",
            "下表横向比较三类离线最优策略在每个C上的结果；只有最终全局赢家进入默认在线验证。",
            "",
        ]
    )
    family_c_rows = []
    for family in ("local_ratio", "direct_rank", "global_fractional"):
        candidate = best.get(family)
        if not candidate:
            continue
        for concurrency, profile in (candidate.get("profiles") or {}).items():
            family_c_rows.append(
                [
                    candidate.get("family_zh"),
                    concurrency,
                    number(profile.get("rho"), 6),
                    pct(profile.get("relative_gain_vs_designated_fixed")),
                    pct(profile.get("throughput_equivalent_time_saving")),
                    pct(profile.get("oracle_agreement")),
                    number(profile.get("mean_local_regret"), 6),
                ]
            )
    lines.extend(
        markdown_table(
            ["策略族", "C", "ρ", "增益", "等效省时", "一致", "局遗"],
            family_c_rows,
        )
    )
    winner = search.get("winner") or {}
    lines.extend(
        [
            "",
            f"全局赢家：**{winner.get('family_zh','—')}**；信号 `{winner.get('signal','—')}`；截断状态 `{'启用' if winner.get('use_censor_state') else '不启用'}`。该选择来自八数据集等权、七并发度等权的离线目标，在线阶段只运行这一冻结策略。",
            "",
            "下表中`块占比`按sample内轮次占比先聚合，再做数据集等权；`一致`是局部oracle完全一致率；`等效省时`是由吞吐增益换算的同token量时间节省，在线完整轨迹结果优先于这一离线等效量。",
            "",
        ]
    )
    rows = []
    for concurrency, profile in (winner.get("profiles") or {}).items():
        rows.append(
            [
                concurrency,
                number(profile.get("rho"), 6),
                pct(profile.get("relative_gain_vs_designated_fixed")),
                pct(profile.get("throughput_equivalent_time_saving")),
                pct(profile.get("oracle_agreement")),
                number(profile.get("mean_local_regret"), 6),
                pct(profile.get("l8_rate")),
                pct(profile.get("l16_rate")),
                pct(profile.get("l32_rate")),
            ]
        )
    lines.extend(markdown_table(["C", "ρ", "增益", "等效省时", "一致", "局遗", "B8", "B16", "B32"], rows))
    lines.extend(
        [
            "",
            "下表给出离线赢家在每个数据集、每个C上的sample内聚合结果；`增益`仍使用该C指定固定基线，而不是事后选择的最佳固定块。",
            "",
        ]
    )
    dataset_rows = []
    for concurrency, profile in (search.get("offline_winner_profiles") or {}).items():
        for dataset, row in (profile.get("summary", {}).get("datasets", {})).items():
            dataset_rows.append(
                [
                    dataset,
                    concurrency,
                    row.get("samples"),
                    number(row.get("decode_tpf")),
                    number(row.get("score_token_ms_req"), 5),
                    pct(row.get("relative_gain_vs_designated_fixed")),
                    pct(row.get("throughput_equivalent_time_saving")),
                    pct(row.get("l8_rate")),
                    pct(row.get("l16_rate")),
                    pct(row.get("l32_rate")),
                ]
            )
    lines.extend(
        markdown_table(
            ["数据集", "C", "样本", "TPF", "tok/ms/req", "增益", "等效省时", "B8", "B16", "B32"],
            dataset_rows,
        )
    )


def render_validation(lines: List[str], run_dir: Path, settings: Mapping[str, Any]) -> None:
    lines.extend(
        [
            "",
            "## 获胜策略真实SGLang动态验证",
            "",
            "每个request按照冻结策略独立改变block size；B200理论成本始终查满名义并发度C。`TPF`只统计decode，两个模型forward构成一轮，所以等于每轮有效前进token除以2。`批吞吐`为C×token/ms/request，不代表含调度、prefill的实测wall TPS。",
            "",
        ]
    )
    validation_rows = []
    dataset_rows = []
    search = load_json(run_dir / "search/search_results.json", {})
    winner_family = (search.get("winner") or {}).get("family")
    for concurrency in parse_concurrencies(settings):
        result = (
            load_json(
                run_dir
                / f"search/validation_{winner_family}_c{concurrency}.json"
            )
            if winner_family
            else None
        )
        if not result:
            continue
        dynamic = result["dynamic_policy"]
        macro = dynamic["macro"]
        validation_rows.append(
            [
                concurrency,
                macro.get("datasets"),
                macro.get("samples"),
                number(macro.get("decode_tpf")),
                number(macro.get("score_token_ms_req"), 5),
                number(macro.get("pure_forward_token_ms_batch"), 4),
                pct(macro.get("relative_gain_vs_designated_fixed")),
                pct(macro.get("throughput_equivalent_time_saving")),
                pct(macro.get("l8_rate")),
                pct(macro.get("l16_rate")),
                pct(macro.get("l32_rate")),
            ]
        )
        for dataset, row in dynamic.get("datasets", {}).items():
            dataset_rows.append(
                [
                    dataset,
                    concurrency,
                    row.get("samples"),
                    number(row.get("decode_tpf")),
                    number(row.get("score_token_ms_req"), 5),
                    pct(row.get("relative_gain_vs_designated_fixed")),
                    pct(row.get("throughput_equivalent_time_saving")),
                    pct(row.get("l8_rate")),
                    pct(row.get("l16_rate")),
                    pct(row.get("l32_rate")),
                ]
            )
    if validation_rows:
        lines.extend(markdown_table(["C", "集", "样本", "TPF", "tok/ms/req", "批吞吐", "增益", "等效省时", "B8", "B16", "B32"], validation_rows))
        lines.extend(
            [
                "",
                "下表是已经完成的单数据集结果；实验顺序为数据集外层、C内层，因此同一数据集的七种C会连续出现。",
                "",
            ]
        )
        lines.extend(markdown_table(["数据集", "C", "样本", "TPF", "tok/ms/req", "增益", "等效省时", "B8", "B16", "B32"], dataset_rows))
    else:
        lines.append("尚未产生在线验证结果。")


def render(run_dir: Path) -> None:
    settings = load_json(run_dir / "settings.json", {})
    state = load_json(run_dir / "run_state.json", {"events": []})
    lines = [
        "# B200延迟感知verify历史动态block实验报告",
        "",
        f"> 创建：{settings.get('created_at','—')}　更新时间：{now()}",
        "",
        "本报告从实验初始化开始实时存在。目标是在排除AIME24/MMLU的八个数据集上，以sample内聚合、数据集等权、七并发度等权选择一个全局策略。策略只读取已完成轮次的verify信息，三类策略全部离线比较，真实SGLang全量验证只运行唯一赢家。",
        "",
        "## 协议与当前状态",
        "",
        "`记录`是该事件提交的trace行数；失败尝试不会覆盖已提交trace。下表用于断点恢复审计。",
        "",
    ]
    event_rows = [
        [row.get("at"), row.get("phase"), row.get("dataset"), row.get("status"), row.get("records", 0), row.get("message", "")]
        for row in state.get("events", [])[-80:]
    ]
    lines.extend(markdown_table(["时间", "阶段", "对象", "状态", "记录", "说明"], event_rows))
    search_result = load_json(run_dir / "search/search_results.json")
    if search_result:
        lines.append("")
        render_offline(lines, search_result)
    else:
        lines.extend(["", "## 离线搜索结果", "", "尚未完成；模板已建立，进度见 `progress.md`。"])
    render_validation(lines, run_dir, settings)
    lines.extend(
        [
            "",
            "## 变量速查",
            "",
            "- `V_B(s)`：历史verify状态s下使用B预计前进的token数。例如V16=10表示类似历史状态下B16平均推进约10个token。",
            "- `ρ`：全局分式对照自动求得的token/ms时间影子价格；另外两类策略不使用ρ。",
            "- `oracle`：在同一真实上下文直接看到A8/A16/A32后选择A_B/T_C(B)最大的局部上界，不是可部署策略。",
            "- `full`：该轮候选块全部验证通过；与上一轮B一起用于识别接收长度是否被块长截断。",
            "- `增益`：相对用户指定固定块基线的token/ms提升，不是accuracy变化。",
        ]
    )
    atomic_text(run_dir / "report.md", "\n".join(lines))
    render_progress(run_dir, settings, state)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    initialize = subparsers.add_parser("init")
    initialize.add_argument("--run-dir", type=Path, required=True)
    initialize.add_argument("--settings-json", required=True)
    event = subparsers.add_parser("event")
    event.add_argument("--run-dir", type=Path, required=True)
    event.add_argument("--event-json", required=True)
    update = subparsers.add_parser("update-settings")
    update.add_argument("--run-dir", type=Path, required=True)
    update.add_argument("--settings-json", required=True)
    render_parser = subparsers.add_parser("render")
    render_parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "init":
        init_run(args.run_dir, json.loads(args.settings_json))
    elif args.command == "event":
        add_event(args.run_dir, json.loads(args.event_json))
    elif args.command == "update-settings":
        update_settings(args.run_dir, json.loads(args.settings_json))
    else:
        render(args.run_dir)


if __name__ == "__main__":
    main()
