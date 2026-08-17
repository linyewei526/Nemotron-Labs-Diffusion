#!/usr/bin/env python3
"""Summarize LinearSpec draft-vs-final alignment traces."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


ROUND_EVENT = "linearspec_draft_alignment_round"
FINAL_EVENT = "linearspec_draft_alignment_final"


def normalize_request_id(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def is_internal_request_id(request_id: str) -> bool:
    return (not request_id) or request_id.startswith("HEALTH_CHECK_")


def load_records(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                yield {"event": "_decode_error", "line_no": line_no, "error": str(exc)}


def ratio(num: int | float, den: int | float) -> float | None:
    return (num / den) if den else None


def blank_counter() -> dict[str, int]:
    return {"aligned_count": 0, "total_count": 0}


def add_counter(counter: dict[str, int], aligned: bool) -> None:
    counter["total_count"] += 1
    if aligned:
        counter["aligned_count"] += 1


def finalize_counter(counter: dict[str, int]) -> dict[str, Any]:
    aligned = int(counter["aligned_count"])
    total = int(counter["total_count"])
    return {
        "aligned_count": aligned,
        "total_count": total,
        "alignment_rate": ratio(aligned, total),
    }


def get_final_token(
    final: dict[str, Any],
    sequence_position: int,
) -> int | None:
    prompt_length = int(final.get("prompt_length") or 0)
    output_index = sequence_position - prompt_length
    output_ids = final.get("output_ids") or []
    if output_index < 0 or output_index >= len(output_ids):
        return None
    return int(output_ids[output_index])


def summarize_trace(trace_file: Path) -> dict[str, Any]:
    rounds: list[dict[str, Any]] = []
    finals: dict[str, dict[str, Any]] = {}
    decode_errors: list[dict[str, Any]] = []
    internal_records_skipped = 0

    for row in load_records(trace_file) or []:
        event = row.get("event")
        if event == "_decode_error":
            decode_errors.append(row)
        elif event == ROUND_EVENT:
            request_id = normalize_request_id(row.get("request_id"))
            if is_internal_request_id(request_id):
                internal_records_skipped += 1
                continue
            row["request_id"] = request_id
            rounds.append(row)
        elif event == FINAL_EVENT:
            request_id = normalize_request_id(row.get("request_id"))
            if is_internal_request_id(request_id):
                internal_records_skipped += 1
                continue
            row["request_id"] = request_id
            finals[request_id] = row

    block_position = defaultdict(blank_counter)
    post_rejection_offset = defaultdict(blank_counter)

    rounds_with_final = 0
    rounds_with_compared_candidates = 0
    round_alignment_counts: list[int] = []
    round_alignment_rates: list[float] = []
    total_draft_candidates = 0
    compared_candidates = 0
    aligned_candidates = 0
    missing_final_tokens = 0
    rounds_with_rejection = 0
    rounds_with_post_rejection_candidates = 0

    for row in rounds:
        request_id = normalize_request_id(row.get("request_id"))
        final = finals.get(request_id)
        candidate_tokens = row.get("draft_token_ids") or []
        candidate_positions = row.get("candidate_sequence_positions") or []
        candidate_start_local = int(row.get("candidate_start_local_position") or 1)
        candidate_count = min(len(candidate_tokens), len(candidate_positions))
        total_draft_candidates += int(row.get("candidate_count") or candidate_count)

        if final is None:
            missing_final_tokens += candidate_count
            continue
        rounds_with_final += 1

        round_aligned = 0
        round_compared = 0
        rejected_candidate_index = row.get("rejected_candidate_index")
        if rejected_candidate_index is not None:
            try:
                rejected_candidate_index = int(rejected_candidate_index)
                rounds_with_rejection += 1
            except (TypeError, ValueError):
                rejected_candidate_index = None

        has_post_rejection_compared = False

        for idx in range(candidate_count):
            draft_token = int(candidate_tokens[idx])
            sequence_position = int(candidate_positions[idx])
            final_token = get_final_token(final, sequence_position)
            if final_token is None:
                missing_final_tokens += 1
                continue

            aligned = draft_token == final_token
            round_compared += 1
            compared_candidates += 1
            if aligned:
                round_aligned += 1
                aligned_candidates += 1

            local_position = candidate_start_local + idx
            add_counter(block_position[f"position_{local_position}"], aligned)

            if rejected_candidate_index is not None and idx > rejected_candidate_index:
                offset = idx - rejected_candidate_index
                add_counter(post_rejection_offset[f"offset_{offset}"], aligned)
                has_post_rejection_compared = True

        if round_compared > 0:
            rounds_with_compared_candidates += 1
            round_alignment_counts.append(round_aligned)
            round_alignment_rates.append(round_aligned / round_compared)
        if has_post_rejection_compared:
            rounds_with_post_rejection_candidates += 1

    mean_alignment_count = (
        sum(round_alignment_counts) / len(round_alignment_counts)
        if round_alignment_counts
        else None
    )
    mean_alignment_rate = (
        sum(round_alignment_rates) / len(round_alignment_rates)
        if round_alignment_rates
        else None
    )

    def sorted_position_items(items: dict[str, dict[str, int]]) -> dict[str, Any]:
        def key_fn(item: tuple[str, dict[str, int]]) -> int:
            return int(item[0].split("_", 1)[1])

        return {
            key: finalize_counter(counter)
            for key, counter in sorted(items.items(), key=key_fn)
        }

    return {
        "schema_version": 1,
        "trace_file": str(trace_file),
        "definition": {
            "candidate_scope": "draft candidates only; seed position_0 is excluded",
            "block_position_alignment": "alignment rate grouped by block-local draft candidate position",
            "post_rejection_offset_alignment": "alignment rate grouped by offset after first rejected candidate; offset_0 is intentionally excluded",
            "alignment": "draft token id equals final output token id at the same sequence position",
        },
        "requests": {
            "final_records": len(finals),
            "internal_records_skipped": internal_records_skipped,
        },
        "rounds": {
            "total": len(rounds),
            "with_final_truth": rounds_with_final,
            "with_compared_candidates": rounds_with_compared_candidates,
            "with_rejection": rounds_with_rejection,
            "with_post_rejection_compared_candidates": rounds_with_post_rejection_candidates,
        },
        "tokens": {
            "total_draft_candidates": total_draft_candidates,
            "compared_candidates": compared_candidates,
            "aligned_candidates": aligned_candidates,
            "missing_final_tokens": missing_final_tokens,
        },
        "alignment": {
            "mean_alignment_count": mean_alignment_count,
            "mean_alignment_rate": mean_alignment_rate,
            "micro_alignment_rate": ratio(aligned_candidates, compared_candidates),
        },
        "block_position_alignment": sorted_position_items(block_position),
        "post_rejection_offset_alignment": sorted_position_items(post_rejection_offset),
        "decode_errors": decode_errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-file", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--benchmark", default="")
    parser.add_argument("--benchmark-spec", default="")
    args = parser.parse_args()

    payload = summarize_trace(args.trace_file)
    payload["benchmark"] = args.benchmark
    payload["benchmark_spec"] = args.benchmark_spec

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


if __name__ == "__main__":
    main()
