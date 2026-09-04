#!/usr/bin/env python3
"""Align LinearSpec trace request groups and compare common per-round fields."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any


CORE_FIELDS = (
    "block_size",
    "gen_start",
    "gen_len",
    "matched_draft_tokens",
    "emitted_tokens",
    "accepted_draft_tokens",
    "eos_hit",
    "has_rejection",
    "accepted_positions",
    "rejected_position",
    "rejected_draft_token_id",
    "rejected_correct_token_id",
    "rejected_correct_token_rank",
)


def grouped_rows(path: Path) -> list[list[dict[str, Any]]]:
    groups: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("event") != "linearspec_confidence_round":
                continue
            groups.setdefault(str(row.get("request_id", "")), []).append(row)
    return list(groups.values())


def distribution(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "max": None}
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "max": max(values),
    }


def aggregate(groups: list[list[dict[str, Any]]]) -> dict[str, Any]:
    rows = [row for group in groups for row in group]
    accepted_confidences = [
        float(value)
        for row in rows
        for value in (row.get("accepted_draft_confidences") or [])
    ]
    rejected_confidences = [
        float(row["rejected_draft_confidence"])
        for row in rows
        if row.get("rejected_draft_confidence") is not None
    ]
    return {
        "requests": len(groups),
        "rounds": len(rows),
        "matched_draft_tokens": sum(
            int(row.get("matched_draft_tokens") or 0) for row in rows
        ),
        "accepted_draft_tokens": sum(
            int(row.get("accepted_draft_tokens") or 0) for row in rows
        ),
        "emitted_tokens_in_trace": sum(
            int(row.get("emitted_tokens") or 0) for row in rows
        ),
        "accepted_confidence": distribution(accepted_confidences),
        "rejected_confidence": distribution(rejected_confidences),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--drop-left-leading-requests", type=int, default=0)
    parser.add_argument("--drop-right-leading-requests", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    left_groups_all = grouped_rows(args.left)
    right_groups_all = grouped_rows(args.right)
    left_groups = left_groups_all[args.drop_left_leading_requests :]
    right_groups = right_groups_all[args.drop_right_leading_requests :]

    core_mismatches: list[dict[str, Any]] = []
    accepted_conf_abs: list[float] = []
    rejected_conf_abs: list[float] = []
    paired_rounds = 0
    for sample_index, (left_group, right_group) in enumerate(
        zip(left_groups, right_groups)
    ):
        for round_index, (left, right) in enumerate(zip(left_group, right_group)):
            paired_rounds += 1
            differences = {
                field: {"left": left.get(field), "right": right.get(field)}
                for field in CORE_FIELDS
                if left.get(field) != right.get(field)
            }
            if differences:
                core_mismatches.append(
                    {
                        "sample_index": sample_index,
                        "round_index": round_index,
                        "differences": differences,
                    }
                )
                continue

            left_accepted = left.get("accepted_draft_confidences") or []
            right_accepted = right.get("accepted_draft_confidences") or []
            if len(left_accepted) == len(right_accepted):
                accepted_conf_abs.extend(
                    abs(float(a) - float(b))
                    for a, b in zip(left_accepted, right_accepted)
                )
            left_rejected = left.get("rejected_draft_confidence")
            right_rejected = right.get("rejected_draft_confidence")
            if left_rejected is not None and right_rejected is not None:
                rejected_conf_abs.append(
                    abs(float(left_rejected) - float(right_rejected))
                )

    payload = {
        "left": str(args.left.resolve()),
        "right": str(args.right.resolve()),
        "left_all_request_round_counts": [len(group) for group in left_groups_all],
        "right_all_request_round_counts": [len(group) for group in right_groups_all],
        "left_compared_request_round_counts": [len(group) for group in left_groups],
        "right_compared_request_round_counts": [len(group) for group in right_groups],
        "left_compared_aggregate": aggregate(left_groups),
        "right_compared_aggregate": aggregate(right_groups),
        "same_request_count": len(left_groups) == len(right_groups),
        "same_round_counts": [len(group) for group in left_groups]
        == [len(group) for group in right_groups],
        "paired_rounds": paired_rounds,
        "core_exact_match": not core_mismatches
        and len(left_groups) == len(right_groups)
        and [len(group) for group in left_groups]
        == [len(group) for group in right_groups],
        "core_mismatch_count": len(core_mismatches),
        "core_mismatch_field_counts": dict(
            sorted(
                Counter(
                    field
                    for mismatch in core_mismatches
                    for field in mismatch["differences"]
                ).items()
            )
        ),
        "first_core_mismatch": core_mismatches[0] if core_mismatches else None,
        "core_mismatches": core_mismatches[:20],
        "accepted_confidence_abs_diff": distribution(accepted_conf_abs),
        "rejected_confidence_abs_diff": distribution(rejected_conf_abs),
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
