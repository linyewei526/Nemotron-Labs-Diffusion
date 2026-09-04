#!/usr/bin/env python3
"""Compare two LinearSpec confidence traces after removing volatile metadata."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


VOLATILE_FIELDS = {"created_at_unix", "request_id", "backend", "mode"}


def load_trace(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("event") != "linearspec_confidence_round":
                continue
            row["_source_line"] = line_number
            rows.append(row)
    return rows


def normalize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    request_ordinals: dict[str, int] = {}
    normalized: list[dict[str, Any]] = []
    round_ordinals: dict[int, int] = {}
    for row in rows:
        request_id = str(row.get("request_id", ""))
        if request_id not in request_ordinals:
            request_ordinals[request_id] = len(request_ordinals)
        request_ordinal = request_ordinals[request_id]
        round_ordinal = round_ordinals.get(request_ordinal, 0)
        round_ordinals[request_ordinal] = round_ordinal + 1
        clean = {
            key: value
            for key, value in row.items()
            if key not in VOLATILE_FIELDS and key != "_source_line"
        }
        clean["request_ordinal"] = request_ordinal
        clean["round_ordinal"] = round_ordinal
        normalized.append(clean)
    return normalized


def numeric_leaves(value: Any, prefix: str = "") -> list[tuple[str, float]]:
    leaves: list[tuple[str, float]] = []
    if isinstance(value, dict):
        for key in sorted(value):
            leaves.extend(numeric_leaves(value[key], f"{prefix}.{key}" if prefix else key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            leaves.extend(numeric_leaves(item, f"{prefix}[{index}]"))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        if math.isfinite(number):
            leaves.append((prefix, number))
    return leaves


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    left_raw = load_trace(args.left)
    right_raw = load_trace(args.right)
    left = normalize(left_raw)
    right = normalize(right_raw)

    first_mismatch: dict[str, Any] | None = None
    exact_rows = 0
    for index, (left_row, right_row) in enumerate(zip(left, right)):
        if left_row == right_row:
            exact_rows += 1
            continue
        differing_keys = sorted(
            key
            for key in set(left_row) | set(right_row)
            if left_row.get(key) != right_row.get(key)
        )
        first_mismatch = {
            "row_index": index,
            "left_source_line": left_raw[index]["_source_line"],
            "right_source_line": right_raw[index]["_source_line"],
            "differing_keys": differing_keys,
            "left": {key: left_row.get(key) for key in differing_keys},
            "right": {key: right_row.get(key) for key in differing_keys},
        }
        break

    left_numbers = numeric_leaves(left)
    right_numbers = numeric_leaves(right)
    paired_numeric = min(len(left_numbers), len(right_numbers))
    comparable_numeric = 0
    max_abs_numeric_diff = 0.0
    for (left_key, left_value), (right_key, right_value) in zip(
        left_numbers, right_numbers
    ):
        if left_key != right_key:
            continue
        comparable_numeric += 1
        max_abs_numeric_diff = max(
            max_abs_numeric_diff, abs(left_value - right_value)
        )

    payload = {
        "left": str(args.left.resolve()),
        "right": str(args.right.resolve()),
        "left_rows": len(left),
        "right_rows": len(right),
        "same_row_count": len(left) == len(right),
        "exact_normalized_match": left == right,
        "exact_prefix_rows": exact_rows,
        "first_mismatch": first_mismatch,
        "numeric_leaf_pairs": paired_numeric,
        "comparable_numeric_leaf_pairs": comparable_numeric,
        "max_abs_numeric_diff_for_matching_paths": max_abs_numeric_diff,
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
