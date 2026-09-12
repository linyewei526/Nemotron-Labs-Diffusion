#!/usr/bin/env python3
"""B200 latency table parser and protocol constants for this experiment."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence


BLOCKS = (8, 16, 32)
DEFAULT_CONCURRENCIES = (2, 4, 8, 16, 32, 64, 128)
COLD_START_BLOCKS = {2: 32, 4: 32, 8: 16, 16: 16, 32: 16, 64: 8, 128: 8}
SECTION = re.compile(r"^### B(8|16|32)(?:\b|（)")


def cold_start_block(concurrency: int) -> int:
    try:
        return COLD_START_BLOCKS[int(concurrency)]
    except KeyError as exc:
        raise ValueError(
            f"unsupported C={concurrency}; expected {list(DEFAULT_CONCURRENCIES)}"
        ) from exc


def parse_cost_document(path: Path) -> Dict[int, Dict[int, float]]:
    text = path.read_text(encoding="utf-8")
    by_block: Dict[int, Dict[int, float]] = {block: {} for block in BLOCKS}
    block = None
    for line_number, line in enumerate(text.splitlines(), 1):
        match = SECTION.match(line.strip())
        if match:
            block = int(match.group(1))
            continue
        if block is None or not line.startswith("|"):
            continue
        fields = [field.strip() for field in line.strip().strip("|").split("|")]
        if len(fields) != 9 or not fields[0].isdigit() or fields[-1] == "N/A":
            continue
        concurrency = int(fields[0])
        if concurrency == 0:
            continue
        try:
            value = float(fields[-1])
        except ValueError as exc:
            raise RuntimeError(f"{path}:{line_number}: invalid latency") from exc
        previous = by_block[block].get(concurrency)
        if previous is not None and abs(previous - value) > 1e-9:
            raise RuntimeError(
                f"{path}:{line_number}: conflicting B{block}/C{concurrency} latency"
            )
        by_block[block][concurrency] = value
    for candidate in BLOCKS:
        missing = sorted(set(range(1, 129)) - set(by_block[candidate]))
        if missing:
            raise RuntimeError(f"B{candidate} latency table misses C={missing}")
    costs = {
        concurrency: {block: by_block[block][concurrency] for block in BLOCKS}
        for concurrency in range(1, 129)
    }
    if any(value <= 0 for row in costs.values() for value in row.values()):
        raise RuntimeError("latencies must be positive")
    return costs


def parse_int_list(text: str) -> list[int]:
    values = sorted({int(item.strip()) for item in text.split(",") if item.strip()})
    if not values or values[0] < 1 or values[-1] > 128:
        raise ValueError("integer list must be nonempty and contained in [1,128]")
    return values


def snapshot(path: Path, concurrencies: Sequence[int]) -> Dict[str, Any]:
    costs = parse_cost_document(path)
    selected = {
        str(concurrency): {
            str(block): costs[int(concurrency)][block] for block in BLOCKS
        }
        for concurrency in concurrencies
    }
    return {
        "schema_version": 1,
        "source": str(path.resolve()),
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "units": "ms per one draft plus one causal-verify forward",
        "blocks": list(BLOCKS),
        "concurrencies": list(concurrencies),
        "selected_costs_ms": selected,
        "all_costs_ms": {
            str(concurrency): {
                str(block): costs[concurrency][block] for block in BLOCKS
            }
            for concurrency in costs
        },
        "scope": (
            "pure B200 full-C bucket forward latency; no occupancy, scheduling, "
            "prefill, or wall-time model"
        ),
    }


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
