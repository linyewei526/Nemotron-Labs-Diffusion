#!/usr/bin/env python3
"""Select a physical GPU with sufficient free memory for an isolated run."""

from __future__ import annotations

import argparse
import csv
import io
import subprocess
import time


def query() -> list[dict[str, float | int]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    output = subprocess.check_output(command, text=True)
    rows = []
    for fields in csv.reader(io.StringIO(output)):
        index, free_mib, utilization = (field.strip() for field in fields)
        rows.append(
            {
                "index": int(index),
                "free_gib": float(free_mib) / 1024.0,
                "utilization": float(utilization),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-free-gb", type=float, default=24.0)
    parser.add_argument("--candidates", default="")
    parser.add_argument("--wait-seconds", type=int, default=0)
    parser.add_argument("--poll-seconds", type=int, default=15)
    args = parser.parse_args()
    allowed = (
        {int(item.strip()) for item in args.candidates.split(",") if item.strip()}
        if args.candidates
        else None
    )
    deadline = time.monotonic() + args.wait_seconds
    while True:
        candidates = [
            row
            for row in query()
            if (allowed is None or int(row["index"]) in allowed)
            and float(row["free_gib"]) >= args.min_free_gb
        ]
        if candidates:
            candidates.sort(
                key=lambda row: (
                    float(row["free_gib"]),
                    -float(row["utilization"]),
                ),
                reverse=True,
            )
            print(int(candidates[0]["index"]))
            return 0
        if time.monotonic() >= deadline:
            visible = ", ".join(
                f"gpu{row['index']}={row['free_gib']:.2f}GiB/{row['utilization']:.0f}%"
                for row in query()
                if allowed is None or int(row["index"]) in allowed
            )
            raise SystemExit(
                f"no candidate GPU has {args.min_free_gb:.2f} GiB free; observed: {visible}"
            )
        time.sleep(max(1, args.poll_seconds))


if __name__ == "__main__":
    raise SystemExit(main())
