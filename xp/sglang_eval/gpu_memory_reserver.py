#!/usr/bin/env python3
"""Reserve GPU memory in a long-running helper process."""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gb", type=float, required=True, help="GiB to reserve per visible GPU.")
    parser.add_argument("--chunk-mb", type=int, default=512, help="Allocation chunk size in MiB.")
    parser.add_argument("--ready-file", default="", help="Path written after successful allocation.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.gb <= 0:
        return 0
    if args.chunk_mb <= 0:
        print("--chunk-mb must be positive", file=sys.stderr)
        return 2

    import torch

    if not torch.cuda.is_available():
        print("CUDA is not available; cannot reserve GPU memory.", file=sys.stderr)
        return 2

    device_count = torch.cuda.device_count()
    if device_count <= 0:
        print("No visible CUDA devices; cannot reserve GPU memory.", file=sys.stderr)
        return 2

    target_bytes = int(args.gb * (1024**3))
    chunk_bytes = int(args.chunk_mb * (1024**2))
    allocations: list[list[torch.Tensor]] = []
    allocation_summary = []

    for device_idx in range(device_count):
        torch.cuda.set_device(device_idx)
        device_allocations: list[torch.Tensor] = []
        remaining = target_bytes
        allocated = 0
        try:
            while remaining > 0:
                alloc_bytes = min(chunk_bytes, remaining)
                tensor = torch.empty(alloc_bytes, dtype=torch.uint8, device=f"cuda:{device_idx}")
                tensor[0] = 1
                device_allocations.append(tensor)
                allocated += alloc_bytes
                remaining -= alloc_bytes
            torch.cuda.synchronize(device_idx)
        except torch.cuda.OutOfMemoryError as exc:
            print(
                f"Failed to reserve {args.gb} GiB on visible CUDA device {device_idx}; "
                f"allocated {allocated / (1024**3):.3f} GiB before OOM.",
                file=sys.stderr,
            )
            print(str(exc), file=sys.stderr)
            return 2

        allocations.append(device_allocations)
        allocation_summary.append(
            {
                "visible_device": device_idx,
                "reserved_gib": round(allocated / (1024**3), 6),
                "chunks": len(device_allocations),
            }
        )
        print(
            f"Reserved {allocated / (1024**3):.3f} GiB on visible CUDA device {device_idx}.",
            flush=True,
        )

    ready_payload = {
        "pid": os.getpid(),
        "visible_cuda_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "requested_gib_per_gpu": args.gb,
        "allocations": allocation_summary,
    }
    if args.ready_file:
        ready_path = Path(args.ready_file)
        ready_path.parent.mkdir(parents=True, exist_ok=True)
        ready_path.write_text(json.dumps(ready_payload, indent=2), encoding="utf-8")

    running = True

    def stop(_signum: int, _frame: object) -> None:
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    while running:
        time.sleep(1)

    allocations.clear()
    if torch.cuda.is_available():
        for device_idx in range(device_count):
            torch.cuda.set_device(device_idx)
            torch.cuda.empty_cache()
    print("Released reserved GPU memory.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
