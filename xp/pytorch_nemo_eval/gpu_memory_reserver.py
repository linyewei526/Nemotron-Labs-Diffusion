#!/usr/bin/env python3
"""Reserve a fixed amount of memory on every CUDA device visible to this process."""

from __future__ import annotations

import argparse
import json
import os
import signal
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gb", type=float, required=True, help="GiB to reserve per visible GPU")
    parser.add_argument("--chunk-mb", type=int, default=512)
    parser.add_argument("--ready-file", default="")
    args = parser.parse_args()

    if args.gb <= 0:
        return 0
    if args.chunk_mb <= 0:
        raise SystemExit("--chunk-mb must be positive")

    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        raise SystemExit("CUDA is not available; cannot reserve GPU memory")

    target_bytes = int(args.gb * 1024**3)
    chunk_bytes = int(args.chunk_mb * 1024**2)
    all_allocations: list[list[torch.Tensor]] = []
    summaries = []
    for device_idx in range(torch.cuda.device_count()):
        torch.cuda.set_device(device_idx)
        allocations: list[torch.Tensor] = []
        allocated = 0
        try:
            while allocated < target_bytes:
                size = min(chunk_bytes, target_bytes - allocated)
                tensor = torch.empty(size, dtype=torch.uint8, device=f"cuda:{device_idx}")
                tensor[0] = 1
                allocations.append(tensor)
                allocated += size
            torch.cuda.synchronize(device_idx)
        except torch.cuda.OutOfMemoryError as exc:
            raise SystemExit(
                f"Failed to reserve {args.gb} GiB on visible CUDA device {device_idx}; "
                f"reserved {allocated / 1024**3:.3f} GiB before OOM: {exc}"
            ) from exc
        all_allocations.append(allocations)
        summaries.append(
            {
                "visible_device": device_idx,
                "reserved_gib": round(allocated / 1024**3, 6),
                "chunks": len(allocations),
            }
        )
        print(f"Reserved {allocated / 1024**3:.3f} GiB on visible CUDA device {device_idx}", flush=True)

    if args.ready_file:
        path = Path(args.ready_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "visible_cuda_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                    "requested_gib_per_gpu": args.gb,
                    "allocations": summaries,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    running = True

    def stop(_signum: int, _frame: object) -> None:
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    while running:
        time.sleep(1)

    all_allocations.clear()
    for device_idx in range(torch.cuda.device_count()):
        torch.cuda.set_device(device_idx)
        torch.cuda.empty_cache()
    print("Released reserved GPU memory", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
