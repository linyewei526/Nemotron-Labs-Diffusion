#!/usr/bin/env python3
"""Reserve configured GPU memory while this experiment performs CPU search."""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hold-gb", type=float, required=True, help="GiB reserved per visible GPU"
    )
    parser.add_argument("--chunk-gb", type=float, default=1.0)
    parser.add_argument("--ready-file", type=Path, required=True)
    parser.add_argument("--released-file", type=Path, required=True)
    args = parser.parse_args()
    if args.hold_gb <= 0 or args.chunk_gb <= 0:
        parser.error("--hold-gb and --chunk-gb must be positive")

    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("CUDA is unavailable in the GPU guard process")
    requested = int(args.hold_gb * (1024**3))
    chunk = max(1, int(args.chunk_gb * (1024**3)))
    allocations: list[list[torch.Tensor]] = []
    devices: list[dict[str, Any]] = []
    for index in range(torch.cuda.device_count()):
        torch.cuda.set_device(index)
        free, total = torch.cuda.mem_get_info(index)
        # Leave room for CUDA context initialization and bookkeeping.
        margin = 256 * 1024**2
        if requested > free - margin:
            raise RuntimeError(
                f"GPU {index} has only {free / 1024**3:.2f} GiB free; "
                f"cannot reserve {args.hold_gb:.2f} GiB safely"
            )
        tensors: list[torch.Tensor] = []
        remaining = requested
        while remaining > 0:
            size = min(chunk, remaining)
            tensors.append(torch.empty(size, dtype=torch.uint8, device=index))
            remaining -= size
        torch.cuda.synchronize(index)
        allocations.append(tensors)
        devices.append(
            {
                "visible_index": index,
                "name": torch.cuda.get_device_name(index),
                "requested_gib": args.hold_gb,
                "free_before_gib": free / 1024**3,
                "total_gib": total / 1024**3,
            }
        )
        print(
            f"[GPU显存守护] visible GPU {index}: reserved {args.hold_gb:.2f} GiB",
            flush=True,
        )

    stopping = False

    def stop(_signum=None, _frame=None) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    atomic_json(
        args.ready_file,
        {
            "status": "ready",
            "pid": os.getpid(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "devices": devices,
            "ready_at": datetime.now().astimezone().isoformat(),
        },
    )
    while not stopping:
        time.sleep(1)

    allocations.clear()
    for index in range(torch.cuda.device_count()):
        torch.cuda.set_device(index)
        torch.cuda.empty_cache()
    atomic_json(
        args.released_file,
        {
            "status": "released",
            "pid": os.getpid(),
            "released_at": datetime.now().astimezone().isoformat(),
        },
    )
    print("[GPU显存守护] all allocations released", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"GPU memory guard failed: {exc}", file=sys.stderr, flush=True)
        raise
