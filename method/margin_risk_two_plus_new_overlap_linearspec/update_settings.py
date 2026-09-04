#!/usr/bin/env python3
"""Atomically update resolved runtime fields in an experiment Settings.json."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("settings")
    parser.add_argument("--resolved-port", type=int)
    parser.add_argument("--resolved-gpu", type=int)
    parser.add_argument("--status", default="")
    args = parser.parse_args()
    path = Path(args.settings).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    resolved = payload.setdefault("resolved_runtime", {})
    if args.resolved_port is not None:
        resolved["port"] = args.resolved_port
    if args.resolved_gpu is not None:
        resolved["gpu_device"] = args.resolved_gpu
    if args.status:
        payload["status"] = args.status
    payload["updated_at"] = datetime.now().astimezone().isoformat()
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

