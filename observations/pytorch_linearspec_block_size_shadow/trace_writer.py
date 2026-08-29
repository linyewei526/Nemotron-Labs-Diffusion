#!/usr/bin/env python3
"""Thread-safe JSONL writer for paired block-size shadow rounds."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any


class ShadowTraceWriter:
    """Append complete round records without allowing interleaved JSON lines."""

    def __init__(
        self,
        path: str | Path,
        *,
        benchmark: str,
        trace_detail: str = "position",
    ) -> None:
        if trace_detail not in {"scalar", "position", "tokens"}:
            raise ValueError("trace_detail must be scalar, position, or tokens")
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)
        self.benchmark = str(benchmark)
        self.trace_detail = trace_detail
        self._lock = threading.Lock()

    def write(self, record: dict[str, Any]) -> None:
        payload = dict(record)
        payload.setdefault("benchmark", self.benchmark)
        line = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        with self._lock:
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(line + "\n")
