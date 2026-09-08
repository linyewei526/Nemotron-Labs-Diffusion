#!/usr/bin/env python3
"""Isolated entry point for the tested unified-policy GPU memory guard."""

from __future__ import annotations

import sys
from pathlib import Path


OBSERVATIONS = Path(__file__).resolve().parents[1]
if str(OBSERVATIONS) not in sys.path:
    sys.path.insert(0, str(OBSERVATIONS))

from sglang_unified_scalar_block_policy.gpu_memory_guard import main  # noqa: E402


if __name__ == "__main__":
    main()

