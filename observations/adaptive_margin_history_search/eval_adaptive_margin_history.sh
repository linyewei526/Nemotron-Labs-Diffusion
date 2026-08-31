#!/bin/bash
# Isolated entrypoint for offline trace replay or fresh PyTorch trace collection.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_DEFAULT="/data/home/wly/.conda/envs/nld_sglang/bin/python"
[[ -x "$PYTHON_DEFAULT" ]] || PYTHON_DEFAULT="python"
exec "$PYTHON_DEFAULT" "$SCRIPT_DIR/run.py" "$@"

