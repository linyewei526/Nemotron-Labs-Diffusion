#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REAL_PYTHON="/data/home/wly/.conda/envs/nld_sglang/bin/python"

if [[ "${1:-}" == "-m" && "${2:-}" == "sglang.launch_server" ]]; then
    export NLD_DIAG_APPLY_PREDRAFT_HOOK=1
    export PYTHONPATH="$SCRIPT_DIR/diagnostic_site${PYTHONPATH:+:$PYTHONPATH}"
fi

exec "$REAL_PYTHON" "$@"
