#!/bin/bash
# Offline low-confidence threshold curves from an existing PyTorch confidence trace.

set -euo pipefail

OBSERVATIONS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$OBSERVATIONS_DIR/.." && pwd)"
OBSERVATION_RESULTS_ROOT="${NLD_OBSERVATION_RESULTS_ROOT:-/data/home/wly/dLLM/NLD_results/observations}"
ANALYZER="$PROJECT_DIR/xp/pytorch_linearspec_low_confidence_offline/analyze_existing_confidence_traces.py"
DEFAULT_INPUT_RUN="$OBSERVATION_RESULTS_ROOT/pytorch_linearspec_confidence_results/linearspec_confidence_20260804_165154"
DEFAULT_OUTPUT_PATH="$OBSERVATION_RESULTS_ROOT/pytorch_linearspec_low_confidence_offline_results"

usage() {
    cat <<EOF
Usage: bash $0 [options]

Read completed native-PyTorch LinearSpec confidence traces and reconstruct the
SGLang-compatible low-confidence threshold curves. This is CPU-only and does not
load the model or import torch.

  --input-run DIR          Source PyTorch confidence run (default: current completed block=16 run)
  --output-path DIR        Result root; a timestamped run is created below it
  --benchmarks LIST        Optional comma-separated subset, e.g. gsm8k:1,math-500:1
  --require-block-size N   Reject a source run with another block size (default: 16)
  --abs-start V            drop_abs start, inclusive (default: 0.140; calibrated on PyTorch block=16)
  --abs-end V              drop_abs end, inclusive (default: 0.300; calibrated on PyTorch block=16)
  --abs-step V             drop_abs step (default: 0.005)
  --pct-start V            drop_pct start, inclusive (default: 0.15; calibrated on PyTorch block=16)
  --pct-end V              drop_pct end, inclusive (default: 0.33; calibrated on PyTorch block=16)
  --pct-step V             drop_pct step (default: 0.01)
  --python PATH            Python interpreter (default: current python3/python)
  --dry-run                Print the resolved command without creating output
  -h, --help               Show this help
EOF
}

INPUT_RUN="$DEFAULT_INPUT_RUN"
OUTPUT_PATH="$DEFAULT_OUTPUT_PATH"
BENCHMARKS=""
REQUIRE_BLOCK_SIZE="16"
ABS_START="0.140"
ABS_END="0.300"
ABS_STEP="0.005"
PCT_START="0.15"
PCT_END="0.33"
PCT_STEP="0.01"
PYTHON_BIN=""
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --input-run) INPUT_RUN="$2"; shift 2 ;;
        --output-path|--out-dir) OUTPUT_PATH="$2"; shift 2 ;;
        --benchmarks) BENCHMARKS="$2"; shift 2 ;;
        --require-block-size) REQUIRE_BLOCK_SIZE="$2"; shift 2 ;;
        --abs-start) ABS_START="$2"; shift 2 ;;
        --abs-end) ABS_END="$2"; shift 2 ;;
        --abs-step) ABS_STEP="$2"; shift 2 ;;
        --pct-start) PCT_START="$2"; shift 2 ;;
        --pct-end) PCT_END="$2"; shift 2 ;;
        --pct-step) PCT_STEP="$2"; shift 2 ;;
        --python) PYTHON_BIN="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "ERROR: unknown option: $1" >&2; usage; exit 1 ;;
    esac
done

if [[ -z "$PYTHON_BIN" ]]; then
    PYTHON_BIN="$(command -v python3 || command -v python)"
fi
if [[ ! -f "$ANALYZER" ]]; then
    echo "ERROR: analyzer does not exist: $ANALYZER" >&2
    exit 1
fi
if [[ ! -d "$INPUT_RUN/traces" || ! -f "$INPUT_RUN/Settings.json" ]]; then
    echo "ERROR: invalid input run: $INPUT_RUN" >&2
    exit 1
fi

RUN_NAME="offline_low_confidence_$(date +%Y%m%d_%H%M%S)"
FINAL_DIR="$OUTPUT_PATH/$RUN_NAME"
suffix=1
while [[ -e "$FINAL_DIR" ]]; do
    FINAL_DIR="$OUTPUT_PATH/${RUN_NAME}_$(printf '%02d' "$suffix")"
    suffix=$((suffix + 1))
done

COMMAND=(
    "$PYTHON_BIN" "$ANALYZER"
    --input-run "$INPUT_RUN"
    --output-dir "$FINAL_DIR"
    --require-block-size "$REQUIRE_BLOCK_SIZE"
    --abs-start "$ABS_START" --abs-end "$ABS_END" --abs-step "$ABS_STEP"
    --pct-start "$PCT_START" --pct-end "$PCT_END" --pct-step "$PCT_STEP"
)
if [[ -n "$BENCHMARKS" ]]; then
    COMMAND+=(--benchmarks "$BENCHMARKS")
fi

printf 'Resolved command:'
printf ' %q' "${COMMAND[@]}"
printf '\n'
printf 'GPU/model use: none\n'
if [[ "$DRY_RUN" -eq 1 ]]; then
    exit 0
fi

"${COMMAND[@]}"
