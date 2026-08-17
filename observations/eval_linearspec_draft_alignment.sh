#!/bin/bash
# LinearSpec draft-vs-final alignment diagnostic experiment.

set -euo pipefail

ORIGINAL_ARGS=("$@")
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OBSERVATIONS_DIR="$SCRIPT_DIR"
PROJECT_DIR="$(cd "$OBSERVATIONS_DIR/.." && pwd)"
OBSERVATION_RESULTS_ROOT="${NLD_OBSERVATION_RESULTS_ROOT:-/data/home/wly/dLLM/NLD_results/observations}"
EVAL_SGLANG="$OBSERVATIONS_DIR/eval_sglang.sh"
SUMMARY_SCRIPT="$PROJECT_DIR/xp/sglang_eval/summarize_linearspec_draft_alignment.py"

DEFAULT_OUTPUT_PATH="$OBSERVATION_RESULTS_ROOT/sglang_linearspec_draft_alignment_results"
DEFAULT_MODEL="/data1/linyewei/models/Nemotron-Labs-Diffusion-8B"
DEFAULT_SGLANG_PYTHON="/data/home/wly/.conda/envs/nld_sglang/bin/python"
if [[ ! -x "$DEFAULT_SGLANG_PYTHON" ]]; then
    DEFAULT_SGLANG_PYTHON="python"
fi

usage() {
    cat <<EOF
Usage: $0 --benchmarks LIST [options]

Purpose:
  Run LinearSpec draft-vs-final alignment diagnostics for one or more NeMo-Skills benchmarks.
  The experiment records every draft candidate token in each round and compares it
  with the final output token at the same sequence position after the request finishes.

Core:
  --benchmarks LIST           Required. Comma-separated benchmarks, e.g. gsm8k:1,math-500:1
  --mode MODE                 linearspec_lora or linearspec_base (default: linearspec_lora)
  --output-path DIR           Output root (default: $DEFAULT_OUTPUT_PATH)

SGLang / eval controls:
  --model PATH                Model path (default: $DEFAULT_MODEL)
  --served-model-name NAME    OpenAI model name (default: nemotron-labs-diffusion-8b)
  --gpu-devices LIST          CUDA_VISIBLE_DEVICES, e.g. 3 or 0,1 (default: 0)
  --tp-size N                 Tensor parallel size (default: inferred by eval_sglang.sh)
  --batch-size N              Alias for SGLang max running requests (default: 1)
  --client-concurrency N      NeMo client concurrency and proxy max in-flight requests (default: 1)
  --gpu-memory-reserve-gb V   Reserve V GiB on each selected GPU before server start (default: 0)
  --block-size N              LinearSpec block_size in generated YAML
  --tokens N                  Max generated tokens (default: 8192)
  --context-length N          SGLang context length; omit to use eval_sglang.sh auto rule
  --mem-fraction V            SGLang mem-fraction-static (default: 0.55)
  --cuda-graph-bs LIST        CUDA graph batch sizes, quoted when needed (default: 1)
  --port N                    SGLang server port; omit to auto-pick if default is busy
  --proxy-port N              Timing proxy port; omit to auto-pick if default is busy
  --max-samples N             Limit number of samples per benchmark for smoke tests
  --temperature V             Sampling temperature passed to NeMo-Skills (default: 0)
  --top-p V                   top_p passed to NeMo-Skills (default: 0.95)
  --nemo-skills-data-dir DIR  Persistent NeMo-Skills dataset dir
  --sglang-python PATH        Python for SGLang and summary script
  --eval-python PATH          Python for NeMo-Skills eval
  --sglang-src DIR            SGLang source root
  --sglang-work-dir DIR       SGLang work/cache root
  --lora-path DIR             LoRA adapter dir for linearspec_lora
  --lora-mode MODE            draft_only or both (default: draft_only)
  --extra-server-args "..."   Extra args appended to sglang.launch_server
  --dry-run                   Validate relocated dependencies and resolved paths, then exit
  -h, --help                  Show this help

Example:
  bash $0 --benchmarks gsm8k:1 --gpu-devices 3 --batch-size 1 --client-concurrency 1 --gpu-memory-reserve-gb 10 --block-size 32
EOF
}

MODE="linearspec_lora"
BENCHMARKS=""
OUTPUT_PATH="$DEFAULT_OUTPUT_PATH"
MODEL="$DEFAULT_MODEL"
SERVED_MODEL_NAME="nemotron-labs-diffusion-8b"
GPU_DEVICES="0"
TP_SIZE=""
BATCH_SIZE="1"
CLIENT_CONCURRENCY="1"
GPU_MEMORY_RESERVE_GB="0"
BLOCK_SIZE=""
TOKENS="8192"
CONTEXT_LENGTH=""
MEM_FRACTION="0.55"
CUDA_GRAPH_BS="1"
PORT=""
PROXY_PORT=""
MAX_SAMPLES=""
TEMPERATURE="0"
TOP_P="0.95"
NEMO_SKILLS_DATA_DIR=""
SGLANG_PYTHON="$DEFAULT_SGLANG_PYTHON"
EVAL_PYTHON=""
SGLANG_SRC=""
SGLANG_WORK_DIR=""
LORA_PATH=""
LORA_MODE="draft_only"
EXTRA_SERVER_ARGS=""
DRY_RUN="false"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --benchmarks) BENCHMARKS="$2"; shift 2 ;;
        --mode) MODE="$2"; shift 2 ;;
        --output-path|--out-dir) OUTPUT_PATH="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --served-model-name|--model-name) SERVED_MODEL_NAME="$2"; shift 2 ;;
        --gpu-devices) GPU_DEVICES="$2"; shift 2 ;;
        --tp-size|--tensor-parallel-size) TP_SIZE="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --client-concurrency) CLIENT_CONCURRENCY="$2"; shift 2 ;;
        --gpu-memory-reserve-gb) GPU_MEMORY_RESERVE_GB="$2"; shift 2 ;;
        --block-size) BLOCK_SIZE="$2"; shift 2 ;;
        --tokens) TOKENS="$2"; shift 2 ;;
        --context-length) CONTEXT_LENGTH="$2"; shift 2 ;;
        --mem-fraction) MEM_FRACTION="$2"; shift 2 ;;
        --cuda-graph-bs) CUDA_GRAPH_BS="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --proxy-port) PROXY_PORT="$2"; shift 2 ;;
        --max-samples) MAX_SAMPLES="$2"; shift 2 ;;
        --temperature) TEMPERATURE="$2"; shift 2 ;;
        --top-p) TOP_P="$2"; shift 2 ;;
        --nemo-skills-data-dir) NEMO_SKILLS_DATA_DIR="$2"; shift 2 ;;
        --sglang-python) SGLANG_PYTHON="$2"; shift 2 ;;
        --eval-python) EVAL_PYTHON="$2"; shift 2 ;;
        --sglang-src) SGLANG_SRC="$2"; shift 2 ;;
        --sglang-work-dir) SGLANG_WORK_DIR="$2"; shift 2 ;;
        --lora-path) LORA_PATH="$2"; shift 2 ;;
        --lora-mode) LORA_MODE="$2"; shift 2 ;;
        --extra-server-args) EXTRA_SERVER_ARGS="$2"; shift 2 ;;
        --dry-run) DRY_RUN="true"; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "ERROR: unknown option: $1" >&2; usage; exit 1 ;;
    esac
done

if [[ -z "$BENCHMARKS" ]]; then
    echo "ERROR: --benchmarks is required" >&2
    usage
    exit 1
fi

case "$MODE" in
    linearspec_lora|linear_spec_lora) MODE="linearspec_lora" ;;
    linearspec_base|linear_spec_base|linearspec) MODE="linearspec_base" ;;
    *) echo "ERROR: --mode must be linearspec_lora or linearspec_base, got: $MODE" >&2; exit 1 ;;
esac

if command -v realpath >/dev/null; then
    OUTPUT_PATH="$(realpath -m "$OUTPUT_PATH")"
fi

if [[ "$DRY_RUN" == "true" ]]; then
    [[ -f "$EVAL_SGLANG" ]] || { echo "ERROR: missing nested entrypoint: $EVAL_SGLANG" >&2; exit 1; }
    [[ -f "$SUMMARY_SCRIPT" ]] || { echo "ERROR: missing summary script: $SUMMARY_SCRIPT" >&2; exit 1; }
    printf 'Dry run OK\n  observation entry: %s\n  project root: %s\n  nested eval: %s\n  summary script: %s\n  output root: %s\n  benchmarks: %s\n  mode: %s\n' \
        "$0" "$PROJECT_DIR" "$EVAL_SGLANG" "$SUMMARY_SCRIPT" "$OUTPUT_PATH" "$BENCHMARKS" "$MODE"
    exit 0
fi

RUN_NAME="linearspec_draft_alignment_$(date +%Y%m%d_%H%M%S)"
FINAL_DIR="$OUTPUT_PATH/$RUN_NAME"
if [[ -e "$FINAL_DIR" ]]; then
    suffix=1
    while [[ -e "$OUTPUT_PATH/${RUN_NAME}_$(printf '%02d' "$suffix")" ]]; do
        suffix=$((suffix + 1))
    done
    FINAL_DIR="$OUTPUT_PATH/${RUN_NAME}_$(printf '%02d' "$suffix")"
fi

TRACE_DIR="$FINAL_DIR/traces"
SUMMARY_DIR="$FINAL_DIR/summaries"
EVAL_RUNS_DIR="$FINAL_DIR/eval_runs"
mkdir -p "$TRACE_DIR" "$SUMMARY_DIR" "$EVAL_RUNS_DIR"

write_settings() {
    "$SGLANG_PYTHON" - "$FINAL_DIR/Settings.json" "${ORIGINAL_ARGS[@]}" <<'PY'
import json
import os
import shlex
import sys
from datetime import datetime

settings = {
    "created_at": datetime.now().astimezone().isoformat(),
    "entrypoint": "observations/eval_linearspec_draft_alignment.sh",
    "argv": sys.argv[2:],
    "command": "bash observations/eval_linearspec_draft_alignment.sh " + " ".join(shlex.quote(x) for x in sys.argv[2:]),
    "project_dir": os.environ["PROJECT_DIR"],
    "mode": os.environ["MODE"],
    "benchmarks": os.environ["BENCHMARKS"],
    "output_dir": os.environ["FINAL_DIR"],
    "trace_dir": os.environ["TRACE_DIR"],
    "summary_dir": os.environ["SUMMARY_DIR"],
    "eval_runs_dir": os.environ["EVAL_RUNS_DIR"],
    "model": os.environ["MODEL"],
    "served_model_name": os.environ["SERVED_MODEL_NAME"],
    "gpu_devices": os.environ["GPU_DEVICES"],
    "tp_size": os.environ["TP_SIZE"],
    "batch_size": os.environ["BATCH_SIZE"],
    "client_concurrency": os.environ["CLIENT_CONCURRENCY"],
    "gpu_memory_reserve_gb": os.environ["GPU_MEMORY_RESERVE_GB"],
    "block_size": os.environ["BLOCK_SIZE"],
    "tokens": os.environ["TOKENS"],
    "context_length": os.environ["CONTEXT_LENGTH"],
    "mem_fraction": os.environ["MEM_FRACTION"],
    "cuda_graph_bs": os.environ["CUDA_GRAPH_BS"],
    "port": os.environ["PORT"],
    "proxy_port": os.environ["PROXY_PORT"],
    "max_samples": os.environ["MAX_SAMPLES"],
    "temperature": os.environ["TEMPERATURE"],
    "top_p": os.environ["TOP_P"],
    "nemo_skills_data_dir": os.environ["NEMO_SKILLS_DATA_DIR"],
    "sglang_python": os.environ["SGLANG_PYTHON"],
    "eval_python": os.environ["EVAL_PYTHON"],
    "sglang_src": os.environ["SGLANG_SRC"],
    "sglang_work_dir": os.environ["SGLANG_WORK_DIR"],
    "lora_path": os.environ["LORA_PATH"],
    "lora_mode": os.environ["LORA_MODE"],
    "extra_server_args": os.environ["EXTRA_SERVER_ARGS"],
}
with open(sys.argv[1], "w", encoding="utf-8") as f:
    json.dump(settings, f, indent=2, ensure_ascii=False, sort_keys=True)
    f.write("\n")
PY
}

safe_name() {
    local name="$1"
    name="${name%%:*}"
    name="${name//\//_}"
    name="${name//:/_}"
    name="${name//[[:space:]]/_}"
    echo "$name"
}

append_status() {
    local bench_spec="$1"
    local bench_name="$2"
    local status="$3"
    local trace_file="$4"
    local summary_file="$5"
    local eval_parent="$6"
    "$SGLANG_PYTHON" - "$FINAL_DIR/benchmark_status.jsonl" <<PY
import json
from datetime import datetime
payload = {
    "created_at": datetime.now().astimezone().isoformat(),
    "benchmark_spec": "$bench_spec",
    "benchmark": "$bench_name",
    "exit_code": int("$status"),
    "trace_file": "$trace_file",
    "summary_file": "$summary_file",
    "eval_parent": "$eval_parent",
}
with open("$FINAL_DIR/benchmark_status.jsonl", "a", encoding="utf-8") as f:
    f.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\\n")
PY
}

PROJECT_DIR="$PROJECT_DIR" MODE="$MODE" BENCHMARKS="$BENCHMARKS" FINAL_DIR="$FINAL_DIR" TRACE_DIR="$TRACE_DIR" SUMMARY_DIR="$SUMMARY_DIR" EVAL_RUNS_DIR="$EVAL_RUNS_DIR" MODEL="$MODEL" SERVED_MODEL_NAME="$SERVED_MODEL_NAME" GPU_DEVICES="$GPU_DEVICES" TP_SIZE="$TP_SIZE" BATCH_SIZE="$BATCH_SIZE" CLIENT_CONCURRENCY="$CLIENT_CONCURRENCY" GPU_MEMORY_RESERVE_GB="$GPU_MEMORY_RESERVE_GB" BLOCK_SIZE="$BLOCK_SIZE" TOKENS="$TOKENS" CONTEXT_LENGTH="$CONTEXT_LENGTH" MEM_FRACTION="$MEM_FRACTION" CUDA_GRAPH_BS="$CUDA_GRAPH_BS" PORT="$PORT" PROXY_PORT="$PROXY_PORT" MAX_SAMPLES="$MAX_SAMPLES" TEMPERATURE="$TEMPERATURE" TOP_P="$TOP_P" NEMO_SKILLS_DATA_DIR="$NEMO_SKILLS_DATA_DIR" SGLANG_PYTHON="$SGLANG_PYTHON" EVAL_PYTHON="$EVAL_PYTHON" SGLANG_SRC="$SGLANG_SRC" SGLANG_WORK_DIR="$SGLANG_WORK_DIR" LORA_PATH="$LORA_PATH" LORA_MODE="$LORA_MODE" EXTRA_SERVER_ARGS="$EXTRA_SERVER_ARGS" write_settings

echo "================================================================"
echo "  LinearSpec draft-vs-final alignment diagnostic eval"
echo "================================================================"
echo "  Mode:          $MODE"
echo "  Benchmarks:    $BENCHMARKS"
echo "  GPU devices:   $GPU_DEVICES"
echo "  Output dir:    $FINAL_DIR"
echo "  Trace dir:     $TRACE_DIR"
echo "  Summary dir:   $SUMMARY_DIR"
echo "================================================================"

IFS=',' read -ra BENCH_ARRAY <<< "$BENCHMARKS"
ANY_FAILURE=0

for bench_spec in "${BENCH_ARRAY[@]}"; do
    bench_spec="${bench_spec//[[:space:]]/}"
    [[ -z "$bench_spec" ]] && continue
    bench_name="$(safe_name "$bench_spec")"
    trace_file="$TRACE_DIR/raw_draft_alignment_trace_${bench_name}.jsonl"
    summary_file="$SUMMARY_DIR/draft_alignment_${bench_name}.json"
    eval_parent="$EVAL_RUNS_DIR/$bench_name"
    mkdir -p "$eval_parent"
    : > "$trace_file"

    cmd=(bash "$EVAL_SGLANG"
        --mode "$MODE"
        --benchmarks "$bench_spec"
        --model "$MODEL"
        --served-model-name "$SERVED_MODEL_NAME"
        --gpu-devices "$GPU_DEVICES"
        --batch-size "$BATCH_SIZE"
        --client-concurrency "$CLIENT_CONCURRENCY"
        --gpu-memory-reserve-gb "$GPU_MEMORY_RESERVE_GB"
        --tokens "$TOKENS"
        --temperature "$TEMPERATURE"
        --top-p "$TOP_P"
        --mem-fraction "$MEM_FRACTION"
        --cuda-graph-bs "$CUDA_GRAPH_BS"
        --lora-mode "$LORA_MODE"
        --output-path "$eval_parent")

    [[ -n "$TP_SIZE" ]] && cmd+=(--tp-size "$TP_SIZE")
    [[ -n "$BLOCK_SIZE" ]] && cmd+=(--block-size "$BLOCK_SIZE")
    [[ -n "$CONTEXT_LENGTH" ]] && cmd+=(--context-length "$CONTEXT_LENGTH")
    [[ -n "$PORT" ]] && cmd+=(--port "$PORT")
    [[ -n "$PROXY_PORT" ]] && cmd+=(--proxy-port "$PROXY_PORT")
    [[ -n "$MAX_SAMPLES" ]] && cmd+=(--max-samples "$MAX_SAMPLES")
    [[ -n "$NEMO_SKILLS_DATA_DIR" ]] && cmd+=(--nemo-skills-data-dir "$NEMO_SKILLS_DATA_DIR")
    [[ -n "$SGLANG_PYTHON" ]] && cmd+=(--sglang-python "$SGLANG_PYTHON")
    [[ -n "$EVAL_PYTHON" ]] && cmd+=(--eval-python "$EVAL_PYTHON")
    [[ -n "$SGLANG_SRC" ]] && cmd+=(--sglang-src "$SGLANG_SRC")
    [[ -n "$SGLANG_WORK_DIR" ]] && cmd+=(--sglang-work-dir "$SGLANG_WORK_DIR")
    [[ -n "$LORA_PATH" ]] && cmd+=(--lora-path "$LORA_PATH")
    [[ -n "$EXTRA_SERVER_ARGS" ]] && cmd+=(--extra-server-args "$EXTRA_SERVER_ARGS")

    echo ""
    echo "--- benchmark: $bench_spec ---"
    echo "Trace:   $trace_file"
    echo "Summary: $summary_file"

    status=0
    SGLANG_CONFIDENCE_TRACE_FILE="" SGLANG_LOW_CONFIDENCE_TRACE_FILE="" SGLANG_DRAFT_ALIGNMENT_TRACE_FILE="$trace_file" "${cmd[@]}" || status=$?
    if [[ "$status" != "0" ]]; then
        ANY_FAILURE=1
        echo "WARNING: benchmark $bench_spec exited with status $status; summarizing trace if present." >&2
    fi

    "$SGLANG_PYTHON" "$SUMMARY_SCRIPT" \
        --trace-file "$trace_file" \
        --output-json "$summary_file" \
        --benchmark "$bench_name" \
        --benchmark-spec "$bench_spec"
    append_status "$bench_spec" "$bench_name" "$status" "$trace_file" "$summary_file" "$eval_parent"
    echo "Wrote summary: $summary_file"
done

echo ""
echo "Completed LinearSpec draft-vs-final alignment diagnostic eval."
echo "Output dir: $FINAL_DIR"
if [[ "$ANY_FAILURE" != "0" ]]; then
    echo "One or more benchmarks failed; see benchmark_status.jsonl and nested eval_runs outputs."
fi
