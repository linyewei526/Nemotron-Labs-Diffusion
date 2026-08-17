#!/bin/bash
# Native PyTorch + NeMo-Skills LinearSpec confidence/rank diagnostics.

set -euo pipefail

ORIGINAL_ARGS=("$@")
OBSERVATIONS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$OBSERVATIONS_DIR/.." && pwd)"
OBSERVATION_RESULTS_ROOT="${NLD_OBSERVATION_RESULTS_ROOT:-/data/home/wly/dLLM/NLD_results/observations}"
EVAL_SCRIPT="$PROJECT_DIR/xp/nemo-skills/eval_dlm.py"
SERVER_SCRIPT="$PROJECT_DIR/xp/pytorch_linearspec_confidence/pytorch_confidence_openai_server.py"
SUMMARY_SCRIPT="$PROJECT_DIR/xp/pytorch_linearspec_confidence/summarize_linearspec_confidence_trace.py"
METRICS_MERGER="$PROJECT_DIR/xp/pytorch_nemo_eval/add_pytorch_metrics_to_metrics.py"
MEMORY_RESERVER="$PROJECT_DIR/xp/pytorch_nemo_eval/gpu_memory_reserver.py"

DEFAULT_MODEL="/data1/linyewei/models/Nemotron-Labs-Diffusion-8B"
DEFAULT_OUTPUT="$OBSERVATION_RESULTS_ROOT/pytorch_linearspec_confidence_results"
DEFAULT_DATA_DIR="/data1/linyewei/datasets/NLD"
DEFAULT_PYTHON="/data/home/wly/.conda/envs/nld_sglang/bin/python"
[[ -x "$DEFAULT_PYTHON" ]] || DEFAULT_PYTHON="python"

usage() {
    cat <<EOF
Usage: $0 --benchmarks LIST [options]

Native PyTorch LinearSpec confidence/rank diagnostics using NeMo-Skills.
This entry never starts or imports SGLang.

Core:
  --benchmarks LIST           Required comma-separated NeMo-Skills specs
  --mode MODE                 linearspec_lora or linearspec_base (default: linearspec_lora)
  --output-path DIR           Result root (default: $DEFAULT_OUTPUT)
  --include-values            Include raw value arrays in summary JSON
  --bins N                    Histogram bin count (default: 100)

Model/decoding:
  --model PATH                Model path (default: $DEFAULT_MODEL)
  --served-model-name NAME    OpenAI model label
  --lora-path DIR             LoRA path (default: <model>/linear_spec_lora)
  --block-size N              Native LinearSpec block length (default: 32)
  --block-length N            Alias for --block-size
  --threshold V               Native draft unmask threshold (default: 0)
  --temperature V             Native sampling temperature (default: 0)
  --top-p V                   Recorded/forwarded for protocol parity; native method does not apply it (default: 0.95)
  --tokens N                  Maximum returned completion tokens (default: 8192)
  --context-length N          Prompt + internal generation limit (default: tokens+2048)
  --dtype DTYPE               bfloat16, float16, or float32 (default: bfloat16)
  --max-thinking-tokens N     Force </think> after this generated-token budget

GPU/client:
  --gpu-device ID             One physical GPU (default: 0)
  --gpu-devices ID            Alias for --gpu-device; comma lists are rejected
  --batch-size N              Compatibility flag; only 1 is supported (default: 1)
  --client-concurrency N      NeMo concurrent requests; model remains serialized (default: 1)
  --num-chunks N              NeMo client chunks (default: client concurrency)
  --gpu-memory-reserve-gb V   Reserve V GiB before model load (default: 0)
  --port N                    Diagnostic server port (default: auto from 33000+GPU)

Benchmark/prompt:
  --max-samples N             Limit samples per benchmark
  --quick-test                NeMo-Skills quick test
  --enable-thinking           Enable thinking in the native chat template
  --disable-thinking          Explicit non-thinking configuration
  --keep-thinking             Keep thinking in NeMo output/scoring flow
  --strip-thinking            Strip thinking and re-score where supported
  --math-prompt-config NAME   Override NeMo math prompt config

Environment:
  --pytorch-python PATH       Model server Python (default: nld_sglang)
  --eval-python PATH          NeMo-Skills Python (default: PyTorch Python)
  --nemo-skills-data-dir DIR  Prepared dataset/cache root (default: $DEFAULT_DATA_DIR)
  --google-research-dir DIR   IFEval google-research checkout
  --dry-run                   Validate/print settings without writing or loading a model
  -h, --help                  Show this help

Example:
  bash $0 --benchmarks gsm8k:1 --gpu-devices 2 --batch-size 1 --client-concurrency 1 --gpu-memory-reserve-gb 1 --block-size 16 --tokens 128 --max-samples 1
EOF
}

MODE="linearspec_lora"
BENCHMARKS=""
OUTPUT_PATH="$DEFAULT_OUTPUT"
MODEL="$DEFAULT_MODEL"
SERVED_MODEL_NAME="nemotron-labs-diffusion-8b"
LORA_PATH=""
BLOCK_SIZE="32"
THRESHOLD="0"
TEMPERATURE="0"
TOP_P="0.95"
TOKENS="8192"
CONTEXT_LENGTH=""
DTYPE="bfloat16"
MAX_THINKING_TOKENS=""
GPU_DEVICE="0"
BATCH_SIZE="1"
CLIENT_CONCURRENCY="1"
NUM_CHUNKS=""
GPU_MEMORY_RESERVE_GB="0"
PORT=""
PORT_USER_SET="false"
MAX_SAMPLES=""
QUICK_TEST="false"
ENABLE_THINKING="false"
DISABLE_THINKING="false"
KEEP_THINKING="false"
STRIP_THINKING="false"
MATH_PROMPT_CONFIG=""
INCLUDE_VALUES="false"
BINS="100"
PYTORCH_PYTHON="$DEFAULT_PYTHON"
EVAL_PYTHON=""
NEMO_SKILLS_DATA_DIR_ARG="${NEMO_SKILLS_DATA_DIR:-$DEFAULT_DATA_DIR}"
GOOGLE_RESEARCH_DIR=""
DRY_RUN="false"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --benchmarks) BENCHMARKS="$2"; shift 2 ;;
        --mode) MODE="$2"; shift 2 ;;
        --output-path|--out-dir) OUTPUT_PATH="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --served-model-name|--model-name) SERVED_MODEL_NAME="$2"; shift 2 ;;
        --lora-path) LORA_PATH="$2"; shift 2 ;;
        --block-size|--block-length) BLOCK_SIZE="$2"; shift 2 ;;
        --threshold) THRESHOLD="$2"; shift 2 ;;
        --temperature) TEMPERATURE="$2"; shift 2 ;;
        --top-p) TOP_P="$2"; shift 2 ;;
        --tokens) TOKENS="$2"; shift 2 ;;
        --context-length) CONTEXT_LENGTH="$2"; shift 2 ;;
        --dtype) DTYPE="$2"; shift 2 ;;
        --max-thinking-tokens) MAX_THINKING_TOKENS="$2"; shift 2 ;;
        --gpu-device|--gpu-devices) GPU_DEVICE="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --client-concurrency) CLIENT_CONCURRENCY="$2"; shift 2 ;;
        --num-chunks) NUM_CHUNKS="$2"; shift 2 ;;
        --gpu-memory-reserve-gb) GPU_MEMORY_RESERVE_GB="$2"; shift 2 ;;
        --port) PORT="$2"; PORT_USER_SET="true"; shift 2 ;;
        --max-samples) MAX_SAMPLES="$2"; shift 2 ;;
        --quick-test) QUICK_TEST="true"; shift ;;
        --enable-thinking) ENABLE_THINKING="true"; shift ;;
        --disable-thinking) DISABLE_THINKING="true"; shift ;;
        --keep-thinking) KEEP_THINKING="true"; shift ;;
        --strip-thinking) STRIP_THINKING="true"; shift ;;
        --math-prompt-config) MATH_PROMPT_CONFIG="$2"; shift 2 ;;
        --include-values) INCLUDE_VALUES="true"; shift ;;
        --no-values) INCLUDE_VALUES="false"; shift ;;
        --bins) BINS="$2"; shift 2 ;;
        --pytorch-python) PYTORCH_PYTHON="$2"; shift 2 ;;
        --eval-python) EVAL_PYTHON="$2"; shift 2 ;;
        --nemo-skills-data-dir) NEMO_SKILLS_DATA_DIR_ARG="$2"; shift 2 ;;
        --google-research-dir) GOOGLE_RESEARCH_DIR="$2"; shift 2 ;;
        --dry-run) DRY_RUN="true"; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "ERROR: unknown option: $1" >&2; usage; exit 1 ;;
    esac
done

[[ -n "$BENCHMARKS" ]] || { echo "ERROR: --benchmarks is required" >&2; usage; exit 1; }
case "$MODE" in
    linearspec_lora|linear_spec_lora) MODE="linearspec_lora" ;;
    linearspec_base|linear_spec_base|linearspec) MODE="linearspec_base" ;;
    *) echo "ERROR: --mode must be linearspec_lora or linearspec_base" >&2; exit 1 ;;
esac
[[ -n "$LORA_PATH" ]] || LORA_PATH="$MODEL/linear_spec_lora"
[[ "$MODE" == "linearspec_lora" ]] || LORA_PATH=""
[[ -n "$NUM_CHUNKS" ]] || NUM_CHUNKS="$CLIENT_CONCURRENCY"
[[ -n "$EVAL_PYTHON" ]] || EVAL_PYTHON="$PYTORCH_PYTHON"
[[ -n "$GOOGLE_RESEARCH_DIR" ]] || GOOGLE_RESEARCH_DIR="$NEMO_SKILLS_DATA_DIR_ARG/google-research"
[[ -n "$CONTEXT_LENGTH" ]] || CONTEXT_LENGTH=$((TOKENS + 2048))

is_positive_int() { [[ "${1:-}" =~ ^[0-9]+$ ]] && (( 10#$1 > 0 )); }
is_nonnegative_number() {
    "$PYTORCH_PYTHON" - "$1" <<'PY'
import math, sys
try:
    value = float(sys.argv[1])
except (TypeError, ValueError):
    raise SystemExit(1)
raise SystemExit(0 if math.isfinite(value) and value >= 0 else 1)
PY
}
port_is_free() {
    "$PYTORCH_PYTHON" - "$1" <<'PY'
import socket, sys
with socket.socket() as sock:
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind(("127.0.0.1", int(sys.argv[1])))
    except OSError:
        raise SystemExit(1)
PY
}
find_free_port() {
    "$PYTORCH_PYTHON" - "$1" <<'PY'
import socket, sys
for port in range(int(sys.argv[1]), 65535):
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            continue
        print(port)
        break
else:
    raise SystemExit("no free port")
PY
}

[[ "$GPU_DEVICE" =~ ^[0-9]+$ ]] || { echo "ERROR: exactly one numeric GPU ID is required" >&2; exit 1; }
[[ "$GPU_DEVICE" != *","* ]] || { echo "ERROR: multi-GPU is not supported" >&2; exit 1; }
[[ "$BATCH_SIZE" == "1" ]] || { echo "ERROR: native LinearSpec requires --batch-size 1" >&2; exit 1; }
is_positive_int "$TOKENS" || { echo "ERROR: --tokens must be positive" >&2; exit 1; }
is_positive_int "$CONTEXT_LENGTH" || { echo "ERROR: --context-length must be positive" >&2; exit 1; }
is_positive_int "$BLOCK_SIZE" || { echo "ERROR: --block-size must be positive" >&2; exit 1; }
is_positive_int "$CLIENT_CONCURRENCY" || { echo "ERROR: --client-concurrency must be positive" >&2; exit 1; }
is_positive_int "$NUM_CHUNKS" || { echo "ERROR: --num-chunks must be positive" >&2; exit 1; }
is_positive_int "$BINS" || { echo "ERROR: --bins must be positive" >&2; exit 1; }
[[ -z "$MAX_SAMPLES" ]] || is_positive_int "$MAX_SAMPLES" || { echo "ERROR: --max-samples must be positive" >&2; exit 1; }
[[ -z "$MAX_THINKING_TOKENS" ]] || is_positive_int "$MAX_THINKING_TOKENS" || { echo "ERROR: --max-thinking-tokens must be positive" >&2; exit 1; }
(( TOKENS <= CONTEXT_LENGTH )) || { echo "ERROR: --tokens cannot exceed --context-length" >&2; exit 1; }
is_nonnegative_number "$GPU_MEMORY_RESERVE_GB" || { echo "ERROR: --gpu-memory-reserve-gb must be non-negative" >&2; exit 1; }
"$PYTORCH_PYTHON" - "$THRESHOLD" "$TEMPERATURE" "$TOP_P" <<'PY'
import math, sys
try:
    threshold, temperature, top_p = map(float, sys.argv[1:])
except (TypeError, ValueError):
    raise SystemExit("threshold, temperature and top-p must be numeric")
if not all(math.isfinite(value) for value in (threshold, temperature, top_p)):
    raise SystemExit("threshold, temperature and top-p must be finite")
if threshold < 0 or temperature < 0 or not 0 <= top_p <= 1:
    raise SystemExit("require threshold>=0, temperature>=0 and 0<=top-p<=1")
PY
case "$DTYPE" in
    bfloat16|bf16|float16|fp16|float32|fp32) ;;
    *) echo "ERROR: unsupported --dtype $DTYPE" >&2; exit 1 ;;
esac
[[ "$ENABLE_THINKING" != "true" || "$DISABLE_THINKING" != "true" ]] || { echo "ERROR: thinking flags are mutually exclusive" >&2; exit 1; }
[[ "$KEEP_THINKING" != "true" || "$STRIP_THINKING" != "true" ]] || { echo "ERROR: keep/strip thinking are mutually exclusive" >&2; exit 1; }
if [[ "$MODE" == "linearspec_lora" && ! -f "$LORA_PATH/adapter_config.json" ]]; then
    echo "ERROR: LoRA adapter_config.json not found: $LORA_PATH" >&2
    exit 1
fi

if [[ -z "$PORT" ]]; then
    PORT="$(find_free_port "$((33000 + GPU_DEVICE))")"
else
    is_positive_int "$PORT" && (( PORT <= 65535 )) || { echo "ERROR: invalid --port" >&2; exit 1; }
    port_is_free "$PORT" || { echo "ERROR: requested port is busy: $PORT" >&2; exit 1; }
fi

OUTPUT_PATH="$(realpath -m "$OUTPUT_PATH")"
NEMO_SKILLS_DATA_DIR_ARG="$(realpath -m "$NEMO_SKILLS_DATA_DIR_ARG")"
GOOGLE_RESEARCH_DIR="$(realpath -m "$GOOGLE_RESEARCH_DIR")"
RUN_NAME="linearspec_confidence_$(date +%Y%m%d_%H%M%S)"
FINAL_DIR="$OUTPUT_PATH/$RUN_NAME"
if [[ -e "$FINAL_DIR" ]]; then
    suffix=1
    while [[ -e "$OUTPUT_PATH/${RUN_NAME}_$(printf '%02d' "$suffix")" ]]; do suffix=$((suffix + 1)); done
    FINAL_DIR="$OUTPUT_PATH/${RUN_NAME}_$(printf '%02d' "$suffix")"
fi
TRACE_DIR="$FINAL_DIR/traces"
SUMMARY_DIR="$FINAL_DIR/summaries"
EVAL_RUNS_DIR="$FINAL_DIR/eval_runs"
RUNTIME_DIR="$FINAL_DIR/runtime"

echo "================================================================"
echo " Native PyTorch LinearSpec confidence/rank diagnostic"
echo "================================================================"
echo " Mode:                $MODE"
echo " Benchmarks:          $BENCHMARKS"
echo " GPU / reserve GiB:   $GPU_DEVICE / $GPU_MEMORY_RESERVE_GB"
echo " Block / threshold:   $BLOCK_SIZE / $THRESHOLD"
echo " Tokens / context:    $TOKENS / $CONTEXT_LENGTH"
echo " Client / batch:      $CLIENT_CONCURRENCY / $BATCH_SIZE"
echo " Port:                $PORT"
echo " Output:              $FINAL_DIR"
echo "================================================================"
if [[ "$DRY_RUN" == "true" ]]; then
    echo "[dry-run] No files, processes, datasets, or models were created."
    exit 0
fi

mkdir -p "$TRACE_DIR" "$SUMMARY_DIR" "$EVAL_RUNS_DIR" "$RUNTIME_DIR" "$NEMO_SKILLS_DATA_DIR_ARG"
if [[ "$EVAL_PYTHON" == */* ]]; then
    EVAL_PYTHON_DIR="$(cd "$(dirname "$EVAL_PYTHON")" && pwd)"
else
    EVAL_PYTHON_DIR="$(dirname "$(command -v "$EVAL_PYTHON")")"
fi
export PATH="$EVAL_PYTHON_DIR:$PATH"
export NEMO_SKILLS_DATA_DIR="$NEMO_SKILLS_DATA_DIR_ARG"
export NLD_GOOGLE_RESEARCH_DIR="$GOOGLE_RESEARCH_DIR"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$NEMO_SKILLS_DATA_DIR_ARG/hf_datasets_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$NEMO_SKILLS_DATA_DIR_ARG/xdg_cache}"
mkdir -p "$HF_DATASETS_CACHE" "$XDG_CACHE_HOME"

SETTINGS_MODE="$MODE" SETTINGS_BENCHMARKS="$BENCHMARKS" SETTINGS_OUTPUT="$FINAL_DIR" SETTINGS_MODEL="$MODEL" SETTINGS_MODEL_TAG="$SERVED_MODEL_NAME" SETTINGS_LORA="$LORA_PATH" SETTINGS_GPU="$GPU_DEVICE" SETTINGS_RESERVE="$GPU_MEMORY_RESERVE_GB" SETTINGS_BLOCK="$BLOCK_SIZE" SETTINGS_THRESHOLD="$THRESHOLD" SETTINGS_TEMPERATURE="$TEMPERATURE" SETTINGS_TOP_P="$TOP_P" SETTINGS_TOKENS="$TOKENS" SETTINGS_CONTEXT="$CONTEXT_LENGTH" SETTINGS_DTYPE="$DTYPE" SETTINGS_BATCH="$BATCH_SIZE" SETTINGS_CLIENT="$CLIENT_CONCURRENCY" SETTINGS_CHUNKS="$NUM_CHUNKS" SETTINGS_PORT="$PORT" SETTINGS_MAX_SAMPLES="$MAX_SAMPLES" SETTINGS_QUICK="$QUICK_TEST" SETTINGS_ENABLE_THINKING="$ENABLE_THINKING" SETTINGS_DISABLE_THINKING="$DISABLE_THINKING" SETTINGS_KEEP_THINKING="$KEEP_THINKING" SETTINGS_STRIP_THINKING="$STRIP_THINKING" SETTINGS_MAX_THINKING="$MAX_THINKING_TOKENS" SETTINGS_MATH_PROMPT="$MATH_PROMPT_CONFIG" SETTINGS_INCLUDE_VALUES="$INCLUDE_VALUES" SETTINGS_BINS="$BINS" SETTINGS_PYTORCH_PYTHON="$PYTORCH_PYTHON" SETTINGS_EVAL_PYTHON="$EVAL_PYTHON" SETTINGS_DATA_DIR="$NEMO_SKILLS_DATA_DIR_ARG" SETTINGS_GOOGLE_RESEARCH="$GOOGLE_RESEARCH_DIR" \
"$PYTORCH_PYTHON" - "$FINAL_DIR/Settings.json" "${ORIGINAL_ARGS[@]}" <<'PY'
import json, os, shlex, sys
from datetime import datetime
def value(name): return os.environ.get("SETTINGS_" + name, "")
def integer(name): return int(value(name)) if value(name) else None
def number(name): return float(value(name)) if value(name) else None
def boolean(name): return value(name).lower() in {"true", "1", "yes"}
payload = {
    "created_at": datetime.now().astimezone().isoformat(),
    "entrypoint": "observations/eval_pytorch_linearspec_confidence.sh",
    "argv": sys.argv[2:],
    "command": "bash observations/eval_pytorch_linearspec_confidence.sh " + " ".join(shlex.quote(arg) for arg in sys.argv[2:]),
    "backend": "native_pytorch",
    "mode": value("MODE"), "benchmarks": value("BENCHMARKS"),
    "output_dir": value("OUTPUT"), "model": value("MODEL"),
    "served_model_name": value("MODEL_TAG"), "lora_path": value("LORA"),
    "gpu_device": integer("GPU"), "gpu_memory_reserve_gb": number("RESERVE"),
    "block_size": integer("BLOCK"), "threshold": number("THRESHOLD"),
    "temperature": number("TEMPERATURE"), "top_p": number("TOP_P"),
    "tokens": integer("TOKENS"), "context_length": integer("CONTEXT"),
    "dtype": value("DTYPE"), "batch_size": integer("BATCH"),
    "client_concurrency": integer("CLIENT"), "num_chunks": integer("CHUNKS"),
    "port": integer("PORT"), "max_samples": integer("MAX_SAMPLES"),
    "quick_test": boolean("QUICK"), "enable_thinking": boolean("ENABLE_THINKING"),
    "disable_thinking": boolean("DISABLE_THINKING"), "keep_thinking": boolean("KEEP_THINKING"),
    "strip_thinking": boolean("STRIP_THINKING"), "max_thinking_tokens": integer("MAX_THINKING"),
    "math_prompt_config": value("MATH_PROMPT"), "include_values": boolean("INCLUDE_VALUES"),
    "bins": integer("BINS"), "pytorch_python": value("PYTORCH_PYTHON"),
    "eval_python": value("EVAL_PYTHON"), "nemo_skills_data_dir": value("DATA_DIR"),
    "google_research_dir": value("GOOGLE_RESEARCH"),
}
with open(sys.argv[1], "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2, ensure_ascii=False, sort_keys=True)
    f.write("\n")
PY

if ! "$PYTORCH_PYTHON" -c "import torch, transformers, peft, fastapi, uvicorn" >/dev/null 2>&1; then
    echo "ERROR: PyTorch Python lacks torch/transformers/peft/FastAPI/Uvicorn" >&2
    exit 1
fi
if ! "$EVAL_PYTHON" -c "from nemo_skills.pipeline.eval import eval" >/dev/null 2>&1; then
    echo "ERROR: eval Python cannot import NeMo-Skills" >&2
    exit 1
fi

NEMO_PACKAGE_DATASET_DIR="$("$EVAL_PYTHON" - <<'PY'
from pathlib import Path
import nemo_skills.dataset
print(Path(nemo_skills.dataset.__file__).resolve().parent)
PY
)"
dataset_has_jsonl() { [[ -d "$1" ]] && find "$1" -maxdepth 1 -type f -name '*.jsonl' -print -quit | grep -q .; }
benchmark_requested() {
    local wanted="$1" spec
    IFS=',' read -ra _CHECK <<< "$BENCHMARKS"
    for spec in "${_CHECK[@]}"; do
        spec="${spec//[[:space:]]/}"
        [[ "${spec%%:*}" == "$wanted" ]] && return 0
    done
    return 1
}
if benchmark_requested ifeval; then
    [[ -d "$GOOGLE_RESEARCH_DIR/instruction_following_eval" ]] || { echo "ERROR: IFEval scorer not found: $GOOGLE_RESEARCH_DIR" >&2; exit 1; }
    export PYTHONPATH="$GOOGLE_RESEARCH_DIR:${PYTHONPATH:-}"
    "$EVAL_PYTHON" -c "import instruction_following_eval.evaluation_main, langdetect, immutabledict, nltk" >/dev/null
fi
IFS=',' read -ra _PREP <<< "$BENCHMARKS"
for spec in "${_PREP[@]}"; do
    spec="${spec//[[:space:]]/}"; [[ -z "$spec" ]] && continue
    name="${spec%%:*}"
    package_dir="$NEMO_PACKAGE_DATASET_DIR/$name"
    cached_dir="$NEMO_SKILLS_DATA_DIR_ARG/$name"
    if dataset_has_jsonl "$package_dir"; then continue; fi
    if dataset_has_jsonl "$cached_dir"; then mkdir -p "$package_dir"; cp -a "$cached_dir"/. "$package_dir"/; continue; fi
    echo "Preparing NeMo-Skills dataset: $name"
    if command -v flock >/dev/null; then
        flock "$NEMO_SKILLS_DATA_DIR_ARG/.prepare.lock" "$EVAL_PYTHON" -m nemo_skills.dataset.prepare "$name" --parallelism 20 --retries 3 || true
    else
        "$EVAL_PYTHON" -m nemo_skills.dataset.prepare "$name" --parallelism 20 --retries 3 || true
    fi
    if dataset_has_jsonl "$package_dir"; then mkdir -p "$cached_dir"; cp -a "$package_dir"/. "$cached_dir"/; fi
done

SERVER_PID=""
RESERVER_PID=""
stop_server() {
    if [[ -n "$SERVER_PID" ]]; then kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true; SERVER_PID=""; fi
}
cleanup() {
    stop_server
    if [[ -n "$RESERVER_PID" ]]; then kill "$RESERVER_PID" 2>/dev/null || true; wait "$RESERVER_PID" 2>/dev/null || true; RESERVER_PID=""; fi
}
trap cleanup EXIT INT TERM
wait_for_health() {
    local pid="$1" log="$2"
    for _ in $(seq 1 300); do
        kill -0 "$pid" 2>/dev/null || { echo "ERROR: server exited during startup" >&2; tail -100 "$log" >&2 || true; return 1; }
        curl -sf --max-time 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && return 0
        sleep 2
    done
    echo "ERROR: server health timeout" >&2; tail -100 "$log" >&2 || true; return 1
}
if "$PYTORCH_PYTHON" - "$GPU_MEMORY_RESERVE_GB" <<'PY'
import sys
raise SystemExit(0 if float(sys.argv[1]) > 0 else 1)
PY
then
    ready="$RUNTIME_DIR/gpu_memory_reserver_ready.json"
    (
        export CUDA_VISIBLE_DEVICES="$GPU_DEVICE"
        exec "$PYTORCH_PYTHON" -u "$MEMORY_RESERVER" --gb "$GPU_MEMORY_RESERVE_GB" --ready-file "$ready"
    ) > "$RUNTIME_DIR/gpu_memory_reserver.log" 2>&1 &
    RESERVER_PID=$!
    for _ in $(seq 1 120); do
        kill -0 "$RESERVER_PID" 2>/dev/null || { tail -80 "$RUNTIME_DIR/gpu_memory_reserver.log" >&2; exit 1; }
        [[ -s "$ready" ]] && break
        sleep 1
    done
    [[ -s "$ready" ]] || { echo "ERROR: GPU memory reserver did not become ready" >&2; exit 1; }
fi

safe_name() { local value="${1%%:*}"; value="${value//\//_}"; value="${value//:/_}"; value="${value//[[:space:]]/_}"; echo "$value"; }
write_error() {
    local bench="$1" stage="$2" code="$3" message="$4" log="$5"
    ERROR_BENCH="$bench" ERROR_STAGE="$stage" ERROR_CODE="$code" ERROR_MESSAGE="$message" ERROR_LOG="$log" "$EVAL_PYTHON" - "$FINAL_DIR/error_$(safe_name "$bench").json" <<'PY'
import json, os, sys
from datetime import datetime
from pathlib import Path
log = Path(os.environ["ERROR_LOG"])
tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-80:] if log.is_file() else []
payload = {"status":"failed", "created_at":datetime.now().astimezone().isoformat(), "benchmark":os.environ["ERROR_BENCH"], "stage":os.environ["ERROR_STAGE"], "exit_code":int(os.environ["ERROR_CODE"]), "message":os.environ["ERROR_MESSAGE"], "log_tail":tail}
Path(sys.argv[1]).write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
PY
}
append_status() {
    local spec="$1" name="$2" status="$3" eval_status="$4" merge_status="$5" summary_status="$6" trace="$7" summary="$8" metrics="$9"
    STATUS_SPEC="$spec" STATUS_NAME="$name" STATUS_VALUE="$status" STATUS_EVAL="$eval_status" STATUS_MERGE="$merge_status" STATUS_SUMMARY="$summary_status" STATUS_TRACE="$trace" STATUS_SUMMARY_FILE="$summary" STATUS_METRICS="$metrics" "$EVAL_PYTHON" - "$FINAL_DIR/benchmark_status.jsonl" <<'PY'
import json, os, sys
from datetime import datetime
from pathlib import Path
trace = Path(os.environ["STATUS_TRACE"])
records = sum(1 for line in trace.open(encoding="utf-8")) if trace.is_file() else 0
payload = {"created_at":datetime.now().astimezone().isoformat(), "benchmark_spec":os.environ["STATUS_SPEC"], "benchmark":os.environ["STATUS_NAME"], "status":os.environ["STATUS_VALUE"], "eval_exit_code":int(os.environ["STATUS_EVAL"]), "merge_exit_code":int(os.environ["STATUS_MERGE"]), "summary_exit_code":int(os.environ["STATUS_SUMMARY"]), "trace_file":os.environ["STATUS_TRACE"], "trace_records":records, "summary_file":os.environ["STATUS_SUMMARY_FILE"], "metrics_file":os.environ["STATUS_METRICS"]}
with Path(sys.argv[1]).open("a", encoding="utf-8") as f: f.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
PY
}

ANY_FAILURE=0
IFS=',' read -ra BENCH_ARRAY <<< "$BENCHMARKS"
for bench_spec in "${BENCH_ARRAY[@]}"; do
    bench_spec="${bench_spec//[[:space:]]/}"; [[ -z "$bench_spec" ]] && continue
    bench_name="${bench_spec%%:*}"
    safe="$(safe_name "$bench_spec")"
    trace_file="$TRACE_DIR/raw_trace_${safe}.jsonl"
    summary_file="$SUMMARY_DIR/confidence_distribution_${safe}.json"
    eval_parent="$EVAL_RUNS_DIR/$safe"
    eval_dir="$eval_parent/eval-results/$bench_name"
    bench_runtime="$RUNTIME_DIR/$safe"
    stats_file="$bench_runtime/pytorch_request_stats.jsonl"
    server_log="$bench_runtime/pytorch_confidence_server.log"
    bench_log="$bench_runtime/nemo_skills_benchmark.log"
    mkdir -p "$eval_parent" "$bench_runtime"
    : > "$trace_file"; : > "$stats_file"; : > "$server_log"; : > "$bench_log"

    server_args=("$SERVER_SCRIPT" --model-path "$MODEL" --served-model-name "$SERVED_MODEL_NAME" --mode "$MODE" --dtype "$DTYPE" --block-length "$BLOCK_SIZE" --threshold "$THRESHOLD" --default-max-new-tokens "$TOKENS" --context-length "$CONTEXT_LENGTH" --stats-file "$stats_file" --confidence-trace-file "$trace_file" --host 0.0.0.0 --port "$PORT")
    [[ -n "$LORA_PATH" ]] && server_args+=(--lora-path "$LORA_PATH")
    [[ "$ENABLE_THINKING" == "true" ]] && server_args+=(--enable-thinking)
    [[ -n "$MAX_THINKING_TOKENS" ]] && server_args+=(--max-thinking-tokens "$MAX_THINKING_TOKENS")
    echo "--- benchmark: $bench_spec ---"
    (
        export CUDA_VISIBLE_DEVICES="$GPU_DEVICE"
        exec "$PYTORCH_PYTHON" -u "${server_args[@]}"
    ) > "$server_log" 2>&1 &
    SERVER_PID=$!
    if ! wait_for_health "$SERVER_PID" "$server_log"; then
        stop_server; write_error "$bench_name" server_start 1 "Native confidence server failed to start" "$server_log"; append_status "$bench_spec" "$bench_name" failed 1 1 1 "$trace_file" "$summary_file" ""; ANY_FAILURE=1; continue
    fi

    eval_args=("$EVAL_SCRIPT" --server-address "http://127.0.0.1:$PORT/v1" --benchmark "$bench_spec" --output-dir "$eval_parent" --expname pytorch_confidence --model "$SERVED_MODEL_NAME" --tokens-to-generate "$TOKENS" --temperature "$TEMPERATURE" --top-p "$TOP_P" --num-chunks "$NUM_CHUNKS" --max-concurrent-requests "$CLIENT_CONCURRENCY" --generation-algorithm nemotron --no-extra-body)
    [[ -n "$MAX_SAMPLES" ]] && eval_args+=(--max-samples "$MAX_SAMPLES")
    [[ "$QUICK_TEST" == "true" ]] && eval_args+=(--quick-test)
    [[ "$KEEP_THINKING" == "true" ]] && eval_args+=(--keep-thinking)
    [[ "$STRIP_THINKING" == "true" ]] && eval_args+=(--strip-thinking)
    [[ "$DISABLE_THINKING" == "true" ]] && eval_args+=(--disable-thinking)
    [[ -n "$MATH_PROMPT_CONFIG" ]] && eval_args+=(--math-prompt-config "$MATH_PROMPT_CONFIG")
    start_time="$(date +%s.%N)"
    set +e
    "$EVAL_PYTHON" "${eval_args[@]}" 2>&1 | tee "$bench_log"
    eval_status=${PIPESTATUS[0]}
    set -e
    end_time="$(date +%s.%N)"
    wall_time="$("$EVAL_PYTHON" - "$start_time" "$end_time" <<'PY'
import sys
print(f"{float(sys.argv[2]) - float(sys.argv[1]):.6f}")
PY
)"
    merge_status=1
    metrics_file=""
    if [[ "$eval_status" == "0" && -f "$eval_dir/metrics.json" ]]; then
        cp "$stats_file" "$eval_dir/pytorch_request_stats.jsonl"
        set +e
        "$EVAL_PYTHON" "$METRICS_MERGER" --metrics-json "$eval_dir/metrics.json" --request-stats-file "$stats_file" --benchmark "$bench_name" --wall-time-s "$wall_time" 2>&1 | tee -a "$bench_log"
        merge_status=${PIPESTATUS[0]}
        set -e
        if [[ "$merge_status" == "0" ]]; then metrics_file="$FINAL_DIR/metrics_${safe}.json"; cp "$eval_dir/metrics.json" "$metrics_file"; fi
    fi
    stop_server

    summary_args=("$SUMMARY_SCRIPT" --trace-file "$trace_file" --output-json "$summary_file" --benchmark "$bench_name" --benchmark-spec "$bench_spec" --bins "$BINS")
    [[ "$INCLUDE_VALUES" == "true" ]] && summary_args+=(--include-values)
    set +e
    "$EVAL_PYTHON" "${summary_args[@]}" 2>&1 | tee -a "$bench_log"
    summary_status=${PIPESTATUS[0]}
    set -e
    if [[ "$eval_status" != "0" ]]; then
        write_error "$bench_name" nemo_eval "$eval_status" "NeMo-Skills evaluation failed" "$bench_log"; ANY_FAILURE=1
    elif [[ ! -f "$eval_dir/metrics.json" ]]; then
        write_error "$bench_name" metrics_missing 1 "NeMo-Skills did not produce metrics.json" "$bench_log"; ANY_FAILURE=1
    elif [[ "$merge_status" != "0" ]]; then
        write_error "$bench_name" metrics_merge "$merge_status" "PyTorch metrics merge failed" "$bench_log"; ANY_FAILURE=1
    elif [[ "$summary_status" != "0" ]]; then
        write_error "$bench_name" trace_summary "$summary_status" "Confidence trace summary validation failed" "$bench_log"; ANY_FAILURE=1
    fi
    if [[ "$eval_status" == "0" && "$merge_status" == "0" && "$summary_status" == "0" ]]; then bench_status=completed; else bench_status=failed; fi
    append_status "$bench_spec" "$bench_name" "$bench_status" "$eval_status" "$merge_status" "$summary_status" "$trace_file" "$summary_file" "$metrics_file"
done

cleanup
trap - EXIT INT TERM
echo "Completed native PyTorch LinearSpec confidence/rank diagnostics."
echo "Output: $FINAL_DIR"
if [[ "$ANY_FAILURE" != "0" ]]; then
    echo "One or more benchmarks failed; inspect benchmark_status.jsonl and error_*.json." >&2
    exit 1
fi
