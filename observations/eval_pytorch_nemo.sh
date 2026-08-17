#!/bin/bash
# Native PyTorch/Hugging Face inference with NeMo-Skills benchmark organization.

set -euo pipefail

ORIGINAL_ARGS=("$@")
OBSERVATIONS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$OBSERVATIONS_DIR/.." && pwd)"
OBSERVATION_RESULTS_ROOT="${NLD_OBSERVATION_RESULTS_ROOT:-/data/home/wly/dLLM/NLD_results/observations}"
PIPELINE="${PYTORCH_NEMO_PIPELINE_OVERRIDE:-$PROJECT_DIR/xp/examples/run_pytorch_nemo_eval_pipeline_gpu_only.sh}"

DEFAULT_MODEL="/data1/linyewei/models/Nemotron-Labs-Diffusion-8B"
DEFAULT_OUTPUT_PATH="$OBSERVATION_RESULTS_ROOT/pytorch_nemo_eval_results"
DEFAULT_DATA_DIR="/data1/linyewei/datasets/NLD"
DEFAULT_BENCHMARKS="gsm8k:1,human-eval:1,mbpp:1,math-500:1,aime24:1,aime25:1,gpqa:1,mmlu:1,ifeval:1,livecodebench-cpp:1"
DEFAULT_PYTHON="/data/home/wly/.conda/envs/nld_sglang/bin/python"
[[ -x "$DEFAULT_PYTHON" ]] || DEFAULT_PYTHON="python"

usage() {
    cat <<EOF
Usage: $0 --mode MODE [options]

Modes:
  --mode ar                  Native model.ar_generate()
  --mode dlm                 Native block-diffusion model.generate()
  --mode linearspec_base     Native Linear SS without LoRA
  --mode linearspec_lora     Native Linear SS with draft-only LoRA

Benchmark:
  --benchmarks LIST          Comma-separated benchmarks; mt-bench/alpaca-eval use dedicated runners
  --tokens N                 Maximum returned completion tokens (default: 8192)
  --temperature V            Native sampling temperature (default: 0)
  --top-p V                  OpenAI/NeMo parameter recorded for parity; native model methods do not apply top-p (default: 0.95)
  --num-chunks N             NeMo-Skills client-side chunk count (default: client concurrency)
  --client-concurrency N     Concurrent client requests; GPU generation remains serialized (default: 1)
  --max-samples N            Limit the number of benchmark problems
  --quick-test               NeMo-Skills quick test mode
  --keep-thinking            Keep thinking text in NeMo-Skills output
  --strip-thinking           Strip thinking text and re-score where supported
  --disable-thinking         Explicitly keep chat-template thinking disabled (default behavior)
  --enable-thinking          Enable thinking in the model chat template
  --math-prompt-config NAME  NeMo-Skills prompt_config override for math tasks

Judge-based benchmarks (Arena-Hard, MT-Bench, and AlpacaEval):
  --judge-model NAME         Override the benchmark's default judge
  --judge-server-address URL Override the default https://api.openai.com/v1 judge endpoint
  --judge-server-type TYPE   Judge server type (currently openai-compatible)
  --judge-concurrency N      Concurrent dedicated-runner judge requests (default: 4)
  --mt-bench-max-tokens N    Completion budget per MT-Bench turn (default: 1024)
  --alpaca-eval-max-tokens N Candidate completion budget for AlpacaEval (default: 2048)
  --skip-judge-api-key-check Skip the OPENAI_API_KEY preflight check for downstream-injected credentials

Native PyTorch decoding:
  --model PATH               Local/HF model (default: $DEFAULT_MODEL)
  --served-model-name NAME   OpenAI API model label
  --gpu-device ID            One physical GPU ID (default: 0)
  --gpu-devices ID           Alias for --gpu-device; lists are rejected because this backend is single-GPU
  --gpu-memory-reserve-gb V  Reserve V GiB on the selected GPU before model load (default: 0)
  --dtype DTYPE              bfloat16, float16, or float32 (default: bfloat16)
  --block-length N           dLM/Linear SS block length (mode default: 8/32)
  --block-size N             Alias for --block-length
  --threshold V              dLM/Linear draft confidence threshold (mode default: 0.9/0.0)
  --causal-context           Enable clean-prefix causal KV context for dLM (default)
  --no-causal-context        Disable it
  --context-length N         Reject requests exceeding this prompt+completion length (default: tokens+2048)
  --lora-path DIR            LinearSpec LoRA (default: <model>/linear_spec_lora)
  --max-thinking-tokens N    Force </think> after this generated-token budget

Runtime/output:
  --port N                   Local PyTorch OpenAI server port (default: 32000; auto-pick if busy)
  --output-path DIR          Compact result root (default: $DEFAULT_OUTPUT_PATH)
  --pytorch-python PATH      Python used by model server
  --eval-python PATH         Python used by NeMo-Skills (default: PyTorch Python)
  --nemo-skills-data-dir DIR Persistent prepared dataset/cache root (default: $DEFAULT_DATA_DIR)
  --google-research-dir DIR  google-research checkout used by IFEval
  --keep-runtime             Preserve per-run server logs, raw outputs, and request stats
  --dry-run                  Resolve and print settings without loading a model or writing results
  -h, --help                 Show this help

Example:
  bash $0 --mode linearspec_lora --benchmarks gsm8k:1 --gpu-device 2 --gpu-memory-reserve-gb 30 --block-length 16 --max-samples 5
EOF
}

MODE=""
BENCHMARKS="$DEFAULT_BENCHMARKS"
MODEL="$DEFAULT_MODEL"
SERVED_MODEL_NAME="nemotron-labs-diffusion-8b"
TOKENS="8192"
TEMPERATURE="0"
TOP_P="0.95"
NUM_CHUNKS=""
CLIENT_CONCURRENCY="1"
MAX_SAMPLES=""
QUICK_TEST="false"
KEEP_THINKING="false"
STRIP_THINKING="false"
DISABLE_THINKING="false"
ENABLE_THINKING="false"
MATH_PROMPT_CONFIG=""
JUDGE_MODEL=""
JUDGE_SERVER_ADDRESS=""
JUDGE_SERVER_TYPE=""
JUDGE_CONCURRENCY="4"
MT_BENCH_MAX_TOKENS="1024"
ALPACA_EVAL_MAX_TOKENS="2048"
SKIP_JUDGE_API_KEY_CHECK="false"

GPU_DEVICE="0"
GPU_MEMORY_RESERVE_GB="0"
DTYPE="bfloat16"
BLOCK_LENGTH=""
THRESHOLD=""
CAUSAL_CONTEXT="true"
CONTEXT_LENGTH="2048"
CONTEXT_LENGTH_USER_SET="false"
LORA_PATH=""
MAX_THINKING_TOKENS=""
PORT="32000"
PORT_USER_SET="false"

OUTPUT_PATH="${OUTPUT_PATH:-$DEFAULT_OUTPUT_PATH}"
PYTORCH_PYTHON="$DEFAULT_PYTHON"
EVAL_PYTHON=""
NEMO_SKILLS_DATA_DIR_ARG="${NEMO_SKILLS_DATA_DIR:-$DEFAULT_DATA_DIR}"
GOOGLE_RESEARCH_DIR=""
KEEP_RUNTIME="false"
DRY_RUN="false"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode) MODE="$2"; shift 2 ;;
        --benchmarks) BENCHMARKS="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --served-model-name|--model-name) SERVED_MODEL_NAME="$2"; shift 2 ;;
        --tokens) TOKENS="$2"; shift 2 ;;
        --temperature) TEMPERATURE="$2"; shift 2 ;;
        --top-p) TOP_P="$2"; shift 2 ;;
        --num-chunks) NUM_CHUNKS="$2"; shift 2 ;;
        --client-concurrency) CLIENT_CONCURRENCY="$2"; shift 2 ;;
        --max-samples) MAX_SAMPLES="$2"; shift 2 ;;
        --quick-test) QUICK_TEST="true"; shift ;;
        --keep-thinking) KEEP_THINKING="true"; shift ;;
        --strip-thinking) STRIP_THINKING="true"; shift ;;
        --disable-thinking) DISABLE_THINKING="true"; shift ;;
        --enable-thinking) ENABLE_THINKING="true"; shift ;;
        --math-prompt-config) MATH_PROMPT_CONFIG="$2"; shift 2 ;;
        --judge-model) JUDGE_MODEL="$2"; shift 2 ;;
        --judge-server-address) JUDGE_SERVER_ADDRESS="$2"; shift 2 ;;
        --judge-server-type) JUDGE_SERVER_TYPE="$2"; shift 2 ;;
        --judge-concurrency) JUDGE_CONCURRENCY="$2"; shift 2 ;;
        --mt-bench-max-tokens) MT_BENCH_MAX_TOKENS="$2"; shift 2 ;;
        --alpaca-eval-max-tokens) ALPACA_EVAL_MAX_TOKENS="$2"; shift 2 ;;
        --skip-judge-api-key-check) SKIP_JUDGE_API_KEY_CHECK="true"; shift ;;
        --gpu-device|--gpu-devices) GPU_DEVICE="$2"; shift 2 ;;
        --gpu-memory-reserve-gb) GPU_MEMORY_RESERVE_GB="$2"; shift 2 ;;
        --dtype) DTYPE="$2"; shift 2 ;;
        --block-length|--block-size) BLOCK_LENGTH="$2"; shift 2 ;;
        --threshold) THRESHOLD="$2"; shift 2 ;;
        --causal-context) CAUSAL_CONTEXT="true"; shift ;;
        --no-causal-context) CAUSAL_CONTEXT="false"; shift ;;
        --context-length) CONTEXT_LENGTH="$2"; CONTEXT_LENGTH_USER_SET="true"; shift 2 ;;
        --lora-path) LORA_PATH="$2"; shift 2 ;;
        --max-thinking-tokens) MAX_THINKING_TOKENS="$2"; shift 2 ;;
        --port) PORT="$2"; PORT_USER_SET="true"; shift 2 ;;
        --output-path|--out-dir) OUTPUT_PATH="$2"; shift 2 ;;
        --pytorch-python) PYTORCH_PYTHON="$2"; shift 2 ;;
        --eval-python) EVAL_PYTHON="$2"; shift 2 ;;
        --nemo-skills-data-dir) NEMO_SKILLS_DATA_DIR_ARG="$2"; shift 2 ;;
        --google-research-dir) GOOGLE_RESEARCH_DIR="$2"; shift 2 ;;
        --keep-runtime) KEEP_RUNTIME="true"; shift ;;
        --dry-run) DRY_RUN="true"; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "ERROR: unknown option: $1" >&2; usage; exit 1 ;;
    esac
done

[[ -n "$MODE" ]] || { echo "ERROR: --mode is required" >&2; usage; exit 1; }
case "$MODE" in
    ar) MODE="ar" ;;
    dlm|fastdiffuser) MODE="dlm" ;;
    linearspec|linearspec_base|linear_spec|linear_spec_base) MODE="linearspec_base" ;;
    linearspec_lora|linear_spec_lora) MODE="linearspec_lora" ;;
    *) echo "ERROR: unknown --mode $MODE" >&2; exit 1 ;;
esac

if [[ -z "$BLOCK_LENGTH" ]]; then
    case "$MODE" in
        ar) BLOCK_LENGTH="1" ;;
        dlm) BLOCK_LENGTH="8" ;;
        linearspec_base|linearspec_lora) BLOCK_LENGTH="32" ;;
    esac
fi
if [[ -z "$THRESHOLD" ]]; then
    case "$MODE" in
        ar) THRESHOLD="" ;;
        dlm) THRESHOLD="0.9" ;;
        linearspec_base|linearspec_lora) THRESHOLD="0.0" ;;
    esac
elif [[ "${THRESHOLD,,}" == "none" || "${THRESHOLD,,}" == "null" ]]; then
    THRESHOLD=""
fi
if [[ -z "$LORA_PATH" ]]; then
    LORA_PATH="$MODEL/linear_spec_lora"
fi
if [[ "$MODE" != "linearspec_lora" ]]; then
    LORA_PATH=""
fi
[[ -n "$NUM_CHUNKS" ]] || NUM_CHUNKS="$CLIENT_CONCURRENCY"
[[ -n "$EVAL_PYTHON" ]] || EVAL_PYTHON="$PYTORCH_PYTHON"
[[ -n "$GOOGLE_RESEARCH_DIR" ]] || GOOGLE_RESEARCH_DIR="$NEMO_SKILLS_DATA_DIR_ARG/google-research"

if [[ "$GPU_DEVICE" == *","* || "$GPU_DEVICE" =~ [[:space:]] ]]; then
    echo "ERROR: native PyTorch backend supports exactly one --gpu-device, got: $GPU_DEVICE" >&2
    exit 1
fi
if [[ "$KEEP_THINKING" == "true" && "$STRIP_THINKING" == "true" ]]; then
    echo "ERROR: --keep-thinking and --strip-thinking are mutually exclusive" >&2
    exit 1
fi
if [[ "$ENABLE_THINKING" == "true" && "$DISABLE_THINKING" == "true" ]]; then
    echo "ERROR: --enable-thinking and --disable-thinking are mutually exclusive" >&2
    exit 1
fi

is_positive_int() {
    [[ "${1:-}" =~ ^[0-9]+$ ]] && (( 10#$1 > 0 ))
}
arena_hard_requested() {
    local spec name
    local -a specs
    IFS=',' read -ra specs <<< "$BENCHMARKS"
    for spec in "${specs[@]}"; do
        spec="${spec//[[:space:]]/}"
        name="${spec%%:*}"
        [[ "$name" == "arena-hard" || "$name" == "arena-hard-v2" ]] && return 0
    done
    return 1
}
mt_bench_requested() {
    local spec name
    local -a specs
    IFS=',' read -ra specs <<< "$BENCHMARKS"
    for spec in "${specs[@]}"; do
        spec="${spec//[[:space:]]/}"
        name="${spec%%:*}"
        [[ "$name" == "mt-bench" ]] && return 0
    done
    return 1
}
alpaca_eval_requested() {
    local spec name
    local -a specs
    IFS=',' read -ra specs <<< "$BENCHMARKS"
    for spec in "${specs[@]}"; do
        spec="${spec//[[:space:]]/}"
        name="${spec%%:*}"
        [[ "$name" == "alpaca-eval" ]] && return 0
    done
    return 1
}
standard_benchmark_requested() {
    local spec name
    local -a specs
    IFS=',' read -ra specs <<< "$BENCHMARKS"
    for spec in "${specs[@]}"; do
        spec="${spec//[[:space:]]/}"
        name="${spec%%:*}"
        [[ -z "$name" ]] && continue
        [[ "$name" != "mt-bench" && "$name" != "alpaca-eval" ]] && return 0
    done
    return 1
}
judge_benchmark_requested() {
    arena_hard_requested || mt_bench_requested || alpaca_eval_requested
}
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

is_positive_int "$TOKENS" || { echo "ERROR: --tokens must be a positive integer" >&2; exit 1; }
is_positive_int "$BLOCK_LENGTH" || { echo "ERROR: --block-length must be a positive integer" >&2; exit 1; }
is_positive_int "$CLIENT_CONCURRENCY" || { echo "ERROR: --client-concurrency must be positive" >&2; exit 1; }
is_positive_int "$NUM_CHUNKS" || { echo "ERROR: --num-chunks must be positive" >&2; exit 1; }
is_positive_int "$JUDGE_CONCURRENCY" || { echo "ERROR: --judge-concurrency must be positive" >&2; exit 1; }
is_positive_int "$MT_BENCH_MAX_TOKENS" || { echo "ERROR: --mt-bench-max-tokens must be positive" >&2; exit 1; }
is_positive_int "$ALPACA_EVAL_MAX_TOKENS" || { echo "ERROR: --alpaca-eval-max-tokens must be positive" >&2; exit 1; }
if { mt_bench_requested || alpaca_eval_requested; } && [[ -n "$JUDGE_SERVER_TYPE" && "$JUDGE_SERVER_TYPE" != "openai" ]]; then
    echo "ERROR: MT-Bench and AlpacaEval currently support only an OpenAI-compatible judge endpoint." >&2
    exit 1
fi
is_positive_int "$PORT" || { echo "ERROR: --port must be a positive integer" >&2; exit 1; }
(( PORT <= 65535 )) || { echo "ERROR: --port must be <= 65535" >&2; exit 1; }
[[ "$GPU_DEVICE" =~ ^[0-9]+$ ]] || { echo "ERROR: --gpu-device must be one non-negative integer" >&2; exit 1; }
case "$DTYPE" in
    bfloat16|bf16|float16|fp16|float32|fp32) ;;
    *) echo "ERROR: unsupported --dtype $DTYPE" >&2; exit 1 ;;
esac
if [[ -n "$THRESHOLD" ]]; then
    "$PYTORCH_PYTHON" - "$THRESHOLD" <<'PY'
import math, sys
try:
    value = float(sys.argv[1])
except (TypeError, ValueError):
    raise SystemExit("--threshold must be numeric, none, or null")
if not math.isfinite(value):
    raise SystemExit("--threshold must be finite")
PY
fi
if [[ -n "$MAX_THINKING_TOKENS" ]]; then
    is_positive_int "$MAX_THINKING_TOKENS" || { echo "ERROR: --max-thinking-tokens must be positive" >&2; exit 1; }
fi
if [[ -n "$MAX_SAMPLES" ]]; then
    is_positive_int "$MAX_SAMPLES" || { echo "ERROR: --max-samples must be positive" >&2; exit 1; }
fi
"$PYTORCH_PYTHON" - "$TEMPERATURE" "$TOP_P" <<'PY'
import math, sys
try:
    temperature, top_p = map(float, sys.argv[1:])
except (TypeError, ValueError):
    raise SystemExit("--temperature and --top-p must be numeric")
if not math.isfinite(temperature) or temperature < 0:
    raise SystemExit("--temperature must be finite and non-negative")
if not math.isfinite(top_p) or not 0 <= top_p <= 1:
    raise SystemExit("--top-p must be between 0 and 1")
PY
is_nonnegative_number "$GPU_MEMORY_RESERVE_GB" || { echo "ERROR: --gpu-memory-reserve-gb must be non-negative" >&2; exit 1; }

if judge_benchmark_requested \
    && [[ "$DRY_RUN" != "true" ]] \
    && [[ "$SKIP_JUDGE_API_KEY_CHECK" != "true" ]] \
    && { [[ -z "$JUDGE_SERVER_ADDRESS" ]] || [[ "${JUDGE_SERVER_ADDRESS,,}" == *"api.openai.com"* ]]; } \
    && [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "ERROR: A requested judge-based benchmark uses an OpenAI judge by default, but OPENAI_API_KEY is not set." >&2
    echo "       Export OPENAI_API_KEY, pass a custom --judge-server-address, or use" >&2
    echo "       --skip-judge-api-key-check when the key is injected downstream." >&2
    exit 1
fi

if [[ "$CONTEXT_LENGTH_USER_SET" != "true" ]]; then
    REQUIRED_GENERATION_TOKENS="0"
    if standard_benchmark_requested || mt_bench_requested; then
        REQUIRED_GENERATION_TOKENS="$TOKENS"
    fi
    if mt_bench_requested && (( MT_BENCH_MAX_TOKENS > REQUIRED_GENERATION_TOKENS )); then
        REQUIRED_GENERATION_TOKENS="$MT_BENCH_MAX_TOKENS"
    fi
    if alpaca_eval_requested && (( ALPACA_EVAL_MAX_TOKENS > REQUIRED_GENERATION_TOKENS )); then
        REQUIRED_GENERATION_TOKENS="$ALPACA_EVAL_MAX_TOKENS"
    fi
    (( REQUIRED_GENERATION_TOKENS > 0 )) || REQUIRED_GENERATION_TOKENS="$TOKENS"
    CONTEXT_LENGTH=$((REQUIRED_GENERATION_TOKENS + 2048))
fi
is_positive_int "$CONTEXT_LENGTH" || { echo "ERROR: --context-length must be positive" >&2; exit 1; }
if { standard_benchmark_requested || mt_bench_requested; } && (( TOKENS > CONTEXT_LENGTH )); then
    echo "ERROR: --tokens cannot exceed --context-length" >&2
    exit 1
fi
if [[ "$MODE" == "linearspec_lora" && ! -f "$LORA_PATH/adapter_config.json" ]]; then
    echo "ERROR: linearspec_lora adapter_config.json not found: $LORA_PATH" >&2
    exit 1
fi

if ! port_is_free "$PORT"; then
    if [[ "$PORT_USER_SET" == "true" ]]; then
        echo "ERROR: explicitly requested port is busy: $PORT" >&2
        exit 1
    fi
    PORT="$(find_free_port "$((PORT + 1))")"
fi

OUTPUT_PATH="$(realpath -m "$OUTPUT_PATH")"
NEMO_SKILLS_DATA_DIR_ARG="$(realpath -m "$NEMO_SKILLS_DATA_DIR_ARG")"
GOOGLE_RESEARCH_DIR="$(realpath -m "$GOOGLE_RESEARCH_DIR")"
RUN_NAME="eval_$(date +%Y%m%d_%H%M%S)"
FINAL_JOB_DIR="$OUTPUT_PATH/$RUN_NAME"
if [[ -e "$FINAL_JOB_DIR" ]]; then
    suffix=1
    while [[ -e "$OUTPUT_PATH/${RUN_NAME}_$(printf '%02d' "$suffix")" ]]; do suffix=$((suffix + 1)); done
    FINAL_JOB_DIR="$OUTPUT_PATH/${RUN_NAME}_$(printf '%02d' "$suffix")"
fi
INTERNAL_JOB_DIR="$OUTPUT_PATH/.${RUN_NAME}_work_$$"
EVAL_OUTPUT_DIR="$INTERNAL_JOB_DIR/results"
RUNTIME_DIR="$INTERNAL_JOB_DIR/pytorch_runtime"

echo "================================================================"
echo " Native PyTorch + NeMo-Skills eval"
echo "================================================================"
echo " Mode:                 $MODE"
echo " Model:                $MODEL"
echo " Benchmarks:           $BENCHMARKS"
echo " GPU device:           $GPU_DEVICE"
echo " GPU reserve GiB:      $GPU_MEMORY_RESERVE_GB"
echo " Block length:         $BLOCK_LENGTH"
echo " Threshold:            ${THRESHOLD:-None}"
echo " Tokens/context:       $TOKENS / $CONTEXT_LENGTH"
echo " Client concurrency:   $CLIENT_CONCURRENCY (model execution serialized)"
echo " Port:                 $PORT"
echo " Final output:         $FINAL_JOB_DIR"
echo " Internal work:        $INTERNAL_JOB_DIR"
echo " Python:               $PYTORCH_PYTHON"
if arena_hard_requested; then
    echo " Arena judge model:    ${JUDGE_MODEL:-gpt-4.1 (dataset default)}"
    echo " Arena judge server:   ${JUDGE_SERVER_ADDRESS:-https://api.openai.com/v1 (dataset default)}"
    echo " Arena judge type:     ${JUDGE_SERVER_TYPE:-openai (dataset default)}"
fi
if mt_bench_requested; then
    echo " MT-Bench judge:       ${JUDGE_MODEL:-gpt-4.1}"
    echo " MT-Bench max tokens:  $MT_BENCH_MAX_TOKENS per turn"
    echo " Judge concurrency:    $JUDGE_CONCURRENCY"
fi
if alpaca_eval_requested; then
    echo " AlpacaEval judge:     ${JUDGE_MODEL:-gpt-4-1106-preview (official default)}"
    echo " AlpacaEval max tok:   $ALPACA_EVAL_MAX_TOKENS"
    echo " Judge concurrency:    $JUDGE_CONCURRENCY"
fi
echo "================================================================"

if [[ "$DRY_RUN" == "true" ]]; then
    echo "[dry-run] No directories, server, model, or evaluation were created."
    exit 0
fi

mkdir -p "$FINAL_JOB_DIR" "$EVAL_OUTPUT_DIR" "$RUNTIME_DIR" "$NEMO_SKILLS_DATA_DIR_ARG"

write_settings() {
    SETTINGS_MODE="$MODE" SETTINGS_MODEL="$MODEL" SETTINGS_MODEL_TAG="$SERVED_MODEL_NAME" SETTINGS_BENCHMARKS="$BENCHMARKS" SETTINGS_TOKENS="$TOKENS" SETTINGS_TEMPERATURE="$TEMPERATURE" SETTINGS_TOP_P="$TOP_P" SETTINGS_NUM_CHUNKS="$NUM_CHUNKS" SETTINGS_CLIENT_CONCURRENCY="$CLIENT_CONCURRENCY" SETTINGS_MAX_SAMPLES="$MAX_SAMPLES" SETTINGS_QUICK_TEST="$QUICK_TEST" SETTINGS_KEEP_THINKING="$KEEP_THINKING" SETTINGS_STRIP_THINKING="$STRIP_THINKING" SETTINGS_DISABLE_THINKING="$DISABLE_THINKING" SETTINGS_ENABLE_THINKING="$ENABLE_THINKING" SETTINGS_MATH_PROMPT_CONFIG="$MATH_PROMPT_CONFIG" SETTINGS_GPU_DEVICE="$GPU_DEVICE" SETTINGS_GPU_RESERVE="$GPU_MEMORY_RESERVE_GB" SETTINGS_DTYPE="$DTYPE" SETTINGS_BLOCK_LENGTH="$BLOCK_LENGTH" SETTINGS_THRESHOLD="$THRESHOLD" SETTINGS_CAUSAL_CONTEXT="$CAUSAL_CONTEXT" SETTINGS_CONTEXT_LENGTH="$CONTEXT_LENGTH" SETTINGS_LORA_PATH="$LORA_PATH" SETTINGS_MAX_THINKING="$MAX_THINKING_TOKENS" SETTINGS_PORT="$PORT" SETTINGS_OUTPUT="$OUTPUT_PATH" SETTINGS_FINAL="$FINAL_JOB_DIR" SETTINGS_INTERNAL="$INTERNAL_JOB_DIR" SETTINGS_PIPELINE="$PIPELINE" SETTINGS_PYTORCH_PYTHON="$PYTORCH_PYTHON" SETTINGS_EVAL_PYTHON="$EVAL_PYTHON" SETTINGS_DATA_DIR="$NEMO_SKILLS_DATA_DIR_ARG" SETTINGS_GOOGLE_RESEARCH="$GOOGLE_RESEARCH_DIR" SETTINGS_KEEP_RUNTIME="$KEEP_RUNTIME" \
    SETTINGS_JUDGE_MODEL="$JUDGE_MODEL" SETTINGS_JUDGE_SERVER_ADDRESS="$JUDGE_SERVER_ADDRESS" SETTINGS_JUDGE_SERVER_TYPE="$JUDGE_SERVER_TYPE" SETTINGS_JUDGE_CONCURRENCY="$JUDGE_CONCURRENCY" SETTINGS_MT_BENCH_MAX_TOKENS="$MT_BENCH_MAX_TOKENS" SETTINGS_ALPACA_EVAL_MAX_TOKENS="$ALPACA_EVAL_MAX_TOKENS" SETTINGS_SKIP_JUDGE_API_KEY_CHECK="$SKIP_JUDGE_API_KEY_CHECK" SETTINGS_OPENAI_API_KEY_CONFIGURED="$([[ -n "${OPENAI_API_KEY:-}" ]] && echo true || echo false)" \
    "$PYTORCH_PYTHON" - "$FINAL_JOB_DIR/Settings.json" "${ORIGINAL_ARGS[@]}" <<'PY'
import json, os, shlex, sys
from datetime import datetime
def env(name): return os.environ.get("SETTINGS_" + name, "")
def boolean(name): return env(name).lower() in {"1", "true", "yes"}
def integer(name): return int(env(name)) if env(name) else None
def number(name): return float(env(name)) if env(name) else None
payload = {
    "created_at": datetime.now().astimezone().isoformat(),
    "entrypoint": "observations/eval_pytorch_nemo.sh",
    "original_args": sys.argv[2:],
    "command": "bash observations/eval_pytorch_nemo.sh " + " ".join(shlex.quote(x) for x in sys.argv[2:]),
    "backend": "native_pytorch",
    "benchmark": {
        "benchmarks": env("BENCHMARKS"), "tokens": integer("TOKENS"),
        "temperature": number("TEMPERATURE"), "top_p": number("TOP_P"),
        "num_chunks": integer("NUM_CHUNKS"), "client_concurrency": integer("CLIENT_CONCURRENCY"),
        "max_samples": integer("MAX_SAMPLES"), "quick_test": boolean("QUICK_TEST"),
        "keep_thinking": boolean("KEEP_THINKING"), "strip_thinking": boolean("STRIP_THINKING"),
        "disable_thinking": boolean("DISABLE_THINKING"), "math_prompt_config": env("MATH_PROMPT_CONFIG"),
        "mt_bench_max_tokens": integer("MT_BENCH_MAX_TOKENS"),
        "alpaca_eval_max_tokens": integer("ALPACA_EVAL_MAX_TOKENS"),
    },
    "pytorch": {
        "mode": env("MODE"), "model": env("MODEL"), "served_model_name": env("MODEL_TAG"),
        "gpu_device": env("GPU_DEVICE"), "gpu_memory_reserve_gb": number("GPU_RESERVE"),
        "dtype": env("DTYPE"), "block_length": integer("BLOCK_LENGTH"),
        "threshold": number("THRESHOLD"), "causal_context": boolean("CAUSAL_CONTEXT"),
        "context_length": integer("CONTEXT_LENGTH"), "lora_path": env("LORA_PATH"),
        "enable_thinking": boolean("ENABLE_THINKING"), "max_thinking_tokens": integer("MAX_THINKING"),
        "port": integer("PORT"), "model_execution_serialized": True,
        "top_p_applied_by_native_model": False,
    },
    "judge": {
        "model": env("JUDGE_MODEL") or "dataset default",
        "server_address": env("JUDGE_SERVER_ADDRESS") or "dataset default",
        "server_type": env("JUDGE_SERVER_TYPE") or "dataset default",
        "concurrency": integer("JUDGE_CONCURRENCY"),
        "skip_api_key_check": boolean("SKIP_JUDGE_API_KEY_CHECK"),
        "openai_api_key_configured": boolean("OPENAI_API_KEY_CONFIGURED"),
    },
    "paths": {
        "output_path": env("OUTPUT"), "final_job_dir": env("FINAL"),
        "internal_job_dir": env("INTERNAL"), "pipeline": env("PIPELINE"),
        "pytorch_python": env("PYTORCH_PYTHON"), "eval_python": env("EVAL_PYTHON"),
        "nemo_skills_data_dir": env("DATA_DIR"), "google_research_dir": env("GOOGLE_RESEARCH"),
    },
    "keep_runtime": boolean("KEEP_RUNTIME"),
}
with open(sys.argv[1], "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2, ensure_ascii=False, sort_keys=True)
    f.write("\n")
PY
}
write_settings

export PYTORCH_MODE="$MODE"
export PYTORCH_MODEL="$MODEL"
export PYTORCH_MODEL_TAG="$SERVED_MODEL_NAME"
export PYTORCH_GPU_DEVICE="$GPU_DEVICE"
export PYTORCH_GPU_MEMORY_RESERVE_GB="$GPU_MEMORY_RESERVE_GB"
export PYTORCH_PORT="$PORT"
export PYTORCH_DTYPE="$DTYPE"
export PYTORCH_BLOCK_LENGTH="$BLOCK_LENGTH"
export PYTORCH_THRESHOLD="$THRESHOLD"
export PYTORCH_CAUSAL_CONTEXT="$CAUSAL_CONTEXT"
export PYTORCH_CONTEXT_LENGTH="$CONTEXT_LENGTH"
export PYTORCH_LORA_PATH="$LORA_PATH"
export PYTORCH_ENABLE_THINKING="$ENABLE_THINKING"
export PYTORCH_MAX_THINKING_TOKENS="$MAX_THINKING_TOKENS"
export PYTORCH_RUN_DIR="$RUNTIME_DIR"
export PYTORCH_FINAL_OUTPUT_DIR="$FINAL_JOB_DIR"
export PYTORCH_PYTHON="$PYTORCH_PYTHON"
export EVAL_PYTHON="$EVAL_PYTHON"
export NEMO_SKILLS_DATA_DIR="$NEMO_SKILLS_DATA_DIR_ARG"
export NLD_GOOGLE_RESEARCH_DIR="$GOOGLE_RESEARCH_DIR"
export SEQ_EVAL_BENCHMARK="$BENCHMARKS"
export SEQ_EVAL_EXPNAME="hf_base"
export SEQ_EVAL_OUTPUT_DIR="$EVAL_OUTPUT_DIR"
export SEQ_EVAL_TOKENS_TO_GENERATE="$TOKENS"
export SEQ_EVAL_TEMPERATURE="$TEMPERATURE"
export SEQ_EVAL_TOP_P="$TOP_P"
export SEQ_EVAL_NUM_CHUNKS="$NUM_CHUNKS"
export SEQ_EVAL_CLIENT_CONCURRENCY="$CLIENT_CONCURRENCY"
export SEQ_EVAL_MAX_SAMPLES="$MAX_SAMPLES"
export SEQ_EVAL_QUICK_TEST="$QUICK_TEST"
export SEQ_EVAL_KEEP_THINKING="$KEEP_THINKING"
export SEQ_EVAL_STRIP_THINKING="$STRIP_THINKING"
export SEQ_EVAL_DISABLE_THINKING="$DISABLE_THINKING"
export SEQ_EVAL_MATH_PROMPT_CONFIG="$MATH_PROMPT_CONFIG"
export SEQ_EVAL_JUDGE_MODEL="$JUDGE_MODEL"
export SEQ_EVAL_JUDGE_SERVER_ADDRESS="$JUDGE_SERVER_ADDRESS"
export SEQ_EVAL_JUDGE_SERVER_TYPE="$JUDGE_SERVER_TYPE"
export SEQ_EVAL_JUDGE_CONCURRENCY="$JUDGE_CONCURRENCY"
export SEQ_EVAL_MT_BENCH_MAX_TOKENS="$MT_BENCH_MAX_TOKENS"
export SEQ_EVAL_ALPACA_EVAL_MAX_TOKENS="$ALPACA_EVAL_MAX_TOKENS"
export SEQ_EVAL_SKIP_JUDGE_API_KEY_CHECK="$SKIP_JUDGE_API_KEY_CHECK"

PIPELINE_STATUS=0
bash "$PIPELINE" || PIPELINE_STATUS=$?
if [[ "$PIPELINE_STATUS" != "0" ]]; then
    echo "ERROR: native PyTorch pipeline failed; internal work kept: $INTERNAL_JOB_DIR" >&2
    exit "$PIPELINE_STATUS"
fi

ANY_ERROR=0
IFS=',' read -ra FINAL_BENCHES <<< "$BENCHMARKS"
for spec in "${FINAL_BENCHES[@]}"; do
    spec="${spec//[[:space:]]/}"
    [[ -z "$spec" ]] && continue
    name="${spec%%:*}"
    safe="${name//\//_}"
    safe="${safe//:/_}"
    if [[ -f "$FINAL_JOB_DIR/error_${safe}.json" ]]; then
        ANY_ERROR=1
    elif [[ ! -f "$FINAL_JOB_DIR/metrics_${safe}.json" ]]; then
        echo "ERROR: missing compact result for $name" >&2
        ANY_ERROR=1
    fi
done

if [[ "$KEEP_RUNTIME" == "true" || "$ANY_ERROR" != "0" ]]; then
    echo "Runtime/debug directory kept: $INTERNAL_JOB_DIR"
else
    rm -rf "$INTERNAL_JOB_DIR"
fi

echo "Completed native PyTorch + NeMo-Skills evaluation."
echo "Final output: $FINAL_JOB_DIR"
if [[ "$ANY_ERROR" != "0" ]]; then
    echo "One or more benchmarks failed; inspect error_<benchmark>.json and the retained runtime directory." >&2
fi
