#!/bin/bash
# NeMo-Skills benchmark organization with SGLang as the inference backend.

set -euo pipefail

ORIGINAL_ARGS=("$@")
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OBSERVATIONS_DIR="$SCRIPT_DIR"
PROJECT_DIR="$(cd "$OBSERVATIONS_DIR/.." && pwd)"
OBSERVATION_RESULTS_ROOT="${NLD_OBSERVATION_RESULTS_ROOT:-/data/home/wly/dLLM/NLD_results/observations}"
PIPELINE="${SGLANG_PIPELINE_OVERRIDE:-$PROJECT_DIR/xp/examples/run_sglang_eval_pipeline_gpu_only.sh}"

DEFAULT_MODEL="/data1/linyewei/models/Nemotron-Labs-Diffusion-8B"
DEFAULT_OUT_DIR="$OBSERVATION_RESULTS_ROOT/sglang_nemo_eval_results"
DEFAULT_NEMO_SKILLS_DATA_DIR="/data1/linyewei/datasets/NLD"
DEFAULT_BENCHMARKS="gsm8k:1,human-eval:1,mbpp:1,math-500:1,aime24:1,aime25:1,gpqa:1,mmlu:1,ifeval:1,livecodebench-cpp:1"
DEFAULT_SGLANG_WORK_DIR="$PROJECT_DIR/sglang_dllm"
DEFAULT_SGLANG_SRC="$DEFAULT_SGLANG_WORK_DIR/src/sglang"
DEFAULT_SGLANG_PYTHON="/data/home/wly/.conda/envs/nld_sglang/bin/python"
if [[ ! -x "$DEFAULT_SGLANG_PYTHON" ]]; then
    DEFAULT_SGLANG_PYTHON="python"
fi

usage() {
    cat <<EOF
Usage: $0 --mode MODE [options]

Modes:
  --mode linearspec_lora      SGLang LinearSpec with LoRA draft weights
  --mode linearspec_base      SGLang LinearSpec without LoRA
  --mode fastdiffuser         SGLang FastDiffuser / dLLM block denoising
  --mode ar                   SGLang AR mode via {"ar_mode": true}

Benchmark:
  --benchmarks LIST           Comma-separated benchmarks; mt-bench/alpaca-eval use dedicated runners
  --tokens N                  Max generated tokens sent to NeMo-Skills (default: 8192)
  --temperature V             Sampling temperature (default: 0)
  --top-p V                   top_p (default: 0.95)
  --num-chunks N              NeMo-Skills chunking knob (default: client concurrency)
  --max-samples N             Limit number of samples per benchmark
  --quick-test                NeMo-Skills quick test mode
  --keep-thinking             Keep <think> content in NeMo-Skills output
  --strip-thinking            Strip <think> content before scoring
  --disable-thinking          Do not send thinking-disable extra_body in SGLang mode; kept for config clarity
  --math-prompt-config NAME   NeMo-Skills math prompt_config override

Judge-based benchmarks (Arena-Hard, MT-Bench, and AlpacaEval):
  --judge-model NAME          Override the benchmark's default judge
  --judge-server-address URL  Override the default https://api.openai.com/v1 judge endpoint
  --judge-server-type TYPE    Judge server type (currently openai-compatible)
  --judge-concurrency N       Concurrent dedicated-runner judge requests (default: 4)
  --mt-bench-max-tokens N     Completion budget per MT-Bench turn (default: 1024)
  --alpaca-eval-max-tokens N  Candidate completion budget for AlpacaEval (default: 2048)
  --skip-judge-api-key-check  Skip the OPENAI_API_KEY preflight check for downstream-injected credentials

SGLang server:
  --model PATH                Model path/id (default: $DEFAULT_MODEL)
  --served-model-name NAME    OpenAI model name served by SGLang (default: nemotron-labs-diffusion-8b)
  --gpu-devices LIST          CUDA_VISIBLE_DEVICES, e.g. 0 or 0,1 (default: 0)
  --tp-size N                 Tensor parallel size (default: inferred from --gpu-devices)
  --port N                    SGLang server port (default: 30000; auto-pick if default is busy)
  --proxy-port N              Timing proxy port (default: 31000; auto-pick if default is busy)
  --batch-size N              Alias for --max-running-requests
  --max-running-requests N    SGLang max running requests (default: 1)
  --client-concurrency N      NeMo max_concurrent_requests and proxy max in-flight requests (default: 1)
  --gpu-memory-reserve-gb V   Reserve V GiB on each selected GPU before starting SGLang (default: 0)
  --context-length N          SGLang context length (default: auto; 2048, raised to tokens+2048 if needed)
  --mem-fraction V            SGLang mem-fraction-static (default: 0.55)
  --cuda-graph-bs LIST        Quoted CUDA graph batch sizes, e.g. "1 2 4" (default: "1")
  --dtype DTYPE               SGLang dtype (default: bfloat16)
  --quantization NAME         Optional SGLang quantization, e.g. fp8
  --block-size N              Optional DLLM block_size in generated YAML
  --max-steps N               Optional FastDiffuser max_steps in generated YAML
  --threshold V               Optional FastDiffuser threshold in generated YAML
  --lora-path DIR             LoRA adapter dir for linearspec_lora
  --lora-mode MODE            draft_only or both (default: draft_only)
  --extra-server-args "..."   Extra args appended to sglang.launch_server

Environment / output:
  --output-path DIR           Final compact output dir (default: $DEFAULT_OUT_DIR)
  --out-dir DIR               Backward-compatible alias for --output-path
  --exp-name NAME             Compatibility arg; compact output always uses eval_YYYYMMDD_HHMMSS
  --sglang-python PATH        Python for SGLang server/proxy/eval (default: $DEFAULT_SGLANG_PYTHON)
  --eval-python PATH          Python for NeMo-Skills eval (default: --sglang-python)
  --sglang-src DIR            SGLang source root (default: $DEFAULT_SGLANG_SRC)
  --sglang-work-dir DIR       SGLang work dir/cache root (default: $DEFAULT_SGLANG_WORK_DIR)
  --nemo-skills-data-dir DIR  Persistent NeMo-Skills dataset dir (default: $DEFAULT_NEMO_SKILLS_DATA_DIR)
  --hf-home DIR               HF cache dir (default: <work-dir>/hf_cache)
  --sglang-cache-dir DIR      SGLang cache dir (default: <work-dir>/sglang_cache)
  --keep-server               Do not stop SGLang/proxy after evaluation
  --dry-run                   Print resolved settings without running
  -h, --help                  Show this help

Example:
  bash $0 --mode linearspec_lora --benchmarks gsm8k:1 --gpu-devices 0 --batch-size 1 --client-concurrency 1
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
MAX_SAMPLES=""
QUICK_TEST="false"
KEEP_THINKING="false"
STRIP_THINKING="false"
DISABLE_THINKING="false"
MATH_PROMPT_CONFIG=""
JUDGE_MODEL=""
JUDGE_SERVER_ADDRESS=""
JUDGE_SERVER_TYPE=""
JUDGE_CONCURRENCY="4"
MT_BENCH_MAX_TOKENS="1024"
ALPACA_EVAL_MAX_TOKENS="2048"
SKIP_JUDGE_API_KEY_CHECK="false"

GPU_DEVICES="0"
TP_SIZE="1"
TP_SIZE_USER_SET="false"
PORT="30000"
PROXY_PORT="31000"
PORT_USER_SET="false"
PROXY_PORT_USER_SET="false"
MAX_RUNNING_REQUESTS="1"
CLIENT_CONCURRENCY="1"
GPU_MEMORY_RESERVE_GB="0"
CONTEXT_LENGTH="2048"
CONTEXT_LENGTH_USER_SET="false"
MEM_FRACTION="0.55"
CUDA_GRAPH_BS="1"
DTYPE="bfloat16"
QUANTIZATION=""
BLOCK_SIZE=""
MAX_STEPS=""
THRESHOLD=""
LORA_PATH=""
LORA_MODE="draft_only"
EXTRA_SERVER_ARGS=""

OUT_DIR="${OUTPUT_PATH:-${OUT_DIR:-$DEFAULT_OUT_DIR}}"
EXP_NAME=""
SGLANG_PYTHON="$DEFAULT_SGLANG_PYTHON"
EVAL_PYTHON=""
SGLANG_SRC="$DEFAULT_SGLANG_SRC"
SGLANG_WORK_DIR="$DEFAULT_SGLANG_WORK_DIR"
NEMO_SKILLS_DATA_DIR_ARG="${NEMO_SKILLS_DATA_DIR:-$DEFAULT_NEMO_SKILLS_DATA_DIR}"
HF_HOME_ARG=""
SGLANG_CACHE_ARG=""
KEEP_SERVER="false"
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
        --max-samples) MAX_SAMPLES="$2"; shift 2 ;;
        --quick-test) QUICK_TEST="true"; shift ;;
        --keep-thinking) KEEP_THINKING="true"; shift ;;
        --strip-thinking) STRIP_THINKING="true"; shift ;;
        --disable-thinking) DISABLE_THINKING="true"; shift ;;
        --math-prompt-config) MATH_PROMPT_CONFIG="$2"; shift 2 ;;
        --judge-model) JUDGE_MODEL="$2"; shift 2 ;;
        --judge-server-address) JUDGE_SERVER_ADDRESS="$2"; shift 2 ;;
        --judge-server-type) JUDGE_SERVER_TYPE="$2"; shift 2 ;;
        --judge-concurrency) JUDGE_CONCURRENCY="$2"; shift 2 ;;
        --mt-bench-max-tokens) MT_BENCH_MAX_TOKENS="$2"; shift 2 ;;
        --alpaca-eval-max-tokens) ALPACA_EVAL_MAX_TOKENS="$2"; shift 2 ;;
        --skip-judge-api-key-check) SKIP_JUDGE_API_KEY_CHECK="true"; shift ;;
        --gpu-devices) GPU_DEVICES="$2"; shift 2 ;;
        --tp-size|--tensor-parallel-size) TP_SIZE="$2"; TP_SIZE_USER_SET="true"; shift 2 ;;
        --port) PORT="$2"; PORT_USER_SET="true"; shift 2 ;;
        --proxy-port) PROXY_PORT="$2"; PROXY_PORT_USER_SET="true"; shift 2 ;;
        --batch-size) MAX_RUNNING_REQUESTS="$2"; shift 2 ;;
        --max-running-requests) MAX_RUNNING_REQUESTS="$2"; shift 2 ;;
        --client-concurrency) CLIENT_CONCURRENCY="$2"; shift 2 ;;
        --gpu-memory-reserve-gb) GPU_MEMORY_RESERVE_GB="$2"; shift 2 ;;
        --context-length) CONTEXT_LENGTH="$2"; CONTEXT_LENGTH_USER_SET="true"; shift 2 ;;
        --mem-fraction) MEM_FRACTION="$2"; shift 2 ;;
        --cuda-graph-bs) CUDA_GRAPH_BS="$2"; shift 2 ;;
        --dtype) DTYPE="$2"; shift 2 ;;
        --quantization) QUANTIZATION="$2"; shift 2 ;;
        --block-size) BLOCK_SIZE="$2"; shift 2 ;;
        --max-steps) MAX_STEPS="$2"; shift 2 ;;
        --threshold) THRESHOLD="$2"; shift 2 ;;
        --lora-path) LORA_PATH="$2"; shift 2 ;;
        --lora-mode) LORA_MODE="$2"; shift 2 ;;
        --extra-server-args) EXTRA_SERVER_ARGS="$2"; shift 2 ;;
        --output-path|--out-dir) OUT_DIR="$2"; shift 2 ;;
        --exp-name) EXP_NAME="$2"; shift 2 ;;
        --sglang-python) SGLANG_PYTHON="$2"; shift 2 ;;
        --eval-python) EVAL_PYTHON="$2"; shift 2 ;;
        --sglang-src) SGLANG_SRC="$2"; shift 2 ;;
        --sglang-work-dir) SGLANG_WORK_DIR="$2"; shift 2 ;;
        --nemo-skills-data-dir) NEMO_SKILLS_DATA_DIR_ARG="$2"; shift 2 ;;
        --hf-home) HF_HOME_ARG="$2"; shift 2 ;;
        --sglang-cache-dir) SGLANG_CACHE_ARG="$2"; shift 2 ;;
        --keep-server) KEEP_SERVER="true"; shift ;;
        --dry-run) DRY_RUN="true"; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "ERROR: unknown option: $1" >&2; usage; exit 1 ;;
    esac
done

if [[ -z "$MODE" ]]; then
    echo "ERROR: --mode is required" >&2
    usage
    exit 1
fi

case "$MODE" in
    linearspec_lora|linear_spec_lora) MODE="linearspec_lora" ;;
    linearspec|linearspec_base|linear_spec_base) MODE="linearspec_base" ;;
    fastdiffuser|dlm) MODE="fastdiffuser" ;;
    ar) MODE="ar" ;;
    *) echo "ERROR: unknown --mode $MODE" >&2; usage; exit 1 ;;
esac

if [[ -z "$NUM_CHUNKS" ]]; then
    NUM_CHUNKS="$CLIENT_CONCURRENCY"
fi
if [[ -z "$EVAL_PYTHON" ]]; then
    EVAL_PYTHON="$SGLANG_PYTHON"
fi
if [[ -z "$LORA_PATH" ]]; then
    LORA_PATH="$SGLANG_WORK_DIR/linear_spec_lora"
fi

infer_gpu_count() {
    local devices="$1"
    local count=0
    local item
    local -a GPU_ARRAY
    IFS=',' read -ra GPU_ARRAY <<< "$devices"
    for item in "${GPU_ARRAY[@]}"; do
        item="${item//[[:space:]]/}"
        [[ -n "$item" ]] && count=$((count + 1))
    done
    echo "$count"
}

is_positive_int() {
    [[ "${1:-}" =~ ^[0-9]+$ ]] && (( "$1" > 0 ))
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
    "$SGLANG_PYTHON" - "$1" <<'PY'
import math
import sys

try:
    value = float(sys.argv[1])
except (TypeError, ValueError):
    sys.exit(1)
sys.exit(0 if math.isfinite(value) and value >= 0 else 1)
PY
}

port_is_free() {
    "$SGLANG_PYTHON" - "$1" <<'PY'
import socket
import sys

port = int(sys.argv[1])
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind(("127.0.0.1", port))
    except OSError:
        sys.exit(1)
PY
}

find_free_port() {
    "$SGLANG_PYTHON" - "$1" <<'PY'
import socket
import sys

start = int(sys.argv[1])
for port in range(start, 65535):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            continue
        print(port)
        break
else:
    raise SystemExit("no free port found")
PY
}

write_settings_file() {
    local settings_file="$FINAL_JOB_DIR/Settings.json"
    NLD_SETTINGS_ENTRYPOINT="$0" \
    NLD_SETTINGS_PROJECT_DIR="$PROJECT_DIR" \
    NLD_SETTINGS_PIPELINE="$PIPELINE" \
    NLD_SETTINGS_MODE="$MODE" \
    NLD_SETTINGS_BENCHMARKS="$BENCHMARKS" \
    NLD_SETTINGS_MODEL="$MODEL" \
    NLD_SETTINGS_SERVED_MODEL_NAME="$SERVED_MODEL_NAME" \
    NLD_SETTINGS_TOKENS="$TOKENS" \
    NLD_SETTINGS_TEMPERATURE="$TEMPERATURE" \
    NLD_SETTINGS_TOP_P="$TOP_P" \
    NLD_SETTINGS_NUM_CHUNKS="$NUM_CHUNKS" \
    NLD_SETTINGS_MAX_SAMPLES="$MAX_SAMPLES" \
    NLD_SETTINGS_QUICK_TEST="$QUICK_TEST" \
    NLD_SETTINGS_KEEP_THINKING="$KEEP_THINKING" \
    NLD_SETTINGS_STRIP_THINKING="$STRIP_THINKING" \
    NLD_SETTINGS_DISABLE_THINKING="$DISABLE_THINKING" \
    NLD_SETTINGS_MATH_PROMPT_CONFIG="$MATH_PROMPT_CONFIG" \
    NLD_SETTINGS_JUDGE_MODEL="$JUDGE_MODEL" \
    NLD_SETTINGS_JUDGE_SERVER_ADDRESS="$JUDGE_SERVER_ADDRESS" \
    NLD_SETTINGS_JUDGE_SERVER_TYPE="$JUDGE_SERVER_TYPE" \
    NLD_SETTINGS_JUDGE_CONCURRENCY="$JUDGE_CONCURRENCY" \
    NLD_SETTINGS_MT_BENCH_MAX_TOKENS="$MT_BENCH_MAX_TOKENS" \
    NLD_SETTINGS_ALPACA_EVAL_MAX_TOKENS="$ALPACA_EVAL_MAX_TOKENS" \
    NLD_SETTINGS_SKIP_JUDGE_API_KEY_CHECK="$SKIP_JUDGE_API_KEY_CHECK" \
    NLD_SETTINGS_OPENAI_API_KEY_CONFIGURED="$([[ -n "${OPENAI_API_KEY:-}" ]] && echo true || echo false)" \
    NLD_SETTINGS_GPU_DEVICES="$GPU_DEVICES" \
    NLD_SETTINGS_TP_SIZE="$TP_SIZE" \
    NLD_SETTINGS_TP_SIZE_USER_SET="$TP_SIZE_USER_SET" \
    NLD_SETTINGS_PORT="$PORT" \
    NLD_SETTINGS_PORT_USER_SET="$PORT_USER_SET" \
    NLD_SETTINGS_PROXY_PORT="$PROXY_PORT" \
    NLD_SETTINGS_PROXY_PORT_USER_SET="$PROXY_PORT_USER_SET" \
    NLD_SETTINGS_MAX_RUNNING_REQUESTS="$MAX_RUNNING_REQUESTS" \
    NLD_SETTINGS_CLIENT_CONCURRENCY="$CLIENT_CONCURRENCY" \
    NLD_SETTINGS_GPU_MEMORY_RESERVE_GB="$GPU_MEMORY_RESERVE_GB" \
    NLD_SETTINGS_CONTEXT_LENGTH="$CONTEXT_LENGTH" \
    NLD_SETTINGS_CONTEXT_LENGTH_USER_SET="$CONTEXT_LENGTH_USER_SET" \
    NLD_SETTINGS_MEM_FRACTION="$MEM_FRACTION" \
    NLD_SETTINGS_CUDA_GRAPH_BS="$CUDA_GRAPH_BS" \
    NLD_SETTINGS_DTYPE="$DTYPE" \
    NLD_SETTINGS_QUANTIZATION="$QUANTIZATION" \
    NLD_SETTINGS_BLOCK_SIZE="$BLOCK_SIZE" \
    NLD_SETTINGS_MAX_STEPS="$MAX_STEPS" \
    NLD_SETTINGS_THRESHOLD="$THRESHOLD" \
    NLD_SETTINGS_LORA_PATH="$LORA_PATH" \
    NLD_SETTINGS_LORA_MODE="$LORA_MODE" \
    NLD_SETTINGS_EXTRA_SERVER_ARGS="$EXTRA_SERVER_ARGS" \
    NLD_SETTINGS_OUTPUT_PATH="$OUT_DIR" \
    NLD_SETTINGS_FINAL_JOB_DIR="$FINAL_JOB_DIR" \
    NLD_SETTINGS_INTERNAL_JOB_DIR="$INTERNAL_JOB_DIR" \
    NLD_SETTINGS_EVAL_OUTPUT_DIR="$EVAL_OUTPUT_DIR" \
    NLD_SETTINGS_RUNTIME_DIR="$RUNTIME_DIR" \
    NLD_SETTINGS_CHECKPOINT_NAME="$CHECKPOINT_NAME" \
    NLD_SETTINGS_EVAL_DIR_NAME="$EVAL_DIR_NAME" \
    NLD_SETTINGS_SGLANG_PYTHON="$SGLANG_PYTHON" \
    NLD_SETTINGS_EVAL_PYTHON="$EVAL_PYTHON" \
    NLD_SETTINGS_SGLANG_SRC="$SGLANG_SRC" \
    NLD_SETTINGS_SGLANG_WORK_DIR="$SGLANG_WORK_DIR" \
    NLD_SETTINGS_NEMO_SKILLS_DATA_DIR="$NEMO_SKILLS_DATA_DIR_ARG" \
    NLD_SETTINGS_HF_HOME="${HF_HOME_ARG:-$SGLANG_WORK_DIR/hf_cache}" \
    NLD_SETTINGS_HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$NEMO_SKILLS_DATA_DIR_ARG/hf_datasets_cache}" \
    NLD_SETTINGS_XDG_CACHE_HOME="${XDG_CACHE_HOME:-$NEMO_SKILLS_DATA_DIR_ARG/xdg_cache}" \
    NLD_SETTINGS_SGLANG_CACHE_DIR="${SGLANG_CACHE_ARG:-$SGLANG_WORK_DIR/sglang_cache}" \
    NLD_SETTINGS_KEEP_SERVER="$KEEP_SERVER" \
    "$SGLANG_PYTHON" - "$settings_file" "${ORIGINAL_ARGS[@]}" <<'PY'
import json
import os
import shlex
import sys
from datetime import datetime


def env(name: str) -> str:
    return os.environ.get(f"NLD_SETTINGS_{name}", "")


def maybe_int(value: str):
    if value == "":
        return None
    try:
        return int(value)
    except ValueError:
        return value


def maybe_float(value: str):
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return value


def as_bool(value: str) -> bool:
    return value.lower() in {"1", "true", "yes", "y"}


settings_file = sys.argv[1]
original_args = sys.argv[2:]
entrypoint = env("ENTRYPOINT")
settings = {
    "created_at": datetime.now().astimezone().isoformat(),
    "entrypoint": entrypoint,
    "original_args": original_args,
    "command": " ".join([shlex.quote(entrypoint), *[shlex.quote(arg) for arg in original_args]]),
    "benchmark": {
        "benchmarks": env("BENCHMARKS"),
        "tokens": maybe_int(env("TOKENS")),
        "temperature": maybe_float(env("TEMPERATURE")),
        "top_p": maybe_float(env("TOP_P")),
        "num_chunks": maybe_int(env("NUM_CHUNKS")),
        "max_samples": maybe_int(env("MAX_SAMPLES")),
        "quick_test": as_bool(env("QUICK_TEST")),
        "keep_thinking": as_bool(env("KEEP_THINKING")),
        "strip_thinking": as_bool(env("STRIP_THINKING")),
        "disable_thinking": as_bool(env("DISABLE_THINKING")),
        "math_prompt_config": env("MATH_PROMPT_CONFIG"),
        "mt_bench_max_tokens": maybe_int(env("MT_BENCH_MAX_TOKENS")),
        "alpaca_eval_max_tokens": maybe_int(env("ALPACA_EVAL_MAX_TOKENS")),
    },
    "sglang": {
        "mode": env("MODE"),
        "model": env("MODEL"),
        "served_model_name": env("SERVED_MODEL_NAME"),
        "gpu_devices": env("GPU_DEVICES"),
        "tp_size": maybe_int(env("TP_SIZE")),
        "tp_size_user_set": as_bool(env("TP_SIZE_USER_SET")),
        "port": maybe_int(env("PORT")),
        "port_user_set": as_bool(env("PORT_USER_SET")),
        "proxy_port": maybe_int(env("PROXY_PORT")),
        "proxy_port_user_set": as_bool(env("PROXY_PORT_USER_SET")),
        "max_running_requests": maybe_int(env("MAX_RUNNING_REQUESTS")),
        "client_concurrency": maybe_int(env("CLIENT_CONCURRENCY")),
        "gpu_memory_reserve_gb": maybe_float(env("GPU_MEMORY_RESERVE_GB")),
        "context_length": maybe_int(env("CONTEXT_LENGTH")),
        "context_length_user_set": as_bool(env("CONTEXT_LENGTH_USER_SET")),
        "mem_fraction": maybe_float(env("MEM_FRACTION")),
        "cuda_graph_bs": env("CUDA_GRAPH_BS"),
        "dtype": env("DTYPE"),
        "quantization": env("QUANTIZATION"),
        "block_size": maybe_int(env("BLOCK_SIZE")),
        "max_steps": maybe_int(env("MAX_STEPS")),
        "threshold": maybe_float(env("THRESHOLD")),
        "lora_path": env("LORA_PATH"),
        "lora_mode": env("LORA_MODE"),
        "extra_server_args": env("EXTRA_SERVER_ARGS"),
    },
    "judge": {
        "model": env("JUDGE_MODEL") or "dataset default",
        "server_address": env("JUDGE_SERVER_ADDRESS") or "dataset default",
        "server_type": env("JUDGE_SERVER_TYPE") or "dataset default",
        "concurrency": maybe_int(env("JUDGE_CONCURRENCY")),
        "skip_api_key_check": as_bool(env("SKIP_JUDGE_API_KEY_CHECK")),
        "openai_api_key_configured": as_bool(env("OPENAI_API_KEY_CONFIGURED")),
    },
    "paths": {
        "project_dir": env("PROJECT_DIR"),
        "pipeline": env("PIPELINE"),
        "output_path": env("OUTPUT_PATH"),
        "final_job_dir": env("FINAL_JOB_DIR"),
        "internal_job_dir": env("INTERNAL_JOB_DIR"),
        "eval_output_dir": env("EVAL_OUTPUT_DIR"),
        "runtime_dir": env("RUNTIME_DIR"),
        "checkpoint_name": env("CHECKPOINT_NAME"),
        "eval_dir_name": env("EVAL_DIR_NAME"),
        "sglang_src": env("SGLANG_SRC"),
        "sglang_work_dir": env("SGLANG_WORK_DIR"),
        "nemo_skills_data_dir": env("NEMO_SKILLS_DATA_DIR"),
        "hf_home": env("HF_HOME"),
        "hf_datasets_cache": env("HF_DATASETS_CACHE"),
        "xdg_cache_home": env("XDG_CACHE_HOME"),
        "sglang_cache_dir": env("SGLANG_CACHE_DIR"),
        "lora_path": env("LORA_PATH"),
    },
    "environment": {
        "sglang_python": env("SGLANG_PYTHON"),
        "eval_python": env("EVAL_PYTHON"),
        "keep_server": as_bool(env("KEEP_SERVER")),
        "pwd": os.getcwd(),
    },
}
with open(settings_file, "w", encoding="utf-8") as f:
    json.dump(settings, f, indent=2, ensure_ascii=False, sort_keys=True)
    f.write("\n")
print(f"Settings: {settings_file}")
PY
}

is_positive_int "$JUDGE_CONCURRENCY" || { echo "ERROR: --judge-concurrency must be positive" >&2; exit 1; }
is_positive_int "$MT_BENCH_MAX_TOKENS" || { echo "ERROR: --mt-bench-max-tokens must be positive" >&2; exit 1; }
is_positive_int "$ALPACA_EVAL_MAX_TOKENS" || { echo "ERROR: --alpaca-eval-max-tokens must be positive" >&2; exit 1; }
if { mt_bench_requested || alpaca_eval_requested; } && [[ -n "$JUDGE_SERVER_TYPE" && "$JUDGE_SERVER_TYPE" != "openai" ]]; then
    echo "ERROR: MT-Bench and AlpacaEval currently support only an OpenAI-compatible judge endpoint." >&2
    exit 1
fi

if mt_bench_requested && [[ "$CONTEXT_LENGTH_USER_SET" != "true" ]] \
    && (( CONTEXT_LENGTH < MT_BENCH_MAX_TOKENS + 2048 )); then
    CONTEXT_LENGTH=$((MT_BENCH_MAX_TOKENS + 2048))
    echo "INFO: auto-adjusted context length to $CONTEXT_LENGTH for MT-Bench."
fi

if alpaca_eval_requested && [[ "$CONTEXT_LENGTH_USER_SET" != "true" ]] \
    && (( CONTEXT_LENGTH < ALPACA_EVAL_MAX_TOKENS + 2048 )); then
    CONTEXT_LENGTH=$((ALPACA_EVAL_MAX_TOKENS + 2048))
    echo "INFO: auto-adjusted context length to $CONTEXT_LENGTH for AlpacaEval."
fi

if { standard_benchmark_requested || mt_bench_requested; } \
    && is_positive_int "$TOKENS" && is_positive_int "$CONTEXT_LENGTH"; then
    if (( TOKENS > CONTEXT_LENGTH )); then
        if [[ "$CONTEXT_LENGTH_USER_SET" == "true" ]]; then
            echo "ERROR: --tokens ($TOKENS) is larger than --context-length ($CONTEXT_LENGTH)." >&2
            echo "       Increase --context-length or reduce --tokens." >&2
            exit 1
        fi
        CONTEXT_LENGTH=$((TOKENS + 2048))
        echo "INFO: auto-adjusted context length to $CONTEXT_LENGTH for tokens=$TOKENS."
    fi
fi

if [[ "$TP_SIZE_USER_SET" != "true" ]]; then
    GPU_COUNT="$(infer_gpu_count "$GPU_DEVICES")"
    if is_positive_int "$GPU_COUNT"; then
        TP_SIZE="$GPU_COUNT"
    fi
fi

if ! is_nonnegative_number "$GPU_MEMORY_RESERVE_GB"; then
    echo "ERROR: --gpu-memory-reserve-gb must be a non-negative number, got: $GPU_MEMORY_RESERVE_GB" >&2
    exit 1
fi

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

if ! port_is_free "$PORT"; then
    if [[ "$PORT_USER_SET" == "true" ]]; then
        echo "ERROR: SGLang server port $PORT is already in use. Pass a free --port." >&2
        exit 1
    fi
    OLD_PORT="$PORT"
    PORT="$(find_free_port "$((PORT + 1))")"
    echo "INFO: default SGLang port $OLD_PORT is in use; using free port $PORT."
fi
if ! port_is_free "$PROXY_PORT"; then
    if [[ "$PROXY_PORT_USER_SET" == "true" ]]; then
        echo "ERROR: timing proxy port $PROXY_PORT is already in use. Pass a free --proxy-port." >&2
        exit 1
    fi
    OLD_PROXY_PORT="$PROXY_PORT"
    PROXY_PORT="$(find_free_port "$((PROXY_PORT + 1))")"
    echo "INFO: default proxy port $OLD_PROXY_PORT is in use; using free port $PROXY_PORT."
fi

if command -v realpath >/dev/null; then
    OUT_DIR="$(realpath -m "$OUT_DIR")"
    SGLANG_WORK_DIR="$(realpath -m "$SGLANG_WORK_DIR")"
    SGLANG_SRC="$(realpath -m "$SGLANG_SRC")"
    NEMO_SKILLS_DATA_DIR_ARG="$(realpath -m "$NEMO_SKILLS_DATA_DIR_ARG")"
fi

EVAL_DIR_NAME="eval_$(date +%Y%m%d_%H%M%S)"
CHECKPOINT_NAME="hf_base"
FINAL_JOB_DIR="$OUT_DIR/$EVAL_DIR_NAME"
if [[ -e "$FINAL_JOB_DIR" ]]; then
    suffix=1
    while [[ -e "$OUT_DIR/${EVAL_DIR_NAME}_$(printf '%02d' "$suffix")" ]]; do
        suffix=$((suffix + 1))
    done
    FINAL_JOB_DIR="$OUT_DIR/${EVAL_DIR_NAME}_$(printf '%02d' "$suffix")"
fi
INTERNAL_JOB_DIR="$OUT_DIR/.${EVAL_DIR_NAME}_work_$$"
EVAL_OUTPUT_DIR="$INTERNAL_JOB_DIR/results"
RUNTIME_DIR="$INTERNAL_JOB_DIR/sglang_runtime"

echo "================================================================"
echo "  SGLang NeMo-Skills eval"
echo "================================================================"
echo "  Mode:                $MODE"
echo "  Model:               $MODEL"
echo "  Served model name:   $SERVED_MODEL_NAME"
echo "  Benchmarks:          $BENCHMARKS"
echo "  Final output:        $FINAL_JOB_DIR"
echo "  Internal work dir:   $INTERNAL_JOB_DIR"
echo "  GPU devices:         $GPU_DEVICES"
echo "  TP size:             $TP_SIZE"
echo "  Server port:         $PORT"
echo "  Proxy port:          $PROXY_PORT"
echo "  Max running reqs:    $MAX_RUNNING_REQUESTS"
echo "  Client concurrency:  $CLIENT_CONCURRENCY"
echo "  GPU reserve GB:      $GPU_MEMORY_RESERVE_GB"
echo "  Num chunks:          $NUM_CHUNKS"
echo "  CUDA graph bs:       $CUDA_GRAPH_BS"
echo "  Tokens:              $TOKENS"
echo "  Context length:      $CONTEXT_LENGTH"
echo "  Python:              $SGLANG_PYTHON"
if arena_hard_requested; then
    echo "  Arena judge model:   ${JUDGE_MODEL:-gpt-4.1 (dataset default)}"
    echo "  Arena judge server:  ${JUDGE_SERVER_ADDRESS:-https://api.openai.com/v1 (dataset default)}"
    echo "  Arena judge type:    ${JUDGE_SERVER_TYPE:-openai (dataset default)}"
fi
if mt_bench_requested; then
    echo "  MT-Bench judge:      ${JUDGE_MODEL:-gpt-4.1}"
    echo "  MT-Bench max tokens: $MT_BENCH_MAX_TOKENS per turn"
    echo "  Judge concurrency:   $JUDGE_CONCURRENCY"
fi
if alpaca_eval_requested; then
    echo "  AlpacaEval judge:    ${JUDGE_MODEL:-gpt-4-1106-preview (official default)}"
    echo "  AlpacaEval max tok:  $ALPACA_EVAL_MAX_TOKENS"
    echo "  Judge concurrency:   $JUDGE_CONCURRENCY"
fi
echo "================================================================"

if [[ "$DRY_RUN" == "true" ]]; then
    echo "[dry-run] No server or evaluation started."
    exit 0
fi

mkdir -p \
    "$FINAL_JOB_DIR" \
    "$EVAL_OUTPUT_DIR" \
    "$RUNTIME_DIR" \
    "$NEMO_SKILLS_DATA_DIR_ARG" \
    "${HF_DATASETS_CACHE:-$NEMO_SKILLS_DATA_DIR_ARG/hf_datasets_cache}" \
    "${XDG_CACHE_HOME:-$NEMO_SKILLS_DATA_DIR_ARG/xdg_cache}"
write_settings_file

export SGLANG_MODE="$MODE"
export SGLANG_MODEL="$MODEL"
export SGLANG_MODEL_TAG="$SERVED_MODEL_NAME"
export SGLANG_GPU_DEVICES="$GPU_DEVICES"
export SGLANG_TP_SIZE="$TP_SIZE"
export SGLANG_PORT="$PORT"
export SGLANG_PROXY_PORT="$PROXY_PORT"
export SGLANG_MAX_RUNNING_REQUESTS="$MAX_RUNNING_REQUESTS"
export SGLANG_CLIENT_CONCURRENCY="$CLIENT_CONCURRENCY"
export SGLANG_GPU_MEMORY_RESERVE_GB="$GPU_MEMORY_RESERVE_GB"
export SGLANG_CONTEXT_LENGTH="$CONTEXT_LENGTH"
export SGLANG_MEM_FRACTION="$MEM_FRACTION"
export SGLANG_CUDA_GRAPH_BS="$CUDA_GRAPH_BS"
export SGLANG_DTYPE="$DTYPE"
export SGLANG_QUANTIZATION="$QUANTIZATION"
export SGLANG_LORA_PATH="$LORA_PATH"
export SGLANG_LORA_MODE="$LORA_MODE"
export SGLANG_BLOCK_SIZE="$BLOCK_SIZE"
export SGLANG_MAX_STEPS="$MAX_STEPS"
export SGLANG_THRESHOLD="$THRESHOLD"
export SGLANG_EXTRA_SERVER_ARGS="$EXTRA_SERVER_ARGS"
export SGLANG_KEEP_SERVER="$KEEP_SERVER"
export SGLANG_PYTHON="$SGLANG_PYTHON"
export EVAL_PYTHON="$EVAL_PYTHON"
export SGLANG_SRC="$SGLANG_SRC"
export SGLANG_WORK_DIR="$SGLANG_WORK_DIR"
export SGLANG_RUN_DIR="$RUNTIME_DIR"
export SGLANG_FINAL_OUTPUT_DIR="$FINAL_JOB_DIR"
export NEMO_SKILLS_DATA_DIR="$NEMO_SKILLS_DATA_DIR_ARG"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$NEMO_SKILLS_DATA_DIR_ARG/hf_datasets_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$NEMO_SKILLS_DATA_DIR_ARG/xdg_cache}"
export SGLANG_HF_HOME="${HF_HOME_ARG:-$SGLANG_WORK_DIR/hf_cache}"
export SGLANG_CACHE_DIR="${SGLANG_CACHE_ARG:-$SGLANG_WORK_DIR/sglang_cache}"

export SEQ_EVAL_BENCHMARK="$BENCHMARKS"
export SEQ_EVAL_EXPNAME="$CHECKPOINT_NAME"
export SEQ_EVAL_OUTPUT_DIR="$EVAL_OUTPUT_DIR"
export SEQ_EVAL_TOKENS_TO_GENERATE="$TOKENS"
export SEQ_EVAL_TEMPERATURE="$TEMPERATURE"
export SEQ_EVAL_TOP_P="$TOP_P"
export SEQ_EVAL_NUM_CHUNKS="$NUM_CHUNKS"
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
"$PIPELINE" || PIPELINE_STATUS=$?
if [[ "$PIPELINE_STATUS" != "0" ]]; then
    echo "ERROR: SGLang NeMo-Skills eval failed. Internal work dir kept for debugging: $INTERNAL_JOB_DIR" >&2
    echo "       Completed compact metrics, if any, are in: $FINAL_JOB_DIR" >&2
    exit "$PIPELINE_STATUS"
fi

metrics_filename_for_benchmark() {
    local bench_name="$1"
    bench_name="${bench_name//\//_}"
    bench_name="${bench_name//:/_}"
    bench_name="${bench_name//[[:space:]]/_}"
    echo "metrics_${bench_name}.json"
}

error_filename_for_benchmark() {
    local bench_name="$1"
    bench_name="${bench_name//\//_}"
    bench_name="${bench_name//:/_}"
    bench_name="${bench_name//[[:space:]]/_}"
    echo "error_${bench_name}.json"
}

BENCHMARK_ERRORS_RECORDED=0
IFS=',' read -ra FINAL_BENCH_ARRAY <<< "$BENCHMARKS"
for bench_spec in "${FINAL_BENCH_ARRAY[@]}"; do
    bench_name="${bench_spec%%:*}"
    expected_metrics="$FINAL_JOB_DIR/$(metrics_filename_for_benchmark "$bench_name")"
    expected_error="$FINAL_JOB_DIR/$(error_filename_for_benchmark "$bench_name")"
    if [[ -f "$expected_error" ]]; then
        BENCHMARK_ERRORS_RECORDED=1
        continue
    fi
    if [[ ! -f "$expected_metrics" ]]; then
        echo "ERROR: expected compact result file not found: $expected_metrics or $expected_error" >&2
        echo "       Internal work dir kept for debugging: $INTERNAL_JOB_DIR" >&2
        exit 1
    fi
done

if [[ "${KEEP_SERVER,,}" == "true" || "$KEEP_SERVER" == "1" ]]; then
    echo "INFO: --keep-server is set; internal work dir kept: $INTERNAL_JOB_DIR"
elif [[ "$BENCHMARK_ERRORS_RECORDED" != "0" && "${NLD_KEEP_FAILED_WORK_DIR:-false}" =~ ^(1|true|TRUE|yes|YES)$ ]]; then
    echo "INFO: benchmark failure detected; internal work dir kept for diagnostics: $INTERNAL_JOB_DIR"
else
    rm -rf "$INTERNAL_JOB_DIR"
fi

echo ""
echo "Completed SGLang NeMo-Skills eval."
echo "Final output dir: $FINAL_JOB_DIR"
if [[ "$BENCHMARK_ERRORS_RECORDED" != "0" ]]; then
    echo "One or more benchmarks failed; see error_<benchmark>.json files in the final output dir."
    if [[ "${NLD_FAIL_ON_BENCHMARK_ERROR:-false}" =~ ^(1|true|TRUE|yes|YES)$ ]]; then
        echo "ERROR: benchmark failure is fatal because NLD_FAIL_ON_BENCHMARK_ERROR is enabled." >&2
        exit 86
    fi
fi
