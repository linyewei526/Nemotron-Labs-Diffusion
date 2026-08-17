#!/bin/bash
#
# Local GPU pipeline: NeMo-Skills benchmark organization + SGLang backend.
#
# This script intentionally does not submit sbatch jobs and does not start the
# legacy xp/dlm_api worker/load-balancer. Run it directly in an environment that
# can import both SGLang and NeMo-Skills, or inside an existing SLURM allocation.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_DIR="$(cd "$ROOT_DIR/.." && pwd)"

EVAL_SCRIPT="$ROOT_DIR/nemo-skills/eval_dlm.py"
MT_BENCH_SCRIPT="$ROOT_DIR/mt_bench/eval_mt_bench.py"
ALPACA_EVAL_SCRIPT="$ROOT_DIR/alpaca_eval/eval_alpaca_eval.py"
DICTCONFIG_PATCH="$ROOT_DIR/nemo-skills/patch_dictconfig_serialization.py"
IFEVAL_PATH_PATCH="$ROOT_DIR/nemo-skills/patch_ifeval_google_research_path.py"
TIMING_PROXY_SCRIPT="$ROOT_DIR/sglang_eval/openai_timing_proxy.py"
MERGE_SCRIPT="$ROOT_DIR/sglang_eval/add_sglang_metrics_to_metrics.py"
GPU_MEMORY_RESERVER_SCRIPT="$ROOT_DIR/sglang_eval/gpu_memory_reserver.py"

SGLANG_PYTHON="${SGLANG_PYTHON:-python}"
EVAL_PYTHON="${EVAL_PYTHON:-$SGLANG_PYTHON}"
if [[ "$EVAL_PYTHON" == */* ]]; then
    EVAL_PYTHON_DIR="$(cd "$(dirname "$EVAL_PYTHON")" && pwd)"
else
    EVAL_PYTHON_DIR="$(dirname "$(command -v "$EVAL_PYTHON")")"
fi
export PATH="$EVAL_PYTHON_DIR:$PATH"
SGLANG_SRC="${SGLANG_SRC:-$PROJECT_DIR/sglang_dllm/src/sglang}"
SGLANG_WORK_DIR="${SGLANG_WORK_DIR:-$PROJECT_DIR/sglang_dllm}"
SGLANG_HF_HOME="${SGLANG_HF_HOME:-$SGLANG_WORK_DIR/hf_cache}"
SGLANG_CACHE_DIR="${SGLANG_CACHE_DIR:-$SGLANG_WORK_DIR/sglang_cache}"
NEMO_SKILLS_DATA_DIR="${NEMO_SKILLS_DATA_DIR:-}"
if [[ -n "$NEMO_SKILLS_DATA_DIR" ]]; then
    mkdir -p "$NEMO_SKILLS_DATA_DIR"
    NEMO_SKILLS_DATA_DIR="$(realpath -m "$NEMO_SKILLS_DATA_DIR")"
    export NEMO_SKILLS_DATA_DIR
    export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$NEMO_SKILLS_DATA_DIR/hf_datasets_cache}"
    export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$NEMO_SKILLS_DATA_DIR/xdg_cache}"
    mkdir -p "$HF_DATASETS_CACHE" "$XDG_CACHE_HOME"
fi

SGLANG_MODE="${SGLANG_MODE:-linearspec_lora}"
SGLANG_MODEL="${SGLANG_MODEL:-/data1/linyewei/models/Nemotron-Labs-Diffusion-8B}"
SGLANG_MODEL_TAG="${SGLANG_MODEL_TAG:-nemotron-labs-diffusion-8b}"
SGLANG_GPU_DEVICES="${SGLANG_GPU_DEVICES:-0}"
SGLANG_TP_SIZE="${SGLANG_TP_SIZE:-1}"
SGLANG_PORT="${SGLANG_PORT:-30000}"
SGLANG_PROXY_PORT="${SGLANG_PROXY_PORT:-31000}"
SGLANG_HOST="${SGLANG_HOST:-127.0.0.1}"
SGLANG_CONTEXT_LENGTH="${SGLANG_CONTEXT_LENGTH:-2048}"
SGLANG_MEM_FRACTION="${SGLANG_MEM_FRACTION:-0.55}"
SGLANG_MAX_RUNNING_REQUESTS="${SGLANG_MAX_RUNNING_REQUESTS:-1}"
SGLANG_CLIENT_CONCURRENCY="${SGLANG_CLIENT_CONCURRENCY:-1}"
SGLANG_CUDA_GRAPH_BS="${SGLANG_CUDA_GRAPH_BS:-1}"
SGLANG_DTYPE="${SGLANG_DTYPE:-bfloat16}"
SGLANG_ATTENTION_BACKEND="${SGLANG_ATTENTION_BACKEND:-flashinfer}"
SGLANG_SAMPLING_BACKEND="${SGLANG_SAMPLING_BACKEND:-flashinfer}"
SGLANG_QUANTIZATION="${SGLANG_QUANTIZATION:-}"
SGLANG_LORA_PATH="${SGLANG_LORA_PATH:-$SGLANG_WORK_DIR/linear_spec_lora}"
SGLANG_LORA_MODE="${SGLANG_LORA_MODE:-draft_only}"
SGLANG_BLOCK_SIZE="${SGLANG_BLOCK_SIZE:-}"
SGLANG_MAX_STEPS="${SGLANG_MAX_STEPS:-}"
SGLANG_THRESHOLD="${SGLANG_THRESHOLD:-}"
SGLANG_ALGO_TEMPERATURE="${SGLANG_ALGO_TEMPERATURE:-}"
SGLANG_EXTRA_SERVER_ARGS="${SGLANG_EXTRA_SERVER_ARGS:-}"
SGLANG_CONFIDENCE_TRACE_FILE="${SGLANG_CONFIDENCE_TRACE_FILE:-}"
SGLANG_LOW_CONFIDENCE_TRACE_FILE="${SGLANG_LOW_CONFIDENCE_TRACE_FILE:-}"
SGLANG_DRAFT_ALIGNMENT_TRACE_FILE="${SGLANG_DRAFT_ALIGNMENT_TRACE_FILE:-}"
SGLANG_KEEP_SERVER="${SGLANG_KEEP_SERVER:-false}"
SGLANG_GPU_MEMORY_RESERVE_GB="${SGLANG_GPU_MEMORY_RESERVE_GB:-0}"
NLD_GOOGLE_RESEARCH_DIR="${NLD_GOOGLE_RESEARCH_DIR:-${GOOGLE_RESEARCH_DIR:-/data1/linyewei/datasets/NLD/google-research}}"
export NLD_GOOGLE_RESEARCH_DIR

SEQ_EVAL_BENCHMARK="${SEQ_EVAL_BENCHMARK:-gsm8k:1,human-eval:1,mbpp:1,math-500:1,aime24:1,aime25:1,gpqa:1,mmlu:1,ifeval:1,livecodebench-cpp:1}"
SEQ_EVAL_EXPNAME="${SEQ_EVAL_EXPNAME:-hf_base}"
SEQ_EVAL_OUTPUT_DIR="${SEQ_EVAL_OUTPUT_DIR:-$PROJECT_DIR/sglang_eval_results}"
SEQ_EVAL_TOKENS_TO_GENERATE="${SEQ_EVAL_TOKENS_TO_GENERATE:-8192}"
SEQ_EVAL_TEMPERATURE="${SEQ_EVAL_TEMPERATURE:-0}"
SEQ_EVAL_TOP_P="${SEQ_EVAL_TOP_P:-0.95}"
SEQ_EVAL_NUM_CHUNKS="${SEQ_EVAL_NUM_CHUNKS:-$SGLANG_CLIENT_CONCURRENCY}"
SEQ_EVAL_MAX_SAMPLES="${SEQ_EVAL_MAX_SAMPLES:-}"
SEQ_EVAL_QUICK_TEST="${SEQ_EVAL_QUICK_TEST:-false}"
SEQ_EVAL_KEEP_THINKING="${SEQ_EVAL_KEEP_THINKING:-false}"
SEQ_EVAL_STRIP_THINKING="${SEQ_EVAL_STRIP_THINKING:-false}"
SEQ_EVAL_DISABLE_THINKING="${SEQ_EVAL_DISABLE_THINKING:-false}"
SEQ_EVAL_MATH_PROMPT_CONFIG="${SEQ_EVAL_MATH_PROMPT_CONFIG:-}"
SEQ_EVAL_JUDGE_MODEL="${SEQ_EVAL_JUDGE_MODEL:-}"
SEQ_EVAL_JUDGE_SERVER_ADDRESS="${SEQ_EVAL_JUDGE_SERVER_ADDRESS:-}"
SEQ_EVAL_JUDGE_SERVER_TYPE="${SEQ_EVAL_JUDGE_SERVER_TYPE:-}"
SEQ_EVAL_SKIP_JUDGE_API_KEY_CHECK="${SEQ_EVAL_SKIP_JUDGE_API_KEY_CHECK:-false}"
SEQ_EVAL_JUDGE_CONCURRENCY="${SEQ_EVAL_JUDGE_CONCURRENCY:-4}"
SEQ_EVAL_MT_BENCH_MAX_TOKENS="${SEQ_EVAL_MT_BENCH_MAX_TOKENS:-1024}"
SEQ_EVAL_MT_BENCH_CHAT_TEMPLATE_SHA256="${SEQ_EVAL_MT_BENCH_CHAT_TEMPLATE_SHA256:-}"
SEQ_EVAL_ALPACA_EVAL_MAX_TOKENS="${SEQ_EVAL_ALPACA_EVAL_MAX_TOKENS:-2048}"
SEQ_EVAL_ALPACA_EVAL_CHAT_TEMPLATE_SHA256="${SEQ_EVAL_ALPACA_EVAL_CHAT_TEMPLATE_SHA256:-}"
SEQ_EVAL_EXTRA_ARGS="${SEQ_EVAL_EXTRA_ARGS:-}"
SGLANG_FINAL_OUTPUT_DIR="${SGLANG_FINAL_OUTPUT_DIR:-}"

SGLANG_RUN_DIR="${SGLANG_RUN_DIR:-$SEQ_EVAL_OUTPUT_DIR/../sglang_runtime}"
mkdir -p "$SEQ_EVAL_OUTPUT_DIR" "$SGLANG_RUN_DIR" "$SGLANG_HF_HOME" "$SGLANG_CACHE_DIR"
SEQ_EVAL_OUTPUT_DIR="$(realpath -m "$SEQ_EVAL_OUTPUT_DIR")"
SGLANG_RUN_DIR="$(realpath -m "$SGLANG_RUN_DIR")"
if [[ -n "$NEMO_SKILLS_DATA_DIR" ]]; then
    MT_BENCH_DATA_DIR="${MT_BENCH_DATA_DIR:-$NEMO_SKILLS_DATA_DIR/mt-bench}"
    ALPACA_EVAL_DATA_DIR="${ALPACA_EVAL_DATA_DIR:-$NEMO_SKILLS_DATA_DIR/alpaca-eval}"
else
    MT_BENCH_DATA_DIR="${MT_BENCH_DATA_DIR:-$SEQ_EVAL_OUTPUT_DIR/mt-bench-data}"
    ALPACA_EVAL_DATA_DIR="${ALPACA_EVAL_DATA_DIR:-$SEQ_EVAL_OUTPUT_DIR/alpaca-eval-data}"
fi
if [[ -n "$SGLANG_FINAL_OUTPUT_DIR" ]]; then
    mkdir -p "$SGLANG_FINAL_OUTPUT_DIR"
    SGLANG_FINAL_OUTPUT_DIR="$(realpath -m "$SGLANG_FINAL_OUTPUT_DIR")"
fi

if ! "$EVAL_PYTHON" -c "import fastapi, httpx, uvicorn; from nemo_skills.pipeline.eval import eval" >/dev/null 2>&1; then
    echo "ERROR: EVAL_PYTHON cannot import one or more required packages: nemo_skills.pipeline.eval, fastapi, httpx, uvicorn." >&2
    echo "       Current EVAL_PYTHON: $EVAL_PYTHON" >&2
    echo "       Install NeMo-Skills into this environment or pass --eval-python pointing to a NeMo-Skills-capable Python." >&2
    exit 1
fi

ALGO_CONFIG="$SGLANG_RUN_DIR/sglang_algorithm.yaml"
DECODE_STATS_FILE="$SGLANG_RUN_DIR/sglang_decode_stats.jsonl"
TIMING_LOG="$SGLANG_RUN_DIR/sglang_timing.jsonl"
SERVER_LOG="$SGLANG_RUN_DIR/sglang_server.log"
PROXY_LOG="$SGLANG_RUN_DIR/sglang_timing_proxy.log"
GPU_MEMORY_RESERVER_LOG="$SGLANG_RUN_DIR/gpu_memory_reserver.log"
GPU_MEMORY_RESERVER_READY="$SGLANG_RUN_DIR/gpu_memory_reserver_ready.json"
NEMO_SKILLS_PACKAGE_DATASET_DIR="$("$EVAL_PYTHON" - <<'PY'
from pathlib import Path
import nemo_skills.dataset

print(Path(nemo_skills.dataset.__file__).resolve().parent)
PY
)"

reserve_memory_enabled() {
    "$SGLANG_PYTHON" - "$SGLANG_GPU_MEMORY_RESERVE_GB" <<'PY'
import math
import sys

try:
    value = float(sys.argv[1])
except (TypeError, ValueError):
    sys.exit(2)
sys.exit(0 if math.isfinite(value) and value > 0 else 1)
PY
}

benchmark_requested() {
    local wanted="$1"
    local bench_spec
    IFS=',' read -ra _BENCH_CHECK_ARRAY <<< "$SEQ_EVAL_BENCHMARK"
    for bench_spec in "${_BENCH_CHECK_ARRAY[@]}"; do
        bench_spec="${bench_spec//[[:space:]]/}"
        [[ "${bench_spec%%:*}" == "$wanted" ]] && return 0
    done
    return 1
}

ensure_ifeval_runtime() {
    benchmark_requested "ifeval" || return 0

    if [[ ! -d "$NLD_GOOGLE_RESEARCH_DIR/instruction_following_eval" ]]; then
        echo "ERROR: IFEval requested, but Google Research scorer was not found." >&2
        echo "       Expected: $NLD_GOOGLE_RESEARCH_DIR/instruction_following_eval" >&2
        echo "       Set NLD_GOOGLE_RESEARCH_DIR to a google-research checkout." >&2
        return 1
    fi

    export PYTHONPATH="$NLD_GOOGLE_RESEARCH_DIR:${PYTHONPATH:-}"

    if ! "$EVAL_PYTHON" -c "import langdetect, immutabledict, nltk; import instruction_following_eval.evaluation_main" >/dev/null 2>&1; then
        echo "  Installing/checking IFEval runtime deps (langdetect, immutabledict, nltk)..."
        "$EVAL_PYTHON" -m pip install langdetect immutabledict nltk
    fi

    "$EVAL_PYTHON" - <<'PY'
import nltk

for package in ("punkt_tab", "punkt"):
    try:
        nltk.data.find(f"tokenizers/{package}")
    except LookupError:
        nltk.download(package, quiet=True)
PY

    "$EVAL_PYTHON" -c "import instruction_following_eval.evaluation_main, langdetect, immutabledict, nltk" >/dev/null
    echo "  IFEval scorer: $NLD_GOOGLE_RESEARCH_DIR"
}

ensure_arena_hard_runtime() {
    local name
    for name in arena-hard arena-hard-v2; do
        benchmark_requested "$name" || continue
        if [[ ! -f "$NEMO_SKILLS_PACKAGE_DATASET_DIR/$name/__init__.py" \
            || ! -f "$NEMO_SKILLS_PACKAGE_DATASET_DIR/$name/prepare.py" ]]; then
            echo "ERROR: benchmark $name requires a NeMo-Skills installation with its Arena-Hard dataset adapter." >&2
            echo "       Missing under: $NEMO_SKILLS_PACKAGE_DATASET_DIR/$name" >&2
            echo "       The validated NLD environment uses nemo-skills 0.7.0." >&2
            return 1
        fi
    done
}

compact_metrics_filename() {
    local bench_name="$1"
    bench_name="${bench_name//\//_}"
    bench_name="${bench_name//:/_}"
    bench_name="${bench_name//[[:space:]]/_}"
    echo "metrics_${bench_name}.json"
}

compact_error_filename() {
    local bench_name="$1"
    bench_name="${bench_name//\//_}"
    bench_name="${bench_name//:/_}"
    bench_name="${bench_name//[[:space:]]/_}"
    echo "error_${bench_name}.json"
}

dataset_has_prepared_jsonl() {
    local dataset_dir="$1"
    [[ -d "$dataset_dir" ]] && find "$dataset_dir" -maxdepth 1 -type f -name '*.jsonl' -print -quit | grep -q .
}

sync_dataset_dir() {
    local src_dir="$1"
    local dst_dir="$2"
    [[ -d "$src_dir" ]] || return 1
    mkdir -p "$dst_dir"
    cp -a "$src_dir"/. "$dst_dir"/
}

write_compact_metrics() {
    local bench_name="$1"
    local metrics_json="$2"
    if [[ -z "$SGLANG_FINAL_OUTPUT_DIR" ]]; then
        return
    fi
    local dst_metrics="$SGLANG_FINAL_OUTPUT_DIR/$(compact_metrics_filename "$bench_name")"
    local metrics_dir artifact_dir artifact_name artifact
    metrics_dir="$(dirname "$metrics_json")"
    artifact_name="${bench_name//\//_}"
    artifact_name="${artifact_name//:/_}"
    artifact_name="${artifact_name//[[:space:]]/_}"
    artifact_dir="$SGLANG_FINAL_OUTPUT_DIR/artifacts/$artifact_name"
    cp "$metrics_json" "$dst_metrics"

    # The top-level entry deletes its internal work directory after a successful
    # run. Preserve benchmark-native answers/judgments/preflight files so the
    # compact metrics never point at artifacts that disappear during cleanup.
    mkdir -p "$artifact_dir"
    for artifact in "$metrics_dir"/*; do
        [[ -f "$artifact" ]] || continue
        [[ "$(basename "$artifact")" == "metrics.json" ]] && continue
        cp "$artifact" "$artifact_dir/"
    done

    "$EVAL_PYTHON" - "$dst_metrics" "$artifact_dir" <<'PY'
import json
import os
import sys
from pathlib import Path

metrics_path = Path(sys.argv[1])
artifact_dir = Path(sys.argv[2]).resolve()
payload = json.loads(metrics_path.read_text(encoding="utf-8"))
artifacts = payload.get("artifacts")
if isinstance(artifacts, dict):
    for key, original in list(artifacts.items()):
        if key == "data_dir" or not isinstance(original, str):
            continue
        compact_artifact = artifact_dir / Path(original).name
        if compact_artifact.is_file():
            artifacts[key] = str(compact_artifact)

temporary = metrics_path.with_name(metrics_path.name + ".tmp")
temporary.write_text(
    json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
os.replace(temporary, metrics_path)
PY
    echo "  Wrote compact metrics: $dst_metrics"
}

write_compact_error() {
    local bench_name="$1"
    local bench_spec="$2"
    local stage="$3"
    local exit_code="$4"
    local message="$5"
    local bench_log="$6"
    local metrics_json="${7:-}"
    if [[ -z "$SGLANG_FINAL_OUTPUT_DIR" ]]; then
        return
    fi
    local dst_error="$SGLANG_FINAL_OUTPUT_DIR/$(compact_error_filename "$bench_name")"
    NLD_ERROR_BENCHMARK="$bench_name" \
    NLD_ERROR_BENCHMARK_SPEC="$bench_spec" \
    NLD_ERROR_STAGE="$stage" \
    NLD_ERROR_EXIT_CODE="$exit_code" \
    NLD_ERROR_MESSAGE="$message" \
    NLD_ERROR_BENCH_LOG="$bench_log" \
    NLD_ERROR_METRICS_JSON="$metrics_json" \
    "$EVAL_PYTHON" - "$dst_error" <<'PY'
import json
import os
import sys
from datetime import datetime
from pathlib import Path


def env(name: str) -> str:
    return os.environ.get(f"NLD_ERROR_{name}", "")


def log_tail(path: str, max_lines: int = 80) -> list[str]:
    if not path:
        return []
    log_path = Path(path)
    if not log_path.is_file():
        return []
    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    return [line.rstrip("\n") for line in lines[-max_lines:]]


exit_code_raw = env("EXIT_CODE")
try:
    exit_code = int(exit_code_raw) if exit_code_raw != "" else None
except ValueError:
    exit_code = exit_code_raw

payload = {
    "status": "failed",
    "benchmark": env("BENCHMARK"),
    "benchmark_spec": env("BENCHMARK_SPEC"),
    "stage": env("STAGE"),
    "exit_code": exit_code,
    "message": env("MESSAGE"),
    "created_at": datetime.now().astimezone().isoformat(),
    "internal_benchmark_log": env("BENCH_LOG"),
    "internal_metrics_json": env("METRICS_JSON"),
    "log_tail": log_tail(env("BENCH_LOG")),
}
with open(sys.argv[1], "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2, ensure_ascii=False, sort_keys=True)
    f.write("\n")
PY
    echo "  Wrote compact error: $dst_error"
}

write_algorithm_config() {
    : > "$DECODE_STATS_FILE"
    case "$SGLANG_MODE" in
        linearspec|linearspec_base)
            {
                echo "algorithm: LinearSpec"
                echo "causal_context: true"
                if [[ -n "$SGLANG_BLOCK_SIZE" ]]; then
                    echo "block_size: $SGLANG_BLOCK_SIZE"
                fi
                echo "stats_file: $DECODE_STATS_FILE"
                if [[ -n "$SGLANG_CONFIDENCE_TRACE_FILE" ]]; then
                    echo "confidence_trace_file: $SGLANG_CONFIDENCE_TRACE_FILE"
                fi
                if [[ -n "$SGLANG_LOW_CONFIDENCE_TRACE_FILE" ]]; then
                    echo "low_confidence_trace_file: $SGLANG_LOW_CONFIDENCE_TRACE_FILE"
                fi
                if [[ -n "$SGLANG_DRAFT_ALIGNMENT_TRACE_FILE" ]]; then
                    echo "draft_alignment_trace_file: $SGLANG_DRAFT_ALIGNMENT_TRACE_FILE"
                fi
            } > "$ALGO_CONFIG"
            ;;
        linearspec_lora|linear_spec_lora)
            {
                echo "algorithm: LinearSpec"
                echo "causal_context: true"
                if [[ -n "$SGLANG_BLOCK_SIZE" ]]; then
                    echo "block_size: $SGLANG_BLOCK_SIZE"
                fi
                echo "lora_path: $SGLANG_LORA_PATH"
                echo "lora_mode: $SGLANG_LORA_MODE"
                echo "stats_file: $DECODE_STATS_FILE"
                if [[ -n "$SGLANG_CONFIDENCE_TRACE_FILE" ]]; then
                    echo "confidence_trace_file: $SGLANG_CONFIDENCE_TRACE_FILE"
                fi
                if [[ -n "$SGLANG_LOW_CONFIDENCE_TRACE_FILE" ]]; then
                    echo "low_confidence_trace_file: $SGLANG_LOW_CONFIDENCE_TRACE_FILE"
                fi
                if [[ -n "$SGLANG_DRAFT_ALIGNMENT_TRACE_FILE" ]]; then
                    echo "draft_alignment_trace_file: $SGLANG_DRAFT_ALIGNMENT_TRACE_FILE"
                fi
            } > "$ALGO_CONFIG"
            ;;
        fastdiffuser|dlm)
            {
                echo "algorithm: FastDiffuser"
                echo "causal_context: true"
                if [[ -n "$SGLANG_BLOCK_SIZE" ]]; then
                    echo "block_size: $SGLANG_BLOCK_SIZE"
                fi
                if [[ -n "$SGLANG_MAX_STEPS" ]]; then
                    echo "max_steps: $SGLANG_MAX_STEPS"
                fi
                if [[ -n "$SGLANG_THRESHOLD" ]]; then
                    echo "threshold: $SGLANG_THRESHOLD"
                fi
                if [[ -n "$SGLANG_ALGO_TEMPERATURE" ]]; then
                    echo "temperature: $SGLANG_ALGO_TEMPERATURE"
                fi
                echo "stats_file: $DECODE_STATS_FILE"
            } > "$ALGO_CONFIG"
            ;;
        ar)
            : > "$ALGO_CONFIG"
            ;;
        *)
            echo "ERROR: unknown SGLANG_MODE=$SGLANG_MODE" >&2
            exit 1
            ;;
    esac
}

wait_for_health() {
    local url="$1"
    local name="$2"
    local log_file="$3"
    for retry in $(seq 1 180); do
        if ! kill -0 "${4:-0}" 2>/dev/null; then
            echo "ERROR: $name process exited while waiting for health. Log tail:" >&2
            tail -80 "$log_file" >&2 || true
            return 1
        fi
        if curl -sf --max-time 2 "$url" >/dev/null 2>&1; then
            echo "  $name healthy after $retry checks"
            return 0
        fi
        sleep 2
    done
    echo "ERROR: $name did not become healthy. Log tail:" >&2
    tail -120 "$log_file" >&2 || true
    return 1
}

wait_for_ready_file() {
    local ready_file="$1"
    local name="$2"
    local log_file="$3"
    local pid="$4"
    for retry in $(seq 1 120); do
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "ERROR: $name process exited while waiting for readiness. Log tail:" >&2
            tail -80 "$log_file" >&2 || true
            return 1
        fi
        if [[ -s "$ready_file" ]]; then
            echo "  $name ready after $retry checks"
            return 0
        fi
        sleep 1
    done
    echo "ERROR: $name did not become ready. Log tail:" >&2
    tail -120 "$log_file" >&2 || true
    return 1
}

cleanup() {
    if [[ "${SGLANG_KEEP_SERVER,,}" == "true" || "$SGLANG_KEEP_SERVER" == "1" ]]; then
        echo "[cleanup] keep-server requested; SGLang PID=${SGLANG_PID:-}, proxy PID=${PROXY_PID:-}, memory reserve PID=${GPU_MEMORY_RESERVER_PID:-}"
        return
    fi
    [[ -n "${PROXY_PID:-}" ]] && kill "$PROXY_PID" 2>/dev/null || true
    [[ -n "${SGLANG_PID:-}" ]] && kill "$SGLANG_PID" 2>/dev/null || true
    [[ -n "${GPU_MEMORY_RESERVER_PID:-}" ]] && kill "$GPU_MEMORY_RESERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

write_algorithm_config

echo "=============================================================="
echo "[sglang-eval] NeMo-Skills benchmark organization + SGLang backend"
echo "=============================================================="
echo "  Mode:              $SGLANG_MODE"
echo "  Model:             $SGLANG_MODEL"
echo "  Benchmarks:        $SEQ_EVAL_BENCHMARK"
echo "  Output dir:        $SEQ_EVAL_OUTPUT_DIR"
echo "  Runtime dir:       $SGLANG_RUN_DIR"
echo "  GPU devices:       $SGLANG_GPU_DEVICES"
echo "  TP size:           $SGLANG_TP_SIZE"
echo "  Max running reqs:  $SGLANG_MAX_RUNNING_REQUESTS"
echo "  Client concurrency:$SGLANG_CLIENT_CONCURRENCY"
echo "  CUDA graph bs:     $SGLANG_CUDA_GRAPH_BS"
echo "  Context length:    $SGLANG_CONTEXT_LENGTH"
echo "  Tokens to gen:     $SEQ_EVAL_TOKENS_TO_GENERATE"
echo "  GPU reserve GB:    $SGLANG_GPU_MEMORY_RESERVE_GB"
[[ -n "$NEMO_SKILLS_DATA_DIR" ]] && echo "  Data cache dir:    $NEMO_SKILLS_DATA_DIR"
echo "  Algorithm config:  $ALGO_CONFIG"
[[ -n "$SGLANG_CONFIDENCE_TRACE_FILE" ]] && echo "  Confidence trace:  $SGLANG_CONFIDENCE_TRACE_FILE"
[[ -n "$SGLANG_LOW_CONFIDENCE_TRACE_FILE" ]] && echo "  Low-conf trace:    $SGLANG_LOW_CONFIDENCE_TRACE_FILE"
[[ -n "$SGLANG_DRAFT_ALIGNMENT_TRACE_FILE" ]] && echo "  Draft align trace: $SGLANG_DRAFT_ALIGNMENT_TRACE_FILE"
[[ -n "$SGLANG_FINAL_OUTPUT_DIR" ]] && echo "  Final output dir:  $SGLANG_FINAL_OUTPUT_DIR"
echo "=============================================================="

SGLANG_ARGS=(
    -m sglang.launch_server
    --model-path "$SGLANG_MODEL"
    --served-model-name "$SGLANG_MODEL_TAG"
    --trust-remote-code
    --dtype "$SGLANG_DTYPE"
    --tensor-parallel-size "$SGLANG_TP_SIZE"
    --mem-fraction-static "$SGLANG_MEM_FRACTION"
    --max-running-requests "$SGLANG_MAX_RUNNING_REQUESTS"
    --attention-backend "$SGLANG_ATTENTION_BACKEND"
    --sampling-backend "$SGLANG_SAMPLING_BACKEND"
    --cuda-graph-bs
)
read -r -a CUDA_GRAPH_BS_ARRAY <<< "$SGLANG_CUDA_GRAPH_BS"
SGLANG_ARGS+=("${CUDA_GRAPH_BS_ARRAY[@]}")
SGLANG_ARGS+=(
    --context-length "$SGLANG_CONTEXT_LENGTH"
    --host "0.0.0.0"
    --port "$SGLANG_PORT"
)
if [[ -n "$SGLANG_QUANTIZATION" ]]; then
    SGLANG_ARGS+=(--quantization "$SGLANG_QUANTIZATION")
fi
case "$SGLANG_MODE" in
    ar)
        SGLANG_ARGS+=(--json-model-override-args '{"ar_mode": true}')
        ;;
    *)
        SGLANG_ARGS+=(--dllm-algorithm "$(grep '^algorithm:' "$ALGO_CONFIG" | awk '{print $2}')" --dllm-algorithm-config "$ALGO_CONFIG")
        ;;
esac
if [[ -n "$SGLANG_EXTRA_SERVER_ARGS" ]]; then
    read -r -a EXTRA_SERVER_ARGS_ARRAY <<< "$SGLANG_EXTRA_SERVER_ARGS"
    SGLANG_ARGS+=("${EXTRA_SERVER_ARGS_ARRAY[@]}")
fi

if reserve_memory_enabled; then
    echo "[0/5] Reserving GPU memory..."
    rm -f "$GPU_MEMORY_RESERVER_READY"
    (
        export CUDA_VISIBLE_DEVICES="$SGLANG_GPU_DEVICES"
        exec "$SGLANG_PYTHON" -u "$GPU_MEMORY_RESERVER_SCRIPT" \
            --gb "$SGLANG_GPU_MEMORY_RESERVE_GB" \
            --ready-file "$GPU_MEMORY_RESERVER_READY"
    ) > "$GPU_MEMORY_RESERVER_LOG" 2>&1 &
    GPU_MEMORY_RESERVER_PID=$!
    wait_for_ready_file "$GPU_MEMORY_RESERVER_READY" "GPU memory reserver" "$GPU_MEMORY_RESERVER_LOG" "$GPU_MEMORY_RESERVER_PID"
fi

ensure_arena_hard_runtime

echo "[1/5] Starting SGLang server..."
(
    export CUDA_VISIBLE_DEVICES="$SGLANG_GPU_DEVICES"
    export HF_HOME="$SGLANG_HF_HOME"
    export SGLANG_CACHE_DIR="$SGLANG_CACHE_DIR"
    export PYTHONPATH="$SGLANG_SRC/python:${PYTHONPATH:-}"
    exec "$SGLANG_PYTHON" "${SGLANG_ARGS[@]}"
) > "$SERVER_LOG" 2>&1 &
SGLANG_PID=$!
wait_for_health "http://127.0.0.1:$SGLANG_PORT/health" "SGLang" "$SERVER_LOG" "$SGLANG_PID"

echo "[2/5] Starting timing proxy..."
: > "$TIMING_LOG"
"$EVAL_PYTHON" -u "$TIMING_PROXY_SCRIPT" \
    --host "$SGLANG_HOST" \
    --port "$SGLANG_PROXY_PORT" \
    --upstream-base-url "http://127.0.0.1:$SGLANG_PORT/v1" \
    --timing-log "$TIMING_LOG" \
    --max-concurrency "$SGLANG_CLIENT_CONCURRENCY" > "$PROXY_LOG" 2>&1 &
PROXY_PID=$!
wait_for_health "http://127.0.0.1:$SGLANG_PROXY_PORT/health" "timing proxy" "$PROXY_LOG" "$PROXY_PID"

echo "[3/5] Applying NeMo-Skills compatibility patches if available..."
"$EVAL_PYTHON" "$DICTCONFIG_PATCH" >/dev/null 2>&1 || true
"$EVAL_PYTHON" "$IFEVAL_PATH_PATCH" >/dev/null 2>&1 || true
ensure_ifeval_runtime

IFS=',' read -ra PREP_BENCHES <<< "$SEQ_EVAL_BENCHMARK"
for bench_spec in "${PREP_BENCHES[@]}"; do
    bench_spec="${bench_spec//[[:space:]]/}"
    [[ -z "$bench_spec" ]] && continue
    bench_name="${bench_spec%%:*}"
    if [[ "$bench_name" == "mt-bench" ]]; then
        "$EVAL_PYTHON" "$MT_BENCH_SCRIPT" \
            --data-dir "$MT_BENCH_DATA_DIR" \
            --prepare-only
        continue
    fi
    if [[ "$bench_name" == "alpaca-eval" ]]; then
        "$EVAL_PYTHON" "$ALPACA_EVAL_SCRIPT" \
            --data-dir "$ALPACA_EVAL_DATA_DIR" \
            --prepare-only
        continue
    fi
    package_dataset_dir="$NEMO_SKILLS_PACKAGE_DATASET_DIR/$bench_name"
    cached_dataset_dir="${NEMO_SKILLS_DATA_DIR:+$NEMO_SKILLS_DATA_DIR/$bench_name}"

    if [[ -n "$NEMO_SKILLS_DATA_DIR" ]] && dataset_has_prepared_jsonl "$cached_dataset_dir"; then
        if [[ ( "$bench_name" == "arena-hard" || "$bench_name" == "arena-hard-v2" ) \
            && ! -f "$cached_dataset_dir/test.jsonl" ]]; then
            echo "  Ignoring incomplete Arena-Hard cache without test.jsonl: $cached_dataset_dir"
        else
            echo "  Restoring cached data for $bench_name from $cached_dataset_dir"
            sync_dataset_dir "$cached_dataset_dir" "$package_dataset_dir" || true
            continue
        fi
    fi

    if dataset_has_prepared_jsonl "$package_dataset_dir"; then
        if [[ ( "$bench_name" == "arena-hard" || "$bench_name" == "arena-hard-v2" ) \
            && ! -f "$package_dataset_dir/test.jsonl" ]]; then
            echo "  Existing Arena-Hard package data is incomplete; regenerating $bench_name"
        else
            echo "  Reusing existing prepared data for $bench_name"
            if [[ -n "$NEMO_SKILLS_DATA_DIR" ]]; then
                sync_dataset_dir "$package_dataset_dir" "$cached_dataset_dir" || true
            fi
            continue
        fi
    fi

    echo "  Preparing data for $bench_name..."
    prepare_status=0
    "$EVAL_PYTHON" -m nemo_skills.dataset.prepare "$bench_name" --parallelism 20 --retries 3 || prepare_status=$?
    if [[ "$prepare_status" != "0" && ( "$bench_name" == "arena-hard" || "$bench_name" == "arena-hard-v2" ) ]]; then
        echo "ERROR: failed to prepare required Arena-Hard data for $bench_name." >&2
        exit "$prepare_status"
    fi
    if [[ -n "$NEMO_SKILLS_DATA_DIR" ]] && dataset_has_prepared_jsonl "$package_dataset_dir"; then
        echo "  Caching prepared data for $bench_name into $cached_dataset_dir"
        sync_dataset_dir "$package_dataset_dir" "$cached_dataset_dir" || true
    fi
    if [[ ( "$bench_name" == "arena-hard" || "$bench_name" == "arena-hard-v2" ) \
        && ! -f "$package_dataset_dir/test.jsonl" ]]; then
        echo "ERROR: Arena-Hard preparation did not produce $package_dataset_dir/test.jsonl" >&2
        exit 1
    fi
done

if [[ -n "$SGLANG_DRAFT_ALIGNMENT_TRACE_FILE" ]]; then
    : > "$SGLANG_DRAFT_ALIGNMENT_TRACE_FILE"
    echo "  Cleared draft-alignment startup trace records: $SGLANG_DRAFT_ALIGNMENT_TRACE_FILE"
fi

echo "[4/5] Running benchmark evaluation through timing proxy..."
PIPELINE_FAILED=0
IFS=',' read -ra BENCH_ARRAY <<< "$SEQ_EVAL_BENCHMARK"
for bench_spec in "${BENCH_ARRAY[@]}"; do
    bench_spec="${bench_spec//[[:space:]]/}"
    [[ -z "$bench_spec" ]] && continue
    bench_name="${bench_spec%%:*}"
    echo ""
    echo "--- benchmark: $bench_spec ---"
    : > "$DECODE_STATS_FILE"
    : > "$TIMING_LOG"
    eval_dir="$SEQ_EVAL_OUTPUT_DIR/eval-results/$bench_name"
    mkdir -p "$eval_dir"
    bench_log="$eval_dir/sglang_benchmark.log"
    : > "$bench_log"

    if [[ "$bench_name" == "mt-bench" ]]; then
        if [[ -n "$SEQ_EVAL_JUDGE_SERVER_TYPE" && "$SEQ_EVAL_JUDGE_SERVER_TYPE" != "openai" ]]; then
            echo "ERROR: MT-Bench currently supports only an OpenAI-compatible judge endpoint." >&2
            exit 1
        fi
        EVAL_ARGS=(
            "$MT_BENCH_SCRIPT"
            --candidate-server-address "http://127.0.0.1:$SGLANG_PROXY_PORT/v1"
            --candidate-model "$SGLANG_MODEL_TAG"
            --candidate-tokenizer "$SGLANG_MODEL"
            --candidate-concurrency "$SGLANG_CLIENT_CONCURRENCY"
            --judge-concurrency "$SEQ_EVAL_JUDGE_CONCURRENCY"
            --data-dir "$MT_BENCH_DATA_DIR"
            --output-dir "$eval_dir"
            --max-tokens "$SEQ_EVAL_MT_BENCH_MAX_TOKENS"
            --resume
        )
        [[ -n "$SEQ_EVAL_MAX_SAMPLES" ]] && EVAL_ARGS+=(--max-samples "$SEQ_EVAL_MAX_SAMPLES")
        if [[ -z "$SEQ_EVAL_MAX_SAMPLES" && ( "${SEQ_EVAL_QUICK_TEST,,}" == "true" || "$SEQ_EVAL_QUICK_TEST" == "1" ) ]]; then
            EVAL_ARGS+=(--max-samples 10)
        fi
        [[ -n "$SEQ_EVAL_JUDGE_MODEL" ]] && EVAL_ARGS+=(--judge-model "$SEQ_EVAL_JUDGE_MODEL")
        [[ -n "$SEQ_EVAL_JUDGE_SERVER_ADDRESS" ]] && EVAL_ARGS+=(--judge-server-address "$SEQ_EVAL_JUDGE_SERVER_ADDRESS")
        [[ -n "$SEQ_EVAL_MT_BENCH_CHAT_TEMPLATE_SHA256" ]] && EVAL_ARGS+=(--expected-chat-template-sha256 "$SEQ_EVAL_MT_BENCH_CHAT_TEMPLATE_SHA256")
        if [[ "${SEQ_EVAL_STRIP_THINKING,,}" == "true" || "$SEQ_EVAL_STRIP_THINKING" == "1" ]]; then
            EVAL_ARGS+=(--strip-thinking)
        fi
        if [[ "${SEQ_EVAL_SKIP_JUDGE_API_KEY_CHECK,,}" == "true" || "$SEQ_EVAL_SKIP_JUDGE_API_KEY_CHECK" == "1" ]]; then
            EVAL_ARGS+=(--skip-judge-api-key-check)
        fi
    elif [[ "$bench_name" == "alpaca-eval" ]]; then
        if [[ -n "$SEQ_EVAL_JUDGE_SERVER_TYPE" && "$SEQ_EVAL_JUDGE_SERVER_TYPE" != "openai" ]]; then
            echo "ERROR: AlpacaEval currently supports only an OpenAI-compatible judge endpoint." >&2
            exit 1
        fi
        EVAL_ARGS=(
            "$ALPACA_EVAL_SCRIPT"
            --candidate-server-address "http://127.0.0.1:$SGLANG_PROXY_PORT/v1"
            --candidate-model "$SGLANG_MODEL_TAG"
            --candidate-tokenizer "$SGLANG_MODEL"
            --candidate-concurrency "$SGLANG_CLIENT_CONCURRENCY"
            --judge-concurrency "$SEQ_EVAL_JUDGE_CONCURRENCY"
            --data-dir "$ALPACA_EVAL_DATA_DIR"
            --output-dir "$eval_dir"
            --max-tokens "$SEQ_EVAL_ALPACA_EVAL_MAX_TOKENS"
            --resume
        )
        [[ -n "$SEQ_EVAL_MAX_SAMPLES" ]] && EVAL_ARGS+=(--max-samples "$SEQ_EVAL_MAX_SAMPLES")
        if [[ -z "$SEQ_EVAL_MAX_SAMPLES" && ( "${SEQ_EVAL_QUICK_TEST,,}" == "true" || "$SEQ_EVAL_QUICK_TEST" == "1" ) ]]; then
            EVAL_ARGS+=(--max-samples 10)
        fi
        [[ -n "$SEQ_EVAL_JUDGE_MODEL" ]] && EVAL_ARGS+=(--judge-model "$SEQ_EVAL_JUDGE_MODEL")
        [[ -n "$SEQ_EVAL_JUDGE_SERVER_ADDRESS" ]] && EVAL_ARGS+=(--judge-server-address "$SEQ_EVAL_JUDGE_SERVER_ADDRESS")
        [[ -n "$SEQ_EVAL_ALPACA_EVAL_CHAT_TEMPLATE_SHA256" ]] && EVAL_ARGS+=(--expected-chat-template-sha256 "$SEQ_EVAL_ALPACA_EVAL_CHAT_TEMPLATE_SHA256")
        if [[ "${SEQ_EVAL_STRIP_THINKING,,}" == "true" || "$SEQ_EVAL_STRIP_THINKING" == "1" ]]; then
            EVAL_ARGS+=(--strip-thinking)
        fi
        if [[ "${SEQ_EVAL_SKIP_JUDGE_API_KEY_CHECK,,}" == "true" || "$SEQ_EVAL_SKIP_JUDGE_API_KEY_CHECK" == "1" ]]; then
            EVAL_ARGS+=(--skip-judge-api-key-check)
        fi
    else
        EVAL_ARGS=(
            "$EVAL_SCRIPT"
            --server-address "http://127.0.0.1:$SGLANG_PROXY_PORT/v1"
            --benchmark "$bench_spec"
            --output-dir "$SEQ_EVAL_OUTPUT_DIR"
            --expname "$SEQ_EVAL_EXPNAME"
            --model "$SGLANG_MODEL_TAG"
            --tokens-to-generate "$SEQ_EVAL_TOKENS_TO_GENERATE"
            --temperature "$SEQ_EVAL_TEMPERATURE"
            --top-p "$SEQ_EVAL_TOP_P"
            --num-chunks "$SEQ_EVAL_NUM_CHUNKS"
            --max-concurrent-requests "$SGLANG_CLIENT_CONCURRENCY"
            --generation-algorithm nemotron
            --no-extra-body
        )
        [[ -n "$SEQ_EVAL_MAX_SAMPLES" ]] && EVAL_ARGS+=(--max-samples "$SEQ_EVAL_MAX_SAMPLES")
        if [[ "${SEQ_EVAL_QUICK_TEST,,}" == "true" || "$SEQ_EVAL_QUICK_TEST" == "1" ]]; then EVAL_ARGS+=(--quick-test); fi
        if [[ "${SEQ_EVAL_KEEP_THINKING,,}" == "true" || "$SEQ_EVAL_KEEP_THINKING" == "1" ]]; then EVAL_ARGS+=(--keep-thinking); fi
        if [[ "${SEQ_EVAL_STRIP_THINKING,,}" == "true" || "$SEQ_EVAL_STRIP_THINKING" == "1" ]]; then EVAL_ARGS+=(--strip-thinking); fi
        if [[ "${SEQ_EVAL_DISABLE_THINKING,,}" == "true" || "$SEQ_EVAL_DISABLE_THINKING" == "1" ]]; then EVAL_ARGS+=(--disable-thinking); fi
        [[ -n "$SEQ_EVAL_MATH_PROMPT_CONFIG" ]] && EVAL_ARGS+=(--math-prompt-config "$SEQ_EVAL_MATH_PROMPT_CONFIG")
        [[ -n "$SEQ_EVAL_JUDGE_MODEL" ]] && EVAL_ARGS+=(--judge-model "$SEQ_EVAL_JUDGE_MODEL")
        [[ -n "$SEQ_EVAL_JUDGE_SERVER_ADDRESS" ]] && EVAL_ARGS+=(--judge-server-address "$SEQ_EVAL_JUDGE_SERVER_ADDRESS")
        [[ -n "$SEQ_EVAL_JUDGE_SERVER_TYPE" ]] && EVAL_ARGS+=(--judge-server-type "$SEQ_EVAL_JUDGE_SERVER_TYPE")
        if [[ "${SEQ_EVAL_SKIP_JUDGE_API_KEY_CHECK,,}" == "true" || "$SEQ_EVAL_SKIP_JUDGE_API_KEY_CHECK" == "1" ]]; then EVAL_ARGS+=(--skip-judge-api-key-check); fi
        if [[ -n "$SEQ_EVAL_EXTRA_ARGS" ]]; then
            read -r -a EXTRA_EVAL_ARGS_ARRAY <<< "$SEQ_EVAL_EXTRA_ARGS"
            EVAL_ARGS+=("${EXTRA_EVAL_ARGS_ARRAY[@]}")
        fi
    fi

    bench_start="$(date +%s.%N)"
    set +e
    "$EVAL_PYTHON" "${EVAL_ARGS[@]}" 2>&1 | tee "$bench_log"
    eval_status=${PIPESTATUS[0]}
    set -e
    bench_end="$(date +%s.%N)"
    wall_time="$("$EVAL_PYTHON" - <<PY
start = float("$bench_start")
end = float("$bench_end")
print(f"{end - start:.6f}")
PY
)"

    metrics_json="$eval_dir/metrics.json"
    if [[ "$eval_status" != "0" ]]; then
        echo "ERROR: benchmark $bench_spec failed during evaluation with exit code $eval_status. Continuing." >&2
        write_compact_error "$bench_name" "$bench_spec" "benchmark_eval" "$eval_status" "Benchmark evaluation command failed." "$bench_log" "$metrics_json"
        PIPELINE_FAILED=1
        continue
    fi

    if [[ ! -f "$metrics_json" ]]; then
        echo "ERROR: metrics file not found for $bench_name: $metrics_json. Continuing." >&2
        write_compact_error "$bench_name" "$bench_spec" "metrics_missing" "0" "Benchmark evaluation completed but metrics.json was not produced." "$bench_log" "$metrics_json"
        PIPELINE_FAILED=1
        continue
    fi

    cp "$DECODE_STATS_FILE" "$eval_dir/sglang_decode_stats.jsonl" 2>/dev/null || true
    cp "$TIMING_LOG" "$eval_dir/sglang_timing.jsonl" 2>/dev/null || true
    MERGE_ARGS=(
        "$MERGE_SCRIPT"
        --metrics-json "$metrics_json"
        --eval-results-dir "$eval_dir"
        --decode-stats-file "$DECODE_STATS_FILE"
        --timing-log "$TIMING_LOG"
        --benchmark "$bench_name"
        --wall-time-s "$wall_time"
    )
    if [[ "$SGLANG_MODE" == "ar" ]]; then
        MERGE_ARGS+=(--assume-ar-forward-pass)
    fi
    set +e
    "$EVAL_PYTHON" "${MERGE_ARGS[@]}" 2>&1 | tee -a "$bench_log"
    merge_status=${PIPESTATUS[0]}
    set -e
    if [[ "$merge_status" != "0" ]]; then
        echo "ERROR: benchmark $bench_spec failed during SGLang metrics merge with exit code $merge_status. Continuing." >&2
        write_compact_error "$bench_name" "$bench_spec" "metrics_merge" "$merge_status" "SGLang metrics merge failed after NeMo-Skills metrics were produced." "$bench_log" "$metrics_json"
        PIPELINE_FAILED=1
        continue
    fi

    if ! write_compact_metrics "$bench_name" "$metrics_json"; then
        echo "ERROR: benchmark $bench_spec failed while writing compact metrics. Continuing." >&2
        write_compact_error "$bench_name" "$bench_spec" "compact_metrics_write" "1" "Failed to copy merged metrics.json to the final output directory." "$bench_log" "$metrics_json"
        PIPELINE_FAILED=1
        continue
    fi
done

if [[ "$PIPELINE_FAILED" != "0" ]]; then
    echo "WARNING: one or more benchmarks failed. Error details were written to error_<benchmark>.json files; completed benchmarks have metrics_<benchmark>.json." >&2
fi

echo "[5/5] Completed SGLang eval pipeline."
echo "Results: $SEQ_EVAL_OUTPUT_DIR"
echo "Runtime logs: $SGLANG_RUN_DIR"
