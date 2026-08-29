#!/bin/bash
# Independent L4/L8/L16/L32 paired shadow observation on native LinearSpec.

set -euo pipefail

ORIGINAL_ARGS=("$@")
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULTS_ROOT_DEFAULT="/data/home/wly/dLLM/NLD_results/observations/pytorch_linearspec_block_size_shadow_results"
MODEL_DEFAULT="/data1/linyewei/models/Nemotron-Labs-Diffusion-8B"
DATA_DEFAULT="/data1/linyewei/datasets/NLD"
PYTHON_DEFAULT="/data/home/wly/.conda/envs/nld_sglang/bin/python"
[[ -x "$PYTHON_DEFAULT" ]] || PYTHON_DEFAULT="python"
BENCHMARKS_DEFAULT="gsm8k:1,human-eval:1,mbpp:1,math-500:1,aime24:1,aime25:1,gpqa:1,mmlu:1,ifeval:1,livecodebench-cpp:1"

SERVER_SCRIPT="$SCRIPT_DIR/server.py"
SUMMARY_SCRIPT="$SCRIPT_DIR/summarize.py"
RUN_MANAGER="$SCRIPT_DIR/run_manager.py"
RUNTIME_UTILS="$SCRIPT_DIR/runtime_utils.py"
EVAL_SCRIPT="$PROJECT_DIR/xp/nemo-skills/eval_dlm.py"
MEMORY_RESERVER="$PROJECT_DIR/xp/pytorch_nemo_eval/gpu_memory_reserver.py"

usage() {
    sed -n '1,240p' "$SCRIPT_DIR/USAGE.txt"
}

MODE="linearspec_lora"
BENCHMARKS="$BENCHMARKS_DEFAULT"
OUTPUT_PATH="$RESULTS_ROOT_DEFAULT"
MODEL="$MODEL_DEFAULT"
SERVED_MODEL_NAME="nemotron-labs-diffusion-8b"
LORA_PATH=""
DTYPE="bfloat16"
BLOCK_SIZES="4,8,16,32"
ANCHOR_BLOCK_SIZE="16"
HISTORY_WINDOWS="1,2,4,8"
THRESHOLD="0"
TEMPERATURE="0"
TOP_P="0.95"
TOKENS="8192"
CONTEXT_LENGTH=""
GPU_DEVICE="auto"
GPU_CANDIDATES="all"
GPU_MIN_FREE_GB="24"
GPU_WAIT_TIMEOUT_S="0"
GPU_POLL_INTERVAL_S="30"
GPU_MEMORY_RESERVE_GB="0"
PORT=""
PORT_USER_SET="false"
BATCH_SIZE="1"
CLIENT_CONCURRENCY="1"
NUM_CHUNKS=""
MAX_SAMPLES=""
QUICK_TEST="false"
ENABLE_THINKING="false"
DISABLE_THINKING="false"
KEEP_THINKING="false"
STRIP_THINKING="false"
MAX_THINKING_TOKENS=""
MATH_PROMPT_CONFIG=""
TRACE_DETAIL="position"
INCLUDE_BOUNDARY_ROUNDS="false"
PYTORCH_PYTHON="$PYTHON_DEFAULT"
EVAL_PYTHON=""
NEMO_SKILLS_DATA_DIR_ARG="${NEMO_SKILLS_DATA_DIR:-$DATA_DEFAULT}"
GOOGLE_RESEARCH_DIR=""
PREPARE_MISSING_DATA="false"
KEEP_RUNTIME="false"
DRY_RUN="false"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode) MODE="$2"; shift 2 ;;
        --benchmarks) BENCHMARKS="$2"; shift 2 ;;
        --output-path|--out-dir) OUTPUT_PATH="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --served-model-name|--model-name) SERVED_MODEL_NAME="$2"; shift 2 ;;
        --lora-path) LORA_PATH="$2"; shift 2 ;;
        --dtype) DTYPE="$2"; shift 2 ;;
        --block-sizes) BLOCK_SIZES="$2"; shift 2 ;;
        --anchor-block-size|--block-size|--block-length) ANCHOR_BLOCK_SIZE="$2"; shift 2 ;;
        --history-windows) HISTORY_WINDOWS="$2"; shift 2 ;;
        --threshold) THRESHOLD="$2"; shift 2 ;;
        --temperature) TEMPERATURE="$2"; shift 2 ;;
        --top-p) TOP_P="$2"; shift 2 ;;
        --tokens) TOKENS="$2"; shift 2 ;;
        --context-length) CONTEXT_LENGTH="$2"; shift 2 ;;
        --gpu-device|--gpu-devices) GPU_DEVICE="$2"; shift 2 ;;
        --gpu-candidates) GPU_CANDIDATES="$2"; shift 2 ;;
        --gpu-min-free-gb) GPU_MIN_FREE_GB="$2"; shift 2 ;;
        --gpu-wait-timeout-s) GPU_WAIT_TIMEOUT_S="$2"; shift 2 ;;
        --gpu-poll-interval-s) GPU_POLL_INTERVAL_S="$2"; shift 2 ;;
        --gpu-memory-reserve-gb) GPU_MEMORY_RESERVE_GB="$2"; shift 2 ;;
        --port) PORT="$2"; PORT_USER_SET="true"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --client-concurrency) CLIENT_CONCURRENCY="$2"; shift 2 ;;
        --num-chunks) NUM_CHUNKS="$2"; shift 2 ;;
        --max-samples) MAX_SAMPLES="$2"; shift 2 ;;
        --quick-test) QUICK_TEST="true"; shift ;;
        --enable-thinking) ENABLE_THINKING="true"; shift ;;
        --disable-thinking) DISABLE_THINKING="true"; shift ;;
        --keep-thinking) KEEP_THINKING="true"; shift ;;
        --strip-thinking) STRIP_THINKING="true"; shift ;;
        --max-thinking-tokens) MAX_THINKING_TOKENS="$2"; shift 2 ;;
        --math-prompt-config) MATH_PROMPT_CONFIG="$2"; shift 2 ;;
        --trace-detail) TRACE_DETAIL="$2"; shift 2 ;;
        --include-boundary-rounds) INCLUDE_BOUNDARY_ROUNDS="true"; shift ;;
        --pytorch-python) PYTORCH_PYTHON="$2"; shift 2 ;;
        --eval-python) EVAL_PYTHON="$2"; shift 2 ;;
        --nemo-skills-data-dir) NEMO_SKILLS_DATA_DIR_ARG="$2"; shift 2 ;;
        --google-research-dir) GOOGLE_RESEARCH_DIR="$2"; shift 2 ;;
        --prepare-missing-data) PREPARE_MISSING_DATA="true"; shift ;;
        --keep-runtime) KEEP_RUNTIME="true"; shift ;;
        --dry-run) DRY_RUN="true"; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "ERROR: unknown option: $1" >&2; usage; exit 1 ;;
    esac
done

[[ -n "$LORA_PATH" ]] || LORA_PATH="$MODEL/linear_spec_lora"
[[ "$MODE" == "linearspec_lora" ]] || LORA_PATH=""
[[ -n "$NUM_CHUNKS" ]] || NUM_CHUNKS="$CLIENT_CONCURRENCY"
[[ -n "$EVAL_PYTHON" ]] || EVAL_PYTHON="$PYTORCH_PYTHON"
[[ -n "$GOOGLE_RESEARCH_DIR" ]] || GOOGLE_RESEARCH_DIR="$NEMO_SKILLS_DATA_DIR_ARG/google-research"
[[ -n "$CONTEXT_LENGTH" ]] || CONTEXT_LENGTH=$((TOKENS + 2048))

is_positive_int() { [[ "${1:-}" =~ ^[0-9]+$ ]] && (( 10#$1 > 0 )); }
is_nonnegative_int() { [[ "${1:-}" =~ ^[0-9]+$ ]]; }
is_nonnegative_number() { "$PYTORCH_PYTHON" -c 'import math,sys; x=float(sys.argv[1]); raise SystemExit(0 if math.isfinite(x) and x>=0 else 1)' "$1"; }
validate_float_triplet() { "$PYTORCH_PYTHON" -c 'import math,sys; t,u,p=map(float,sys.argv[1:]); raise SystemExit(0 if all(map(math.isfinite,(t,u,p))) and t>=0 and u>=0 and 0<=p<=1 else 1)' "$THRESHOLD" "$TEMPERATURE" "$TOP_P"; }
validate_int_list() { "$PYTORCH_PYTHON" -c 'import sys; xs=[int(x) for x in sys.argv[1].split(",") if x.strip()]; raise SystemExit(0 if xs and len(xs)==len(set(xs)) and all(x>=int(sys.argv[2]) for x in xs) else 1)' "$1" "$2"; }
list_contains() { "$PYTORCH_PYTHON" -c 'import sys; raise SystemExit(0 if int(sys.argv[2]) in [int(x) for x in sys.argv[1].split(",")] else 1)' "$1" "$2"; }

case "$MODE" in linearspec_lora|linearspec_base) ;; *) echo "ERROR: --mode must be linearspec_lora or linearspec_base" >&2; exit 1 ;; esac
case "$DTYPE" in bfloat16|bf16|float16|fp16|float32|fp32) ;; *) echo "ERROR: unsupported --dtype" >&2; exit 1 ;; esac
case "$TRACE_DETAIL" in scalar|position|tokens) ;; *) echo "ERROR: --trace-detail must be scalar, position, or tokens" >&2; exit 1 ;; esac
[[ -n "$BENCHMARKS" ]] || { echo "ERROR: --benchmarks cannot be empty" >&2; exit 1; }
[[ "$BATCH_SIZE" == "1" ]] || { echo "ERROR: native paired observation requires --batch-size 1" >&2; exit 1; }
is_positive_int "$TOKENS" || { echo "ERROR: --tokens must be positive" >&2; exit 1; }
is_positive_int "$CONTEXT_LENGTH" || { echo "ERROR: --context-length must be positive" >&2; exit 1; }
is_positive_int "$ANCHOR_BLOCK_SIZE" || { echo "ERROR: anchor block must be positive" >&2; exit 1; }
is_positive_int "$CLIENT_CONCURRENCY" || { echo "ERROR: client concurrency must be positive" >&2; exit 1; }
is_positive_int "$NUM_CHUNKS" || { echo "ERROR: num chunks must be positive" >&2; exit 1; }
is_nonnegative_int "$GPU_WAIT_TIMEOUT_S" || { echo "ERROR: GPU wait timeout must be nonnegative" >&2; exit 1; }
is_positive_int "$GPU_POLL_INTERVAL_S" || { echo "ERROR: GPU poll interval must be positive" >&2; exit 1; }
[[ -z "$MAX_SAMPLES" ]] || is_positive_int "$MAX_SAMPLES" || { echo "ERROR: max samples must be positive" >&2; exit 1; }
[[ -z "$MAX_THINKING_TOKENS" ]] || is_positive_int "$MAX_THINKING_TOKENS" || { echo "ERROR: max thinking tokens must be positive" >&2; exit 1; }
validate_int_list "$BLOCK_SIZES" 2 || { echo "ERROR: block sizes must be unique integers >=2" >&2; exit 1; }
validate_int_list "$HISTORY_WINDOWS" 1 || { echo "ERROR: history windows must be unique positive integers" >&2; exit 1; }
list_contains "$BLOCK_SIZES" "$ANCHOR_BLOCK_SIZE" || { echo "ERROR: anchor block must occur in block sizes" >&2; exit 1; }
validate_float_triplet || { echo "ERROR: require threshold>=0, temperature>=0, 0<=top-p<=1" >&2; exit 1; }
is_nonnegative_number "$GPU_MIN_FREE_GB" || { echo "ERROR: gpu-min-free-gb must be nonnegative" >&2; exit 1; }
is_nonnegative_number "$GPU_MEMORY_RESERVE_GB" || { echo "ERROR: gpu-memory-reserve-gb must be nonnegative" >&2; exit 1; }
[[ "$ENABLE_THINKING" != "true" || "$DISABLE_THINKING" != "true" ]] || { echo "ERROR: enable/disable thinking are mutually exclusive" >&2; exit 1; }
[[ "$KEEP_THINKING" != "true" || "$STRIP_THINKING" != "true" ]] || { echo "ERROR: keep/strip thinking are mutually exclusive" >&2; exit 1; }
if [[ "$MODE" == "linearspec_lora" && ! -f "$LORA_PATH/adapter_config.json" ]]; then echo "ERROR: LoRA adapter missing: $LORA_PATH" >&2; exit 1; fi

select_auto_gpu() {
    local started now_s
    started="$(date +%s)"
    while true; do
        if selected="$($PYTORCH_PYTHON "$RUNTIME_UTILS" gpu --candidates "$GPU_CANDIDATES" --min-free-gb "$GPU_MIN_FREE_GB" 2>/dev/null)"; then
            echo "$selected"
            return 0
        fi
        now_s="$(date +%s)"
        if (( GPU_WAIT_TIMEOUT_S == 0 || now_s - started >= GPU_WAIT_TIMEOUT_S )); then
            "$PYTORCH_PYTHON" "$RUNTIME_UTILS" gpu --candidates "$GPU_CANDIDATES" --min-free-gb "$GPU_MIN_FREE_GB" >&2 || true
            return 1
        fi
        echo "Waiting for a candidate GPU with >=$GPU_MIN_FREE_GB GiB free..." >&2
        sleep "$GPU_POLL_INTERVAL_S"
    done
}

if [[ "$GPU_DEVICE" == "auto" ]]; then GPU_DEVICE="$(select_auto_gpu)" || exit 1; fi
[[ "$GPU_DEVICE" =~ ^[0-9]+$ ]] || { echo "ERROR: --gpu-device must be one ID or auto" >&2; exit 1; }
if [[ -z "$PORT" ]]; then
    PORT="$($PYTORCH_PYTHON "$RUNTIME_UTILS" port --start "$((34000 + GPU_DEVICE))")"
else
    is_positive_int "$PORT" && (( PORT <= 65535 )) || { echo "ERROR: invalid port" >&2; exit 1; }
    "$PYTORCH_PYTHON" "$RUNTIME_UTILS" check-port --port "$PORT" || { echo "ERROR: requested port is busy: $PORT" >&2; exit 1; }
fi
OUTPUT_PATH="$(realpath -m "$OUTPUT_PATH")"
NEMO_SKILLS_DATA_DIR_ARG="$(realpath -m "$NEMO_SKILLS_DATA_DIR_ARG")"
GOOGLE_RESEARCH_DIR="$(realpath -m "$GOOGLE_RESEARCH_DIR")"
RUN_NAME="block_size_shadow_$(date +%Y%m%d_%H%M%S)"
FINAL_DIR="$OUTPUT_PATH/$RUN_NAME"
suffix=1
while [[ -e "$FINAL_DIR" ]]; do FINAL_DIR="$OUTPUT_PATH/${RUN_NAME}_$(printf '%02d' "$suffix")"; suffix=$((suffix + 1)); done

printf -v ORIGINAL_COMMAND ' %q' "${ORIGINAL_ARGS[@]}"
ORIGINAL_COMMAND="bash observations/pytorch_linearspec_block_size_shadow/eval_block_size_shadow.sh${ORIGINAL_COMMAND}"

show_resolution() {
    echo "================================================================"
    echo " LinearSpec paired block-size shadow observation"
    echo "================================================================"
    echo " Benchmarks:        $BENCHMARKS"
    echo " Blocks / anchor:   $BLOCK_SIZES / $ANCHOR_BLOCK_SIZE"
    echo " GPU / reserve:     $GPU_DEVICE / $GPU_MEMORY_RESERVE_GB GiB"
    echo " Tokens / context:  $TOKENS / $CONTEXT_LENGTH"
    echo " Port:              $PORT"
    echo " Output:            $FINAL_DIR"
    echo "================================================================"
}
if [[ "$DRY_RUN" == "true" ]]; then
    show_resolution
    [[ "$TEMPERATURE" == "0" || "$TEMPERATURE" == "0.0" ]] || echo "WARNING: Q1 formal comparison should use temperature=0; stochastic arms use common starting RNG." >&2
    echo "[dry-run] Validated only; no directory, model, or dataset was created."
    exit 0
fi

PORT_LOCK_FD=""
if command -v flock >/dev/null; then
    while true; do
        exec {PORT_LOCK_FD}>"/tmp/nld_block_size_shadow_port_${PORT}.lock"
        if flock -n "$PORT_LOCK_FD"; then break; fi
        exec {PORT_LOCK_FD}>&-
        if [[ "$PORT_USER_SET" == "true" ]]; then echo "ERROR: requested port is reserved by another block-shadow run: $PORT" >&2; exit 1; fi
        PORT="$($PYTORCH_PYTHON "$RUNTIME_UTILS" port --start "$((PORT + 1))")"
    done
fi

mkdir -p "$OUTPUT_PATH"
while ! mkdir "$FINAL_DIR" 2>/dev/null; do
    FINAL_DIR="$OUTPUT_PATH/${RUN_NAME}_$(printf '%02d' "$suffix")"
    suffix=$((suffix + 1))
done
show_resolution
[[ "$TEMPERATURE" == "0" || "$TEMPERATURE" == "0.0" ]] || echo "WARNING: Q1 formal comparison should use temperature=0; stochastic arms use common starting RNG." >&2
init_args=("$RUN_MANAGER" init --run-dir "$FINAL_DIR" --entrypoint "observations/pytorch_linearspec_block_size_shadow/eval_block_size_shadow.sh" --command "$ORIGINAL_COMMAND" --mode "$MODE" --benchmarks "$BENCHMARKS" --model "$MODEL" --served-model-name "$SERVED_MODEL_NAME" --lora-path "$LORA_PATH" --dtype "$DTYPE" --block-sizes "$BLOCK_SIZES" --anchor-block-size "$ANCHOR_BLOCK_SIZE" --history-windows "$HISTORY_WINDOWS" --threshold "$THRESHOLD" --temperature "$TEMPERATURE" --top-p "$TOP_P" --tokens "$TOKENS" --context-length "$CONTEXT_LENGTH" --gpu-device "$GPU_DEVICE" --gpu-candidates "$GPU_CANDIDATES" --gpu-min-free-gb "$GPU_MIN_FREE_GB" --gpu-memory-reserve-gb "$GPU_MEMORY_RESERVE_GB" --port "$PORT" --client-concurrency "$CLIENT_CONCURRENCY" --num-chunks "$NUM_CHUNKS" --trace-detail "$TRACE_DETAIL" --pytorch-python "$PYTORCH_PYTHON" --eval-python "$EVAL_PYTHON" --nemo-skills-data-dir "$NEMO_SKILLS_DATA_DIR_ARG" --google-research-dir "$GOOGLE_RESEARCH_DIR")
[[ -n "$MAX_SAMPLES" ]] && init_args+=(--max-samples "$MAX_SAMPLES")
[[ "$QUICK_TEST" == "true" ]] && init_args+=(--quick-test)
[[ "$ENABLE_THINKING" == "true" ]] && init_args+=(--enable-thinking)
[[ "$DISABLE_THINKING" == "true" ]] && init_args+=(--disable-thinking)
[[ "$KEEP_THINKING" == "true" ]] && init_args+=(--keep-thinking)
[[ "$STRIP_THINKING" == "true" ]] && init_args+=(--strip-thinking)
[[ -n "$MAX_THINKING_TOKENS" ]] && init_args+=(--max-thinking-tokens "$MAX_THINKING_TOKENS")
[[ -n "$MATH_PROMPT_CONFIG" ]] && init_args+=(--math-prompt-config "$MATH_PROMPT_CONFIG")
[[ "$INCLUDE_BOUNDARY_ROUNDS" == "true" ]] && init_args+=(--include-boundary-rounds)
"$PYTORCH_PYTHON" "${init_args[@]}"

if ! "$PYTORCH_PYTHON" -c 'import torch,transformers,peft,fastapi,uvicorn' >/dev/null 2>&1; then echo "ERROR: model Python dependencies missing" >&2; exit 1; fi
if ! "$EVAL_PYTHON" -c 'from nemo_skills.pipeline.eval import eval' >/dev/null 2>&1; then echo "ERROR: NeMo-Skills unavailable" >&2; exit 1; fi
if [[ "$EVAL_PYTHON" == */* ]]; then EVAL_PYTHON_DIR="$(cd "$(dirname "$EVAL_PYTHON")" && pwd)"; else EVAL_PYTHON_DIR="$(dirname "$(command -v "$EVAL_PYTHON")")"; fi
export PATH="$EVAL_PYTHON_DIR:$PATH"
export NEMO_SKILLS_DATA_DIR="$NEMO_SKILLS_DATA_DIR_ARG"
export NLD_GOOGLE_RESEARCH_DIR="$GOOGLE_RESEARCH_DIR"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$NEMO_SKILLS_DATA_DIR_ARG/hf_datasets_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$NEMO_SKILLS_DATA_DIR_ARG/xdg_cache}"
mkdir -p "$NEMO_SKILLS_DATA_DIR_ARG" "$HF_DATASETS_CACHE" "$XDG_CACHE_HOME"

NEMO_PACKAGE_DATASET_DIR="$($EVAL_PYTHON -c 'from pathlib import Path; import nemo_skills.dataset; print(Path(nemo_skills.dataset.__file__).resolve().parent)')"
dataset_has_jsonl() { [[ -d "$1" ]] && find "$1" -maxdepth 1 -type f -name '*.jsonl' -print -quit | grep -q .; }
if [[ "$BENCHMARKS" == *"ifeval"* ]]; then
    [[ -d "$GOOGLE_RESEARCH_DIR/instruction_following_eval" ]] || { echo "ERROR: IFEval scorer missing: $GOOGLE_RESEARCH_DIR" >&2; exit 1; }
    export PYTHONPATH="$GOOGLE_RESEARCH_DIR:${PYTHONPATH:-}"
    "$EVAL_PYTHON" -c 'import instruction_following_eval.evaluation_main,langdetect,immutabledict,nltk' >/dev/null
fi

IFS=',' read -ra PREP_SPECS <<< "$BENCHMARKS"
for spec in "${PREP_SPECS[@]}"; do
    spec="${spec//[[:space:]]/}"; [[ -z "$spec" ]] && continue
    name="${spec%%:*}"
    package_dir="$NEMO_PACKAGE_DATASET_DIR/$name"
    cached_dir="$NEMO_SKILLS_DATA_DIR_ARG/$name"
    if dataset_has_jsonl "$package_dir"; then continue; fi
    if dataset_has_jsonl "$cached_dir"; then
        mkdir -p "$package_dir"
        if command -v flock >/dev/null; then flock "$NEMO_SKILLS_DATA_DIR_ARG/.prepare.lock" cp -a "$cached_dir"/. "$package_dir"/; else cp -a "$cached_dir"/. "$package_dir"/; fi
        continue
    fi
    if [[ "$PREPARE_MISSING_DATA" != "true" ]]; then echo "ERROR: dataset $name is neither installed nor cached; pass --prepare-missing-data to allow preparation" >&2; exit 1; fi
    if command -v flock >/dev/null; then flock "$NEMO_SKILLS_DATA_DIR_ARG/.prepare.lock" "$EVAL_PYTHON" -m nemo_skills.dataset.prepare "$name" --parallelism 20 --retries 3; else "$EVAL_PYTHON" -m nemo_skills.dataset.prepare "$name" --parallelism 20 --retries 3; fi
    dataset_has_jsonl "$package_dir" || { echo "ERROR: preparation produced no JSONL for $name" >&2; exit 1; }
    mkdir -p "$cached_dir"; cp -a "$package_dir"/. "$cached_dir"/
done

SERVER_PID=""
RESERVER_PID=""
stop_server() { if [[ -n "$SERVER_PID" ]]; then kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true; SERVER_PID=""; fi; }
cleanup() { stop_server; if [[ -n "$RESERVER_PID" ]]; then kill "$RESERVER_PID" 2>/dev/null || true; wait "$RESERVER_PID" 2>/dev/null || true; RESERVER_PID=""; fi; }
trap cleanup EXIT INT TERM
wait_for_health() {
    local pid="$1" log="$2"
    for _ in $(seq 1 300); do
        kill -0 "$pid" 2>/dev/null || { echo "ERROR: server exited" >&2; tail -100 "$log" >&2 || true; return 1; }
        curl -sf --max-time 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && return 0
        sleep 2
    done
    tail -100 "$log" >&2 || true; return 1
}
status_update() {
    local spec="$1" status="$2" stage="$3" code="$4" message="$5" trace="$6" summary="$7" metrics="$8"
    "$EVAL_PYTHON" "$RUN_MANAGER" status --run-dir "$FINAL_DIR" --benchmark-spec "$spec" --status "$status" --stage "$stage" --exit-code "$code" --message "$message" --trace-file "$trace" --summary-file "$summary" --metrics-file "$metrics"
}

if "$PYTORCH_PYTHON" -c 'import sys; raise SystemExit(0 if float(sys.argv[1])>0 else 1)' "$GPU_MEMORY_RESERVE_GB"; then
    ready="$FINAL_DIR/runtime/gpu_memory_reserver_ready.json"
    ( export CUDA_VISIBLE_DEVICES="$GPU_DEVICE"; exec "$PYTORCH_PYTHON" -u "$MEMORY_RESERVER" --gb "$GPU_MEMORY_RESERVE_GB" --ready-file "$ready" ) > "$FINAL_DIR/runtime/gpu_memory_reserver.log" 2>&1 &
    RESERVER_PID=$!
    for _ in $(seq 1 120); do kill -0 "$RESERVER_PID" 2>/dev/null || { tail -80 "$FINAL_DIR/runtime/gpu_memory_reserver.log" >&2; exit 1; }; [[ -s "$ready" ]] && break; sleep 1; done
    [[ -s "$ready" ]] || { echo "ERROR: GPU memory reserver timeout" >&2; exit 1; }
fi

ANY_FAILURE=0
IFS=',' read -ra BENCH_SPECS <<< "$BENCHMARKS"
for spec in "${BENCH_SPECS[@]}"; do
    spec="${spec//[[:space:]]/}"; [[ -z "$spec" ]] && continue
    name="${spec%%:*}"
    safe="${name//\//_}"
    trace="$FINAL_DIR/traces/block_size_shadow_${safe}.jsonl"
    summary="$FINAL_DIR/summaries/block_size_shadow_${safe}.json"
    metrics="$FINAL_DIR/metrics/metrics_${safe}.json"
    eval_parent="$FINAL_DIR/eval_runs/$safe"
    eval_dir="$eval_parent/eval-results/$name"
    runtime="$FINAL_DIR/runtime/$safe"
    stats="$runtime/pytorch_request_stats.jsonl"
    server_log="$runtime/server.log"
    eval_log="$runtime/nemo_skills.log"
    mkdir -p "$eval_parent" "$runtime"
    : > "$trace"; : > "$stats"; : > "$server_log"; : > "$eval_log"
    status_update "$spec" running server_start 0 "starting paired-shadow server" "$trace" "$summary" "$metrics"

    server_args=("$SERVER_SCRIPT" --model-path "$MODEL" --served-model-name "$SERVED_MODEL_NAME" --mode "$MODE" --dtype "$DTYPE" --block-sizes "$BLOCK_SIZES" --anchor-block-size "$ANCHOR_BLOCK_SIZE" --history-windows "$HISTORY_WINDOWS" --threshold "$THRESHOLD" --default-max-new-tokens "$TOKENS" --context-length "$CONTEXT_LENGTH" --stats-file "$stats" --trace-file "$trace" --benchmark "$name" --trace-detail "$TRACE_DETAIL" --host 0.0.0.0 --port "$PORT")
    [[ -n "$LORA_PATH" ]] && server_args+=(--lora-path "$LORA_PATH")
    [[ "$ENABLE_THINKING" == "true" ]] && server_args+=(--enable-thinking)
    [[ -n "$MAX_THINKING_TOKENS" ]] && server_args+=(--max-thinking-tokens "$MAX_THINKING_TOKENS")
    ( export CUDA_VISIBLE_DEVICES="$GPU_DEVICE"; exec "$PYTORCH_PYTHON" -u "${server_args[@]}" ) > "$server_log" 2>&1 &
    SERVER_PID=$!
    if ! wait_for_health "$SERVER_PID" "$server_log"; then
        stop_server; status_update "$spec" failed server_start 1 "server failed to become healthy" "$trace" "$summary" "$metrics"; ANY_FAILURE=1; continue
    fi

    eval_args=("$EVAL_SCRIPT" --server-address "http://127.0.0.1:$PORT/v1" --benchmark "$spec" --output-dir "$eval_parent" --expname block_size_shadow --model "$SERVED_MODEL_NAME" --tokens-to-generate "$TOKENS" --temperature "$TEMPERATURE" --top-p "$TOP_P" --num-chunks "$NUM_CHUNKS" --max-concurrent-requests "$CLIENT_CONCURRENCY" --generation-algorithm nemotron --no-extra-body)
    [[ -n "$MAX_SAMPLES" ]] && eval_args+=(--max-samples "$MAX_SAMPLES")
    [[ "$QUICK_TEST" == "true" ]] && eval_args+=(--quick-test)
    [[ "$KEEP_THINKING" == "true" ]] && eval_args+=(--keep-thinking)
    [[ "$STRIP_THINKING" == "true" ]] && eval_args+=(--strip-thinking)
    [[ "$DISABLE_THINKING" == "true" ]] && eval_args+=(--disable-thinking)
    [[ -n "$MATH_PROMPT_CONFIG" ]] && eval_args+=(--math-prompt-config "$MATH_PROMPT_CONFIG")
    set +e
    "$EVAL_PYTHON" "${eval_args[@]}" 2>&1 | tee "$eval_log"
    eval_status=${PIPESTATUS[0]}
    set -e
    stop_server
    if [[ "$eval_status" != "0" || ! -f "$eval_dir/metrics.json" ]]; then
        status_update "$spec" failed nemo_eval "$eval_status" "NeMo-Skills evaluation or metrics failed" "$trace" "$summary" "$metrics"; ANY_FAILURE=1; continue
    fi
    cp "$eval_dir/metrics.json" "$metrics"
    summary_args=("$SUMMARY_SCRIPT" --trace-file "$trace" --output-json "$summary" --benchmark "$name" --benchmark-spec "$spec")
    [[ "$INCLUDE_BOUNDARY_ROUNDS" == "true" ]] && summary_args+=(--include-boundary-rounds)
    set +e
    "$EVAL_PYTHON" "${summary_args[@]}" 2>&1 | tee -a "$eval_log"
    summary_status=${PIPESTATUS[0]}
    set -e
    if [[ "$summary_status" == "0" ]]; then
        status_update "$spec" completed summary 0 "dataset completed and report refreshed" "$trace" "$summary" "$metrics"
    else
        status_update "$spec" failed summary "$summary_status" "trace summary validation failed" "$trace" "$summary" "$metrics"; ANY_FAILURE=1
    fi
done

cleanup
trap - EXIT INT TERM
"$EVAL_PYTHON" "$RUN_MANAGER" report --run-dir "$FINAL_DIR"
echo "Output: $FINAL_DIR"
if [[ "$ANY_FAILURE" != "0" ]]; then echo "One or more datasets failed; completed datasets remain summarized." >&2; exit 1; fi
