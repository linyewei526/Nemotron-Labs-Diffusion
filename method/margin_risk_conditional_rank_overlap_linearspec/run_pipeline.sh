#!/bin/bash
# Isolated fixed-margin-risk conditional-rank PyTorch + NeMo-Skills pipeline.

set -euo pipefail

METHOD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$METHOD_DIR/../.." && pwd)"
ROOT_DIR="$PROJECT_DIR/xp"

EVAL_SCRIPT="$ROOT_DIR/nemo-skills/eval_dlm.py"
MT_BENCH_SCRIPT="$ROOT_DIR/mt_bench/eval_mt_bench.py"
ALPACA_EVAL_SCRIPT="$ROOT_DIR/alpaca_eval/eval_alpaca_eval.py"
SERVER_SCRIPT="$METHOD_DIR/server.py"
MERGE_SCRIPT="$METHOD_DIR/merge_metrics.py"
MEMORY_RESERVER_SCRIPT="$ROOT_DIR/pytorch_nemo_eval/gpu_memory_reserver.py"
SETTINGS_UPDATER="$METHOD_DIR/update_settings.py"
REPORT_SCRIPT="$METHOD_DIR/report.py"

PYTORCH_PYTHON="${PYTORCH_PYTHON:-python}"
EVAL_PYTHON="${EVAL_PYTHON:-$PYTORCH_PYTHON}"
if [[ "$EVAL_PYTHON" == */* ]]; then
    EVAL_PYTHON_DIR="$(cd "$(dirname "$EVAL_PYTHON")" && pwd)"
else
    EVAL_PYTHON_DIR="$(dirname "$(command -v "$EVAL_PYTHON")")"
fi
export PATH="$EVAL_PYTHON_DIR:$PATH"
PYTORCH_MODE="${PYTORCH_MODE:-overlap_lora}"
PYTORCH_MODEL="${PYTORCH_MODEL:-/data1/linyewei/models/Nemotron-Labs-Diffusion-8B}"
PYTORCH_MODEL_TAG="${PYTORCH_MODEL_TAG:-nemotron-labs-diffusion-8b}"
PYTORCH_GPU_DEVICE="${PYTORCH_GPU_DEVICE:-0}"
PYTORCH_GPU_MEMORY_RESERVE_GB="${PYTORCH_GPU_MEMORY_RESERVE_GB:-0}"
PYTORCH_PORT="${PYTORCH_PORT:-0}"
PYTORCH_DTYPE="${PYTORCH_DTYPE:-bfloat16}"
PYTORCH_BLOCK_LENGTH="${PYTORCH_BLOCK_LENGTH:-16}"
PYTORCH_THRESHOLD="${PYTORCH_THRESHOLD-}"
OVERLAP_MARGIN_RISK_THRESHOLD="${OVERLAP_MARGIN_RISK_THRESHOLD:-0.5}"
OVERLAP_SETTINGS_FILE="${OVERLAP_SETTINGS_FILE:-}"
OVERLAP_BASELINE_BLOCK16_DIR="${OVERLAP_BASELINE_BLOCK16_DIR:-/data/home/wly/dLLM/NLD_results/observations/pytorch_nemo_eval_results/eval_20260804_120138}"
OVERLAP_BASELINE_BLOCK32_DIR="${OVERLAP_BASELINE_BLOCK32_DIR:-/data/home/wly/dLLM/NLD_results/observations/pytorch_nemo_eval_results/eval_20260804_114935}"
PYTORCH_CONTEXT_LENGTH="${PYTORCH_CONTEXT_LENGTH:-10240}"
PYTORCH_LORA_PATH="${PYTORCH_LORA_PATH:-}"
PYTORCH_ENABLE_THINKING="${PYTORCH_ENABLE_THINKING:-false}"
PYTORCH_EFFICIENCY_ONLY="${PYTORCH_EFFICIENCY_ONLY:-true}"
PYTORCH_MAX_THINKING_TOKENS="${PYTORCH_MAX_THINKING_TOKENS:-}"
PYTORCH_RUN_DIR="${PYTORCH_RUN_DIR:-$PROJECT_DIR/results/pytorch_nemo_runtime}"
PYTORCH_FINAL_OUTPUT_DIR="${PYTORCH_FINAL_OUTPUT_DIR:-}"
NEMO_SKILLS_DATA_DIR="${NEMO_SKILLS_DATA_DIR:-/data1/linyewei/datasets/NLD}"
NLD_GOOGLE_RESEARCH_DIR="${NLD_GOOGLE_RESEARCH_DIR:-$NEMO_SKILLS_DATA_DIR/google-research}"
export NLD_GOOGLE_RESEARCH_DIR
export NEMO_SKILLS_DATA_DIR
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$NEMO_SKILLS_DATA_DIR/hf_datasets_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$NEMO_SKILLS_DATA_DIR/xdg_cache}"

SEQ_EVAL_BENCHMARK="${SEQ_EVAL_BENCHMARK:-gsm8k:1,human-eval:1,mbpp:1,math-500:1,aime24:1,aime25:1,gpqa:1,ifeval:1,livecodebench-cpp:1,mmlu:1}"
SEQ_EVAL_EXPNAME="${SEQ_EVAL_EXPNAME:-margin_risk_conditional_rank_overlap}"
SEQ_EVAL_OUTPUT_DIR="${SEQ_EVAL_OUTPUT_DIR:-$PROJECT_DIR/results/pytorch_nemo_eval_internal}"
SEQ_EVAL_TOKENS_TO_GENERATE="${SEQ_EVAL_TOKENS_TO_GENERATE:-8192}"
SEQ_EVAL_TEMPERATURE="${SEQ_EVAL_TEMPERATURE:-0}"
SEQ_EVAL_TOP_P="${SEQ_EVAL_TOP_P:-0.95}"
SEQ_EVAL_NUM_CHUNKS="${SEQ_EVAL_NUM_CHUNKS:-1}"
SEQ_EVAL_CLIENT_CONCURRENCY="${SEQ_EVAL_CLIENT_CONCURRENCY:-1}"
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
MT_BENCH_DATA_DIR="${MT_BENCH_DATA_DIR:-$NEMO_SKILLS_DATA_DIR/mt-bench}"
ALPACA_EVAL_DATA_DIR="${ALPACA_EVAL_DATA_DIR:-$NEMO_SKILLS_DATA_DIR/alpaca-eval}"

mkdir -p "$PYTORCH_RUN_DIR" "$SEQ_EVAL_OUTPUT_DIR" "$NEMO_SKILLS_DATA_DIR" "$HF_DATASETS_CACHE" "$XDG_CACHE_HOME"
PYTORCH_RUN_DIR="$(realpath -m "$PYTORCH_RUN_DIR")"
SEQ_EVAL_OUTPUT_DIR="$(realpath -m "$SEQ_EVAL_OUTPUT_DIR")"
if [[ -n "$PYTORCH_FINAL_OUTPUT_DIR" ]]; then
    mkdir -p "$PYTORCH_FINAL_OUTPUT_DIR"
    PYTORCH_FINAL_OUTPUT_DIR="$(realpath -m "$PYTORCH_FINAL_OUTPUT_DIR")"
fi

if ! "$EVAL_PYTHON" -c "from nemo_skills.pipeline.eval import eval; import fastapi, uvicorn" >/dev/null 2>&1; then
    echo "ERROR: EVAL_PYTHON cannot import NeMo-Skills/FastAPI/Uvicorn: $EVAL_PYTHON" >&2
    exit 1
fi
if ! "$PYTORCH_PYTHON" -c "import torch, transformers, fastapi, uvicorn, safetensors" >/dev/null 2>&1; then
    echo "ERROR: PYTORCH_PYTHON cannot import torch/transformers/FastAPI/Uvicorn/Safetensors: $PYTORCH_PYTHON" >&2
    exit 1
fi

REQUEST_STATS_FILE="$PYTORCH_RUN_DIR/pytorch_request_stats.jsonl"
SERVER_LOG="$PYTORCH_RUN_DIR/pytorch_server.log"
SERVER_PORT_FILE="$PYTORCH_RUN_DIR/server_port.json"
MEMORY_RESERVER_LOG="$PYTORCH_RUN_DIR/gpu_memory_reserver.log"
MEMORY_RESERVER_READY="$PYTORCH_RUN_DIR/gpu_memory_reserver_ready.json"
: > "$REQUEST_STATS_FILE"

NEMO_SKILLS_PACKAGE_DATASET_DIR="$("$EVAL_PYTHON" - <<'PY'
from pathlib import Path
import nemo_skills.dataset
print(Path(nemo_skills.dataset.__file__).resolve().parent)
PY
)"

is_enabled() {
    [[ "${1,,}" == "true" || "$1" == "1" || "${1,,}" == "yes" ]]
}

request_stats_has_success() {
    "$EVAL_PYTHON" - "$REQUEST_STATS_FILE" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(1)
for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
    try:
        if json.loads(line).get("ok") is True:
            raise SystemExit(0)
    except json.JSONDecodeError:
        pass
raise SystemExit(1)
PY
}

request_stats_failures_are_oom() {
    "$EVAL_PYTHON" - "$REQUEST_STATS_FILE" <<'PY'
import json, sys
path = sys.argv[1]
try:
    rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
except (OSError, json.JSONDecodeError):
    raise SystemExit(1)
for row in rows:
    if row.get("ok") is True:
        continue
    error_type = str(row.get("error_type") or "")
    if row.get("oom_skipped_for_efficiency") is not True and "OutOfMemory" not in error_type:
        raise SystemExit(1)
raise SystemExit(0)
PY
}

generation_outputs_complete() {
    local eval_dir="$1"
    local markers=("$eval_dir"/output*.jsonl.done)
    [[ -e "${markers[0]}" ]]
}

reserve_memory_enabled() {
    "$PYTORCH_PYTHON" - "$PYTORCH_GPU_MEMORY_RESERVE_GB" <<'PY'
import math, sys
try:
    value = float(sys.argv[1])
except (TypeError, ValueError):
    raise SystemExit(2)
raise SystemExit(0 if math.isfinite(value) and value > 0 else 1)
PY
}

wait_for_health() {
    local url="$1"
    local pid="$2"
    local log_file="$3"
    for _ in $(seq 1 300); do
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "ERROR: PyTorch server exited during startup. Log tail:" >&2
            tail -100 "$log_file" >&2 || true
            return 1
        fi
        if curl -sf --max-time 2 "$url" >/dev/null 2>&1; then
            return 0
        fi
        sleep 2
    done
    echo "ERROR: PyTorch server did not become healthy. Log tail:" >&2
    tail -120 "$log_file" >&2 || true
    return 1
}

wait_for_server_port() {
    local path="$1"
    local pid="$2"
    local log_file="$3"
    for _ in $(seq 1 600); do
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "ERROR: overlap server exited before publishing its port. Log tail:" >&2
            tail -120 "$log_file" >&2 || true
            return 1
        fi
        if [[ -s "$path" ]]; then
            "$PYTORCH_PYTHON" - "$path" <<'PY'
import json, sys
port = int(json.load(open(sys.argv[1], encoding="utf-8"))["port"])
raise SystemExit(0 if 0 < port <= 65535 else 1)
PY
            return $?
        fi
        sleep 2
    done
    echo "ERROR: overlap server did not publish a port. Log tail:" >&2
    tail -120 "$log_file" >&2 || true
    return 1
}

wait_for_ready_file() {
    local path="$1"
    local pid="$2"
    local log_file="$3"
    for _ in $(seq 1 120); do
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "ERROR: GPU memory reserver exited during startup. Log tail:" >&2
            tail -80 "$log_file" >&2 || true
            return 1
        fi
        [[ -s "$path" ]] && return 0
        sleep 1
    done
    echo "ERROR: GPU memory reserver did not become ready." >&2
    return 1
}

cleanup() {
    [[ -n "${PYTORCH_SERVER_PID:-}" ]] && kill "$PYTORCH_SERVER_PID" 2>/dev/null || true
    [[ -n "${MEMORY_RESERVER_PID:-}" ]] && kill "$MEMORY_RESERVER_PID" 2>/dev/null || true
    [[ -n "${PYTORCH_SERVER_PID:-}" ]] && wait "$PYTORCH_SERVER_PID" 2>/dev/null || true
    [[ -n "${MEMORY_RESERVER_PID:-}" ]] && wait "$MEMORY_RESERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

benchmark_requested() {
    local wanted="$1"
    local spec
    IFS=',' read -ra _CHECK_BENCHES <<< "$SEQ_EVAL_BENCHMARK"
    for spec in "${_CHECK_BENCHES[@]}"; do
        spec="${spec//[[:space:]]/}"
        [[ "${spec%%:*}" == "$wanted" ]] && return 0
    done
    return 1
}

ensure_ifeval_runtime() {
    benchmark_requested ifeval || return 0
    if [[ ! -d "$NLD_GOOGLE_RESEARCH_DIR/instruction_following_eval" ]]; then
        echo "ERROR: IFEval scorer not found: $NLD_GOOGLE_RESEARCH_DIR/instruction_following_eval" >&2
        return 1
    fi
    export PYTHONPATH="$NLD_GOOGLE_RESEARCH_DIR:${PYTHONPATH:-}"
    if ! "$EVAL_PYTHON" -c "import instruction_following_eval.evaluation_main, langdetect, immutabledict, nltk" >/dev/null 2>&1; then
        echo "ERROR: IFEval Python dependencies are missing. This isolated pipeline will not install packages automatically." >&2
        echo "       Required: instruction_following_eval, langdetect, immutabledict, nltk" >&2
        return 1
    fi
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

dataset_has_jsonl() {
    local path="$1"
    [[ -d "$path" ]] && find "$path" -maxdepth 1 -type f -name '*.jsonl' -print -quit | grep -q .
}

prepare_datasets() {
    local spec name package_dir cached_dir prepare_status
    IFS=',' read -ra _PREP_BENCHES <<< "$SEQ_EVAL_BENCHMARK"
    for spec in "${_PREP_BENCHES[@]}"; do
        spec="${spec//[[:space:]]/}"
        [[ -z "$spec" ]] && continue
        name="${spec%%:*}"
        if [[ "$name" == "mt-bench" ]]; then
            "$EVAL_PYTHON" "$MT_BENCH_SCRIPT" \
                --data-dir "$MT_BENCH_DATA_DIR" \
                --prepare-only
            continue
        fi
        if [[ "$name" == "alpaca-eval" ]]; then
            "$EVAL_PYTHON" "$ALPACA_EVAL_SCRIPT" \
                --data-dir "$ALPACA_EVAL_DATA_DIR" \
                --prepare-only
            continue
        fi
        package_dir="$NEMO_SKILLS_PACKAGE_DATASET_DIR/$name"
        cached_dir="$NEMO_SKILLS_DATA_DIR/$name"
        if dataset_has_jsonl "$package_dir"; then
            if [[ ( "$name" == "arena-hard" || "$name" == "arena-hard-v2" ) \
                && ! -f "$package_dir/test.jsonl" ]]; then
                echo "  Existing Arena-Hard package data is incomplete; regenerating $name"
            else
                continue
            fi
        fi
        if dataset_has_jsonl "$cached_dir"; then
            if [[ ( "$name" == "arena-hard" || "$name" == "arena-hard-v2" ) \
                && ! -f "$cached_dir/test.jsonl" ]]; then
                echo "  Ignoring incomplete Arena-Hard cache without test.jsonl: $cached_dir"
            else
                echo "  Restoring prepared $name data from $cached_dir"
                mkdir -p "$package_dir"
                cp -a "$cached_dir"/. "$package_dir"/
                continue
            fi
        fi
        echo "  Preparing NeMo-Skills dataset: $name"
        prepare_status=0
        if command -v flock >/dev/null; then
            flock "$NEMO_SKILLS_DATA_DIR/.prepare.lock" "$EVAL_PYTHON" -m nemo_skills.dataset.prepare "$name" --parallelism 20 --retries 3 || prepare_status=$?
        else
            "$EVAL_PYTHON" -m nemo_skills.dataset.prepare "$name" --parallelism 20 --retries 3 || prepare_status=$?
        fi
        if [[ "$prepare_status" != "0" && ( "$name" == "arena-hard" || "$name" == "arena-hard-v2" ) ]]; then
            echo "ERROR: failed to prepare required Arena-Hard data for $name." >&2
            return "$prepare_status"
        fi
        if dataset_has_jsonl "$package_dir"; then
            mkdir -p "$cached_dir"
            cp -a "$package_dir"/. "$cached_dir"/
        fi
        if [[ ( "$name" == "arena-hard" || "$name" == "arena-hard-v2" ) \
            && ! -f "$package_dir/test.jsonl" ]]; then
            echo "ERROR: Arena-Hard preparation did not produce $package_dir/test.jsonl" >&2
            return 1
        fi
    done
}

safe_filename() {
    local name="$1"
    name="${name//\//_}"
    name="${name//:/_}"
    name="${name//[[:space:]]/_}"
    echo "$name"
}

write_compact_metrics() {
    local benchmark="$1" metrics_json="$2"
    [[ -z "$PYTORCH_FINAL_OUTPUT_DIR" ]] && return 0

    local safe metrics_dir artifact_dir artifact dst_metrics
    safe="$(safe_filename "$benchmark")"
    metrics_dir="$(dirname "$metrics_json")"
    artifact_dir="$PYTORCH_FINAL_OUTPUT_DIR/artifacts/$safe"
    dst_metrics="$PYTORCH_FINAL_OUTPUT_DIR/metrics_${safe}.json"
    cp "$metrics_json" "$dst_metrics"

    # The top-level entry removes its internal work directory after success.
    # Preserve benchmark-native outputs before it does so and make every
    # recorded artifact path point at the durable compact-output copy.
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
    echo "Wrote compact metrics: $dst_metrics"
}

write_compact_error() {
    local benchmark="$1" stage="$2" exit_code="$3" message="$4" log_file="$5"
    [[ -z "$PYTORCH_FINAL_OUTPUT_DIR" ]] && return 0
    local path="$PYTORCH_FINAL_OUTPUT_DIR/error_$(safe_filename "$benchmark").json"
    ERROR_BENCHMARK="$benchmark" ERROR_STAGE="$stage" ERROR_EXIT_CODE="$exit_code" ERROR_MESSAGE="$message" ERROR_LOG_FILE="$log_file" \
    "$EVAL_PYTHON" - "$path" <<'PY'
import json, os, sys
from datetime import datetime
from pathlib import Path
log = Path(os.environ["ERROR_LOG_FILE"])
tail = []
if log.is_file():
    tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-80:]
payload = {
    "status": "failed",
    "benchmark": os.environ["ERROR_BENCHMARK"],
    "stage": os.environ["ERROR_STAGE"],
    "exit_code": int(os.environ["ERROR_EXIT_CODE"]),
    "message": os.environ["ERROR_MESSAGE"],
    "created_at": datetime.now().astimezone().isoformat(),
    "log_tail": tail,
}
Path(sys.argv[1]).write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
PY
}

echo "=============================================================="
echo " Margin-risk-conditional-rank-overlap LinearSpec + NeMo-Skills evaluation"
echo "=============================================================="
echo " Mode:               $PYTORCH_MODE"
echo " Margin-risk thresh: $OVERLAP_MARGIN_RISK_THRESHOLD"
echo " Model:              $PYTORCH_MODEL"
echo " Benchmarks:         $SEQ_EVAL_BENCHMARK"
echo " GPU device:         $PYTORCH_GPU_DEVICE"
echo " GPU reserve GiB:    $PYTORCH_GPU_MEMORY_RESERVE_GB"
echo " Output:             $SEQ_EVAL_OUTPUT_DIR"
echo " Runtime:            $PYTORCH_RUN_DIR"
echo "=============================================================="

ensure_arena_hard_runtime

if reserve_memory_enabled; then
    echo "[1/5] Reserving GPU memory..."
    rm -f "$MEMORY_RESERVER_READY"
    (
        export CUDA_VISIBLE_DEVICES="$PYTORCH_GPU_DEVICE"
        exec "$PYTORCH_PYTHON" -u "$MEMORY_RESERVER_SCRIPT" --gb "$PYTORCH_GPU_MEMORY_RESERVE_GB" --ready-file "$MEMORY_RESERVER_READY"
    ) > "$MEMORY_RESERVER_LOG" 2>&1 &
    MEMORY_RESERVER_PID=$!
    wait_for_ready_file "$MEMORY_RESERVER_READY" "$MEMORY_RESERVER_PID" "$MEMORY_RESERVER_LOG"
else
    echo "[1/5] GPU memory reservation disabled."
fi

SERVER_ARGS=(
    "$SERVER_SCRIPT"
    --model-path "$PYTORCH_MODEL"
    --served-model-name "$PYTORCH_MODEL_TAG"
    --mode "$PYTORCH_MODE"
    --dtype "$PYTORCH_DTYPE"
    --block-length "$PYTORCH_BLOCK_LENGTH"
    --draft-threshold "${PYTORCH_THRESHOLD:-0.0}"
    --margin-risk-threshold "$OVERLAP_MARGIN_RISK_THRESHOLD"
    --default-max-new-tokens "$SEQ_EVAL_TOKENS_TO_GENERATE"
    --context-length "$PYTORCH_CONTEXT_LENGTH"
    --stats-file "$REQUEST_STATS_FILE"
    --host 127.0.0.1
    --port "$PYTORCH_PORT"
    --port-file "$SERVER_PORT_FILE"
)
[[ -n "$PYTORCH_LORA_PATH" ]] && SERVER_ARGS+=(--lora-path "$PYTORCH_LORA_PATH")
is_enabled "$PYTORCH_ENABLE_THINKING" && SERVER_ARGS+=(--enable-thinking)
is_enabled "$PYTORCH_EFFICIENCY_ONLY" && SERVER_ARGS+=(--efficiency-only)
[[ -n "$PYTORCH_MAX_THINKING_TOKENS" ]] && SERVER_ARGS+=(--max-thinking-tokens "$PYTORCH_MAX_THINKING_TOKENS")

echo "[2/5] Starting native PyTorch server..."
rm -f "$SERVER_PORT_FILE"
(
    export CUDA_VISIBLE_DEVICES="$PYTORCH_GPU_DEVICE"
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
    exec "$PYTORCH_PYTHON" -u "${SERVER_ARGS[@]}"
) > "$SERVER_LOG" 2>&1 &
PYTORCH_SERVER_PID=$!
wait_for_server_port "$SERVER_PORT_FILE" "$PYTORCH_SERVER_PID" "$SERVER_LOG"
PYTORCH_PORT="$("$PYTORCH_PYTHON" - "$SERVER_PORT_FILE" <<'PY'
import json, sys
print(int(json.load(open(sys.argv[1], encoding="utf-8"))["port"]))
PY
)"
if [[ -n "$OVERLAP_SETTINGS_FILE" ]]; then
    "$PYTORCH_PYTHON" "$SETTINGS_UPDATER" "$OVERLAP_SETTINGS_FILE" --resolved-port "$PYTORCH_PORT" --status server_ready
fi
wait_for_health "http://127.0.0.1:$PYTORCH_PORT/health" "$PYTORCH_SERVER_PID" "$SERVER_LOG"

echo "[3/5] Checking datasets and scorer dependencies..."
ensure_ifeval_runtime
prepare_datasets

echo "[4/5] Running benchmarks..."
PIPELINE_FAILED=0
IFS=',' read -ra BENCH_ARRAY <<< "$SEQ_EVAL_BENCHMARK"
for bench_spec in "${BENCH_ARRAY[@]}"; do
    bench_spec="${bench_spec//[[:space:]]/}"
    [[ -z "$bench_spec" ]] && continue
    bench_name="${bench_spec%%:*}"
    eval_dir="$SEQ_EVAL_OUTPUT_DIR/eval-results/$bench_name"
    mkdir -p "$eval_dir"
    bench_log="$eval_dir/pytorch_benchmark.log"
    : > "$bench_log"
    : > "$REQUEST_STATS_FILE"

    if [[ "$bench_name" == "mt-bench" ]]; then
        if [[ -n "$SEQ_EVAL_JUDGE_SERVER_TYPE" && "$SEQ_EVAL_JUDGE_SERVER_TYPE" != "openai" ]]; then
            echo "ERROR: MT-Bench currently supports only an OpenAI-compatible judge endpoint." >&2
            exit 1
        fi
        EVAL_ARGS=(
            "$MT_BENCH_SCRIPT"
            --candidate-server-address "http://127.0.0.1:$PYTORCH_PORT/v1"
            --candidate-model "$PYTORCH_MODEL_TAG"
            --candidate-tokenizer "$PYTORCH_MODEL"
            --candidate-concurrency "$SEQ_EVAL_CLIENT_CONCURRENCY"
            --judge-concurrency "$SEQ_EVAL_JUDGE_CONCURRENCY"
            --data-dir "$MT_BENCH_DATA_DIR"
            --output-dir "$eval_dir"
            --max-tokens "$SEQ_EVAL_MT_BENCH_MAX_TOKENS"
            --resume
        )
        [[ -n "$SEQ_EVAL_MAX_SAMPLES" ]] && EVAL_ARGS+=(--max-samples "$SEQ_EVAL_MAX_SAMPLES")
        if [[ -z "$SEQ_EVAL_MAX_SAMPLES" ]] && is_enabled "$SEQ_EVAL_QUICK_TEST"; then EVAL_ARGS+=(--max-samples 10); fi
        [[ -n "$SEQ_EVAL_JUDGE_MODEL" ]] && EVAL_ARGS+=(--judge-model "$SEQ_EVAL_JUDGE_MODEL")
        [[ -n "$SEQ_EVAL_JUDGE_SERVER_ADDRESS" ]] && EVAL_ARGS+=(--judge-server-address "$SEQ_EVAL_JUDGE_SERVER_ADDRESS")
        [[ -n "$SEQ_EVAL_MT_BENCH_CHAT_TEMPLATE_SHA256" ]] && EVAL_ARGS+=(--expected-chat-template-sha256 "$SEQ_EVAL_MT_BENCH_CHAT_TEMPLATE_SHA256")
        is_enabled "$PYTORCH_ENABLE_THINKING" && EVAL_ARGS+=(--candidate-enable-thinking)
        is_enabled "$SEQ_EVAL_STRIP_THINKING" && EVAL_ARGS+=(--strip-thinking)
        is_enabled "$SEQ_EVAL_SKIP_JUDGE_API_KEY_CHECK" && EVAL_ARGS+=(--skip-judge-api-key-check)
    elif [[ "$bench_name" == "alpaca-eval" ]]; then
        if [[ -n "$SEQ_EVAL_JUDGE_SERVER_TYPE" && "$SEQ_EVAL_JUDGE_SERVER_TYPE" != "openai" ]]; then
            echo "ERROR: AlpacaEval currently supports only an OpenAI-compatible judge endpoint." >&2
            exit 1
        fi
        EVAL_ARGS=(
            "$ALPACA_EVAL_SCRIPT"
            --candidate-server-address "http://127.0.0.1:$PYTORCH_PORT/v1"
            --candidate-model "$PYTORCH_MODEL_TAG"
            --candidate-tokenizer "$PYTORCH_MODEL"
            --candidate-concurrency "$SEQ_EVAL_CLIENT_CONCURRENCY"
            --judge-concurrency "$SEQ_EVAL_JUDGE_CONCURRENCY"
            --data-dir "$ALPACA_EVAL_DATA_DIR"
            --output-dir "$eval_dir"
            --max-tokens "$SEQ_EVAL_ALPACA_EVAL_MAX_TOKENS"
            --resume
        )
        [[ -n "$SEQ_EVAL_MAX_SAMPLES" ]] && EVAL_ARGS+=(--max-samples "$SEQ_EVAL_MAX_SAMPLES")
        if [[ -z "$SEQ_EVAL_MAX_SAMPLES" ]] && is_enabled "$SEQ_EVAL_QUICK_TEST"; then EVAL_ARGS+=(--max-samples 10); fi
        [[ -n "$SEQ_EVAL_JUDGE_MODEL" ]] && EVAL_ARGS+=(--judge-model "$SEQ_EVAL_JUDGE_MODEL")
        [[ -n "$SEQ_EVAL_JUDGE_SERVER_ADDRESS" ]] && EVAL_ARGS+=(--judge-server-address "$SEQ_EVAL_JUDGE_SERVER_ADDRESS")
        [[ -n "$SEQ_EVAL_ALPACA_EVAL_CHAT_TEMPLATE_SHA256" ]] && EVAL_ARGS+=(--expected-chat-template-sha256 "$SEQ_EVAL_ALPACA_EVAL_CHAT_TEMPLATE_SHA256")
        is_enabled "$PYTORCH_ENABLE_THINKING" && EVAL_ARGS+=(--candidate-enable-thinking)
        is_enabled "$SEQ_EVAL_STRIP_THINKING" && EVAL_ARGS+=(--strip-thinking)
        is_enabled "$SEQ_EVAL_SKIP_JUDGE_API_KEY_CHECK" && EVAL_ARGS+=(--skip-judge-api-key-check)
    else
        EVAL_ARGS=(
            "$EVAL_SCRIPT"
            --server-address "http://127.0.0.1:$PYTORCH_PORT/v1"
            --benchmark "$bench_spec"
            --output-dir "$SEQ_EVAL_OUTPUT_DIR"
            --expname "$SEQ_EVAL_EXPNAME"
            --model "$PYTORCH_MODEL_TAG"
            --tokens-to-generate "$SEQ_EVAL_TOKENS_TO_GENERATE"
            --temperature "$SEQ_EVAL_TEMPERATURE"
            --top-p "$SEQ_EVAL_TOP_P"
            --num-chunks "$SEQ_EVAL_NUM_CHUNKS"
            --max-concurrent-requests "$SEQ_EVAL_CLIENT_CONCURRENCY"
            --generation-algorithm nemotron
            --no-extra-body
        )
        [[ -n "$SEQ_EVAL_MAX_SAMPLES" ]] && EVAL_ARGS+=(--max-samples "$SEQ_EVAL_MAX_SAMPLES")
        is_enabled "$SEQ_EVAL_QUICK_TEST" && EVAL_ARGS+=(--quick-test)
        is_enabled "$SEQ_EVAL_KEEP_THINKING" && EVAL_ARGS+=(--keep-thinking)
        is_enabled "$SEQ_EVAL_STRIP_THINKING" && EVAL_ARGS+=(--strip-thinking)
        is_enabled "$SEQ_EVAL_DISABLE_THINKING" && EVAL_ARGS+=(--disable-thinking)
        [[ -n "$SEQ_EVAL_MATH_PROMPT_CONFIG" ]] && EVAL_ARGS+=(--math-prompt-config "$SEQ_EVAL_MATH_PROMPT_CONFIG")
        [[ -n "$SEQ_EVAL_JUDGE_MODEL" ]] && EVAL_ARGS+=(--judge-model "$SEQ_EVAL_JUDGE_MODEL")
        [[ -n "$SEQ_EVAL_JUDGE_SERVER_ADDRESS" ]] && EVAL_ARGS+=(--judge-server-address "$SEQ_EVAL_JUDGE_SERVER_ADDRESS")
        [[ -n "$SEQ_EVAL_JUDGE_SERVER_TYPE" ]] && EVAL_ARGS+=(--judge-server-type "$SEQ_EVAL_JUDGE_SERVER_TYPE")
        is_enabled "$SEQ_EVAL_SKIP_JUDGE_API_KEY_CHECK" && EVAL_ARGS+=(--skip-judge-api-key-check)
    fi

    echo "--- benchmark: $bench_spec ---"
    bench_start="$(date +%s.%N)"
    set +e
    "$EVAL_PYTHON" "${EVAL_ARGS[@]}" 2>&1 | tee "$bench_log"
    eval_status=${PIPESTATUS[0]}
    set -e
    bench_end="$(date +%s.%N)"
    wall_time="$("$EVAL_PYTHON" - "$bench_start" "$bench_end" <<'PY'
import sys
print(f"{float(sys.argv[2]) - float(sys.argv[1]):.6f}")
PY
)"

    metrics_json="$eval_dir/metrics.json"
    accuracy_status="available"
    merge_extra_args=()
    if is_enabled "$PYTORCH_EFFICIENCY_ONLY"; then
        accuracy_status="ignored_efficiency_only"
    fi
    if [[ "$eval_status" != "0" || ! -f "$metrics_json" ]]; then
        if is_enabled "$PYTORCH_EFFICIENCY_ONLY" && generation_outputs_complete "$eval_dir" && request_stats_has_success && request_stats_failures_are_oom; then
            accuracy_status="skipped_after_eval_failure"
            merge_extra_args+=(--create-metrics-if-missing)
            echo "WARNING: $bench_name accuracy/scoring did not complete; preserving successful-request efficiency metrics only." | tee -a "$bench_log"
        elif [[ "$eval_status" != "0" ]]; then
            write_compact_error "$bench_name" benchmark_eval "$eval_status" "Benchmark evaluation failed and no successful efficiency stats are available" "$bench_log"
            PIPELINE_FAILED=1
            continue
        else
            write_compact_error "$bench_name" metrics_missing 1 "Benchmark runner did not produce metrics.json and no successful efficiency stats are available" "$bench_log"
            PIPELINE_FAILED=1
            continue
        fi
    fi

    cp "$REQUEST_STATS_FILE" "$eval_dir/pytorch_request_stats.jsonl"
    set +e
    "$EVAL_PYTHON" "$MERGE_SCRIPT" --metrics-json "$metrics_json" --request-stats-file "$REQUEST_STATS_FILE" --benchmark "$bench_name" --wall-time-s "$wall_time" --accuracy-status "$accuracy_status" "${merge_extra_args[@]}" 2>&1 | tee -a "$bench_log"
    merge_status=${PIPESTATUS[0]}
    set -e
    if [[ "$merge_status" != "0" ]]; then
        write_compact_error "$bench_name" metrics_merge "$merge_status" "PyTorch metrics merge failed" "$bench_log"
        PIPELINE_FAILED=1
        continue
    fi
    if ! write_compact_metrics "$bench_name" "$metrics_json"; then
        write_compact_error "$bench_name" compact_metrics_write 1 "Failed to preserve compact metrics and benchmark artifacts" "$bench_log"
        PIPELINE_FAILED=1
        continue
    fi
    if [[ -n "$PYTORCH_FINAL_OUTPUT_DIR" ]]; then
        "$EVAL_PYTHON" "$REPORT_SCRIPT" --result-dir "$PYTORCH_FINAL_OUTPUT_DIR" --baseline-block16-dir "$OVERLAP_BASELINE_BLOCK16_DIR" --baseline-block32-dir "$OVERLAP_BASELINE_BLOCK32_DIR" || {
            write_compact_error "$bench_name" report_update 1 "Failed to update incremental report.md" "$bench_log"
            PIPELINE_FAILED=1
            continue
        }
    fi
done

echo "[5/5] Completed margin-risk-conditional-rank-overlap PyTorch evaluation pipeline."
if [[ "$PIPELINE_FAILED" != "0" ]]; then
    echo "WARNING: one or more benchmarks failed; compact error files were written." >&2
    exit 1
fi
exit 0
