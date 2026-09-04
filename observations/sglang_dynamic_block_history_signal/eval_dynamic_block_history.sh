#!/bin/bash
# Real SGLang dynamic canonical + L8/L16/L32 counterfactual observation.

set -euo pipefail

ORIGINAL_ARGS=("$@")
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
EVAL_SGLANG="${NLD_DYNAMIC_EVAL_SGLANG:-$PROJECT_DIR/observations/eval_sglang.sh}"
REPORTING="$SCRIPT_DIR/reporting.py"
SEARCH="$SCRIPT_DIR/search.py"
RESULTS_ROOT="${NLD_OBSERVATION_RESULTS_ROOT:-/data/home/wly/dLLM/NLD_results/observations}/sglang_dynamic_block_history_signal_results"

DEFAULT_BENCHMARKS="gsm8k:1,human-eval:1,mbpp:1,math-500:1,aime25:1,gpqa:1,ifeval:1,livecodebench-cpp:1,mmlu:1"
DEFAULT_MODEL="/data1/linyewei/models/Nemotron-Labs-Diffusion-8B"
DEFAULT_PYTHON="/data/home/wly/.conda/envs/nld_sglang/bin/python"
[[ -x "$DEFAULT_PYTHON" ]] || DEFAULT_PYTHON="python"

usage() {
    cat <<EOF
Usage: $0 --stage collect|search|validate|remaining|all [options]

Core:
  --stage NAME                collect: exploration; search: CPU search; validate: two frozen reruns
                              remaining: reuse exploration then search+validate; all: all stages
  --run-dir DIR               Existing run for search/validate/remaining; completed datasets are skipped
  --benchmarks LIST           NeMo-Skills specs (default: nine non-AIME24 datasets; MMLU is always moved last)
  --mode MODE                 linearspec_lora or linearspec_base (default: linearspec_lora)
  --block-sizes LIST          Counterfactual candidates (default: 8,16,32)
  --block-size N              Physical SGLang/KV block (default: 32; must cover every candidate)

Evaluation:
  --model PATH                Model path
  --served-model-name NAME    OpenAI served model label
  --tokens N                  Max completion tokens (default: 8192)
  --context-length N          SGLang context length (default: 10240)
  --max-samples N             Limit every non-MMLU dataset; empty means full
  --mmlu-max-samples N        MMLU subset size (default: 2000, about 10%-20%)
  --temperature V             Must be 0 for paired greedy traces (default: 0)
  --top-p V                   NeMo-Skills top_p (default: 0.95)

SGLang/resources:
  --gpu-devices LIST|auto     CUDA_VISIBLE_DEVICES, e.g. 0 or 0,1; auto reselects one GPU per attempt
  --auto-gpu-min-free-gb V   Required free memory for auto selection (default: 48 GiB)
  --tp-size N                 Tensor parallel size; default inferred by eval_sglang.sh
  --batch-size N              SGLang max running requests (default: 1)
  --client-concurrency N      NeMo/proxy concurrency (default: 1)
  --gpu-memory-reserve-gb V   Reserve V GiB before loading model (default: 0)
  --mem-fraction V            SGLang static memory fraction (default: 0.55)
  --dtype NAME                bfloat16 by default
  --port N                    Optional server port; omitted means existing pipeline auto-searches
  --proxy-port N              Optional proxy port; omitted means existing pipeline auto-searches
  --nemo-skills-data-dir DIR  Persistent dataset cache
  --sglang-python PATH        SGLang/search Python
  --eval-python PATH          NeMo-Skills Python (default: SGLang Python)
  --sglang-src DIR            SGLang source root
  --sglang-work-dir DIR       SGLang cache/work root
  --lora-path DIR             Draft LoRA path
  --lora-mode MODE            draft_only or both (default: draft_only)
  --extra-server-args TEXT    Additional launch_server args; observation always adds --disable-cuda-graph

Search/protocol:
  --seed N                    Exploration seed (default: 20260831)
  --split-seed N              Prompt-hash split seed (default: 20260831)
  --min-large-precision V     Min significant-gain precision for promotion (default: 0.80)
  --max-large-waste V         Max fraction of promotions gaining <=1 token (default: 0.10)
  --min-safe8 V               S16 min no-loss rate for downgrade to L8 (default: 0.98)
  --max-invalid-row-rate V    Max trace rows conservatively excluded for integrity (default: 0.05)
  --dataset-max-attempts N    Fresh-server attempts for each dataset (default: 3)
  --dataset-retry-delay-s N   Delay between failed attempts in seconds (default: 10)
  --allow-partial-search      Development only: allow search/validation without all nine datasets
                              CPU search automatically keeps a baseline SGLang server on the selected GPU(s)
  --dry-run                   Validate/print commands without creating output or launching a model

Example:
  bash $0 --stage all --gpu-devices 0 --batch-size 4 --client-concurrency 4
EOF
}

STAGE=""
RUN_DIR=""
BENCHMARKS="$DEFAULT_BENCHMARKS"
MODE="linearspec_lora"
BLOCK_SIZES="8,16,32"
PHYSICAL_BLOCK_SIZE="32"
MODEL="$DEFAULT_MODEL"
SERVED_MODEL_NAME="nemotron-labs-diffusion-8b"
TOKENS="8192"
CONTEXT_LENGTH="10240"
MAX_SAMPLES=""
MMLU_MAX_SAMPLES="2000"
TEMPERATURE="0"
TOP_P="0.95"
GPU_DEVICES="0"
GPU_DEVICES_USER_SET="false"
AUTO_GPU_MIN_FREE_GB="48"
TP_SIZE=""
BATCH_SIZE="1"
CLIENT_CONCURRENCY="1"
GPU_MEMORY_RESERVE_GB="0"
MEM_FRACTION="0.55"
DTYPE="bfloat16"
PORT=""
PROXY_PORT=""
NEMO_SKILLS_DATA_DIR=""
SGLANG_PYTHON="$DEFAULT_PYTHON"
EVAL_PYTHON=""
SGLANG_SRC=""
SGLANG_WORK_DIR=""
LORA_PATH=""
LORA_MODE="draft_only"
EXTRA_SERVER_ARGS=""
SEED="20260831"
SPLIT_SEED="20260831"
MIN_LARGE_PRECISION="0.80"
MAX_LARGE_WASTE="0.10"
MIN_SAFE8="0.98"
MAX_INVALID_ROW_RATE="0.05"
DATASET_MAX_ATTEMPTS="3"
DATASET_RETRY_DELAY_S="10"
ALLOW_PARTIAL_SEARCH="false"
DRY_RUN="false"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --stage) STAGE="$2"; shift 2 ;;
        --run-dir) RUN_DIR="$2"; shift 2 ;;
        --benchmarks) BENCHMARKS="$2"; shift 2 ;;
        --mode) MODE="$2"; shift 2 ;;
        --block-sizes) BLOCK_SIZES="$2"; shift 2 ;;
        --block-size|--physical-block-size) PHYSICAL_BLOCK_SIZE="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --served-model-name|--model-name) SERVED_MODEL_NAME="$2"; shift 2 ;;
        --tokens) TOKENS="$2"; shift 2 ;;
        --context-length) CONTEXT_LENGTH="$2"; shift 2 ;;
        --max-samples) MAX_SAMPLES="$2"; shift 2 ;;
        --mmlu-max-samples) MMLU_MAX_SAMPLES="$2"; shift 2 ;;
        --temperature) TEMPERATURE="$2"; shift 2 ;;
        --top-p) TOP_P="$2"; shift 2 ;;
        --gpu-devices) GPU_DEVICES="$2"; GPU_DEVICES_USER_SET="true"; shift 2 ;;
        --auto-gpu-min-free-gb) AUTO_GPU_MIN_FREE_GB="$2"; shift 2 ;;
        --tp-size) TP_SIZE="$2"; shift 2 ;;
        --batch-size|--max-running-requests) BATCH_SIZE="$2"; shift 2 ;;
        --client-concurrency) CLIENT_CONCURRENCY="$2"; shift 2 ;;
        --gpu-memory-reserve-gb) GPU_MEMORY_RESERVE_GB="$2"; shift 2 ;;
        --mem-fraction) MEM_FRACTION="$2"; shift 2 ;;
        --dtype) DTYPE="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --proxy-port) PROXY_PORT="$2"; shift 2 ;;
        --nemo-skills-data-dir) NEMO_SKILLS_DATA_DIR="$2"; shift 2 ;;
        --sglang-python) SGLANG_PYTHON="$2"; shift 2 ;;
        --eval-python) EVAL_PYTHON="$2"; shift 2 ;;
        --sglang-src) SGLANG_SRC="$2"; shift 2 ;;
        --sglang-work-dir) SGLANG_WORK_DIR="$2"; shift 2 ;;
        --lora-path) LORA_PATH="$2"; shift 2 ;;
        --lora-mode) LORA_MODE="$2"; shift 2 ;;
        --extra-server-args) EXTRA_SERVER_ARGS="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --split-seed) SPLIT_SEED="$2"; shift 2 ;;
        --min-large-precision) MIN_LARGE_PRECISION="$2"; shift 2 ;;
        --max-large-waste) MAX_LARGE_WASTE="$2"; shift 2 ;;
        --min-safe8) MIN_SAFE8="$2"; shift 2 ;;
        --max-invalid-row-rate) MAX_INVALID_ROW_RATE="$2"; shift 2 ;;
        --dataset-max-attempts) DATASET_MAX_ATTEMPTS="$2"; shift 2 ;;
        --dataset-retry-delay-s) DATASET_RETRY_DELAY_S="$2"; shift 2 ;;
        --allow-partial-search) ALLOW_PARTIAL_SEARCH="true"; shift ;;
        --dry-run) DRY_RUN="true"; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "ERROR: unknown option $1" >&2; usage; exit 1 ;;
    esac
done

[[ -n "$STAGE" ]] || { echo "ERROR: --stage is required" >&2; exit 1; }
case "$STAGE" in collect|search|validate|remaining|all) ;; *) echo "ERROR: invalid --stage $STAGE" >&2; exit 1 ;; esac
case "$MODE" in linearspec_lora|linearspec_base) ;; *) echo "ERROR: mode must be linearspec_lora or linearspec_base" >&2; exit 1 ;; esac
[[ "$TEMPERATURE" == "0" || "$TEMPERATURE" == "0.0" ]] || { echo "ERROR: paired counterfactual protocol requires --temperature 0" >&2; exit 1; }
[[ -x "$SGLANG_PYTHON" || "$SGLANG_PYTHON" == "python" ]] || { echo "ERROR: invalid --sglang-python" >&2; exit 1; }
[[ -n "$EVAL_PYTHON" ]] || EVAL_PYTHON="$SGLANG_PYTHON"

arg_was_set() {
    local needle="$1" arg
    for arg in "${ORIGINAL_ARGS[@]}"; do
        [[ "$arg" == "$needle" ]] && return 0
    done
    return 1
}

# A resumed run must use the collection protocol that produced its trace.  An
# explicitly supplied CLI value wins; everything else is restored from the run
# settings instead of silently falling back to this script's defaults.
if [[ -n "$RUN_DIR" && -f "$RUN_DIR/settings.json" ]]; then
    mapfile -d '' -t RESUME_SETTINGS < <("$SGLANG_PYTHON" -c 'import json,sys
x=json.load(open(sys.argv[1],encoding="utf-8"))
keys=("benchmarks","model","served_model_name","mode","block_sizes","physical_block_size","gpu_devices","tp_size","batch_size","client_concurrency","gpu_memory_reserve_gb","tokens","context_length","temperature","top_p","dtype","mem_fraction","max_samples","mmlu_max_samples","lora_path","lora_mode","seed","split_seed","min_large_precision","max_large_waste","min_safe8","max_invalid_row_rate","dataset_max_attempts","dataset_retry_delay_s","allow_partial_search","extra_server_args","auto_gpu_min_free_gb")
for key in keys:
    value=x.get(key,"")
    sys.stdout.write(("" if value is None else str(value))+"\0")' "$RUN_DIR/settings.json")
    arg_was_set --benchmarks || BENCHMARKS="${RESUME_SETTINGS[0]:-$BENCHMARKS}"
    arg_was_set --model || MODEL="${RESUME_SETTINGS[1]:-$MODEL}"
    { arg_was_set --served-model-name || arg_was_set --model-name; } || SERVED_MODEL_NAME="${RESUME_SETTINGS[2]:-$SERVED_MODEL_NAME}"
    arg_was_set --mode || MODE="${RESUME_SETTINGS[3]:-$MODE}"
    arg_was_set --block-sizes || BLOCK_SIZES="${RESUME_SETTINGS[4]:-$BLOCK_SIZES}"
    { arg_was_set --block-size || arg_was_set --physical-block-size; } || PHYSICAL_BLOCK_SIZE="${RESUME_SETTINGS[5]:-$PHYSICAL_BLOCK_SIZE}"
    arg_was_set --gpu-devices || GPU_DEVICES="${RESUME_SETTINGS[6]:-$GPU_DEVICES}"
    if ! arg_was_set --tp-size && [[ "${RESUME_SETTINGS[7]:-}" != "auto" ]]; then TP_SIZE="${RESUME_SETTINGS[7]:-$TP_SIZE}"; fi
    { arg_was_set --batch-size || arg_was_set --max-running-requests; } || BATCH_SIZE="${RESUME_SETTINGS[8]:-$BATCH_SIZE}"
    arg_was_set --client-concurrency || CLIENT_CONCURRENCY="${RESUME_SETTINGS[9]:-$CLIENT_CONCURRENCY}"
    arg_was_set --gpu-memory-reserve-gb || GPU_MEMORY_RESERVE_GB="${RESUME_SETTINGS[10]:-$GPU_MEMORY_RESERVE_GB}"
    arg_was_set --tokens || TOKENS="${RESUME_SETTINGS[11]:-$TOKENS}"
    arg_was_set --context-length || CONTEXT_LENGTH="${RESUME_SETTINGS[12]:-$CONTEXT_LENGTH}"
    arg_was_set --temperature || TEMPERATURE="${RESUME_SETTINGS[13]:-$TEMPERATURE}"
    arg_was_set --top-p || TOP_P="${RESUME_SETTINGS[14]:-$TOP_P}"
    arg_was_set --dtype || DTYPE="${RESUME_SETTINGS[15]:-$DTYPE}"
    arg_was_set --mem-fraction || MEM_FRACTION="${RESUME_SETTINGS[16]:-$MEM_FRACTION}"
    arg_was_set --max-samples || MAX_SAMPLES="${RESUME_SETTINGS[17]:-$MAX_SAMPLES}"
    arg_was_set --mmlu-max-samples || MMLU_MAX_SAMPLES="${RESUME_SETTINGS[18]:-$MMLU_MAX_SAMPLES}"
    arg_was_set --lora-path || LORA_PATH="${RESUME_SETTINGS[19]:-$LORA_PATH}"
    arg_was_set --lora-mode || LORA_MODE="${RESUME_SETTINGS[20]:-$LORA_MODE}"
    arg_was_set --seed || SEED="${RESUME_SETTINGS[21]:-$SEED}"
    arg_was_set --split-seed || SPLIT_SEED="${RESUME_SETTINGS[22]:-$SPLIT_SEED}"
    arg_was_set --min-large-precision || MIN_LARGE_PRECISION="${RESUME_SETTINGS[23]:-$MIN_LARGE_PRECISION}"
    arg_was_set --max-large-waste || MAX_LARGE_WASTE="${RESUME_SETTINGS[24]:-$MAX_LARGE_WASTE}"
    arg_was_set --min-safe8 || MIN_SAFE8="${RESUME_SETTINGS[25]:-$MIN_SAFE8}"
    arg_was_set --max-invalid-row-rate || MAX_INVALID_ROW_RATE="${RESUME_SETTINGS[26]:-$MAX_INVALID_ROW_RATE}"
    arg_was_set --dataset-max-attempts || DATASET_MAX_ATTEMPTS="${RESUME_SETTINGS[27]:-$DATASET_MAX_ATTEMPTS}"
    arg_was_set --dataset-retry-delay-s || DATASET_RETRY_DELAY_S="${RESUME_SETTINGS[28]:-$DATASET_RETRY_DELAY_S}"
    if ! arg_was_set --allow-partial-search && [[ "${RESUME_SETTINGS[29]:-false}" == "true" ]]; then ALLOW_PARTIAL_SEARCH="true"; fi
    arg_was_set --extra-server-args || EXTRA_SERVER_ARGS="${RESUME_SETTINGS[30]:-$EXTRA_SERVER_ARGS}"
    arg_was_set --auto-gpu-min-free-gb || AUTO_GPU_MIN_FREE_GB="${RESUME_SETTINGS[31]:-$AUTO_GPU_MIN_FREE_GB}"
fi

# Validate again after optional resume restoration, because settings.json may
# replace parser defaults when the corresponding option was not supplied.
case "$MODE" in linearspec_lora|linearspec_base) ;; *) echo "ERROR: restored mode must be linearspec_lora or linearspec_base" >&2; exit 1 ;; esac
[[ "$TEMPERATURE" == "0" || "$TEMPERATURE" == "0.0" ]] || { echo "ERROR: paired counterfactual protocol requires restored --temperature 0" >&2; exit 1; }
"$SGLANG_PYTHON" -c 'import sys
x=float(sys.argv[1])
raise SystemExit(0 if 0.0 <= x <= 1.0 else 1)' "$MAX_INVALID_ROW_RATE" || { echo "ERROR: --max-invalid-row-rate must be in [0,1]" >&2; exit 1; }
[[ "$DATASET_MAX_ATTEMPTS" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: --dataset-max-attempts must be a positive integer" >&2; exit 1; }
[[ "$DATASET_RETRY_DELAY_S" =~ ^[0-9]+$ ]] || { echo "ERROR: --dataset-retry-delay-s must be a nonnegative integer" >&2; exit 1; }
"$SGLANG_PYTHON" -c 'import sys
x=float(sys.argv[1])
raise SystemExit(0 if x > 0 else 1)' "$AUTO_GPU_MIN_FREE_GB" || { echo "ERROR: --auto-gpu-min-free-gb must be positive" >&2; exit 1; }
if [[ "$GPU_DEVICES" == "auto" && -n "$TP_SIZE" && "$TP_SIZE" != "1" ]]; then
    echo "ERROR: --gpu-devices auto currently selects one GPU and requires --tp-size 1 or omitted" >&2
    exit 1
fi

ORDERED_BENCHMARKS="$($SGLANG_PYTHON -c 'import sys
items=[x.strip() for x in sys.argv[1].split(",") if x.strip()]
if any(x.split(":",1)[0]=="aime24" for x in items): raise SystemExit("AIME24 is excluded from this experiment")
m=[x for x in items if x.split(":",1)[0]=="mmlu"]
o=[x for x in items if x.split(":",1)[0]!="mmlu"]
print(",".join(o+m))' "$BENCHMARKS")"
[[ -n "$ORDERED_BENCHMARKS" ]] || { echo "ERROR: --benchmarks must contain at least one dataset" >&2; exit 1; }

NORMALIZED_CANDIDATES="$($SGLANG_PYTHON -c 'import sys; print(",".join(map(str,sorted(set(map(int,sys.argv[1].split(",")))))))' "$BLOCK_SIZES")"
[[ "$NORMALIZED_CANDIDATES" == "8,16,32" ]] || { echo "ERROR: this paired protocol requires exactly --block-sizes 8,16,32" >&2; exit 1; }
MAX_CANDIDATE="32"
(( PHYSICAL_BLOCK_SIZE >= MAX_CANDIDATE )) || { echo "ERROR: physical block must cover candidates" >&2; exit 1; }

if [[ -z "$RUN_DIR" && ( "$STAGE" == "search" || "$STAGE" == "validate" || "$STAGE" == "remaining" ) ]]; then
    echo "ERROR: --run-dir is required for $STAGE" >&2
    exit 1
fi

if [[ "$DRY_RUN" == "true" ]]; then
    echo "stage=$STAGE"
    echo "benchmarks=$ORDERED_BENCHMARKS"
    echo "candidates=$BLOCK_SIZES physical=$PHYSICAL_BLOCK_SIZE"
    echo "gpu=$GPU_DEVICES tp=${TP_SIZE:-auto} batch=$BATCH_SIZE concurrency=$CLIENT_CONCURRENCY reserve_gb=$GPU_MEMORY_RESERVE_GB"
    echo "dataset_attempts=$DATASET_MAX_ATTEMPTS retry_delay_s=$DATASET_RETRY_DELAY_S invalid_row_limit=$MAX_INVALID_ROW_RATE auto_gpu_min_free_gb=$AUTO_GPU_MIN_FREE_GB"
    echo "MMLU is last and capped at $MMLU_MAX_SAMPLES samples"
    echo "SGLang launch is process-locally patched and CUDA graphs are disabled for variable-length shadow views"
    echo "Each shadow branch ends with a CUDA correctness barrier; successful attempt traces are atomically promoted"
    [[ -z "$RUN_DIR" ]] || echo "Existing run directories are protected by a nonblocking single-writer flock"
    if [[ "$STAGE" == "search" || "$STAGE" == "remaining" || "$STAGE" == "all" ]]; then
        echo "CPU search will concurrently run the unmodified SGLang+NeMoSkills baseline on nine non-AIME24 datasets"
    fi
    exit 0
fi

if [[ -z "$RUN_DIR" ]]; then
    TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
    RUN_DIR="$RESULTS_ROOT/dynamic_block_history_${TIMESTAMP}"
    COMMAND="bash observations/sglang_dynamic_block_history_signal/eval_dynamic_block_history.sh ${ORIGINAL_ARGS[*]}"
    SETTINGS_JSON="$($SGLANG_PYTHON -c 'import json,sys
keys="stage benchmarks model served_model_name mode block_sizes physical_block_size gpu_devices tp_size batch_size client_concurrency gpu_memory_reserve_gb tokens context_length temperature top_p dtype mem_fraction max_samples mmlu_max_samples lora_path lora_mode seed split_seed min_large_precision max_large_waste min_safe8 max_invalid_row_rate dataset_max_attempts dataset_retry_delay_s allow_partial_search search_guardian shadow_branch_barrier trace_commit_protocol run_lock_protocol extra_server_args auto_gpu_min_free_gb command".split()
print(json.dumps(dict(zip(keys,sys.argv[1:])),ensure_ascii=False))' "$STAGE" "$ORDERED_BENCHMARKS" "$MODEL" "$SERVED_MODEL_NAME" "$MODE" "$BLOCK_SIZES" "$PHYSICAL_BLOCK_SIZE" "$GPU_DEVICES" "${TP_SIZE:-auto}" "$BATCH_SIZE" "$CLIENT_CONCURRENCY" "$GPU_MEMORY_RESERVE_GB" "$TOKENS" "$CONTEXT_LENGTH" "$TEMPERATURE" "$TOP_P" "$DTYPE" "$MEM_FRACTION" "$MAX_SAMPLES" "$MMLU_MAX_SAMPLES" "$LORA_PATH" "$LORA_MODE" "$SEED" "$SPLIT_SEED" "$MIN_LARGE_PRECISION" "$MAX_LARGE_WASTE" "$MIN_SAFE8" "$MAX_INVALID_ROW_RATE" "$DATASET_MAX_ATTEMPTS" "$DATASET_RETRY_DELAY_S" "$ALLOW_PARTIAL_SEARCH" "baseline_9datasets_concurrent_eval" "cuda_synchronize_after_each_shadow_branch" "attempt_trace_then_atomic_move" "exclusive_flock" "$EXTRA_SERVER_ARGS" "$AUTO_GPU_MIN_FREE_GB" "$COMMAND")"
    "$SGLANG_PYTHON" "$REPORTING" init --run-dir "$RUN_DIR" --settings-json "$SETTINGS_JSON"
else
    [[ -d "$RUN_DIR" ]] || { echo "ERROR: run dir does not exist: $RUN_DIR" >&2; exit 1; }
fi

# The run directory is a single-writer state machine.  Its run_state, report,
# and canonical trace files must never be updated by two resumed runners at
# once.
command -v flock >/dev/null 2>&1 || { echo "ERROR: flock is required for per-run locking" >&2; exit 1; }
mkdir -p "$RUN_DIR/runtime"
RUN_LOCK_FILE="$RUN_DIR/runtime/runner.lock"
exec 9>>"$RUN_LOCK_FILE"
if ! flock -n 9; then
    echo "ERROR: another dynamic-block runner already owns this run directory: $RUN_DIR" >&2
    echo "       Wait for it to finish; do not start two --stage remaining commands on one run." >&2
    exit 89
fi
: > "$RUN_LOCK_FILE"
printf 'pid=%s\nstarted_at=%s\ncommand=%q\n' "$$" "$(date --iso-8601=seconds)" "bash observations/sglang_dynamic_block_history_signal/eval_dynamic_block_history.sh ${ORIGINAL_ARGS[*]}" >&9

event() {
    local phase="$1" dataset="$2" status="$3" records="$4" message="$5"
    local payload
    payload="$($SGLANG_PYTHON -c 'import json,sys; print(json.dumps(dict(zip(("phase","dataset","status","records","message"),sys.argv[1:])),ensure_ascii=False))' "$phase" "$dataset" "$status" "$records" "$message")"
    "$SGLANG_PYTHON" "$REPORTING" event --run-dir "$RUN_DIR" --event-json "$payload"
}

progress_bar() {
    local label="$1" current="$2" total="$3" detail="$4"
    local width=30 filled empty filled_text empty_text percent
    filled=$(( current * width / total ))
    empty=$(( width - filled ))
    percent=$(( current * 100 / total ))
    printf -v filled_text "%*s" "$filled" ""
    printf -v empty_text "%*s" "$empty" ""
    filled_text="${filled_text// /#}"
    empty_text="${empty_text// /-}"
    printf '[进度][%s] |%s%s| %d/%d (%3d%%) %s\n' "$label" "$filled_text" "$empty_text" "$current" "$total" "$percent" "$detail"
}

resolve_runtime_gpu() {
    if [[ "$GPU_DEVICES" != "auto" ]]; then
        printf '%s\n' "$GPU_DEVICES"
        return 0
    fi
    local inventory selected
    if [[ -n "${NLD_DYNAMIC_GPU_INVENTORY:-}" ]]; then
        inventory="$NLD_DYNAMIC_GPU_INVENTORY"
    else
        command -v nvidia-smi >/dev/null 2>&1 || {
            echo "ERROR: --gpu-devices auto requires nvidia-smi" >&2
            return 1
        }
        inventory="$(nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits 2>/dev/null)" || {
            echo "ERROR: nvidia-smi failed during automatic GPU selection" >&2
            return 1
        }
    fi
    selected="$(printf '%s\n' "$inventory" | "$SGLANG_PYTHON" -c 'import sys
threshold=float(sys.argv[1])*1024
rows=[]
for line in sys.stdin:
    parts=[item.strip() for item in line.split(",")]
    if len(parts) != 3:
        continue
    try:
        index,free,util=int(parts[0]),float(parts[1]),float(parts[2])
    except ValueError:
        continue
    if free >= threshold:
        rows.append((util,-free,index,free))
if not rows:
    raise SystemExit(1)
rows.sort()
print(rows[0][2])' "$AUTO_GPU_MIN_FREE_GB")" || {
        echo "ERROR: no GPU currently has at least ${AUTO_GPU_MIN_FREE_GB} GiB free; auto selection will retry later" >&2
        return 1
    }
    local selected_line
    selected_line="$(printf '%s\n' "$inventory" | awk -F, -v gpu="$selected" '$1+0==gpu {print; exit}')"
    echo "[GPU自动选择] GPU $selected（index, free MiB, util%: ${selected_line:-unknown}）" >&2
    printf '%s\n' "$selected"
}

SEARCH_GUARDIAN_PGID=""
SEARCH_GUARDIAN_WRAPPER_PID=""
SEARCH_GUARDIAN_LOG=""

stop_search_guardian() {
    local reason="${1:-CPU 搜索结束}"
    [[ -n "$SEARCH_GUARDIAN_PGID" ]] || return 0
    local pgid="$SEARCH_GUARDIAN_PGID"
    echo "[GPU守护] 正在关闭 baseline SGLang 进程组 PGID=$pgid（$reason）"
    kill -TERM -- "-$pgid" 2>/dev/null || true
    local retry
    for retry in $(seq 1 30); do
        kill -0 -- "-$pgid" 2>/dev/null || break
        sleep 1
    done
    if kill -0 -- "-$pgid" 2>/dev/null; then
        echo "[GPU守护] 常规退出超时，强制清理专属进程组 PGID=$pgid"
        kill -KILL -- "-$pgid" 2>/dev/null || true
    fi
    [[ -z "$SEARCH_GUARDIAN_WRAPPER_PID" ]] || wait "$SEARCH_GUARDIAN_WRAPPER_PID" 2>/dev/null || true
    SEARCH_GUARDIAN_PGID=""
    SEARCH_GUARDIAN_WRAPPER_PID=""
    event "search_guardian" "baseline_9datasets" "stopped" "0" "$reason；九集 baseline 已终止，GPU 显存已交还给后续验证"
}

start_search_guardian() {
    command -v setsid >/dev/null 2>&1 || { echo "ERROR: setsid is required for the search GPU guardian" >&2; return 1; }
    command -v grep >/dev/null 2>&1 || { echo "ERROR: grep is required for the search GPU guardian" >&2; return 1; }
    local guardian_dir="$RUN_DIR/runtime/search_guardian_baseline"
    mkdir -p "$guardian_dir"
    SEARCH_GUARDIAN_LOG="$guardian_dir/guardian.log"
    if [[ -s "$SEARCH_GUARDIAN_LOG" ]]; then
        SEARCH_GUARDIAN_LOG="$guardian_dir/guardian_$(date +%Y%m%d_%H%M%S).log"
    fi
    local output_dir="$guardian_dir/baseline_eval"
    local guard_model="$MODEL" guard_model_name="$SERVED_MODEL_NAME" guard_mode="$MODE"
    local guard_gpu="$GPU_DEVICES" guard_tp="$TP_SIZE" guard_batch="$BATCH_SIZE" guard_client="$CLIENT_CONCURRENCY"
    local guard_reserve="$GPU_MEMORY_RESERVE_GB" guard_context="$CONTEXT_LENGTH"
    local guard_mem_fraction="$MEM_FRACTION" guard_dtype="$DTYPE"
    local guard_block="$PHYSICAL_BLOCK_SIZE" guard_lora_path="$LORA_PATH"
    local guard_lora_mode="$LORA_MODE" guard_extra_args="$EXTRA_SERVER_ARGS"
    local guard_tokens="$TOKENS" guard_temperature="$TEMPERATURE" guard_top_p="$TOP_P"
    local settings_file="$RUN_DIR/settings.json"
    if [[ -f "$settings_file" ]]; then
        local -a saved
        mapfile -d '' -t saved < <("$SGLANG_PYTHON" -c 'import json,sys
x=json.load(open(sys.argv[1],encoding="utf-8"))
keys=("model","served_model_name","mode","gpu_devices","tp_size","batch_size","client_concurrency","gpu_memory_reserve_gb","context_length","mem_fraction","dtype","physical_block_size","lora_path","lora_mode","extra_server_args","tokens","temperature","top_p")
for key in keys:
    value=x.get(key,"")
    sys.stdout.write(("" if value is None else str(value))+"\0")' "$settings_file")
        guard_model="${saved[0]:-$guard_model}"
        guard_model_name="${saved[1]:-$guard_model_name}"
        guard_mode="${saved[2]:-$guard_mode}"
        [[ "$GPU_DEVICES_USER_SET" == "true" ]] || guard_gpu="${saved[3]:-$guard_gpu}"
        if [[ -z "$guard_tp" && "${saved[4]:-}" != "auto" ]]; then guard_tp="${saved[4]:-}"; fi
        guard_batch="${saved[5]:-$guard_batch}"
        guard_client="${saved[6]:-$guard_client}"
        guard_reserve="${saved[7]:-$guard_reserve}"
        guard_context="${saved[8]:-$guard_context}"
        guard_mem_fraction="${saved[9]:-$guard_mem_fraction}"
        guard_dtype="${saved[10]:-$guard_dtype}"
        guard_block="${saved[11]:-$guard_block}"
        guard_lora_path="${saved[12]:-$guard_lora_path}"
        guard_lora_mode="${saved[13]:-$guard_lora_mode}"
        guard_extra_args="${saved[14]:-$guard_extra_args}"
        guard_tokens="${saved[15]:-$guard_tokens}"
        guard_temperature="${saved[16]:-$guard_temperature}"
        guard_top_p="${saved[17]:-$guard_top_p}"
    fi
    if [[ "$guard_gpu" == "auto" ]]; then
        guard_gpu="$(resolve_runtime_gpu)" || return 1
    fi
    local -a cmd=(bash "$EVAL_SGLANG" --mode "$guard_mode" --benchmarks "$DEFAULT_BENCHMARKS" --model "$guard_model" --served-model-name "$guard_model_name" --gpu-devices "$guard_gpu" --batch-size "$guard_batch" --client-concurrency "$guard_client" --gpu-memory-reserve-gb "$guard_reserve" --tokens "$guard_tokens" --context-length "$guard_context" --temperature "$guard_temperature" --top-p "$guard_top_p" --mem-fraction "$guard_mem_fraction" --dtype "$guard_dtype" --block-size "$guard_block" --cuda-graph-bs "1" --output-path "$output_dir" --keep-server)
    [[ -n "$guard_tp" ]] && cmd+=(--tp-size "$guard_tp")
    [[ -n "$PORT" ]] && cmd+=(--port "$PORT")
    [[ -n "$PROXY_PORT" ]] && cmd+=(--proxy-port "$PROXY_PORT")
    [[ -n "$NEMO_SKILLS_DATA_DIR" ]] && cmd+=(--nemo-skills-data-dir "$NEMO_SKILLS_DATA_DIR")
    [[ -n "$SGLANG_PYTHON" ]] && cmd+=(--sglang-python "$SGLANG_PYTHON")
    [[ -n "$EVAL_PYTHON" ]] && cmd+=(--eval-python "$EVAL_PYTHON")
    [[ -n "$SGLANG_SRC" ]] && cmd+=(--sglang-src "$SGLANG_SRC")
    [[ -n "$SGLANG_WORK_DIR" ]] && cmd+=(--sglang-work-dir "$SGLANG_WORK_DIR")
    [[ -n "$guard_lora_path" ]] && cmd+=(--lora-path "$guard_lora_path")
    cmd+=(--lora-mode "$guard_lora_mode")
    [[ -n "$guard_extra_args" ]] && cmd+=(--extra-server-args "$guard_extra_args")

    event "search_guardian" "baseline_9datasets" "starting" "0" "启动九个非 AIME24 数据集的正常 baseline 复现；日志：$SEARCH_GUARDIAN_LOG"
    echo "[GPU守护] 启动九集 baseline 复现（GPU=$guard_gpu, tokens=$guard_tokens, concurrency=$guard_client, mem_fraction=$guard_mem_fraction, reserve=${guard_reserve}GiB）"
    setsid "${cmd[@]}" > "$SEARCH_GUARDIAN_LOG" 2>&1 &
    SEARCH_GUARDIAN_PGID=$!
    SEARCH_GUARDIAN_WRAPPER_PID="$SEARCH_GUARDIAN_PGID"
    local wrapper_pid="$SEARCH_GUARDIAN_WRAPPER_PID" elapsed=0 status=0
    local ready_marker="[4/5] Running benchmark evaluation through timing proxy"
    while ! grep -Fq -- "$ready_marker" "$SEARCH_GUARDIAN_LOG" 2>/dev/null; do
        if ! kill -0 "$wrapper_pid" 2>/dev/null; then
            # Close the small race between the marker check and wrapper exit.
            if grep -Fq -- "$ready_marker" "$SEARCH_GUARDIAN_LOG" 2>/dev/null; then
                break
            fi
            wait "$wrapper_pid" || status=$?
            echo "ERROR: nine-dataset baseline exited before benchmark evaluation started; log tail:" >&2
            tail -100 "$SEARCH_GUARDIAN_LOG" >&2 || true
            event "search_guardian" "baseline_9datasets" "failed" "0" "baseline 在真实测评开始前退出，exit=$status；详见 $SEARCH_GUARDIAN_LOG"
            stop_search_guardian "守护启动失败"
            return 1
        fi
        sleep 5
        elapsed=$(( elapsed + 5 ))
        if (( elapsed >= 1800 )); then
            echo "ERROR: nine-dataset baseline did not enter benchmark evaluation within 1800s" >&2
            tail -100 "$SEARCH_GUARDIAN_LOG" >&2 || true
            event "search_guardian" "baseline_9datasets" "failed" "0" "baseline 启动超过 1800 秒；详见 $SEARCH_GUARDIAN_LOG"
            stop_search_guardian "守护启动超时"
            return 1
        fi
        if (( elapsed % 30 == 0 )); then
            echo "[GPU守护] baseline 仍在启动/健康检查（${elapsed}s）；详见 $SEARCH_GUARDIAN_LOG"
        fi
    done
    event "search_guardian" "baseline_9datasets" "active" "0" "九集 SGLang+NeMoSkills baseline 正在与 CPU 搜索并行运行；PGID=$SEARCH_GUARDIAN_PGID"
    echo "[GPU守护] 九集 baseline 已进入真实测评，PGID=$SEARCH_GUARDIAN_PGID；现在开始并行 CPU 搜索"
}

cleanup_on_exit() {
    local status=$?
    set +e
    stop_search_guardian "主脚本退出，兜底清理"
    return "$status"
}

trap cleanup_on_exit EXIT

json_file_valid() {
    local path="$1"
    [[ -s "$path" ]] || return 1
    "$SGLANG_PYTHON" -c 'import json,sys; json.load(open(sys.argv[1],encoding="utf-8"))' "$path" >/dev/null 2>&1
}

phase_dataset_completed() {
    local phase="$1" dataset="$2" trace="$3"
    [[ -s "$trace" ]] || return 1
    "$SGLANG_PYTHON" -c 'import json,sys
x=json.load(open(sys.argv[1],encoding="utf-8"))
phase,dataset=sys.argv[2:4]
raise SystemExit(0 if any(e.get("phase")==phase and e.get("dataset")==dataset and e.get("status")=="completed" for e in x.get("events",[])) else 1)' "$RUN_DIR/run_state.json" "$phase" "$dataset"
}

run_collection() {
    local phase="$1" policy_mode="$2" target="$3" policy_path="$4" trace_subdir="$5"
    local -a benchmarks_array
    IFS=',' read -ra benchmarks_array <<< "$ORDERED_BENCHMARKS"
    local total_datasets="${#benchmarks_array[@]}" completed_datasets=0
    progress_bar "$phase 数据集" "$completed_datasets" "$total_datasets" "准备开始"
    for spec in "${benchmarks_array[@]}"; do
        local name="${spec%%:*}"
        local trace="$RUN_DIR/traces/$trace_subdir/${name}.jsonl"
        local eval_parent="$RUN_DIR/eval_runs/${phase}_${target:-explore}/${name}"
        mkdir -p "$(dirname "$trace")"
        if phase_dataset_completed "$phase" "$name" "$trace"; then
            local existing_records
            existing_records="$(wc -l < "$trace")"
            completed_datasets=$(( completed_datasets + 1 ))
            event "$phase" "$name" "skipped" "$existing_records" "已有 completed 事件和非空 trace，幂等续跑跳过"
            progress_bar "$phase 数据集" "$completed_datasets" "$total_datasets" "复用已完成 $name"
            continue
        fi
        progress_bar "$phase 数据集" "$completed_datasets" "$total_datasets" "正在运行 $name"
        mkdir -p "$eval_parent"
        local -a cmd=(bash "$EVAL_SGLANG" --mode "$MODE" --benchmarks "$spec" --model "$MODEL" --served-model-name "$SERVED_MODEL_NAME" --batch-size "$BATCH_SIZE" --client-concurrency "$CLIENT_CONCURRENCY" --gpu-memory-reserve-gb "$GPU_MEMORY_RESERVE_GB" --tokens "$TOKENS" --context-length "$CONTEXT_LENGTH" --temperature "$TEMPERATURE" --top-p "$TOP_P" --mem-fraction "$MEM_FRACTION" --dtype "$DTYPE" --block-size "$PHYSICAL_BLOCK_SIZE" --cuda-graph-bs "1" --output-path "$eval_parent" --extra-server-args "--disable-cuda-graph $EXTRA_SERVER_ARGS")
        [[ -n "$TP_SIZE" ]] && cmd+=(--tp-size "$TP_SIZE")
        [[ -n "$PORT" ]] && cmd+=(--port "$PORT")
        [[ -n "$PROXY_PORT" ]] && cmd+=(--proxy-port "$PROXY_PORT")
        [[ -n "$NEMO_SKILLS_DATA_DIR" ]] && cmd+=(--nemo-skills-data-dir "$NEMO_SKILLS_DATA_DIR")
        [[ -n "$SGLANG_PYTHON" ]] && cmd+=(--sglang-python "$SGLANG_PYTHON")
        [[ -n "$EVAL_PYTHON" ]] && cmd+=(--eval-python "$EVAL_PYTHON")
        [[ -n "$SGLANG_SRC" ]] && cmd+=(--sglang-src "$SGLANG_SRC")
        [[ -n "$SGLANG_WORK_DIR" ]] && cmd+=(--sglang-work-dir "$SGLANG_WORK_DIR")
        [[ -n "$LORA_PATH" ]] && cmd+=(--lora-path "$LORA_PATH")
        cmd+=(--lora-mode "$LORA_MODE")
        if [[ "$name" == "mmlu" ]]; then
            cmd+=(--max-samples "$MMLU_MAX_SAMPLES")
        elif [[ -n "$MAX_SAMPLES" ]]; then
            cmd+=(--max-samples "$MAX_SAMPLES")
        fi
        local status=0 records=0 attempt attempt_gpu attempt_trace
        local attempt_trace_dir="$RUN_DIR/runtime/attempt_traces/$phase/$name"
        mkdir -p "$attempt_trace_dir"
        for (( attempt=1; attempt<=DATASET_MAX_ATTEMPTS; attempt++ )); do
            # A failed attempt may contain a partial trace.  Never append a
            # fresh-server retry to that trajectory: the request IDs, round
            # history and NeMo outputs all restart from scratch.
            attempt_trace="$attempt_trace_dir/attempt_$(date +%Y%m%d_%H%M%S)_pid${BASHPID}_n${attempt}.jsonl"
            : > "$attempt_trace"
            status=0
            if attempt_gpu="$(resolve_runtime_gpu)"; then
                event "$phase" "$name" "running" "0" "启动真实 SGLang 动态 canonical + 三分支 shadow（尝试 $attempt/$DATASET_MAX_ATTEMPTS，GPU=$attempt_gpu）"
                NLD_FAIL_ON_BENCHMARK_ERROR=1 NLD_KEEP_FAILED_WORK_DIR=1 NLD_DYNAMIC_BLOCK_ENABLE=1 NLD_DYNAMIC_BLOCK_SIZES="$BLOCK_SIZES" NLD_DYNAMIC_BLOCK_POLICY_MODE="$policy_mode" NLD_DYNAMIC_BLOCK_POLICY_TARGET="$target" NLD_DYNAMIC_BLOCK_POLICY_PATH="$policy_path" NLD_DYNAMIC_BLOCK_TRACE_FILE="$attempt_trace" NLD_DYNAMIC_BLOCK_BENCHMARK="$name" NLD_DYNAMIC_BLOCK_SEED="$SEED" SGLANG_CONFIDENCE_TRACE_FILE="$attempt_trace" PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}" "${cmd[@]}" --gpu-devices "$attempt_gpu" || status=$?
            else
                status=88
                event "$phase" "$name" "waiting_gpu" "0" "尝试 $attempt/$DATASET_MAX_ATTEMPTS 时没有达到 ${AUTO_GPU_MIN_FREE_GB} GiB 空闲门槛的 GPU"
            fi
            records="$(wc -l < "$attempt_trace")"
            if [[ "$status" == "0" && "$records" -gt 0 ]]; then
                mv -f "$attempt_trace" "$trace"
                break
            fi
            [[ "$status" != "0" ]] || status=87
            if (( attempt < DATASET_MAX_ATTEMPTS )); then
                local retry_message="第 $attempt 次失败，eval exit=$status；保留失败 runtime 和 $attempt_trace，${DATASET_RETRY_DELAY_S}s 后用全新 server 重试"
                [[ "$status" != "88" ]] || retry_message="第 $attempt 次未找到满足空闲显存门槛的 GPU；${DATASET_RETRY_DELAY_S}s 后重新选择"
                event "$phase" "$name" "retrying" "$records" "$retry_message"
                progress_bar "$phase 数据集" "$completed_datasets" "$total_datasets" "$name 第 $attempt 次失败，即将重试"
                (( DATASET_RETRY_DELAY_S == 0 )) || sleep "$DATASET_RETRY_DELAY_S"
            fi
        done
        if [[ "$status" == "0" && "$records" -gt 0 ]]; then
            event "$phase" "$name" "completed" "$records" "trace 已写入并通过 pipeline（尝试 $attempt/$DATASET_MAX_ATTEMPTS）"
            completed_datasets=$(( completed_datasets + 1 ))
            progress_bar "$phase 数据集" "$completed_datasets" "$total_datasets" "已完成 $name"
        else
            event "$phase" "$name" "failed" "$records" "连续 $DATASET_MAX_ATTEMPTS 次仍失败，最后 eval exit=$status；失败 runtime 和尝试 trace 已保留，canonical trace 未提交，完整数据集不标记完成"
            return "$status"
        fi
    done
}

run_search() {
    local scope="九集全局"
    [[ "$ALLOW_PARTIAL_SEARCH" == "true" ]] && scope="部分数据集开发"
    if json_file_valid "$RUN_DIR/search/search_results.json" \
        && json_file_valid "$RUN_DIR/search/policy_s8.json" \
        && json_file_valid "$RUN_DIR/search/policy_s16.json"; then
        event "search" "$scope" "skipped" "0" "搜索结果与 S8/S16 冻结策略均已存在且 JSON 完整"
        "$SGLANG_PYTHON" "$REPORTING" render --run-dir "$RUN_DIR"
        return 0
    fi
    start_search_guardian
    event "search" "$scope" "running" "0" "开始集等权的逻辑回归/浅树检索与 GBDT 信号上界诊断"
    local -a cmd=("$SGLANG_PYTHON" "$SEARCH" --mode search --trace-root "$RUN_DIR/traces/explore" --output-dir "$RUN_DIR/search" --split-seed "$SPLIT_SEED" --min-large-precision "$MIN_LARGE_PRECISION" --max-large-waste "$MAX_LARGE_WASTE" --min-safe8 "$MIN_SAFE8" --invalid-row-policy exclude --max-invalid-row-rate "$MAX_INVALID_ROW_RATE")
    [[ "$ALLOW_PARTIAL_SEARCH" == "true" ]] && cmd+=(--allow-partial-datasets)
    if ! "${cmd[@]}"; then
        event "search" "$scope" "failed" "0" "离线检索失败；保留已有 trace 和日志"
        stop_search_guardian "CPU 离线搜索失败"
        return 1
    fi
    stop_search_guardian "CPU 离线搜索完成"
    "$SGLANG_PYTHON" "$REPORTING" render --run-dir "$RUN_DIR"
    event "search" "$scope" "completed" "0" "S8/S16 策略已冻结"
}

run_validation_target() {
    local target="$1"
    local policy="$RUN_DIR/search/policy_${target}.json"
    [[ -f "$policy" ]] || { echo "ERROR: missing frozen policy $policy" >&2; exit 1; }
    if json_file_valid "$RUN_DIR/search/validation_${target}.json"; then
        event "validate_${target}" "九集全局" "skipped" "0" "冻结验证汇总已存在且 JSON 完整"
        "$SGLANG_PYTHON" "$REPORTING" render --run-dir "$RUN_DIR"
        return 0
    fi
    run_collection "validate_${target}" "frozen" "$target" "$policy" "validate_${target}"
    local -a cmd=("$SGLANG_PYTHON" "$SEARCH" --mode validate --trace-root "$RUN_DIR/traces/validate_${target}" --output-dir "$RUN_DIR/search" --split-seed "$SPLIT_SEED" --target "$target" --invalid-row-policy exclude --max-invalid-row-rate "$MAX_INVALID_ROW_RATE")
    [[ "$ALLOW_PARTIAL_SEARCH" == "true" ]] && cmd+=(--allow-partial-datasets)
    "${cmd[@]}"
    "$SGLANG_PYTHON" "$REPORTING" render --run-dir "$RUN_DIR"
}

case "$STAGE" in
    collect)
        progress_bar "总体阶段" 0 1 "探索 trace 采集"
        run_collection "explore" "explore" "s8" "" "explore"
        progress_bar "总体阶段" 1 1 "探索 trace 采集完成"
        ;;
    search)
        progress_bar "总体阶段" 0 1 "离线策略搜索"
        run_search
        progress_bar "总体阶段" 1 1 "离线策略搜索完成"
        ;;
    validate)
        progress_bar "总体阶段" 0 2 "准备 S8 冻结验证"
        run_validation_target "s8"
        progress_bar "总体阶段" 1 2 "S8 完成，准备 S16"
        run_validation_target "s16"
        progress_bar "总体阶段" 2 2 "S8/S16 冻结验证完成"
        ;;
    remaining)
        progress_bar "总体阶段" 1 4 "复用探索 trace，准备离线搜索"
        run_search
        progress_bar "总体阶段" 2 4 "搜索完成，准备 S8 验证"
        run_validation_target "s8"
        progress_bar "总体阶段" 3 4 "S8 完成，准备 S16 验证"
        run_validation_target "s16"
        progress_bar "总体阶段" 4 4 "现有 run 的剩余阶段全部完成"
        ;;
    all)
        progress_bar "总体阶段" 0 4 "准备探索 trace 采集"
        run_collection "explore" "explore" "s8" "" "explore"
        progress_bar "总体阶段" 1 4 "探索完成，准备离线搜索"
        run_search
        progress_bar "总体阶段" 2 4 "搜索完成，准备 S8 验证"
        run_validation_target "s8"
        progress_bar "总体阶段" 3 4 "S8 完成，准备 S16 验证"
        run_validation_target "s16"
        progress_bar "总体阶段" 4 4 "全部阶段完成"
        ;;
esac

echo "Completed stage=$STAGE"
echo "Run directory: $RUN_DIR"
echo "Report: $RUN_DIR/report.md"
