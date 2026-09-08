#!/bin/bash
# Reuse existing SGLang shadow traces, search one unified scalar policy, then validate it.

set -euo pipefail

ORIGINAL_ARGS=("$@")
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
EVAL_SGLANG="${NLD_UNIFIED_EVAL_SGLANG:-$PROJECT_DIR/observations/eval_sglang.sh}"
SEARCH="$SCRIPT_DIR/search.py"
REPORTING="$SCRIPT_DIR/reporting.py"
GPU_GUARD="$SCRIPT_DIR/gpu_memory_guard.py"
RESULTS_ROOT="${NLD_OBSERVATION_RESULTS_ROOT:-/data/home/wly/dLLM/NLD_results/observations}/sglang_unified_scalar_block_policy_results"
DEFAULT_TRACE_ROOT="/data/home/wly/dLLM/NLD_results/observations/sglang_dynamic_block_history_signal_results/dynamic_block_history_20260901_032420/traces/explore"
DEFAULT_BENCHMARKS="gsm8k:1,human-eval:1,mbpp:1,math-500:1,aime25:1,gpqa:1,ifeval:1,livecodebench-cpp:1,mmlu:1"
DEFAULT_MODEL="/data1/linyewei/models/Nemotron-Labs-Diffusion-8B"
DEFAULT_PYTHON="/data/home/wly/.conda/envs/nld_sglang/bin/python"
[[ -x "$DEFAULT_PYTHON" ]] || DEFAULT_PYTHON="python"

usage() {
    cat <<EOF
Usage: $0 --stage search|validate|remaining|all|report [options]

Stages:
  --stage search              Reuse existing traces and run CPU-only global search
  --stage validate            Validate an existing run's frozen policy on SGLang
  --stage remaining           Idempotently finish missing search/validation stages
  --stage all                 New timestamp dir: guarded CPU search, release, nine-dataset validation
  --stage report              Regenerate report.md only
  --run-dir DIR               Existing timestamp directory for validate/remaining/report

Trace/search:
  --trace-root DIR            Existing nine-dataset L8/L16/L32 shadow traces
  --cv-folds N                Prompt-hash OOF folds (default: 5)
  --signal-bins N             Monotone survival bins (default: 32)
  --lambda-grid LIST          One-knob search grid
  --policy-lambda V|auto      Validation lambda; auto uses searched default
  --full-gate-grid LIST       Tier-2 global full_streak gates (default: 1,2,3)
  --min-dual-improvement V    Minimum OOF regret gain before allowing signal 2 (default: 0.02)
  --min-large-precision V     Default-lambda constraint (default: 0.80)
  --max-large-waste V         Default-lambda constraint (default: 0.10)
  --max-loss-vs-l32 V         Mean accepted-token loss constraint (default: 2.5)
  --min-gain16 V              Significant L8->L16 gain (default: 2)
  --min-gain32 V              Significant L16->L32 gain (default: 4)
  --max-invalid-row-rate V    Old-trace exclusion guard (default: 0.05)
  --allow-partial-datasets    Development only; formal search requires all nine datasets
  --search-max-rows-per-dataset N Development smoke only; 0 reads every trace row

GPU reservation and SGLang:
  --gpu-devices LIST|auto     GPU(s) used first by memory guard and then validation
  --search-gpu-hold-gb V      GiB reserved per selected GPU during CPU search (default: 48; 0 disables)
  --search-gpu-hold-chunk-gb V Allocation chunk size (default: 1)
  --auto-gpu-min-free-gb V    Auto-selection free-memory floor (default: 52)
  --tp-size N                 Tensor parallel size; default inferred from GPU list
  --batch-size N              SGLang max running requests (default: 1)
  --client-concurrency N      NeMo/proxy concurrency (default: 1)
  --gpu-memory-reserve-gb V   Existing eval_sglang model-time reservation (default: 0)
  --mem-fraction V            SGLang static memory fraction (default: 0.55)
  --model PATH                Model checkpoint
  --served-model-name NAME    OpenAI served model label
  --mode MODE                 linearspec_lora or linearspec_base
  --tokens N                  Completion limit (default: 8192)
  --context-length N          Context length (default: 10240)
  --mmlu-max-samples N        Formal protocol requires 2000
  --max-samples N             Development-only cap for non-MMLU datasets
  --temperature V             Must be 0 (default: 0)
  --top-p V                   Default: 0.95
  --dtype NAME                Default: bfloat16
  --port N                    Optional fixed server port; omitted uses auto-search
  --proxy-port N              Optional fixed proxy port; omitted uses auto-search
  --nemo-skills-data-dir DIR  Persistent NeMo-Skills data root
  --sglang-python PATH        SGLang/search Python
  --eval-python PATH          NeMo-Skills Python
  --sglang-src DIR            SGLang source root
  --sglang-work-dir DIR       SGLang cache/work root
  --lora-path DIR             Draft LoRA adapter
  --lora-mode MODE            draft_only or both (default: draft_only)
  --extra-server-args TEXT    Extra launch_server arguments
  --dataset-max-attempts N    Fresh-server retries per dataset (default: 3)
  --dataset-retry-delay-s N   Delay between retries (default: 10)
  --dry-run                   Validate and print resolved stages without writing or launching
EOF
}

STAGE=""
RUN_DIR=""
TRACE_ROOT="$DEFAULT_TRACE_ROOT"
BENCHMARKS="$DEFAULT_BENCHMARKS"
MODEL="$DEFAULT_MODEL"
SERVED_MODEL_NAME="nemotron-labs-diffusion-8b"
MODE="linearspec_lora"
TOKENS="8192"
CONTEXT_LENGTH="10240"
MAX_SAMPLES=""
MMLU_MAX_SAMPLES="2000"
TEMPERATURE="0"
TOP_P="0.95"
GPU_DEVICES="0"
AUTO_GPU_MIN_FREE_GB="52"
SEARCH_GPU_HOLD_GB="48"
SEARCH_GPU_HOLD_CHUNK_GB="1"
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
CV_FOLDS="5"
SIGNAL_BINS="32"
LAMBDA_GRID="0,0.025,0.05,0.075,0.1,0.125,0.15,0.175,0.2,0.225,0.25,0.275,0.3,0.35,0.4,0.5,0.6,0.75,1"
POLICY_LAMBDA="auto"
FULL_GATE_GRID="1,2,3"
MIN_DUAL_IMPROVEMENT="0.02"
MIN_LARGE_PRECISION="0.80"
MAX_LARGE_WASTE="0.10"
MAX_LOSS_VS_L32="2.5"
MIN_GAIN16="2"
MIN_GAIN32="4"
MAX_INVALID_ROW_RATE="0.05"
SPLIT_SEED="20260906"
DATASET_MAX_ATTEMPTS="3"
DATASET_RETRY_DELAY_S="10"
ALLOW_PARTIAL="false"
SEARCH_MAX_ROWS_PER_DATASET="0"
DRY_RUN="false"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --stage) STAGE="$2"; shift 2 ;;
        --run-dir) RUN_DIR="$2"; shift 2 ;;
        --trace-root) TRACE_ROOT="$2"; shift 2 ;;
        --benchmarks) BENCHMARKS="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --served-model-name|--model-name) SERVED_MODEL_NAME="$2"; shift 2 ;;
        --mode) MODE="$2"; shift 2 ;;
        --tokens) TOKENS="$2"; shift 2 ;;
        --context-length) CONTEXT_LENGTH="$2"; shift 2 ;;
        --max-samples) MAX_SAMPLES="$2"; shift 2 ;;
        --mmlu-max-samples) MMLU_MAX_SAMPLES="$2"; shift 2 ;;
        --temperature) TEMPERATURE="$2"; shift 2 ;;
        --top-p) TOP_P="$2"; shift 2 ;;
        --gpu-devices) GPU_DEVICES="$2"; shift 2 ;;
        --auto-gpu-min-free-gb) AUTO_GPU_MIN_FREE_GB="$2"; shift 2 ;;
        --search-gpu-hold-gb) SEARCH_GPU_HOLD_GB="$2"; shift 2 ;;
        --search-gpu-hold-chunk-gb) SEARCH_GPU_HOLD_CHUNK_GB="$2"; shift 2 ;;
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
        --cv-folds) CV_FOLDS="$2"; shift 2 ;;
        --signal-bins) SIGNAL_BINS="$2"; shift 2 ;;
        --lambda-grid) LAMBDA_GRID="$2"; shift 2 ;;
        --policy-lambda) POLICY_LAMBDA="$2"; shift 2 ;;
        --full-gate-grid) FULL_GATE_GRID="$2"; shift 2 ;;
        --min-dual-improvement) MIN_DUAL_IMPROVEMENT="$2"; shift 2 ;;
        --min-large-precision) MIN_LARGE_PRECISION="$2"; shift 2 ;;
        --max-large-waste) MAX_LARGE_WASTE="$2"; shift 2 ;;
        --max-loss-vs-l32) MAX_LOSS_VS_L32="$2"; shift 2 ;;
        --min-gain16) MIN_GAIN16="$2"; shift 2 ;;
        --min-gain32) MIN_GAIN32="$2"; shift 2 ;;
        --max-invalid-row-rate) MAX_INVALID_ROW_RATE="$2"; shift 2 ;;
        --split-seed) SPLIT_SEED="$2"; shift 2 ;;
        --dataset-max-attempts) DATASET_MAX_ATTEMPTS="$2"; shift 2 ;;
        --dataset-retry-delay-s) DATASET_RETRY_DELAY_S="$2"; shift 2 ;;
        --allow-partial-datasets) ALLOW_PARTIAL="true"; shift ;;
        --search-max-rows-per-dataset) SEARCH_MAX_ROWS_PER_DATASET="$2"; shift 2 ;;
        --dry-run) DRY_RUN="true"; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "ERROR: unknown option $1" >&2; usage; exit 1 ;;
    esac
done

[[ -n "$STAGE" ]] || { echo "ERROR: --stage is required" >&2; exit 1; }
case "$STAGE" in search|validate|remaining|all|report) ;; *) echo "ERROR: invalid --stage $STAGE" >&2; exit 1 ;; esac
case "$MODE" in linearspec_lora|linearspec_base) ;; *) echo "ERROR: invalid --mode" >&2; exit 1 ;; esac
[[ "$TEMPERATURE" == "0" || "$TEMPERATURE" == "0.0" ]] || { echo "ERROR: paired greedy validation requires temperature 0" >&2; exit 1; }
[[ "$MMLU_MAX_SAMPLES" == "2000" || "$ALLOW_PARTIAL" == "true" ]] || { echo "ERROR: formal protocol fixes MMLU to 2000 samples" >&2; exit 1; }
[[ "$DATASET_MAX_ATTEMPTS" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: invalid --dataset-max-attempts" >&2; exit 1; }
[[ "$DATASET_RETRY_DELAY_S" =~ ^[0-9]+$ ]] || { echo "ERROR: invalid --dataset-retry-delay-s" >&2; exit 1; }
[[ "$SEARCH_MAX_ROWS_PER_DATASET" =~ ^[0-9]+$ ]] || { echo "ERROR: invalid --search-max-rows-per-dataset" >&2; exit 1; }
if [[ "$SEARCH_MAX_ROWS_PER_DATASET" != "0" && "$ALLOW_PARTIAL" != "true" ]]; then
    echo "ERROR: --search-max-rows-per-dataset requires --allow-partial-datasets" >&2
    exit 1
fi
[[ "$POLICY_LAMBDA" == "auto" ]] || "$SGLANG_PYTHON" -c 'import sys; assert float(sys.argv[1]) >= 0' "$POLICY_LAMBDA"
[[ -n "$EVAL_PYTHON" ]] || EVAL_PYTHON="$SGLANG_PYTHON"

if [[ -z "$RUN_DIR" && ( "$STAGE" == "validate" || "$STAGE" == "remaining" || "$STAGE" == "report" ) ]]; then
    echo "ERROR: --run-dir is required for $STAGE" >&2
    exit 1
fi

ORDERED_BENCHMARKS="$($SGLANG_PYTHON -c 'import sys
items=[x.strip() for x in sys.argv[1].split(",") if x.strip()]
if any(x.split(":",1)[0]=="aime24" for x in items): raise SystemExit("AIME24 is excluded")
m=[x for x in items if x.split(":",1)[0]=="mmlu"]
o=[x for x in items if x.split(":",1)[0]!="mmlu"]
print(",".join(o+m))' "$BENCHMARKS")"
if [[ "$ALLOW_PARTIAL" != "true" ]]; then
    [[ -z "$MAX_SAMPLES" ]] || { echo "ERROR: formal protocol requires full non-MMLU datasets; remove --max-samples or add --allow-partial-datasets" >&2; exit 1; }
    "$SGLANG_PYTHON" -c 'import sys
expected={"gsm8k","human-eval","mbpp","math-500","aime25","gpqa","ifeval","livecodebench-cpp","mmlu"}
actual={item.split(":",1)[0] for item in sys.argv[1].split(",") if item}
if actual != expected: raise SystemExit(f"formal validation requires exactly nine datasets; missing={sorted(expected-actual)}, extra={sorted(actual-expected)}")' "$ORDERED_BENCHMARKS"
fi

resolve_gpu() {
    if [[ "$GPU_DEVICES" != "auto" ]]; then
        printf '%s\n' "$GPU_DEVICES"
        return
    fi
    command -v nvidia-smi >/dev/null 2>&1 || { echo "ERROR: auto GPU requires nvidia-smi" >&2; return 1; }
    nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits | "$SGLANG_PYTHON" -c 'import sys
floor=float(sys.argv[1])*1024; rows=[]
for line in sys.stdin:
 p=[x.strip() for x in line.split(",")]
 if len(p)!=3: continue
 try: i,free,util=int(p[0]),float(p[1]),float(p[2])
 except ValueError: continue
 if free>=floor: rows.append((util,-free,i))
if not rows: raise SystemExit("no GPU meets free-memory floor")
print(sorted(rows)[0][2])' "$AUTO_GPU_MIN_FREE_GB"
}

if [[ "$DRY_RUN" == "true" ]]; then
    echo "stage=$STAGE run_dir=${RUN_DIR:-<new timestamp>}"
    echo "trace_root=$TRACE_ROOT"
    echo "benchmarks=$ORDERED_BENCHMARKS (AIME24 excluded; MMLU=$MMLU_MAX_SAMPLES and last)"
    echo "search=one scalar; optional acceptance+full_streak; ${CV_FOLDS}-fold OOF; bins=$SIGNAL_BINS"
    echo "search_max_rows_per_dataset=$SEARCH_MAX_ROWS_PER_DATASET (0 means all)"
    echo "lambda_grid=$LAMBDA_GRID policy_lambda=$POLICY_LAMBDA"
    echo "gpu=$GPU_DEVICES hold_gb_per_gpu=$SEARCH_GPU_HOLD_GB then release before validation"
    echo "validation=batch=$BATCH_SIZE concurrency=$CLIENT_CONCURRENCY mem_fraction=$MEM_FRACTION reserve_gb=$GPU_MEMORY_RESERVE_GB"
    echo "ports=server:${PORT:-auto} proxy:${PROXY_PORT:-auto}; process-local SGLang patch; CUDA graph disabled"
    exit 0
fi

if [[ "$GPU_DEVICES" == "auto" && "$STAGE" != "report" ]]; then
    GPU_DEVICES="$(resolve_gpu)"
    echo "[GPU自动选择] 本次 search guard 与 validation 固定使用 GPU $GPU_DEVICES"
fi

arg_was_set() {
    local needle="$1" item
    for item in "${ORIGINAL_ARGS[@]}"; do
        [[ "$item" == "$needle" ]] && return 0
    done
    return 1
}

# Save intentional runtime overrides before an existing run restores its
# search protocol.  Model/data/search settings stay frozen; only execution
# resources and the single documented lambda knob may change before an
# unfinished validation.
CLI_GPU_DEVICES="$GPU_DEVICES"
CLI_TP_SIZE="$TP_SIZE"
CLI_BATCH_SIZE="$BATCH_SIZE"
CLI_CLIENT_CONCURRENCY="$CLIENT_CONCURRENCY"
CLI_GPU_MEMORY_RESERVE_GB="$GPU_MEMORY_RESERVE_GB"
CLI_SEARCH_GPU_HOLD_GB="$SEARCH_GPU_HOLD_GB"
CLI_PORT="$PORT"
CLI_PROXY_PORT="$PROXY_PORT"
CLI_POLICY_LAMBDA="$POLICY_LAMBDA"

if [[ -z "$RUN_DIR" ]]; then
    TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
    RUN_DIR="$RESULTS_ROOT/unified_scalar_block_${TIMESTAMP}"
    COMMAND="bash observations/sglang_unified_scalar_block_policy/eval_unified_scalar_block_policy.sh ${ORIGINAL_ARGS[*]}"
    SETTINGS_JSON="$($SGLANG_PYTHON -c 'import json,sys
keys="stage trace_root benchmarks model served_model_name mode tokens context_length max_samples mmlu_max_samples temperature top_p gpu_devices auto_gpu_min_free_gb search_gpu_hold_gb search_gpu_hold_chunk_gb tp_size batch_size client_concurrency gpu_memory_reserve_gb mem_fraction dtype port proxy_port nemo_skills_data_dir sglang_python eval_python sglang_src sglang_work_dir lora_path lora_mode extra_server_args cv_folds signal_bins lambda_grid policy_lambda full_gate_grid min_dual_improvement min_large_precision max_large_waste max_loss_vs_l32 min_gain16 min_gain32 max_invalid_row_rate split_seed dataset_max_attempts dataset_retry_delay_s allow_partial search_max_rows_per_dataset command".split()
print(json.dumps(dict(zip(keys,sys.argv[1:])),ensure_ascii=False))' "$STAGE" "$TRACE_ROOT" "$ORDERED_BENCHMARKS" "$MODEL" "$SERVED_MODEL_NAME" "$MODE" "$TOKENS" "$CONTEXT_LENGTH" "$MAX_SAMPLES" "$MMLU_MAX_SAMPLES" "$TEMPERATURE" "$TOP_P" "$GPU_DEVICES" "$AUTO_GPU_MIN_FREE_GB" "$SEARCH_GPU_HOLD_GB" "$SEARCH_GPU_HOLD_CHUNK_GB" "${TP_SIZE:-auto}" "$BATCH_SIZE" "$CLIENT_CONCURRENCY" "$GPU_MEMORY_RESERVE_GB" "$MEM_FRACTION" "$DTYPE" "$PORT" "$PROXY_PORT" "$NEMO_SKILLS_DATA_DIR" "$SGLANG_PYTHON" "$EVAL_PYTHON" "$SGLANG_SRC" "$SGLANG_WORK_DIR" "$LORA_PATH" "$LORA_MODE" "$EXTRA_SERVER_ARGS" "$CV_FOLDS" "$SIGNAL_BINS" "$LAMBDA_GRID" "$POLICY_LAMBDA" "$FULL_GATE_GRID" "$MIN_DUAL_IMPROVEMENT" "$MIN_LARGE_PRECISION" "$MAX_LARGE_WASTE" "$MAX_LOSS_VS_L32" "$MIN_GAIN16" "$MIN_GAIN32" "$MAX_INVALID_ROW_RATE" "$SPLIT_SEED" "$DATASET_MAX_ATTEMPTS" "$DATASET_RETRY_DELAY_S" "$ALLOW_PARTIAL" "$SEARCH_MAX_ROWS_PER_DATASET" "$COMMAND")"
    "$SGLANG_PYTHON" "$REPORTING" init --run-dir "$RUN_DIR" --settings-json "$SETTINGS_JSON"
else
    [[ -d "$RUN_DIR" ]] || { echo "ERROR: run dir not found: $RUN_DIR" >&2; exit 1; }
    if [[ -f "$RUN_DIR/settings.json" ]]; then
        mapfile -d '' -t SAVED < <("$SGLANG_PYTHON" -c 'import json,sys
x=json.load(open(sys.argv[1],encoding="utf-8"))
for key in ("trace_root","benchmarks","model","served_model_name","mode","tokens","context_length","max_samples","mmlu_max_samples","temperature","top_p","gpu_devices","auto_gpu_min_free_gb","search_gpu_hold_gb","search_gpu_hold_chunk_gb","tp_size","batch_size","client_concurrency","gpu_memory_reserve_gb","mem_fraction","dtype","port","proxy_port","nemo_skills_data_dir","sglang_python","eval_python","sglang_src","sglang_work_dir","lora_path","lora_mode","extra_server_args","cv_folds","signal_bins","lambda_grid","policy_lambda","full_gate_grid","min_dual_improvement","min_large_precision","max_large_waste","max_loss_vs_l32","min_gain16","min_gain32","max_invalid_row_rate","split_seed","dataset_max_attempts","dataset_retry_delay_s","allow_partial","search_max_rows_per_dataset"):
 print(str(x.get(key,"")),end="\0")' "$RUN_DIR/settings.json")
        # A resumed run restores its exact protocol.  Resource overrides should
        # use a new run; this avoids silently evaluating a different policy.
        TRACE_ROOT="${SAVED[0]}"; ORDERED_BENCHMARKS="${SAVED[1]}"; MODEL="${SAVED[2]}"; SERVED_MODEL_NAME="${SAVED[3]}"; MODE="${SAVED[4]}"
        TOKENS="${SAVED[5]}"; CONTEXT_LENGTH="${SAVED[6]}"; MAX_SAMPLES="${SAVED[7]}"; MMLU_MAX_SAMPLES="${SAVED[8]}"; TEMPERATURE="${SAVED[9]}"; TOP_P="${SAVED[10]}"
        GPU_DEVICES="${SAVED[11]}"; AUTO_GPU_MIN_FREE_GB="${SAVED[12]}"; SEARCH_GPU_HOLD_GB="${SAVED[13]}"; SEARCH_GPU_HOLD_CHUNK_GB="${SAVED[14]}"; TP_SIZE="${SAVED[15]}"; [[ "$TP_SIZE" == "auto" ]] && TP_SIZE=""
        BATCH_SIZE="${SAVED[16]}"; CLIENT_CONCURRENCY="${SAVED[17]}"; GPU_MEMORY_RESERVE_GB="${SAVED[18]}"; MEM_FRACTION="${SAVED[19]}"; DTYPE="${SAVED[20]}"
        PORT="${SAVED[21]}"; PROXY_PORT="${SAVED[22]}"; NEMO_SKILLS_DATA_DIR="${SAVED[23]}"; SGLANG_PYTHON="${SAVED[24]}"; EVAL_PYTHON="${SAVED[25]}"; SGLANG_SRC="${SAVED[26]}"; SGLANG_WORK_DIR="${SAVED[27]}"
        LORA_PATH="${SAVED[28]}"; LORA_MODE="${SAVED[29]}"; EXTRA_SERVER_ARGS="${SAVED[30]}"; CV_FOLDS="${SAVED[31]}"; SIGNAL_BINS="${SAVED[32]}"; LAMBDA_GRID="${SAVED[33]}"; POLICY_LAMBDA="${SAVED[34]}"
        FULL_GATE_GRID="${SAVED[35]}"; MIN_DUAL_IMPROVEMENT="${SAVED[36]}"; MIN_LARGE_PRECISION="${SAVED[37]}"; MAX_LARGE_WASTE="${SAVED[38]}"; MAX_LOSS_VS_L32="${SAVED[39]}"
        MIN_GAIN16="${SAVED[40]}"; MIN_GAIN32="${SAVED[41]}"; MAX_INVALID_ROW_RATE="${SAVED[42]}"; SPLIT_SEED="${SAVED[43]}"; DATASET_MAX_ATTEMPTS="${SAVED[44]}"; DATASET_RETRY_DELAY_S="${SAVED[45]}"; ALLOW_PARTIAL="${SAVED[46]}"; SEARCH_MAX_ROWS_PER_DATASET="${SAVED[47]:-0}"
        arg_was_set --gpu-devices && GPU_DEVICES="$CLI_GPU_DEVICES"
        arg_was_set --tp-size && TP_SIZE="$CLI_TP_SIZE"
        { arg_was_set --batch-size || arg_was_set --max-running-requests; } && BATCH_SIZE="$CLI_BATCH_SIZE"
        arg_was_set --client-concurrency && CLIENT_CONCURRENCY="$CLI_CLIENT_CONCURRENCY"
        arg_was_set --gpu-memory-reserve-gb && GPU_MEMORY_RESERVE_GB="$CLI_GPU_MEMORY_RESERVE_GB"
        arg_was_set --search-gpu-hold-gb && SEARCH_GPU_HOLD_GB="$CLI_SEARCH_GPU_HOLD_GB"
        arg_was_set --port && PORT="$CLI_PORT"
        arg_was_set --proxy-port && PROXY_PORT="$CLI_PROXY_PORT"
        arg_was_set --policy-lambda && POLICY_LAMBDA="$CLI_POLICY_LAMBDA"
    fi
fi

command -v flock >/dev/null 2>&1 || { echo "ERROR: flock is required" >&2; exit 1; }
exec 9>>"$RUN_DIR/runtime/runner.lock"
flock -n 9 || { echo "ERROR: another process owns run dir $RUN_DIR" >&2; exit 89; }
printf 'pid=%s\nstarted_at=%s\n' "$$" "$(date --iso-8601=seconds)" >&9

event() {
    local phase="$1" dataset="$2" status="$3" records="$4" message="$5"
    local payload
    payload="$($SGLANG_PYTHON -c 'import json,sys; print(json.dumps(dict(zip(("phase","dataset","status","records","message"),sys.argv[1:])),ensure_ascii=False))' "$phase" "$dataset" "$status" "$records" "$message")"
    "$SGLANG_PYTHON" "$REPORTING" event --run-dir "$RUN_DIR" --event-json "$payload"
}

json_valid() {
    [[ -s "$1" ]] && "$SGLANG_PYTHON" -c 'import json,sys; json.load(open(sys.argv[1],encoding="utf-8"))' "$1" >/dev/null 2>&1
}

GUARD_PID=""
GUARD_READY=""
GUARD_RELEASED=""
GUARD_LOG=""

stop_guard() {
    local reason="${1:-search ended}"
    [[ -n "$GUARD_PID" ]] || return 0
    echo "[GPU显存守护] releasing PID=$GUARD_PID ($reason)"
    kill -TERM "$GUARD_PID" 2>/dev/null || true
    local count
    for count in $(seq 1 60); do
        kill -0 "$GUARD_PID" 2>/dev/null || break
        sleep 1
    done
    if kill -0 "$GUARD_PID" 2>/dev/null; then
        kill -KILL "$GUARD_PID" 2>/dev/null || true
    fi
    wait "$GUARD_PID" 2>/dev/null || true
    if [[ -f "$GUARD_RELEASED" ]]; then
        event "gpu_guard" "$GPU_DEVICES" "released" "0" "$reason；显存已释放，可启动验证"
    else
        event "gpu_guard" "$GPU_DEVICES" "forced_release" "0" "$reason；进程已退出，CUDA context 已销毁"
    fi
    GUARD_PID=""
    sleep 2
}

cleanup() {
    local status=$?
    set +e
    stop_guard "主入口退出兜底"
    return "$status"
}
trap cleanup EXIT

start_guard() {
    "$SGLANG_PYTHON" -c 'import sys; raise SystemExit(0 if float(sys.argv[1]) > 0 else 1)' "$SEARCH_GPU_HOLD_GB" || {
        event "gpu_guard" "$GPU_DEVICES" "disabled" "0" "search-gpu-hold-gb=0，用户显式关闭"
        return 0
    }
    local guard_dir="$RUN_DIR/runtime/gpu_memory_guard" suffix
    suffix="$(date +%Y%m%d_%H%M%S)_pid$$"
    GUARD_READY="$guard_dir/ready_$suffix.json"
    GUARD_RELEASED="$guard_dir/released_$suffix.json"
    GUARD_LOG="$guard_dir/guard_$suffix.log"
    event "gpu_guard" "$GPU_DEVICES" "starting" "0" "准备每卡预占 ${SEARCH_GPU_HOLD_GB} GiB；日志=$GUARD_LOG"
    CUDA_VISIBLE_DEVICES="$GPU_DEVICES" "$SGLANG_PYTHON" "$GPU_GUARD" --hold-gb "$SEARCH_GPU_HOLD_GB" --chunk-gb "$SEARCH_GPU_HOLD_CHUNK_GB" --ready-file "$GUARD_READY" --released-file "$GUARD_RELEASED" >"$GUARD_LOG" 2>&1 &
    GUARD_PID=$!
    local waited=0
    while [[ ! -s "$GUARD_READY" ]]; do
        if ! kill -0 "$GUARD_PID" 2>/dev/null; then
            wait "$GUARD_PID" || true
            echo "ERROR: GPU memory guard failed" >&2
            tail -80 "$GUARD_LOG" >&2 || true
            GUARD_PID=""
            return 1
        fi
        sleep 1; waited=$((waited+1))
        (( waited < 300 )) || { echo "ERROR: GPU guard readiness timeout" >&2; return 1; }
    done
    event "gpu_guard" "$GPU_DEVICES" "active" "$SEARCH_GPU_HOLD_GB" "显存守护已就绪；CPU 搜索期间持续持有"
    echo "[GPU显存守护] ready: GPU=$GPU_DEVICES, ${SEARCH_GPU_HOLD_GB} GiB per GPU"
}

run_search() {
    if json_valid "$RUN_DIR/search/search_results.json" && json_valid "$RUN_DIR/search/policy.json"; then
        event "search" "九集全局" "skipped" "0" "搜索结果和冻结策略已存在"
        return 0
    fi
    [[ -d "$TRACE_ROOT" ]] || { echo "ERROR: trace root not found: $TRACE_ROOT" >&2; return 1; }
    start_guard
    event "search" "九集全局" "running" "0" "开始 prompt OOF 单信号/受限双信号与 lambda 检索"
    local -a cmd=("$SGLANG_PYTHON" "$SEARCH" --mode search --trace-root "$TRACE_ROOT" --output-dir "$RUN_DIR/search" --run-dir "$RUN_DIR" --split-seed "$SPLIT_SEED" --cv-folds "$CV_FOLDS" --signal-bins "$SIGNAL_BINS" --lambda-grid "$LAMBDA_GRID" --full-gate-grid "$FULL_GATE_GRID" --min-dual-improvement "$MIN_DUAL_IMPROVEMENT" --min-large-precision "$MIN_LARGE_PRECISION" --max-large-waste "$MAX_LARGE_WASTE" --max-loss-vs-l32 "$MAX_LOSS_VS_L32" --min-gain16 "$MIN_GAIN16" --min-gain32 "$MIN_GAIN32" --invalid-row-policy exclude --max-invalid-row-rate "$MAX_INVALID_ROW_RATE" --mmlu-required-requests "$MMLU_MAX_SAMPLES")
    [[ "$ALLOW_PARTIAL" == "true" ]] && cmd+=(--allow-partial-datasets)
    [[ "$SEARCH_MAX_ROWS_PER_DATASET" == "0" ]] || cmd+=(--max-rows-per-dataset "$SEARCH_MAX_ROWS_PER_DATASET")
    if ! "${cmd[@]}"; then
        event "search" "九集全局" "failed" "0" "离线搜索失败；保留 trace、进度和日志"
        stop_guard "CPU 搜索失败"
        return 1
    fi
    event "search" "九集全局" "completed" "0" "统一策略已冻结"
    stop_guard "CPU 搜索完成"
}

dataset_completed() {
    local dataset="$1" trace="$2"
    [[ -s "$trace" ]] || return 1
    "$SGLANG_PYTHON" -c 'import json,sys
x=json.load(open(sys.argv[1],encoding="utf-8")); d=sys.argv[2]
raise SystemExit(0 if any(e.get("phase")=="validate" and e.get("dataset")==d and e.get("status")=="completed" for e in x.get("events",[])) else 1)' "$RUN_DIR/run_state.json" "$dataset"
}

run_validation() {
    local policy="$RUN_DIR/search/policy.json"
    [[ -s "$policy" ]] || { echo "ERROR: missing policy $policy" >&2; return 1; }
    if json_valid "$RUN_DIR/search/validation.json" && "$SGLANG_PYTHON" -c 'import json,sys; x=json.load(open(sys.argv[1],encoding="utf-8")); raise SystemExit(0 if x.get("protocol",{}).get("formal_complete") is True else 1)' "$RUN_DIR/search/validation.json"; then
        event "validate" "九集全局" "skipped" "0" "冻结验证汇总已存在"
        return 0
    fi
    stop_guard "启动 GPU 验证前强制释放"
    local -a specs
    IFS=',' read -ra specs <<< "$ORDERED_BENCHMARKS"
    local total="${#specs[@]}" completed=0 spec name trace eval_parent attempt status records attempt_trace attempt_gpu
    for spec in "${specs[@]}"; do
        name="${spec%%:*}"
        trace="$RUN_DIR/traces/validate/$name.jsonl"
        eval_parent="$RUN_DIR/eval_runs/validate/$name"
        if dataset_completed "$name" "$trace"; then
            completed=$((completed+1)); event "validate" "$name" "skipped" "$(wc -l < "$trace")" "幂等复用已完成 trace"
            continue
        fi
        mkdir -p "$eval_parent" "$RUN_DIR/runtime/attempt_traces/validate/$name"
        status=1; records=0
        for ((attempt=1; attempt<=DATASET_MAX_ATTEMPTS; attempt++)); do
            attempt_trace="$RUN_DIR/runtime/attempt_traces/validate/$name/attempt_$(date +%Y%m%d_%H%M%S)_pid${BASHPID}_n${attempt}.jsonl"
            : > "$attempt_trace"
            attempt_gpu="$GPU_DEVICES"
            event "validate" "$name" "running" "0" "冻结统一策略尝试 $attempt/$DATASET_MAX_ATTEMPTS，GPU=$attempt_gpu"
            local -a cmd=(bash "$EVAL_SGLANG" --mode "$MODE" --benchmarks "$spec" --model "$MODEL" --served-model-name "$SERVED_MODEL_NAME" --gpu-devices "$attempt_gpu" --batch-size "$BATCH_SIZE" --client-concurrency "$CLIENT_CONCURRENCY" --gpu-memory-reserve-gb "$GPU_MEMORY_RESERVE_GB" --tokens "$TOKENS" --context-length "$CONTEXT_LENGTH" --temperature "$TEMPERATURE" --top-p "$TOP_P" --mem-fraction "$MEM_FRACTION" --dtype "$DTYPE" --block-size 32 --cuda-graph-bs "1" --output-path "$eval_parent" --extra-server-args "--disable-cuda-graph $EXTRA_SERVER_ARGS")
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
            if [[ "$name" == "mmlu" ]]; then cmd+=(--max-samples "$MMLU_MAX_SAMPLES"); elif [[ -n "$MAX_SAMPLES" ]]; then cmd+=(--max-samples "$MAX_SAMPLES"); fi
            status=0
            NLD_FAIL_ON_BENCHMARK_ERROR=1 NLD_KEEP_FAILED_WORK_DIR=1 NLD_DYNAMIC_BLOCK_ENABLE=0 NLD_UNIFIED_BLOCK_ENABLE=1 NLD_DYNAMIC_BLOCK_SIZES="8,16,32" NLD_DYNAMIC_BLOCK_POLICY_MODE=frozen NLD_DYNAMIC_BLOCK_POLICY_TARGET=unified NLD_DYNAMIC_BLOCK_POLICY_PATH="$policy" NLD_DYNAMIC_BLOCK_TRACE_FILE="$attempt_trace" NLD_DYNAMIC_BLOCK_BENCHMARK="$name" NLD_UNIFIED_BLOCK_POLICY_PATH="$policy" NLD_UNIFIED_BLOCK_POLICY_LAMBDA="$([[ "$POLICY_LAMBDA" == "auto" ]] && printf '' || printf '%s' "$POLICY_LAMBDA")" SGLANG_CONFIDENCE_TRACE_FILE="$attempt_trace" PYTHONPATH="$SCRIPT_DIR:$PROJECT_DIR/observations:${PYTHONPATH:-}" "${cmd[@]}" || status=$?
            records="$(wc -l < "$attempt_trace")"
            if [[ "$status" == "0" && "$records" -gt 0 ]]; then mv -f "$attempt_trace" "$trace"; break; fi
            [[ "$status" != "0" ]] || status=87
            event "validate" "$name" "retrying" "$records" "尝试 $attempt 失败，exit=$status；使用全新 server 重试"
            ((attempt < DATASET_MAX_ATTEMPTS)) && sleep "$DATASET_RETRY_DELAY_S"
        done
        if [[ "$status" != "0" || "$records" -eq 0 ]]; then
            event "validate" "$name" "failed" "$records" "连续重试仍失败，最后 exit=$status"
            return "$status"
        fi
        completed=$((completed+1))
        event "validate" "$name" "completed" "$records" "schema v2 on-policy trace 与 NeMo-Skills 产物完成 ($completed/$total)"
        # Immediately materialize the completed datasets' metric tables.  The
        # final call below repeats the analysis with the strict nine-dataset
        # guard and marks formal_complete=true.
        local -a partial=("$SGLANG_PYTHON" "$SEARCH" --mode validate --trace-root "$RUN_DIR/traces/validate" --output-dir "$RUN_DIR/search" --run-dir "$RUN_DIR" --policy "$policy" --eval-root "$RUN_DIR/eval_runs/validate" --split-seed "$SPLIT_SEED" --cv-folds "$CV_FOLDS" --invalid-row-policy strict --max-invalid-row-rate 0 --mmlu-required-requests "$MMLU_MAX_SAMPLES" --min-gain16 "$MIN_GAIN16" --min-gain32 "$MIN_GAIN32" --allow-partial-datasets)
        [[ "$POLICY_LAMBDA" == "auto" ]] || partial+=(--policy-lambda "$POLICY_LAMBDA")
        "${partial[@]}"
    done
    local -a analyze=("$SGLANG_PYTHON" "$SEARCH" --mode validate --trace-root "$RUN_DIR/traces/validate" --output-dir "$RUN_DIR/search" --run-dir "$RUN_DIR" --policy "$policy" --eval-root "$RUN_DIR/eval_runs/validate" --split-seed "$SPLIT_SEED" --cv-folds "$CV_FOLDS" --invalid-row-policy strict --max-invalid-row-rate 0 --mmlu-required-requests "$MMLU_MAX_SAMPLES" --min-gain16 "$MIN_GAIN16" --min-gain32 "$MIN_GAIN32")
    [[ "$POLICY_LAMBDA" == "auto" ]] || analyze+=(--policy-lambda "$POLICY_LAMBDA")
    [[ "$ALLOW_PARTIAL" == "true" ]] && analyze+=(--allow-partial-datasets)
    "${analyze[@]}"
    event "validate" "九集全局" "completed" "0" "统一策略冻结验证与官方 decode-only TPF 已汇总"
}

case "$STAGE" in
    search) run_search ;;
    validate) run_validation ;;
    remaining) run_search; run_validation ;;
    all) run_search; run_validation ;;
    report) "$SGLANG_PYTHON" "$REPORTING" render --run-dir "$RUN_DIR" ;;
esac

"$SGLANG_PYTHON" "$REPORTING" render --run-dir "$RUN_DIR"
echo "Completed stage=$STAGE"
echo "Run directory: $RUN_DIR"
echo "Report: $RUN_DIR/report.md"
