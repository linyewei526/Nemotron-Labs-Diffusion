#!/bin/bash
# Reuse SGLang shadow traces, search a shared signal, then validate S8 and S16.

set -euo pipefail

ORIGINAL_ARGS=("$@")
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
EVAL_SGLANG="${NLD_DUAL_EFF_EVAL_SGLANG:-$PROJECT_DIR/observations/eval_sglang.sh}"
SEARCH="$SCRIPT_DIR/search.py"
REPORTING="$SCRIPT_DIR/reporting.py"
GPU_GUARD="$SCRIPT_DIR/gpu_memory_guard.py"
RESULTS_ROOT="${NLD_OBSERVATION_RESULTS_ROOT:-/data/home/wly/dLLM/NLD_results/observations}/sglang_dual_action_efficiency_policy_results"
DEFAULT_TRACE_ROOT="/data/home/wly/dLLM/NLD_results/observations/sglang_dynamic_block_history_signal_results/dynamic_block_history_20260901_032420/traces/explore"
DEFAULT_BENCHMARKS="gsm8k:1,human-eval:1,mbpp:1,math-500:1,aime25:1,gpqa:1,ifeval:1,livecodebench-cpp:1"
DEFAULT_MODEL="/data1/linyewei/models/Nemotron-Labs-Diffusion-8B"
DEFAULT_PYTHON="/data/home/wly/.conda/envs/nld_sglang/bin/python"
[[ -x "$DEFAULT_PYTHON" ]] || DEFAULT_PYTHON="python"

usage() {
    cat <<EOF
Usage: $0 --stage search|validate|remaining|all|report [options]

Stages:
  --stage search              Guard GPU and run exact-lambda CPU search
  --stage validate            Validate selected profile(s) in an existing run
  --stage remaining           Idempotently finish missing search/validation
  --stage all                 New run: guarded search, release, S8 then S16 validation
  --stage report              Rebuild report.md only
  --run-dir DIR               Existing timestamp dir for validate/remaining/report

Search:
  --trace-root DIR            Existing SGLang L8/L16/L32 shadow trace root
  --cv-folds N                Prompt OOF folds (default: 5)
  --signal-bins N             Monotone survival bins (default: 32)
  --lambda-max V              Exact lambda search upper bound (default: 1)
  --diagnostic-lambda-grid L  Human-readable report grid; not the optimizer
  --full-gate-grid LIST       Tier-2 full_streak gates (default: 1,2,3)
  --min-dual-improvement V    Minimum joint-min efficiency gain (default: 0.002)
  --min-gain16 V              Significant S8 L8->L16 gain (default: 2)
  --min-gain32 V              Significant L16->L32 gain (default: 4)
  --max-invalid-row-rate V    Reused old-trace exclusion guard (default: 0.05)
  --allow-partial-datasets    Development/single-dataset mode only
  --search-max-rows-per-dataset N Development smoke only; 0 means all rows

Profiles and validation:
  --profiles s8,s16|s8|s16    Profiles to validate (default: s8,s16)
  --policy-lambda-s8 V|auto   Override searched S8 lambda only for validation
  --policy-lambda-s16 V|auto  Override searched S16 lambda only for validation
  --benchmarks LIST           Single/multi benchmark specs; formal default is eight sets
  --max-samples N             Development cap; formal runs use all samples

GPU/SGLang/NeMo-Skills:
  --gpu-devices LIST|auto     GPU(s) held during search and used for validation
  --search-gpu-hold-gb V      GiB held per GPU during CPU search (default: 48; 0 disables)
  --search-gpu-hold-chunk-gb V Guard allocation chunk (default: 1)
  --auto-gpu-min-free-gb V    Auto GPU free-memory floor (default: 52)
  --tp-size N                 Tensor parallel size; inferred when omitted
  --batch-size N              SGLang max running requests (default: 1)
  --client-concurrency N      NeMo-Skills request concurrency (default: 1)
  --gpu-memory-reserve-gb V   Model-time reserve used by eval_sglang (default: 0)
  --mem-fraction V            SGLang static memory fraction (default: 0.55)
  --model PATH                Model checkpoint
  --served-model-name NAME    OpenAI served model label
  --mode MODE                 linearspec_lora or linearspec_base
  --tokens N                  Completion limit (default: 8192)
  --context-length N          Context length (default: 10240)
  --temperature V             Must be 0
  --top-p V                   Default: 0.95
  --dtype NAME                Default: bfloat16
  --port N                    Fixed server port; omitted searches automatically
  --proxy-port N              Fixed proxy port; omitted searches automatically
  --nemo-skills-data-dir DIR  NeMo-Skills data root
  --sglang-python PATH        SGLang/search Python
  --eval-python PATH          NeMo-Skills Python
  --sglang-src DIR            SGLang source root
  --sglang-work-dir DIR       SGLang cache/work root
  --lora-path DIR             Draft LoRA adapter
  --lora-mode MODE            draft_only or both (default: draft_only)
  --extra-server-args TEXT    Additional launch_server arguments
  --dataset-max-attempts N    Fresh-server attempts per profile/dataset (default: 3)
  --dataset-retry-delay-s N   Retry delay (default: 10)
  --dry-run                   Resolve and print without writes/GPU/processes
EOF
}

STAGE=""
RUN_DIR=""
TRACE_ROOT="$DEFAULT_TRACE_ROOT"
BENCHMARKS="$DEFAULT_BENCHMARKS"
PROFILES="s8,s16"
POLICY_LAMBDA_S8="auto"
POLICY_LAMBDA_S16="auto"
MODEL="$DEFAULT_MODEL"
SERVED_MODEL_NAME="nemotron-labs-diffusion-8b"
MODE="linearspec_lora"
TOKENS="8192"
CONTEXT_LENGTH="10240"
MAX_SAMPLES=""
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
LAMBDA_MAX="1"
DIAGNOSTIC_LAMBDA_GRID="0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.5,0.6,0.75,1"
FULL_GATE_GRID="1,2,3"
MIN_DUAL_IMPROVEMENT="0.002"
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
        --profiles) PROFILES="$2"; shift 2 ;;
        --policy-lambda-s8) POLICY_LAMBDA_S8="$2"; shift 2 ;;
        --policy-lambda-s16) POLICY_LAMBDA_S16="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --served-model-name|--model-name) SERVED_MODEL_NAME="$2"; shift 2 ;;
        --mode) MODE="$2"; shift 2 ;;
        --tokens) TOKENS="$2"; shift 2 ;;
        --context-length) CONTEXT_LENGTH="$2"; shift 2 ;;
        --max-samples) MAX_SAMPLES="$2"; shift 2 ;;
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
        --lambda-max) LAMBDA_MAX="$2"; shift 2 ;;
        --diagnostic-lambda-grid) DIAGNOSTIC_LAMBDA_GRID="$2"; shift 2 ;;
        --full-gate-grid) FULL_GATE_GRID="$2"; shift 2 ;;
        --min-dual-improvement) MIN_DUAL_IMPROVEMENT="$2"; shift 2 ;;
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
case "$STAGE" in search|validate|remaining|all|report) ;; *) echo "ERROR: invalid stage $STAGE" >&2; exit 1 ;; esac
case "$MODE" in linearspec_lora|linearspec_base) ;; *) echo "ERROR: invalid --mode" >&2; exit 1 ;; esac
[[ "$TEMPERATURE" == "0" || "$TEMPERATURE" == "0.0" ]] || { echo "ERROR: validation requires greedy temperature=0" >&2; exit 1; }
[[ "$DATASET_MAX_ATTEMPTS" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: invalid --dataset-max-attempts" >&2; exit 1; }
[[ "$DATASET_RETRY_DELAY_S" =~ ^[0-9]+$ ]] || { echo "ERROR: invalid retry delay" >&2; exit 1; }
[[ "$SEARCH_MAX_ROWS_PER_DATASET" =~ ^[0-9]+$ ]] || { echo "ERROR: invalid search row cap" >&2; exit 1; }
if [[ "$SEARCH_MAX_ROWS_PER_DATASET" != "0" && "$ALLOW_PARTIAL" != "true" ]]; then
    echo "ERROR: row cap requires --allow-partial-datasets" >&2; exit 1
fi
for value in "$POLICY_LAMBDA_S8" "$POLICY_LAMBDA_S16"; do
    [[ "$value" == "auto" ]] || "$SGLANG_PYTHON" -c 'import sys; assert float(sys.argv[1]) >= 0' "$value"
done
[[ -n "$EVAL_PYTHON" ]] || EVAL_PYTHON="$SGLANG_PYTHON"
if [[ -z "$RUN_DIR" && ( "$STAGE" == "validate" || "$STAGE" == "remaining" || "$STAGE" == "report" ) ]]; then
    echo "ERROR: --run-dir is required for $STAGE" >&2; exit 1
fi

ORDERED_BENCHMARKS="$($SGLANG_PYTHON -c 'import sys
order=["gsm8k","human-eval","mbpp","math-500","aime25","gpqa","ifeval","livecodebench-cpp"]
items=[x.strip() for x in sys.argv[1].split(",") if x.strip()]
names=[x.split(":",1)[0] for x in items]
if "aime24" in names or "mmlu" in names: raise SystemExit("AIME24 and MMLU are excluded")
rank={name:i for i,name in enumerate(order)}
print(",".join(sorted(items,key=lambda x:rank.get(x.split(":",1)[0],999))))' "$BENCHMARKS")"
PROFILES="$($SGLANG_PYTHON -c 'import sys
items=[]
for x in sys.argv[1].split(","):
 x=x.strip().lower()
 if x and x not in items: items.append(x)
if not items or any(x not in {"s8","s16"} for x in items): raise SystemExit("profiles must be s8 and/or s16")
print(",".join(items))' "$PROFILES")"
if [[ "$ALLOW_PARTIAL" != "true" ]]; then
    [[ -z "$MAX_SAMPLES" ]] || { echo "ERROR: formal run uses every sample" >&2; exit 1; }
    "$SGLANG_PYTHON" -c 'import sys
expected={"gsm8k","human-eval","mbpp","math-500","aime25","gpqa","ifeval","livecodebench-cpp"}
actual={x.split(":",1)[0] for x in sys.argv[1].split(",") if x}
if actual != expected: raise SystemExit(f"formal run requires exactly eight datasets; missing={sorted(expected-actual)}, extra={sorted(actual-expected)}")' "$ORDERED_BENCHMARKS"
fi

resolve_gpu() {
    if [[ "$GPU_DEVICES" != "auto" ]]; then printf '%s\n' "$GPU_DEVICES"; return; fi
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
    echo "benchmarks=$ORDERED_BENCHMARKS (AIME24/MMLU excluded)"
    echo "profiles=$PROFILES; S8=cold8/actions8,16,32; S16=cold16/actions16,32"
    echo "search=shared signal, ${CV_FOLDS}-fold OOF, exact lambda intersections [0,$LAMBDA_MAX]"
    echo "gpu=$GPU_DEVICES hold=${SEARCH_GPU_HOLD_GB}GiB/GPU, released before validation"
    echo "validation=batch=$BATCH_SIZE concurrency=$CLIENT_CONCURRENCY ports=${PORT:-auto}/${PROXY_PORT:-auto}"
    exit 0
fi

if [[ "$GPU_DEVICES" == "auto" && "$STAGE" != "report" ]]; then
    GPU_DEVICES="$(resolve_gpu)"
    echo "[GPU自动选择] search guard和validation固定使用GPU $GPU_DEVICES"
fi

arg_was_set() {
    local needle="$1" item
    for item in "${ORIGINAL_ARGS[@]}"; do [[ "$item" == "$needle" ]] && return 0; done
    return 1
}

CLI_GPU_DEVICES="$GPU_DEVICES"; CLI_TP_SIZE="$TP_SIZE"; CLI_BATCH_SIZE="$BATCH_SIZE"
CLI_CLIENT_CONCURRENCY="$CLIENT_CONCURRENCY"; CLI_GPU_MEMORY_RESERVE_GB="$GPU_MEMORY_RESERVE_GB"
CLI_SEARCH_GPU_HOLD_GB="$SEARCH_GPU_HOLD_GB"; CLI_PORT="$PORT"; CLI_PROXY_PORT="$PROXY_PORT"
CLI_PROFILES="$PROFILES"; CLI_POLICY_LAMBDA_S8="$POLICY_LAMBDA_S8"; CLI_POLICY_LAMBDA_S16="$POLICY_LAMBDA_S16"
CLI_ORDERED_BENCHMARKS="$ORDERED_BENCHMARKS"; CLI_MAX_SAMPLES="$MAX_SAMPLES"; CLI_ALLOW_PARTIAL="$ALLOW_PARTIAL"

if [[ -z "$RUN_DIR" ]]; then
    TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
    RUN_DIR="$RESULTS_ROOT/dual_action_efficiency_${TIMESTAMP}"
    COMMAND="bash observations/sglang_dual_action_efficiency_policy/eval_dual_action_efficiency.sh ${ORIGINAL_ARGS[*]}"
    SETTINGS_JSON="$($SGLANG_PYTHON -c 'import json,sys
keys="stage trace_root benchmarks profiles policy_lambda_s8 policy_lambda_s16 model served_model_name mode tokens context_length max_samples temperature top_p gpu_devices auto_gpu_min_free_gb search_gpu_hold_gb search_gpu_hold_chunk_gb tp_size batch_size client_concurrency gpu_memory_reserve_gb mem_fraction dtype port proxy_port nemo_skills_data_dir sglang_python eval_python sglang_src sglang_work_dir lora_path lora_mode extra_server_args cv_folds signal_bins lambda_max diagnostic_lambda_grid full_gate_grid min_dual_improvement min_gain16 min_gain32 max_invalid_row_rate split_seed dataset_max_attempts dataset_retry_delay_s allow_partial search_max_rows_per_dataset command".split()
print(json.dumps(dict(zip(keys,sys.argv[1:])),ensure_ascii=False))' "$STAGE" "$TRACE_ROOT" "$ORDERED_BENCHMARKS" "$PROFILES" "$POLICY_LAMBDA_S8" "$POLICY_LAMBDA_S16" "$MODEL" "$SERVED_MODEL_NAME" "$MODE" "$TOKENS" "$CONTEXT_LENGTH" "$MAX_SAMPLES" "$TEMPERATURE" "$TOP_P" "$GPU_DEVICES" "$AUTO_GPU_MIN_FREE_GB" "$SEARCH_GPU_HOLD_GB" "$SEARCH_GPU_HOLD_CHUNK_GB" "${TP_SIZE:-auto}" "$BATCH_SIZE" "$CLIENT_CONCURRENCY" "$GPU_MEMORY_RESERVE_GB" "$MEM_FRACTION" "$DTYPE" "$PORT" "$PROXY_PORT" "$NEMO_SKILLS_DATA_DIR" "$SGLANG_PYTHON" "$EVAL_PYTHON" "$SGLANG_SRC" "$SGLANG_WORK_DIR" "$LORA_PATH" "$LORA_MODE" "$EXTRA_SERVER_ARGS" "$CV_FOLDS" "$SIGNAL_BINS" "$LAMBDA_MAX" "$DIAGNOSTIC_LAMBDA_GRID" "$FULL_GATE_GRID" "$MIN_DUAL_IMPROVEMENT" "$MIN_GAIN16" "$MIN_GAIN32" "$MAX_INVALID_ROW_RATE" "$SPLIT_SEED" "$DATASET_MAX_ATTEMPTS" "$DATASET_RETRY_DELAY_S" "$ALLOW_PARTIAL" "$SEARCH_MAX_ROWS_PER_DATASET" "$COMMAND")"
    "$SGLANG_PYTHON" "$REPORTING" init --run-dir "$RUN_DIR" --settings-json "$SETTINGS_JSON"
else
    [[ -d "$RUN_DIR" ]] || { echo "ERROR: run dir not found: $RUN_DIR" >&2; exit 1; }
    if [[ -f "$RUN_DIR/settings.json" ]]; then
        mapfile -d '' -t SAVED < <("$SGLANG_PYTHON" -c 'import json,sys
x=json.load(open(sys.argv[1],encoding="utf-8"))
for key in ("trace_root","benchmarks","profiles","policy_lambda_s8","policy_lambda_s16","model","served_model_name","mode","tokens","context_length","max_samples","temperature","top_p","gpu_devices","auto_gpu_min_free_gb","search_gpu_hold_gb","search_gpu_hold_chunk_gb","tp_size","batch_size","client_concurrency","gpu_memory_reserve_gb","mem_fraction","dtype","port","proxy_port","nemo_skills_data_dir","sglang_python","eval_python","sglang_src","sglang_work_dir","lora_path","lora_mode","extra_server_args","cv_folds","signal_bins","lambda_max","diagnostic_lambda_grid","full_gate_grid","min_dual_improvement","min_gain16","min_gain32","max_invalid_row_rate","split_seed","dataset_max_attempts","dataset_retry_delay_s","allow_partial","search_max_rows_per_dataset"):
 print(str(x.get(key,"")),end="\0")' "$RUN_DIR/settings.json")
        TRACE_ROOT="${SAVED[0]}"; ORDERED_BENCHMARKS="${SAVED[1]}"; PROFILES="${SAVED[2]}"; POLICY_LAMBDA_S8="${SAVED[3]}"; POLICY_LAMBDA_S16="${SAVED[4]}"
        MODEL="${SAVED[5]}"; SERVED_MODEL_NAME="${SAVED[6]}"; MODE="${SAVED[7]}"; TOKENS="${SAVED[8]}"; CONTEXT_LENGTH="${SAVED[9]}"; MAX_SAMPLES="${SAVED[10]}"; TEMPERATURE="${SAVED[11]}"; TOP_P="${SAVED[12]}"
        GPU_DEVICES="${SAVED[13]}"; AUTO_GPU_MIN_FREE_GB="${SAVED[14]}"; SEARCH_GPU_HOLD_GB="${SAVED[15]}"; SEARCH_GPU_HOLD_CHUNK_GB="${SAVED[16]}"; TP_SIZE="${SAVED[17]}"; [[ "$TP_SIZE" == "auto" ]] && TP_SIZE=""
        BATCH_SIZE="${SAVED[18]}"; CLIENT_CONCURRENCY="${SAVED[19]}"; GPU_MEMORY_RESERVE_GB="${SAVED[20]}"; MEM_FRACTION="${SAVED[21]}"; DTYPE="${SAVED[22]}"; PORT="${SAVED[23]}"; PROXY_PORT="${SAVED[24]}"
        NEMO_SKILLS_DATA_DIR="${SAVED[25]}"; SGLANG_PYTHON="${SAVED[26]}"; EVAL_PYTHON="${SAVED[27]}"; SGLANG_SRC="${SAVED[28]}"; SGLANG_WORK_DIR="${SAVED[29]}"; LORA_PATH="${SAVED[30]}"; LORA_MODE="${SAVED[31]}"; EXTRA_SERVER_ARGS="${SAVED[32]}"
        CV_FOLDS="${SAVED[33]}"; SIGNAL_BINS="${SAVED[34]}"; LAMBDA_MAX="${SAVED[35]}"; DIAGNOSTIC_LAMBDA_GRID="${SAVED[36]}"; FULL_GATE_GRID="${SAVED[37]}"; MIN_DUAL_IMPROVEMENT="${SAVED[38]}"; MIN_GAIN16="${SAVED[39]}"; MIN_GAIN32="${SAVED[40]}"; MAX_INVALID_ROW_RATE="${SAVED[41]}"; SPLIT_SEED="${SAVED[42]}"; DATASET_MAX_ATTEMPTS="${SAVED[43]}"; DATASET_RETRY_DELAY_S="${SAVED[44]}"; ALLOW_PARTIAL="${SAVED[45]}"; SEARCH_MAX_ROWS_PER_DATASET="${SAVED[46]:-0}"
        arg_was_set --gpu-devices && GPU_DEVICES="$CLI_GPU_DEVICES"
        arg_was_set --tp-size && TP_SIZE="$CLI_TP_SIZE"
        { arg_was_set --batch-size || arg_was_set --max-running-requests; } && BATCH_SIZE="$CLI_BATCH_SIZE"
        arg_was_set --client-concurrency && CLIENT_CONCURRENCY="$CLI_CLIENT_CONCURRENCY"
        arg_was_set --gpu-memory-reserve-gb && GPU_MEMORY_RESERVE_GB="$CLI_GPU_MEMORY_RESERVE_GB"
        arg_was_set --search-gpu-hold-gb && SEARCH_GPU_HOLD_GB="$CLI_SEARCH_GPU_HOLD_GB"
        arg_was_set --port && PORT="$CLI_PORT"; arg_was_set --proxy-port && PROXY_PORT="$CLI_PROXY_PORT"
        arg_was_set --profiles && PROFILES="$CLI_PROFILES"
        arg_was_set --policy-lambda-s8 && POLICY_LAMBDA_S8="$CLI_POLICY_LAMBDA_S8"
        arg_was_set --policy-lambda-s16 && POLICY_LAMBDA_S16="$CLI_POLICY_LAMBDA_S16"
        arg_was_set --benchmarks && ORDERED_BENCHMARKS="$CLI_ORDERED_BENCHMARKS"
        arg_was_set --max-samples && MAX_SAMPLES="$CLI_MAX_SAMPLES"
        arg_was_set --allow-partial-datasets && ALLOW_PARTIAL="$CLI_ALLOW_PARTIAL"
    fi
fi

command -v flock >/dev/null 2>&1 || { echo "ERROR: flock is required" >&2; exit 1; }
exec 9>>"$RUN_DIR/runtime/runner.lock"
flock -n 9 || { echo "ERROR: another process owns $RUN_DIR" >&2; exit 89; }
printf 'pid=%s\nstarted_at=%s\n' "$$" "$(date --iso-8601=seconds)" >&9

event() {
    local phase="$1" dataset="$2" status="$3" records="$4" message="$5" payload
    payload="$($SGLANG_PYTHON -c 'import json,sys; print(json.dumps(dict(zip(("phase","dataset","status","records","message"),sys.argv[1:])),ensure_ascii=False))' "$phase" "$dataset" "$status" "$records" "$message")"
    "$SGLANG_PYTHON" "$REPORTING" event --run-dir "$RUN_DIR" --event-json "$payload"
}

json_valid() { [[ -s "$1" ]] && "$SGLANG_PYTHON" -c 'import json,sys; json.load(open(sys.argv[1],encoding="utf-8"))' "$1" >/dev/null 2>&1; }
GUARD_PID=""; GUARD_READY=""; GUARD_RELEASED=""; GUARD_LOG=""

stop_guard() {
    local reason="${1:-search ended}" count
    [[ -n "$GUARD_PID" ]] || return 0
    echo "[GPU显存守护] releasing PID=$GUARD_PID ($reason)"
    kill -TERM "$GUARD_PID" 2>/dev/null || true
    for count in $(seq 1 60); do kill -0 "$GUARD_PID" 2>/dev/null || break; sleep 1; done
    if kill -0 "$GUARD_PID" 2>/dev/null; then kill -KILL "$GUARD_PID" 2>/dev/null || true; fi
    wait "$GUARD_PID" 2>/dev/null || true
    if [[ -f "$GUARD_RELEASED" ]]; then event gpu_guard "$GPU_DEVICES" released 0 "$reason；显存已释放"; else event gpu_guard "$GPU_DEVICES" forced_release 0 "$reason；CUDA context已销毁"; fi
    GUARD_PID=""; sleep 2
}

cleanup() { local status=$?; set +e; stop_guard "主入口退出兜底"; return "$status"; }
trap cleanup EXIT

start_guard() {
    "$SGLANG_PYTHON" -c 'import sys; raise SystemExit(0 if float(sys.argv[1])>0 else 1)' "$SEARCH_GPU_HOLD_GB" || { event gpu_guard "$GPU_DEVICES" disabled 0 "用户设置预占0"; return 0; }
    local guard_dir="$RUN_DIR/runtime/gpu_memory_guard" suffix waited=0
    suffix="$(date +%Y%m%d_%H%M%S)_pid$$"; GUARD_READY="$guard_dir/ready_$suffix.json"; GUARD_RELEASED="$guard_dir/released_$suffix.json"; GUARD_LOG="$guard_dir/guard_$suffix.log"
    event gpu_guard "$GPU_DEVICES" starting 0 "准备每卡预占${SEARCH_GPU_HOLD_GB}GiB；日志=$GUARD_LOG"
    CUDA_VISIBLE_DEVICES="$GPU_DEVICES" "$SGLANG_PYTHON" "$GPU_GUARD" --hold-gb "$SEARCH_GPU_HOLD_GB" --chunk-gb "$SEARCH_GPU_HOLD_CHUNK_GB" --ready-file "$GUARD_READY" --released-file "$GUARD_RELEASED" >"$GUARD_LOG" 2>&1 &
    GUARD_PID=$!
    while [[ ! -s "$GUARD_READY" ]]; do
        if ! kill -0 "$GUARD_PID" 2>/dev/null; then wait "$GUARD_PID" || true; tail -80 "$GUARD_LOG" >&2 || true; GUARD_PID=""; return 1; fi
        sleep 1; waited=$((waited+1)); (( waited < 300 )) || { echo "ERROR: guard timeout" >&2; return 1; }
    done
    event gpu_guard "$GPU_DEVICES" active "$SEARCH_GPU_HOLD_GB" "CPU搜索期间持续持有"
}

run_search() {
    if json_valid "$RUN_DIR/search/search_results.json" && json_valid "$RUN_DIR/search/policy.json"; then event search 八集全局 skipped 0 "搜索与策略已存在"; return; fi
    [[ -d "$TRACE_ROOT" ]] || { echo "ERROR: trace root not found: $TRACE_ROOT" >&2; return 1; }
    start_guard
    event search 八集全局 running 0 "共享信号、S8/S16精确lambda搜索"
    local -a cmd=("$SGLANG_PYTHON" "$SEARCH" --mode search --trace-root "$TRACE_ROOT" --output-dir "$RUN_DIR/search" --run-dir "$RUN_DIR" --split-seed "$SPLIT_SEED" --cv-folds "$CV_FOLDS" --signal-bins "$SIGNAL_BINS" --lambda-max "$LAMBDA_MAX" --diagnostic-lambda-grid "$DIAGNOSTIC_LAMBDA_GRID" --full-gate-grid "$FULL_GATE_GRID" --min-dual-improvement "$MIN_DUAL_IMPROVEMENT" --min-gain16 "$MIN_GAIN16" --min-gain32 "$MIN_GAIN32" --invalid-row-policy exclude --max-invalid-row-rate "$MAX_INVALID_ROW_RATE")
    [[ "$ALLOW_PARTIAL" == true ]] && cmd+=(--allow-partial-datasets)
    [[ "$SEARCH_MAX_ROWS_PER_DATASET" == 0 ]] || cmd+=(--max-rows-per-dataset "$SEARCH_MAX_ROWS_PER_DATASET")
    if ! "${cmd[@]}"; then event search 八集全局 failed 0 "离线搜索失败"; stop_guard "CPU搜索失败"; return 1; fi
    event search 八集全局 completed 0 "公共信号和两套lambda已冻结"; stop_guard "CPU搜索完成"
}

dataset_completed() {
    local profile="$1" dataset="$2" trace="$3"
    [[ -s "$trace" ]] || return 1
    "$SGLANG_PYTHON" -c 'import json,sys
x=json.load(open(sys.argv[1],encoding="utf-8")); key=sys.argv[2]
raise SystemExit(0 if any(e.get("phase")=="validate" and e.get("dataset")==key and e.get("status")=="completed" for e in x.get("events",[])) else 1)' "$RUN_DIR/run_state.json" "$profile/$dataset"
}

analyze_profile() {
    local profile="$1" partial="$2" policy="$RUN_DIR/search/policy.json" lambda_value
    [[ "$profile" == s8 ]] && lambda_value="$POLICY_LAMBDA_S8" || lambda_value="$POLICY_LAMBDA_S16"
    local -a cmd=("$SGLANG_PYTHON" "$SEARCH" --mode validate --profile "$profile" --trace-root "$RUN_DIR/traces/validate/$profile" --output-dir "$RUN_DIR/search" --run-dir "$RUN_DIR" --policy "$policy" --eval-root "$RUN_DIR/eval_runs/validate/$profile" --split-seed "$SPLIT_SEED" --cv-folds "$CV_FOLDS" --invalid-row-policy strict --max-invalid-row-rate 0 --min-gain16 "$MIN_GAIN16" --min-gain32 "$MIN_GAIN32")
    [[ "$lambda_value" == auto ]] || cmd+=(--policy-lambda "$lambda_value")
    [[ "$partial" == true ]] && cmd+=(--allow-partial-datasets)
    "${cmd[@]}"
}

run_validation() {
    local policy="$RUN_DIR/search/policy.json"
    [[ -s "$policy" ]] || { echo "ERROR: missing policy $policy" >&2; return 1; }
    stop_guard "启动GPU验证前强制释放"
    local -a profile_items specs; IFS=',' read -ra profile_items <<< "$PROFILES"; IFS=',' read -ra specs <<< "$ORDERED_BENCHMARKS"
    local profile spec name trace eval_parent attempt attempt_trace status records completed total lambda_value
    for profile in "${profile_items[@]}"; do
        if json_valid "$RUN_DIR/search/validation_${profile}.json" && "$SGLANG_PYTHON" -c 'import json,sys; x=json.load(open(sys.argv[1])); raise SystemExit(0 if x.get("protocol",{}).get("formal_complete") is True else 1)' "$RUN_DIR/search/validation_${profile}.json"; then event validate "$profile/八集" skipped 0 "正式汇总已存在"; continue; fi
        completed=0; total="${#specs[@]}"; [[ "$profile" == s8 ]] && lambda_value="$POLICY_LAMBDA_S8" || lambda_value="$POLICY_LAMBDA_S16"
        for spec in "${specs[@]}"; do
            name="${spec%%:*}"; trace="$RUN_DIR/traces/validate/$profile/$name.jsonl"; eval_parent="$RUN_DIR/eval_runs/validate/$profile/$name"
            if dataset_completed "$profile" "$name" "$trace"; then completed=$((completed+1)); event validate "$profile/$name" skipped "$(wc -l < "$trace")" "幂等复用"; continue; fi
            mkdir -p "$eval_parent" "$RUN_DIR/runtime/attempt_traces/$profile/$name"
            status=1; records=0
            for ((attempt=1; attempt<=DATASET_MAX_ATTEMPTS; attempt++)); do
                attempt_trace="$RUN_DIR/runtime/attempt_traces/$profile/$name/attempt_$(date +%Y%m%d_%H%M%S)_pid${BASHPID}_n${attempt}.jsonl"; : >"$attempt_trace"
                event validate "$profile/$name" running 0 "尝试$attempt/$DATASET_MAX_ATTEMPTS，GPU=$GPU_DEVICES"
                local -a cmd=(bash "$EVAL_SGLANG" --mode "$MODE" --benchmarks "$spec" --model "$MODEL" --served-model-name "$SERVED_MODEL_NAME" --gpu-devices "$GPU_DEVICES" --batch-size "$BATCH_SIZE" --client-concurrency "$CLIENT_CONCURRENCY" --gpu-memory-reserve-gb "$GPU_MEMORY_RESERVE_GB" --tokens "$TOKENS" --context-length "$CONTEXT_LENGTH" --temperature "$TEMPERATURE" --top-p "$TOP_P" --mem-fraction "$MEM_FRACTION" --dtype "$DTYPE" --block-size 32 --cuda-graph-bs 1 --output-path "$eval_parent" --extra-server-args "--disable-cuda-graph $EXTRA_SERVER_ARGS")
                [[ -n "$TP_SIZE" ]] && cmd+=(--tp-size "$TP_SIZE"); [[ -n "$PORT" ]] && cmd+=(--port "$PORT"); [[ -n "$PROXY_PORT" ]] && cmd+=(--proxy-port "$PROXY_PORT")
                [[ -n "$NEMO_SKILLS_DATA_DIR" ]] && cmd+=(--nemo-skills-data-dir "$NEMO_SKILLS_DATA_DIR"); [[ -n "$SGLANG_PYTHON" ]] && cmd+=(--sglang-python "$SGLANG_PYTHON"); [[ -n "$EVAL_PYTHON" ]] && cmd+=(--eval-python "$EVAL_PYTHON")
                [[ -n "$SGLANG_SRC" ]] && cmd+=(--sglang-src "$SGLANG_SRC"); [[ -n "$SGLANG_WORK_DIR" ]] && cmd+=(--sglang-work-dir "$SGLANG_WORK_DIR"); [[ -n "$LORA_PATH" ]] && cmd+=(--lora-path "$LORA_PATH")
                cmd+=(--lora-mode "$LORA_MODE"); [[ -n "$MAX_SAMPLES" ]] && cmd+=(--max-samples "$MAX_SAMPLES")
                status=0
                NLD_FAIL_ON_BENCHMARK_ERROR=1 NLD_KEEP_FAILED_WORK_DIR=1 NLD_DYNAMIC_BLOCK_ENABLE=0 NLD_UNIFIED_BLOCK_ENABLE=0 NLD_DUAL_EFF_ENABLE=1 NLD_DYNAMIC_BLOCK_SIZES="8,16,32" NLD_DYNAMIC_BLOCK_POLICY_MODE=dual_efficiency NLD_DYNAMIC_BLOCK_TRACE_FILE="$attempt_trace" NLD_DYNAMIC_BLOCK_BENCHMARK="$name" NLD_DUAL_EFF_POLICY_PATH="$policy" NLD_DUAL_EFF_PROFILE="$profile" NLD_DUAL_EFF_POLICY_LAMBDA="$([[ "$lambda_value" == auto ]] && printf '' || printf '%s' "$lambda_value")" SGLANG_CONFIDENCE_TRACE_FILE="$attempt_trace" PYTHONPATH="$SCRIPT_DIR:$PROJECT_DIR/observations:${PYTHONPATH:-}" "${cmd[@]}" || status=$?
                records="$(wc -l < "$attempt_trace")"
                if [[ "$status" == 0 && "$records" -gt 0 ]]; then mv -f "$attempt_trace" "$trace"; break; fi
                [[ "$status" != 0 ]] || status=87; event validate "$profile/$name" retrying "$records" "尝试$attempt失败，exit=$status"
                (( attempt < DATASET_MAX_ATTEMPTS )) && sleep "$DATASET_RETRY_DELAY_S"
            done
            if [[ "$status" != 0 || "$records" -eq 0 ]]; then event validate "$profile/$name" failed "$records" "重试耗尽，exit=$status"; return "$status"; fi
            completed=$((completed+1)); event validate "$profile/$name" completed "$records" "schema v2 trace完成($completed/$total)"; analyze_profile "$profile" true
        done
        if [[ "$ALLOW_PARTIAL" == true ]]; then
            analyze_profile "$profile" true
            event validate "$profile/开发子集" completed 0 "开发验证与官方decode-only TPF已汇总；不可作为八集结论"
        else
            analyze_profile "$profile" false
            event validate "$profile/八集" completed 0 "八集冻结验证与官方decode-only TPF已汇总"
        fi
    done
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
