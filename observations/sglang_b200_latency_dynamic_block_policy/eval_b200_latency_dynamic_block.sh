#!/bin/bash
# Search one B200-latency policy, then validate seven virtual concurrencies.

set -euo pipefail

ORIGINAL_ARGS=("$@")
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
EVAL_SGLANG="${NLD_B200_LATENCY_EVAL_SGLANG:-$PROJECT_DIR/observations/eval_sglang.sh}"
SEARCH="$SCRIPT_DIR/search.py"
REPORTING="$SCRIPT_DIR/reporting.py"
GPU_GUARD="$SCRIPT_DIR/gpu_memory_guard.py"
RESULTS_ROOT="${NLD_OBSERVATION_RESULTS_ROOT:-/data/home/wly/dLLM/NLD_results/observations}/sglang_b200_latency_dynamic_block_policy_results"
DEFAULT_TRACE_ROOT="/data/home/wly/dLLM/NLD_results/observations/sglang_dynamic_block_history_signal_results/dynamic_block_history_20260901_032420/traces/explore"
DEFAULT_COST_DOCUMENT="$PROJECT_DIR/configs/NLD_B200_B8_B32_forward_sweep_20260907_zh.md"
DEFAULT_BENCHMARKS="gsm8k:1,human-eval:1,mbpp:1,math-500:1,aime25:1,gpqa:1,ifeval:1,livecodebench-cpp:1"
DEFAULT_CONCURRENCIES="2,4,8,16,32,64,128"
DEFAULT_MODEL="/data1/linyewei/models/Nemotron-Labs-Diffusion-8B"
DEFAULT_PYTHON="/data/home/wly/.conda/envs/nld_sglang/bin/python"
[[ -x "$DEFAULT_PYTHON" ]] || DEFAULT_PYTHON="python"

usage() {
    cat <<EOF
Usage: $0 --stage search|validate|remaining|all|report [options]

Stages:
  --stage search                 New/resumed guarded CPU search; report offline gain immediately
  --stage validate               Run missing C x dataset frozen validations in --run-dir
  --stage remaining              Idempotently finish search then every missing validation
  --stage all                    New run: search, release guard, then all validations
  --stage report                 Atomically rebuild report.md/progress.md only
  --run-dir DIR                  Existing timestamp directory for validate/remaining/report

Offline search and B200 cost:
  --trace-root DIR               Existing SGLang L8/L16/L32 exploration traces
  --cost-document FILE           B200 C1..128 B8/B16/B32 sweep markdown
  --concurrencies LIST           Virtual C values (formal default: 2,4,8,16,32,64,128)
                                 Cold start: C2/4=B32, C8/16/32=B16, C64/128=B8
  --cv-folds N                   Request-level OOF folds (default: 5)
  --signal-bins N                Monotone survival bins (default: 32)
  --lambda-max V                 Exact breakpoint search upper bound (default: 16)
  --full-gate-grid LIST          Optional second signal: global full-streak gates (default: 1,2,3)
  --max-invalid-row-rate V       Exploration shadow exclusion ceiling (default: 0.05)
  --search-max-rows-per-dataset N Development smoke cap; 0 means every row

Dataset/protocol:
  --benchmarks LIST              Single/multi specs; formal default is all eight sets
  --max-samples N                Development validation cap only
  --allow-partial-datasets       Enables reduced datasets/C/rows/sample smoke mode

GPU/SGLang/NeMo-Skills:
  --gpu-devices LIST|auto        GPU held during CPU search and used later (default: 0)
  --search-gpu-hold-gb V         Actual memory held per GPU during search (default: 48; 0 disables)
  --search-gpu-hold-chunk-gb V   Guard allocation chunk (default: 1)
  --auto-gpu-min-free-gb V       Auto-selection free-memory floor (default: 52)
  --tp-size N                    Tensor parallel size; inferred when omitted
  --batch-size auto|N            Validation max-running-requests; formal auto resolves to C
  --client-concurrency auto|N    Validation client concurrency; formal auto resolves to C
  --gpu-memory-reserve-gb V      Reserve during model evaluation (default: 0)
  --mem-fraction V               SGLang static memory fraction (default: 0.55)
  --model PATH                   Model checkpoint
  --served-model-name NAME       OpenAI served model label
  --mode MODE                    linearspec_lora or linearspec_base
  --tokens N                     Completion limit (default: 8192)
  --context-length N             Context length (default: 10240)
  --temperature V                Must be 0
  --top-p V                      Default: 0.95
  --dtype NAME                   Default: bfloat16
  --port N                       Fixed server port; omitted delegates collision-safe auto selection
  --proxy-port N                 Fixed proxy port; omitted delegates collision-safe auto selection
  --nemo-skills-data-dir DIR     NeMo-Skills data root
  --sglang-python PATH           SGLang/search Python
  --eval-python PATH             NeMo-Skills Python
  --sglang-src DIR               SGLang source root
  --sglang-work-dir DIR          SGLang cache/work root
  --lora-path DIR                Draft LoRA adapter
  --lora-mode MODE               draft_only or both (default: draft_only)
  --extra-server-args TEXT       Additional launch_server arguments
  --dataset-max-attempts N       Fresh-server attempts per task (default: 3)
  --dataset-retry-delay-s N      Retry delay seconds (default: 10)
  --dry-run                      Resolve protocol without writes/processes
EOF
}

STAGE=""; RUN_DIR=""; TRACE_ROOT="$DEFAULT_TRACE_ROOT"; COST_DOCUMENT="$DEFAULT_COST_DOCUMENT"
BENCHMARKS="$DEFAULT_BENCHMARKS"; CONCURRENCIES="$DEFAULT_CONCURRENCIES"
MODEL="$DEFAULT_MODEL"; SERVED_MODEL_NAME="nemotron-labs-diffusion-8b"; MODE="linearspec_lora"
TOKENS="8192"; CONTEXT_LENGTH="10240"; MAX_SAMPLES=""; TEMPERATURE="0"; TOP_P="0.95"
GPU_DEVICES="0"; AUTO_GPU_MIN_FREE_GB="52"; SEARCH_GPU_HOLD_GB="48"; SEARCH_GPU_HOLD_CHUNK_GB="1"
TP_SIZE=""; BATCH_SIZE="auto"; CLIENT_CONCURRENCY="auto"; GPU_MEMORY_RESERVE_GB="0"
MEM_FRACTION="0.55"; DTYPE="bfloat16"; PORT=""; PROXY_PORT=""; NEMO_SKILLS_DATA_DIR=""
SGLANG_PYTHON="$DEFAULT_PYTHON"; EVAL_PYTHON=""; SGLANG_SRC=""; SGLANG_WORK_DIR=""
LORA_PATH=""; LORA_MODE="draft_only"; EXTRA_SERVER_ARGS=""
CV_FOLDS="5"; SIGNAL_BINS="32"; LAMBDA_MAX="16"; FULL_GATE_GRID="1,2,3"
MAX_INVALID_ROW_RATE="0.05"; SPLIT_SEED="20260908"; DATASET_MAX_ATTEMPTS="3"
DATASET_RETRY_DELAY_S="10"; ALLOW_PARTIAL="false"; SEARCH_MAX_ROWS_PER_DATASET="0"; DRY_RUN="false"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --stage) STAGE="$2"; shift 2 ;;
        --run-dir) RUN_DIR="$2"; shift 2 ;;
        --trace-root) TRACE_ROOT="$2"; shift 2 ;;
        --cost-document) COST_DOCUMENT="$2"; shift 2 ;;
        --benchmarks) BENCHMARKS="$2"; shift 2 ;;
        --concurrencies) CONCURRENCIES="$2"; shift 2 ;;
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
        --full-gate-grid) FULL_GATE_GRID="$2"; shift 2 ;;
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
[[ "$TEMPERATURE" == "0" || "$TEMPERATURE" == "0.0" ]] || { echo "ERROR: greedy temperature=0 is required" >&2; exit 1; }
[[ "$DATASET_MAX_ATTEMPTS" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: invalid --dataset-max-attempts" >&2; exit 1; }
[[ "$DATASET_RETRY_DELAY_S" =~ ^[0-9]+$ ]] || { echo "ERROR: invalid retry delay" >&2; exit 1; }
[[ "$SEARCH_MAX_ROWS_PER_DATASET" =~ ^[0-9]+$ ]] || { echo "ERROR: invalid search row cap" >&2; exit 1; }
[[ "$BATCH_SIZE" == auto || "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: --batch-size must be auto or positive" >&2; exit 1; }
[[ "$CLIENT_CONCURRENCY" == auto || "$CLIENT_CONCURRENCY" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: --client-concurrency must be auto or positive" >&2; exit 1; }
if [[ "$SEARCH_MAX_ROWS_PER_DATASET" != 0 && "$ALLOW_PARTIAL" != true ]]; then echo "ERROR: row cap requires --allow-partial-datasets" >&2; exit 1; fi
[[ -n "$EVAL_PYTHON" ]] || EVAL_PYTHON="$SGLANG_PYTHON"
if [[ -z "$RUN_DIR" && ( "$STAGE" == validate || "$STAGE" == remaining || "$STAGE" == report ) ]]; then echo "ERROR: --run-dir is required for $STAGE" >&2; exit 1; fi

ORDERED_BENCHMARKS="$($SGLANG_PYTHON -c 'import sys
order=["gsm8k","human-eval","mbpp","math-500","aime25","gpqa","ifeval","livecodebench-cpp"]
items=[x.strip() for x in sys.argv[1].split(",") if x.strip()]; names=[x.split(":",1)[0] for x in items]
if "aime24" in names or "mmlu" in names: raise SystemExit("AIME24/MMLU are excluded")
rank={name:i for i,name in enumerate(order)}; print(",".join(sorted(items,key=lambda x:rank.get(x.split(":",1)[0],999))))' "$BENCHMARKS")"
CONCURRENCIES="$($SGLANG_PYTHON -c 'import sys
v=sorted({int(x) for x in sys.argv[1].split(",") if x.strip()})
allowed={2,4,8,16,32,64,128}
if not v or not set(v)<=allowed: raise SystemExit(f"concurrencies must be a subset of {sorted(allowed)}")
print(",".join(map(str,v)))' "$CONCURRENCIES")"
if [[ "$ALLOW_PARTIAL" != true ]]; then
    [[ -z "$MAX_SAMPLES" ]] || { echo "ERROR: formal validation uses every sample" >&2; exit 1; }
    [[ "$CONCURRENCIES" == "$DEFAULT_CONCURRENCIES" ]] || { echo "ERROR: formal run requires C=$DEFAULT_CONCURRENCIES" >&2; exit 1; }
    [[ "$BATCH_SIZE" == auto && "$CLIENT_CONCURRENCY" == auto ]] || { echo "ERROR: formal run resolves both validation concurrency knobs from C" >&2; exit 1; }
    "$SGLANG_PYTHON" -c 'import sys
e={"gsm8k","human-eval","mbpp","math-500","aime25","gpqa","ifeval","livecodebench-cpp"}; a={x.split(":",1)[0] for x in sys.argv[1].split(",") if x}
if a!=e: raise SystemExit(f"formal run requires exactly eight datasets; missing={sorted(e-a)}, extra={sorted(a-e)}")' "$ORDERED_BENCHMARKS"
fi

resolve_gpu() {
    if [[ "$GPU_DEVICES" != auto ]]; then printf '%s\n' "$GPU_DEVICES"; return; fi
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

if [[ "$DRY_RUN" == true ]]; then
    echo "stage=$STAGE run_dir=${RUN_DIR:-<new timestamp>}"
    echo "trace_root=$TRACE_ROOT"
    echo "cost_document=$COST_DOCUMENT"
    echo "benchmarks=$ORDERED_BENCHMARKS; excluded=AIME24,MMLU"
    echo "virtual_concurrencies=$CONCURRENCIES; request actions are independent"
    echo "cold_start_blocks=C2/4:B32,C8/16/32:B16,C64/128:B8"
    echo "objective=sample total_accept/total_B200_ms -> dataset mean -> eight-dataset mean"
    echo "gpu=$GPU_DEVICES hold=${SEARCH_GPU_HOLD_GB}GiB/GPU; validation batch/client resolve to each C"
    exit 0
fi

if [[ "$GPU_DEVICES" == auto && "$STAGE" != report ]]; then GPU_DEVICES="$(resolve_gpu)"; echo "[GPU自动选择] 固定GPU $GPU_DEVICES"; fi

arg_was_set() { local needle="$1" item; for item in "${ORIGINAL_ARGS[@]}"; do [[ "$item" == "$needle" ]] && return 0; done; return 1; }
CLI_GPU_DEVICES="$GPU_DEVICES"; CLI_TP_SIZE="$TP_SIZE"; CLI_GPU_MEMORY_RESERVE_GB="$GPU_MEMORY_RESERVE_GB"
CLI_SEARCH_GPU_HOLD_GB="$SEARCH_GPU_HOLD_GB"; CLI_PORT="$PORT"; CLI_PROXY_PORT="$PROXY_PORT"
CLI_DATASET_MAX_ATTEMPTS="$DATASET_MAX_ATTEMPTS"; CLI_DATASET_RETRY_DELAY_S="$DATASET_RETRY_DELAY_S"

if [[ -z "$RUN_DIR" ]]; then
    TIMESTAMP="$(date +%Y%m%d_%H%M%S)"; RUN_DIR="$RESULTS_ROOT/b200_latency_dynamic_${TIMESTAMP}"
    COMMAND="bash observations/sglang_b200_latency_dynamic_block_policy/eval_b200_latency_dynamic_block.sh ${ORIGINAL_ARGS[*]}"
    SETTINGS_JSON="$($SGLANG_PYTHON -c 'import json,sys
keys="stage trace_root cost_document benchmarks concurrencies model served_model_name mode tokens context_length max_samples temperature top_p gpu_devices auto_gpu_min_free_gb search_gpu_hold_gb search_gpu_hold_chunk_gb tp_size batch_size client_concurrency gpu_memory_reserve_gb mem_fraction dtype port proxy_port nemo_skills_data_dir sglang_python eval_python sglang_src sglang_work_dir lora_path lora_mode extra_server_args cv_folds signal_bins lambda_max full_gate_grid max_invalid_row_rate split_seed dataset_max_attempts dataset_retry_delay_s allow_partial search_max_rows_per_dataset command".split()
print(json.dumps(dict(zip(keys,sys.argv[1:])),ensure_ascii=False))' "$STAGE" "$TRACE_ROOT" "$COST_DOCUMENT" "$ORDERED_BENCHMARKS" "$CONCURRENCIES" "$MODEL" "$SERVED_MODEL_NAME" "$MODE" "$TOKENS" "$CONTEXT_LENGTH" "$MAX_SAMPLES" "$TEMPERATURE" "$TOP_P" "$GPU_DEVICES" "$AUTO_GPU_MIN_FREE_GB" "$SEARCH_GPU_HOLD_GB" "$SEARCH_GPU_HOLD_CHUNK_GB" "${TP_SIZE:-auto}" "$BATCH_SIZE" "$CLIENT_CONCURRENCY" "$GPU_MEMORY_RESERVE_GB" "$MEM_FRACTION" "$DTYPE" "$PORT" "$PROXY_PORT" "$NEMO_SKILLS_DATA_DIR" "$SGLANG_PYTHON" "$EVAL_PYTHON" "$SGLANG_SRC" "$SGLANG_WORK_DIR" "$LORA_PATH" "$LORA_MODE" "$EXTRA_SERVER_ARGS" "$CV_FOLDS" "$SIGNAL_BINS" "$LAMBDA_MAX" "$FULL_GATE_GRID" "$MAX_INVALID_ROW_RATE" "$SPLIT_SEED" "$DATASET_MAX_ATTEMPTS" "$DATASET_RETRY_DELAY_S" "$ALLOW_PARTIAL" "$SEARCH_MAX_ROWS_PER_DATASET" "$COMMAND")"
    "$SGLANG_PYTHON" "$REPORTING" init --run-dir "$RUN_DIR" --settings-json "$SETTINGS_JSON"
else
    [[ -d "$RUN_DIR" && -f "$RUN_DIR/settings.json" ]] || { echo "ERROR: invalid run dir $RUN_DIR" >&2; exit 1; }
    mapfile -d '' -t SAVED < <("$SGLANG_PYTHON" -c 'import json,sys
x=json.load(open(sys.argv[1],encoding="utf-8")); keys="trace_root cost_document benchmarks concurrencies model served_model_name mode tokens context_length max_samples temperature top_p gpu_devices auto_gpu_min_free_gb search_gpu_hold_gb search_gpu_hold_chunk_gb tp_size batch_size client_concurrency gpu_memory_reserve_gb mem_fraction dtype port proxy_port nemo_skills_data_dir sglang_python eval_python sglang_src sglang_work_dir lora_path lora_mode extra_server_args cv_folds signal_bins lambda_max full_gate_grid max_invalid_row_rate split_seed dataset_max_attempts dataset_retry_delay_s allow_partial search_max_rows_per_dataset".split()
for k in keys: print(str(x.get(k,"")),end="\0")' "$RUN_DIR/settings.json")
    TRACE_ROOT="${SAVED[0]}"; COST_DOCUMENT="${SAVED[1]}"; ORDERED_BENCHMARKS="${SAVED[2]}"; CONCURRENCIES="${SAVED[3]}"
    MODEL="${SAVED[4]}"; SERVED_MODEL_NAME="${SAVED[5]}"; MODE="${SAVED[6]}"; TOKENS="${SAVED[7]}"; CONTEXT_LENGTH="${SAVED[8]}"; MAX_SAMPLES="${SAVED[9]}"; TEMPERATURE="${SAVED[10]}"; TOP_P="${SAVED[11]}"
    GPU_DEVICES="${SAVED[12]}"; AUTO_GPU_MIN_FREE_GB="${SAVED[13]}"; SEARCH_GPU_HOLD_GB="${SAVED[14]}"; SEARCH_GPU_HOLD_CHUNK_GB="${SAVED[15]}"; TP_SIZE="${SAVED[16]}"; [[ "$TP_SIZE" == auto ]] && TP_SIZE=""
    BATCH_SIZE="${SAVED[17]}"; CLIENT_CONCURRENCY="${SAVED[18]}"; GPU_MEMORY_RESERVE_GB="${SAVED[19]}"; MEM_FRACTION="${SAVED[20]}"; DTYPE="${SAVED[21]}"; PORT="${SAVED[22]}"; PROXY_PORT="${SAVED[23]}"
    NEMO_SKILLS_DATA_DIR="${SAVED[24]}"; SGLANG_PYTHON="${SAVED[25]}"; EVAL_PYTHON="${SAVED[26]}"; SGLANG_SRC="${SAVED[27]}"; SGLANG_WORK_DIR="${SAVED[28]}"; LORA_PATH="${SAVED[29]}"; LORA_MODE="${SAVED[30]}"; EXTRA_SERVER_ARGS="${SAVED[31]}"
    CV_FOLDS="${SAVED[32]}"; SIGNAL_BINS="${SAVED[33]}"; LAMBDA_MAX="${SAVED[34]}"; FULL_GATE_GRID="${SAVED[35]}"; MAX_INVALID_ROW_RATE="${SAVED[36]}"; SPLIT_SEED="${SAVED[37]}"; DATASET_MAX_ATTEMPTS="${SAVED[38]}"; DATASET_RETRY_DELAY_S="${SAVED[39]}"; ALLOW_PARTIAL="${SAVED[40]}"; SEARCH_MAX_ROWS_PER_DATASET="${SAVED[41]}"
    arg_was_set --gpu-devices && GPU_DEVICES="$CLI_GPU_DEVICES"; arg_was_set --tp-size && TP_SIZE="$CLI_TP_SIZE"
    arg_was_set --gpu-memory-reserve-gb && GPU_MEMORY_RESERVE_GB="$CLI_GPU_MEMORY_RESERVE_GB"; arg_was_set --search-gpu-hold-gb && SEARCH_GPU_HOLD_GB="$CLI_SEARCH_GPU_HOLD_GB"
    arg_was_set --port && PORT="$CLI_PORT"; arg_was_set --proxy-port && PROXY_PORT="$CLI_PROXY_PORT"
    arg_was_set --dataset-max-attempts && DATASET_MAX_ATTEMPTS="$CLI_DATASET_MAX_ATTEMPTS"; arg_was_set --dataset-retry-delay-s && DATASET_RETRY_DELAY_S="$CLI_DATASET_RETRY_DELAY_S"
fi

command -v flock >/dev/null 2>&1 || { echo "ERROR: flock is required" >&2; exit 1; }
exec 9>>"$RUN_DIR/runtime/runner.lock"; flock -n 9 || { echo "ERROR: another process owns $RUN_DIR" >&2; exit 89; }
printf 'pid=%s\nstarted_at=%s\n' "$$" "$(date --iso-8601=seconds)" >&9

event() {
    local phase="$1" dataset="$2" status="$3" records="$4" message="$5" payload
    payload="$($SGLANG_PYTHON -c 'import json,sys; print(json.dumps(dict(zip(("phase","dataset","status","records","message"),sys.argv[1:])),ensure_ascii=False))' "$phase" "$dataset" "$status" "$records" "$message")"
    "$SGLANG_PYTHON" "$REPORTING" event --run-dir "$RUN_DIR" --event-json "$payload"
}
json_valid() { [[ -s "$1" ]] && "$SGLANG_PYTHON" -c 'import json,sys; json.load(open(sys.argv[1],encoding="utf-8"))' "$1" >/dev/null 2>&1; }
search_outputs_current() {
    json_valid "$RUN_DIR/search/search_results.json" || return 1
    json_valid "$RUN_DIR/search/policy.json" || return 1
    "$SGLANG_PYTHON" -c 'import json,sys
p=json.load(open(sys.argv[1],encoding="utf-8")); p=p.get("policy",p)
expected={"2":32,"4":32,"8":16,"16":16,"32":16,"64":8,"128":8}
selected={key:expected[key] for key in sys.argv[2].split(",")}
profiles=p.get("concurrency_profiles") or {}
actual={key:int(profiles.get(key,{}).get("cold_start_block",-1)) for key in selected}
raise SystemExit(0 if p.get("schema_version")==2 and actual==selected else 1)' "$RUN_DIR/search/policy.json" "$CONCURRENCIES"
}
progress_bar() {
    local label="$1" done="$2" total="$3" note="$4" width=28 filled percent bar="" i
    (( total > 0 )) || total=1; filled=$((done*width/total)); percent=$((done*100/total))
    for ((i=0; i<filled; i++)); do bar+="#"; done
    for ((i=filled; i<width; i++)); do bar+="-"; done
    printf '[%s] |%s| %d/%d (%3d%%) %s\n' "$label" "$bar" "$done" "$total" "$percent" "$note"
}

GUARD_PID=""; GUARD_READY=""; GUARD_RELEASED=""; GUARD_LOG=""
stop_guard() {
    local reason="${1:-search ended}" count
    [[ -n "$GUARD_PID" ]] || return 0
    echo "[GPU显存守护] releasing PID=$GUARD_PID ($reason)"; kill -TERM "$GUARD_PID" 2>/dev/null || true
    for count in $(seq 1 60); do kill -0 "$GUARD_PID" 2>/dev/null || break; sleep 1; done
    kill -0 "$GUARD_PID" 2>/dev/null && kill -KILL "$GUARD_PID" 2>/dev/null || true; wait "$GUARD_PID" 2>/dev/null || true
    event gpu_guard "$GPU_DEVICES" released 0 "$reason；显存已释放"; GUARD_PID=""; sleep 2
}
cleanup() { local status=$?; set +e; stop_guard "主入口退出兜底"; return "$status"; }
trap cleanup EXIT
start_guard() {
    "$SGLANG_PYTHON" -c 'import sys; raise SystemExit(0 if float(sys.argv[1])>0 else 1)' "$SEARCH_GPU_HOLD_GB" || { event gpu_guard "$GPU_DEVICES" disabled 0 "预占0"; return 0; }
    local guard_dir="$RUN_DIR/runtime/gpu_memory_guard" suffix waited=0
    suffix="$(date +%Y%m%d_%H%M%S)_pid$$"; GUARD_READY="$guard_dir/ready_$suffix.json"; GUARD_RELEASED="$guard_dir/released_$suffix.json"; GUARD_LOG="$guard_dir/guard_$suffix.log"
    event gpu_guard "$GPU_DEVICES" starting 0 "每卡预占${SEARCH_GPU_HOLD_GB}GiB；$GUARD_LOG"
    CUDA_VISIBLE_DEVICES="$GPU_DEVICES" "$SGLANG_PYTHON" "$GPU_GUARD" --hold-gb "$SEARCH_GPU_HOLD_GB" --chunk-gb "$SEARCH_GPU_HOLD_CHUNK_GB" --ready-file "$GUARD_READY" --released-file "$GUARD_RELEASED" >"$GUARD_LOG" 2>&1 & GUARD_PID=$!
    while [[ ! -s "$GUARD_READY" ]]; do
        if ! kill -0 "$GUARD_PID" 2>/dev/null; then wait "$GUARD_PID" || true; tail -80 "$GUARD_LOG" >&2 || true; GUARD_PID=""; return 1; fi
        sleep 1; waited=$((waited+1)); (( waited < 300 )) || { echo "ERROR: guard timeout" >&2; return 1; }
    done
    event gpu_guard "$GPU_DEVICES" active "$SEARCH_GPU_HOLD_GB" "CPU搜索期间持续持有"
}

run_search() {
    if search_outputs_current; then event search 八集全局 skipped 0 "离线结果已存在且首轮协议一致"; return; fi
    [[ -d "$TRACE_ROOT" ]] || { echo "ERROR: trace root not found: $TRACE_ROOT" >&2; return 1; }; [[ -f "$COST_DOCUMENT" ]] || { echo "ERROR: cost document not found" >&2; return 1; }
    start_guard; event search 八集全局 running 0 "单/双信号、指定并发度精确lambda搜索"
    local -a cmd=("$SGLANG_PYTHON" "$SEARCH" --mode search --trace-root "$TRACE_ROOT" --output-dir "$RUN_DIR/search" --run-dir "$RUN_DIR" --cost-document "$COST_DOCUMENT" --concurrencies "$CONCURRENCIES" --split-seed "$SPLIT_SEED" --cv-folds "$CV_FOLDS" --signal-bins "$SIGNAL_BINS" --lambda-max "$LAMBDA_MAX" --full-gate-grid "$FULL_GATE_GRID" --invalid-row-policy exclude --max-invalid-row-rate "$MAX_INVALID_ROW_RATE")
    [[ "$ALLOW_PARTIAL" == true ]] && cmd+=(--allow-partial-datasets); [[ "$SEARCH_MAX_ROWS_PER_DATASET" == 0 ]] || cmd+=(--max-rows-per-dataset "$SEARCH_MAX_ROWS_PER_DATASET")
    if ! "${cmd[@]}"; then event search 八集全局 failed 0 "离线搜索失败"; stop_guard "CPU搜索失败"; return 1; fi
    event search 八集全局 completed 0 "离线结果、baseline提升和冻结策略已写入report.md"; stop_guard "CPU搜索完成"
}

dataset_completed() {
    local concurrency="$1" dataset="$2" trace="$3" marker
    marker="$RUN_DIR/runtime/completed_protocol_v2/c$concurrency/$dataset"
    [[ -s "$trace" && -s "$marker" && "$(<"$marker")" == 2 ]] || return 1
    "$SGLANG_PYTHON" -c 'import json,sys
x=json.load(open(sys.argv[1],encoding="utf-8")); key=sys.argv[2]
raise SystemExit(0 if any(e.get("phase")=="validate" and e.get("dataset")==key and e.get("status")=="completed" for e in x.get("events",[])) else 1)' "$RUN_DIR/run_state.json" "C$concurrency/$dataset"
}
analyze_concurrency() {
    local concurrency="$1" partial="$2"; local -a cmd=("$SGLANG_PYTHON" "$SEARCH" --mode validate --trace-root "$RUN_DIR/traces/validate/c$concurrency" --output-dir "$RUN_DIR/search" --run-dir "$RUN_DIR" --policy "$RUN_DIR/search/policy.json" --eval-root "$RUN_DIR/eval_runs/validate/c$concurrency" --cost-document "$COST_DOCUMENT" --concurrencies "$CONCURRENCIES" --concurrency "$concurrency" --split-seed "$SPLIT_SEED" --cv-folds "$CV_FOLDS")
    [[ "$partial" == true ]] && cmd+=(--allow-partial-datasets); "${cmd[@]}"
}

run_validation() {
    local policy="$RUN_DIR/search/policy.json"; [[ -s "$policy" ]] || { echo "ERROR: missing policy $policy" >&2; return 1; }; stop_guard "GPU验证前释放"
    local -a cs specs; IFS=',' read -ra cs <<< "$CONCURRENCIES"; IFS=',' read -ra specs <<< "$ORDERED_BENCHMARKS"
    local total=$((${#cs[@]}*${#specs[@]})) completed=0 concurrency spec name trace eval_parent attempt attempt_trace status records batch client
    for concurrency in "${cs[@]}"; do
        for spec in "${specs[@]}"; do name="${spec%%:*}"; trace="$RUN_DIR/traces/validate/c$concurrency/$name.jsonl"; if dataset_completed "$concurrency" "$name" "$trace"; then completed=$((completed+1)); fi; done
    done
    progress_bar "SGLang冻结验证" "$completed" "$total" "断点扫描完成"
    for concurrency in "${cs[@]}"; do
        if json_valid "$RUN_DIR/search/validation_c$concurrency.json" && "$SGLANG_PYTHON" -c 'import json,sys
x=json.load(open(sys.argv[1])); p=x.get("protocol",{}); c=int(sys.argv[2]); expected={2:32,4:32,8:16,16:16,32:16,64:8,128:8}
ok=p.get("formal_complete") is True and p.get("policy_schema_version")==2 and p.get("cold_start_block")==expected[c]
raise SystemExit(0 if ok else 1)' "$RUN_DIR/search/validation_c$concurrency.json" "$concurrency"; then continue; fi
        batch="$concurrency"; client="$concurrency"; [[ "$BATCH_SIZE" != auto ]] && batch="$BATCH_SIZE"; [[ "$CLIENT_CONCURRENCY" != auto ]] && client="$CLIENT_CONCURRENCY"
        for spec in "${specs[@]}"; do
            name="${spec%%:*}"; trace="$RUN_DIR/traces/validate/c$concurrency/$name.jsonl"; eval_parent="$RUN_DIR/eval_runs/validate/c$concurrency/$name"
            if dataset_completed "$concurrency" "$name" "$trace"; then progress_bar "SGLang冻结验证" "$completed" "$total" "复用 C$concurrency/$name"; continue; fi
            mkdir -p "$eval_parent" "$RUN_DIR/runtime/attempt_traces/c$concurrency/$name" "$(dirname "$trace")"; status=1; records=0
            for ((attempt=1; attempt<=DATASET_MAX_ATTEMPTS; attempt++)); do
                attempt_trace="$RUN_DIR/runtime/attempt_traces/c$concurrency/$name/attempt_$(date +%Y%m%d_%H%M%S)_pid${BASHPID}_n${attempt}.jsonl"; : >"$attempt_trace"
                event validate "C$concurrency/$name" running 0 "尝试$attempt/$DATASET_MAX_ATTEMPTS；max_running=$batch client=$client"
                local -a cmd=(bash "$EVAL_SGLANG" --mode "$MODE" --benchmarks "$spec" --model "$MODEL" --served-model-name "$SERVED_MODEL_NAME" --gpu-devices "$GPU_DEVICES" --batch-size "$batch" --client-concurrency "$client" --gpu-memory-reserve-gb "$GPU_MEMORY_RESERVE_GB" --tokens "$TOKENS" --context-length "$CONTEXT_LENGTH" --temperature "$TEMPERATURE" --top-p "$TOP_P" --mem-fraction "$MEM_FRACTION" --dtype "$DTYPE" --block-size 32 --cuda-graph-bs 1 --output-path "$eval_parent" --extra-server-args "--disable-cuda-graph $EXTRA_SERVER_ARGS")
                [[ -n "$TP_SIZE" ]] && cmd+=(--tp-size "$TP_SIZE"); [[ -n "$PORT" ]] && cmd+=(--port "$PORT"); [[ -n "$PROXY_PORT" ]] && cmd+=(--proxy-port "$PROXY_PORT")
                [[ -n "$NEMO_SKILLS_DATA_DIR" ]] && cmd+=(--nemo-skills-data-dir "$NEMO_SKILLS_DATA_DIR"); cmd+=(--sglang-python "$SGLANG_PYTHON" --eval-python "$EVAL_PYTHON")
                [[ -n "$SGLANG_SRC" ]] && cmd+=(--sglang-src "$SGLANG_SRC"); [[ -n "$SGLANG_WORK_DIR" ]] && cmd+=(--sglang-work-dir "$SGLANG_WORK_DIR"); [[ -n "$LORA_PATH" ]] && cmd+=(--lora-path "$LORA_PATH"); cmd+=(--lora-mode "$LORA_MODE")
                [[ -n "$MAX_SAMPLES" ]] && cmd+=(--max-samples "$MAX_SAMPLES"); status=0
                NLD_FAIL_ON_BENCHMARK_ERROR=1 NLD_KEEP_FAILED_WORK_DIR=1 NLD_DYNAMIC_BLOCK_ENABLE=0 NLD_UNIFIED_BLOCK_ENABLE=0 NLD_DUAL_EFF_ENABLE=0 NLD_B200_LATENCY_POLICY_ENABLE=1 NLD_DYNAMIC_BLOCK_SIZES="8,16,32" NLD_DYNAMIC_BLOCK_TRACE_FILE="$attempt_trace" NLD_DYNAMIC_BLOCK_BENCHMARK="$name" NLD_B200_LATENCY_POLICY_PATH="$policy" NLD_B200_LATENCY_CONCURRENCY="$concurrency" SGLANG_CONFIDENCE_TRACE_FILE="$attempt_trace" PYTHONPATH="$SCRIPT_DIR:$PROJECT_DIR/observations:${PYTHONPATH:-}" "${cmd[@]}" || status=$?
                records="$(wc -l < "$attempt_trace")"; if [[ "$status" == 0 && "$records" -gt 0 ]]; then
                    mv -f "$attempt_trace" "$trace"
                    mkdir -p "$RUN_DIR/runtime/completed_protocol_v2/c$concurrency"
                    printf '2\n' >"$RUN_DIR/runtime/completed_protocol_v2/c$concurrency/.$name.tmp.$BASHPID"
                    mv -f "$RUN_DIR/runtime/completed_protocol_v2/c$concurrency/.$name.tmp.$BASHPID" "$RUN_DIR/runtime/completed_protocol_v2/c$concurrency/$name"
                    break
                fi
                [[ "$status" != 0 ]] || status=87; event validate "C$concurrency/$name" retrying "$records" "尝试失败 exit=$status"; (( attempt < DATASET_MAX_ATTEMPTS )) && sleep "$DATASET_RETRY_DELAY_S"
            done
            if [[ "$status" != 0 || "$records" -eq 0 ]]; then event validate "C$concurrency/$name" failed "$records" "重试耗尽 exit=$status"; return "$status"; fi
            completed=$((completed+1)); event validate "C$concurrency/$name" completed "$records" "committed trace完成($completed/$total)"; progress_bar "SGLang冻结验证" "$completed" "$total" "完成 C$concurrency/$name"; analyze_concurrency "$concurrency" true
        done
        if [[ "$ALLOW_PARTIAL" == true ]]; then analyze_concurrency "$concurrency" true; else analyze_concurrency "$concurrency" false; fi
        event validate "C$concurrency/八集" completed 0 "该并发度动态轨迹、TPF、block占比和B200理论吞吐已汇总"
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
echo "Completed stage=$STAGE"; echo "Run directory: $RUN_DIR"; echo "Progress: $RUN_DIR/progress.md"; echo "Report: $RUN_DIR/report.md"
