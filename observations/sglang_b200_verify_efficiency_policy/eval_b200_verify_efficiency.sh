#!/bin/bash
# Fresh verifier trace -> three-family offline search -> one-winner validation.

set -euo pipefail

ORIGINAL_ARGS=("$@")
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
EVAL_SGLANG="${NLD_B200_VERIFY_EVAL_SGLANG:-$PROJECT_DIR/observations/eval_sglang.sh}"
SEARCH="${NLD_B200_VERIFY_SEARCH:-$SCRIPT_DIR/search.py}"
REPORTING="$SCRIPT_DIR/reporting.py"
GPU_GUARD="$SCRIPT_DIR/gpu_memory_guard.py"
RESULTS_ROOT="${NLD_OBSERVATION_RESULTS_ROOT:-/data/home/wly/dLLM/NLD_results/observations}/sglang_b200_verify_efficiency_policy_results"
DEFAULT_COST_DOCUMENT="$PROJECT_DIR/configs/NLD_B200_B8_B32_forward_sweep_20260907_zh.md"
DEFAULT_BENCHMARKS="gsm8k:1,human-eval:1,mbpp:1,math-500:1,aime25:1,gpqa:1,ifeval:1,livecodebench-cpp:1"
DEFAULT_CONCURRENCIES="2,4,8,16,32,64,128"
MODEL_8B="/data1/linyewei/models/Nemotron-Labs-Diffusion-8B"
MODEL_14B="/data1/linyewei/models/Nemotron-Labs-Diffusion-14B"
DEFAULT_PYTHON="/data/home/wly/.conda/envs/nld_sglang/bin/python"
[[ -x "$DEFAULT_PYTHON" ]] || DEFAULT_PYTHON="python"

usage() {
    cat <<EOF
Usage: $0 --stage collect|search|validate|remaining|all|report [options]

Lifecycle:
  --stage collect                Fresh B8/B16/B32 verifier shadow trace for 8B or 14B
  --stage search                 Guard GPU, compare three offline families, freeze all and winner
  --stage validate               Validate one frozen family; winner is default
  --stage remaining              Idempotently finish collect/search/winner validation
  --stage all                    Complete fresh trace -> guarded search -> winner validation
  --stage report                 Rebuild report.md and progress.md only
  --run-dir DIR                  Existing timestamp run for non-new stages

Model and data:
  --model-size 8b|14b            Defaults, run name, trace audit (default: 8b)
  --model PATH                   Override checkpoint selected by --model-size
  --lora-path DIR                Override MODEL/linear_spec_lora
  --served-model-name NAME       Override served name
  --benchmarks LIST              Formal default is eight datasets, excludes AIME24/MMLU;
                                   collection and validation preserve the supplied order
  --max-samples N                Development only; requires --allow-partial-datasets
  --tokens N                     Completion limit (default: 8192)
  --context-length N             Context length (default: 10240)
  --temperature V                Must be 0
  --top-p V                      Default: 0.95

Search and B200 objective:
  --cost-document FILE           Full C1..128 B8/B16/B32 latency markdown
  --concurrencies LIST           Default: 2,4,8,16,32,64,128
  --cv-folds N                   Request OOF folds (default: 5)
  --signal-bins N                Nonparametric bins (default: 24)
  --rho-grid-size N              Global-fractional deterministic rho grid (default: 33)
  --report-top N                 Saved ranking depth per family (default: 20)
  --max-rows-per-dataset N       Development CPU cap; formal value is 0
  --max-invalid-row-rate V       Excluded shadow-row ceiling (default: 0.05)

SGLang/resources:
  --gpu-devices LIST|auto        GPU selected once and reused (default: 0)
  --auto-gpu-min-free-gb V       Auto selection floor (default: 52)
  --tp-size N                    Default: 1; 14B is still single-GPU unless overridden
  --trace-batch-size N           Fresh exploration max-running (default: 1)
  --trace-client-concurrency N   Fresh exploration client concurrency (default: 1)
  --batch-size auto|N            Physical validation max-running (default: 1);
                                   auto=C is optional, not used by this protocol
  --client-concurrency auto|N    Physical validation client concurrency (default: 1);
                                   auto=C is optional, not used by this protocol
  --gpu-memory-reserve-gb V      Model-load reserve (default: 0)
  --mem-fraction V               SGLang static memory fraction (default: 0.55)
  --search-gpu-hold-gb V         Real CUDA memory held per GPU during CPU search (default: 48)
  --search-gpu-hold-chunk-gb V   Guard allocation chunk (default: 1)
  --policy-family NAME           winner/local_ratio/direct_rank/global_fractional
                                   validate defaults to winner; alternatives stay runnable
  --validation-trace-retention keep|delete-after-analysis
                                   Keep raw validation JSONL (default), or atomically
                                   compact each dataset/C result before deleting its JSONL
  --port N / --proxy-port N      Omit for collision-safe pipeline selection
  --dataset-max-attempts N       Fresh-server retries (default: 3)
  --dataset-retry-delay-s N      Retry delay (default: 10)
  --nemo-skills-data-dir DIR
  --sglang-python PATH
  --eval-python PATH
  --sglang-src DIR
  --sglang-work-dir DIR
  --dtype NAME                   Default: bfloat16
  --lora-mode MODE               draft_only or both (default: draft_only)
  --extra-server-args TEXT
  --allow-partial-datasets       Development/smoke protocol only
  --dry-run                      Resolve without writes or launches
EOF
}

STAGE=""; RUN_DIR=""; MODEL_SIZE="8b"; MODEL=""; LORA_PATH=""; SERVED_MODEL_NAME=""
BENCHMARKS="$DEFAULT_BENCHMARKS"; CONCURRENCIES="$DEFAULT_CONCURRENCIES"
COST_DOCUMENT=""; COST_DOCUMENT_USER_SET="false"
TOKENS="8192"; CONTEXT_LENGTH="10240"; TEMPERATURE="0"; TOP_P="0.95"; MAX_SAMPLES=""
GPU_DEVICES="0"; AUTO_GPU_MIN_FREE_GB="52"; TP_SIZE="1"
TRACE_BATCH_SIZE="1"; TRACE_CLIENT_CONCURRENCY="1"
BATCH_SIZE="1"; CLIENT_CONCURRENCY="1"; GPU_MEMORY_RESERVE_GB="0"
MEM_FRACTION="0.55"; SEARCH_GPU_HOLD_GB="48"; SEARCH_GPU_HOLD_CHUNK_GB="1"
POLICY_FAMILY="winner"; PORT=""; PROXY_PORT=""; DATASET_MAX_ATTEMPTS="3"; DATASET_RETRY_DELAY_S="10"
NEMO_SKILLS_DATA_DIR=""; SGLANG_PYTHON="$DEFAULT_PYTHON"; EVAL_PYTHON=""
SGLANG_SRC=""; SGLANG_WORK_DIR=""; DTYPE="bfloat16"; LORA_MODE="draft_only"; EXTRA_SERVER_ARGS=""
CV_FOLDS="5"; SIGNAL_BINS="24"; RHO_GRID_SIZE="33"; REPORT_TOP="20"; SPLIT_SEED="20260912"
MAX_ROWS_PER_DATASET="0"; ALLOW_PARTIAL="false"; DRY_RUN="false"
MAX_INVALID_ROW_RATE="0.05"
VALIDATION_TRACE_RETENTION="keep"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --stage) STAGE="$2"; shift 2 ;;
        --run-dir) RUN_DIR="$2"; shift 2 ;;
        --model-size) MODEL_SIZE="${2,,}"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --lora-path) LORA_PATH="$2"; shift 2 ;;
        --served-model-name|--model-name) SERVED_MODEL_NAME="$2"; shift 2 ;;
        --benchmarks) BENCHMARKS="$2"; shift 2 ;;
        --concurrencies) CONCURRENCIES="$2"; shift 2 ;;
        --cost-document) COST_DOCUMENT="$2"; COST_DOCUMENT_USER_SET="true"; shift 2 ;;
        --tokens) TOKENS="$2"; shift 2 ;;
        --context-length) CONTEXT_LENGTH="$2"; shift 2 ;;
        --temperature) TEMPERATURE="$2"; shift 2 ;;
        --top-p) TOP_P="$2"; shift 2 ;;
        --max-samples) MAX_SAMPLES="$2"; shift 2 ;;
        --gpu-devices) GPU_DEVICES="$2"; shift 2 ;;
        --auto-gpu-min-free-gb) AUTO_GPU_MIN_FREE_GB="$2"; shift 2 ;;
        --tp-size) TP_SIZE="$2"; shift 2 ;;
        --trace-batch-size) TRACE_BATCH_SIZE="$2"; shift 2 ;;
        --trace-client-concurrency) TRACE_CLIENT_CONCURRENCY="$2"; shift 2 ;;
        --batch-size|--max-running-requests) BATCH_SIZE="$2"; shift 2 ;;
        --client-concurrency) CLIENT_CONCURRENCY="$2"; shift 2 ;;
        --gpu-memory-reserve-gb) GPU_MEMORY_RESERVE_GB="$2"; shift 2 ;;
        --mem-fraction) MEM_FRACTION="$2"; shift 2 ;;
        --search-gpu-hold-gb) SEARCH_GPU_HOLD_GB="$2"; shift 2 ;;
        --search-gpu-hold-chunk-gb) SEARCH_GPU_HOLD_CHUNK_GB="$2"; shift 2 ;;
        --policy-family) POLICY_FAMILY="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --proxy-port) PROXY_PORT="$2"; shift 2 ;;
        --dataset-max-attempts) DATASET_MAX_ATTEMPTS="$2"; shift 2 ;;
        --dataset-retry-delay-s) DATASET_RETRY_DELAY_S="$2"; shift 2 ;;
        --nemo-skills-data-dir) NEMO_SKILLS_DATA_DIR="$2"; shift 2 ;;
        --sglang-python) SGLANG_PYTHON="$2"; shift 2 ;;
        --eval-python) EVAL_PYTHON="$2"; shift 2 ;;
        --sglang-src) SGLANG_SRC="$2"; shift 2 ;;
        --sglang-work-dir) SGLANG_WORK_DIR="$2"; shift 2 ;;
        --dtype) DTYPE="$2"; shift 2 ;;
        --lora-mode) LORA_MODE="$2"; shift 2 ;;
        --extra-server-args) EXTRA_SERVER_ARGS="$2"; shift 2 ;;
        --cv-folds) CV_FOLDS="$2"; shift 2 ;;
        --signal-bins) SIGNAL_BINS="$2"; shift 2 ;;
        --rho-grid-size) RHO_GRID_SIZE="$2"; shift 2 ;;
        --report-top) REPORT_TOP="$2"; shift 2 ;;
        --split-seed) SPLIT_SEED="$2"; shift 2 ;;
        --max-rows-per-dataset) MAX_ROWS_PER_DATASET="$2"; shift 2 ;;
        --max-invalid-row-rate) MAX_INVALID_ROW_RATE="$2"; shift 2 ;;
        --validation-trace-retention) VALIDATION_TRACE_RETENTION="$2"; shift 2 ;;
        --allow-partial-datasets) ALLOW_PARTIAL="true"; shift ;;
        --dry-run) DRY_RUN="true"; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "ERROR: unknown option $1" >&2; usage; exit 1 ;;
    esac
done

[[ -n "$STAGE" ]] || { echo "ERROR: --stage is required" >&2; exit 1; }
case "$STAGE" in collect|search|validate|remaining|all|report) ;; *) echo "ERROR: invalid stage $STAGE" >&2; exit 1 ;; esac
case "$MODEL_SIZE" in 8b|14b) ;; *) echo "ERROR: --model-size must be 8b or 14b" >&2; exit 1 ;; esac
case "$POLICY_FAMILY" in winner|local_ratio|direct_rank|global_fractional) ;; *) echo "ERROR: invalid --policy-family" >&2; exit 1 ;; esac
case "$VALIDATION_TRACE_RETENTION" in keep|delete-after-analysis) ;; *) echo "ERROR: invalid --validation-trace-retention" >&2; exit 1 ;; esac
[[ "$TEMPERATURE" == "0" || "$TEMPERATURE" == "0.0" ]] || { echo "ERROR: temperature must be 0" >&2; exit 1; }
for integer in "$TP_SIZE" "$TRACE_BATCH_SIZE" "$TRACE_CLIENT_CONCURRENCY" "$DATASET_MAX_ATTEMPTS" "$CV_FOLDS" "$SIGNAL_BINS" "$RHO_GRID_SIZE" "$REPORT_TOP"; do
    [[ "$integer" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: expected positive integer, got $integer" >&2; exit 1; }
done
[[ "$DATASET_RETRY_DELAY_S" =~ ^[0-9]+$ && "$MAX_ROWS_PER_DATASET" =~ ^[0-9]+$ ]] || { echo "ERROR: invalid nonnegative integer" >&2; exit 1; }
"$SGLANG_PYTHON" -c 'import sys; x=float(sys.argv[1]); raise SystemExit(0 if 0<=x<=1 else 1)' "$MAX_INVALID_ROW_RATE" || { echo "ERROR: --max-invalid-row-rate must be in [0,1]" >&2; exit 1; }
[[ "$BATCH_SIZE" == auto || "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: invalid --batch-size" >&2; exit 1; }
[[ "$CLIENT_CONCURRENCY" == auto || "$CLIENT_CONCURRENCY" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: invalid --client-concurrency" >&2; exit 1; }
[[ -n "$EVAL_PYTHON" ]] || EVAL_PYTHON="$SGLANG_PYTHON"

if [[ "$MODEL_SIZE" == 8b ]]; then
    [[ -n "$MODEL" ]] || MODEL="$MODEL_8B"
    [[ -n "$SERVED_MODEL_NAME" ]] || SERVED_MODEL_NAME="nemotron-labs-diffusion-8b"
    [[ -n "$COST_DOCUMENT" ]] || COST_DOCUMENT="$DEFAULT_COST_DOCUMENT"
else
    [[ -n "$MODEL" ]] || MODEL="$MODEL_14B"
    [[ -n "$SERVED_MODEL_NAME" ]] || SERVED_MODEL_NAME="nemotron-labs-diffusion-14b"
    if [[ -z "$COST_DOCUMENT" ]]; then
        echo "ERROR: 14B latency-aware search requires its own --cost-document" >&2
        exit 1
    fi
fi
[[ -n "$LORA_PATH" ]] || LORA_PATH="$MODEL/linear_spec_lora"

ORDERED_BENCHMARKS="$($SGLANG_PYTHON -c 'import sys
items=[x.strip() for x in sys.argv[1].split(",") if x.strip()]
names=[x.split(":",1)[0] for x in items]
if len(names)!=len(set(names)): raise SystemExit("duplicate benchmark")
if "aime24" in names or "mmlu" in names: raise SystemExit("AIME24/MMLU are excluded")
print(",".join(items))' "$BENCHMARKS")"
CONCURRENCIES="$($SGLANG_PYTHON -c 'import sys
values=sorted({int(x) for x in sys.argv[1].split(",") if x.strip()}); allowed={2,4,8,16,32,64,128}
if not values or not set(values)<=allowed: raise SystemExit("unsupported concurrency")
print(",".join(map(str,values)))' "$CONCURRENCIES")"

if [[ "$ALLOW_PARTIAL" != true ]]; then
    [[ -z "$MAX_SAMPLES" && "$MAX_ROWS_PER_DATASET" == 0 ]] || { echo "ERROR: formal run requires all samples/rows" >&2; exit 1; }
    [[ "$CONCURRENCIES" == "$DEFAULT_CONCURRENCIES" ]] || { echo "ERROR: formal run requires all seven C" >&2; exit 1; }
    "$SGLANG_PYTHON" -c 'import sys
expected={"gsm8k","human-eval","mbpp","math-500","aime25","gpqa","ifeval","livecodebench-cpp"}
actual={x.split(":",1)[0] for x in sys.argv[1].split(",") if x}
if actual!=expected: raise SystemExit(f"formal run needs exactly eight datasets; missing={sorted(expected-actual)} extra={sorted(actual-expected)}")' "$ORDERED_BENCHMARKS"
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
    echo "stage=$STAGE model_size=$MODEL_SIZE model=$MODEL"
    echo "benchmarks=$ORDERED_BENCHMARKS excluded=AIME24,MMLU"
    echo "concurrencies=$CONCURRENCIES cost=$COST_DOCUMENT"
    echo "fresh_trace=true verifier_logits=true policy_family=$POLICY_FAMILY"
    echo "search_guard=${SEARCH_GPU_HOLD_GB}GiB/GPU validation_order=dataset_outer,C_inner,winner_only validation_trace_retention=$VALIDATION_TRACE_RETENTION"
    exit 0
fi

[[ -d "$MODEL" ]] || { echo "ERROR: model not found: $MODEL" >&2; exit 1; }
[[ -d "$LORA_PATH" ]] || { echo "ERROR: LoRA not found: $LORA_PATH" >&2; exit 1; }
[[ -f "$COST_DOCUMENT" ]] || { echo "ERROR: cost document not found: $COST_DOCUMENT" >&2; exit 1; }
if [[ "$GPU_DEVICES" == auto && "$STAGE" != report ]]; then
    GPU_DEVICES="$(resolve_gpu)"
    echo "[GPU自动选择] 固定使用 GPU $GPU_DEVICES"
fi

arg_set() { local needle="$1" value; for value in "${ORIGINAL_ARGS[@]}"; do [[ "$value" == "$needle" ]] && return 0; done; return 1; }
CLI_GPU="$GPU_DEVICES"; CLI_RESERVE="$GPU_MEMORY_RESERVE_GB"; CLI_HOLD="$SEARCH_GPU_HOLD_GB"
CLI_PORT="$PORT"; CLI_PROXY="$PROXY_PORT"; CLI_ATTEMPTS="$DATASET_MAX_ATTEMPTS"; CLI_RETRY="$DATASET_RETRY_DELAY_S"; CLI_POLICY_FAMILY="$POLICY_FAMILY"
CLI_TRACE_BATCH="$TRACE_BATCH_SIZE"; CLI_TRACE_CLIENT="$TRACE_CLIENT_CONCURRENCY"
CLI_BATCH="$BATCH_SIZE"; CLI_CLIENT="$CLIENT_CONCURRENCY"; CLI_MEM_FRACTION="$MEM_FRACTION"
CLI_BENCHMARKS="$ORDERED_BENCHMARKS"; PERSIST_BENCHMARK_OVERRIDE="false"
CLI_VALIDATION_TRACE_RETENTION="$VALIDATION_TRACE_RETENTION"; PERSIST_RETENTION_OVERRIDE="false"

if [[ -z "$RUN_DIR" ]]; then
    [[ "$STAGE" == collect || "$STAGE" == all ]] || { echo "ERROR: a new run must start with collect or all" >&2; exit 1; }
    TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
    RUN_DIR="$RESULTS_ROOT/b200_verify_efficiency_${MODEL_SIZE}_${TIMESTAMP}"
    COMMAND="bash observations/sglang_b200_verify_efficiency_policy/eval_b200_verify_efficiency.sh ${ORIGINAL_ARGS[*]}"
    SETTINGS_JSON="$($SGLANG_PYTHON -c 'import json,sys
keys="stage model_size model lora_path served_model_name benchmarks concurrencies cost_document tokens context_length temperature top_p max_samples gpu_devices auto_gpu_min_free_gb tp_size trace_batch_size trace_client_concurrency batch_size client_concurrency gpu_memory_reserve_gb mem_fraction search_gpu_hold_gb search_gpu_hold_chunk_gb policy_family port proxy_port dataset_max_attempts dataset_retry_delay_s nemo_skills_data_dir sglang_python eval_python sglang_src sglang_work_dir dtype lora_mode extra_server_args cv_folds signal_bins rho_grid_size report_top split_seed max_rows_per_dataset allow_partial max_invalid_row_rate validation_trace_retention command".split()
print(json.dumps(dict(zip(keys,sys.argv[1:])),ensure_ascii=False))' "$STAGE" "$MODEL_SIZE" "$MODEL" "$LORA_PATH" "$SERVED_MODEL_NAME" "$ORDERED_BENCHMARKS" "$CONCURRENCIES" "$COST_DOCUMENT" "$TOKENS" "$CONTEXT_LENGTH" "$TEMPERATURE" "$TOP_P" "$MAX_SAMPLES" "$GPU_DEVICES" "$AUTO_GPU_MIN_FREE_GB" "$TP_SIZE" "$TRACE_BATCH_SIZE" "$TRACE_CLIENT_CONCURRENCY" "$BATCH_SIZE" "$CLIENT_CONCURRENCY" "$GPU_MEMORY_RESERVE_GB" "$MEM_FRACTION" "$SEARCH_GPU_HOLD_GB" "$SEARCH_GPU_HOLD_CHUNK_GB" "$POLICY_FAMILY" "$PORT" "$PROXY_PORT" "$DATASET_MAX_ATTEMPTS" "$DATASET_RETRY_DELAY_S" "$NEMO_SKILLS_DATA_DIR" "$SGLANG_PYTHON" "$EVAL_PYTHON" "$SGLANG_SRC" "$SGLANG_WORK_DIR" "$DTYPE" "$LORA_MODE" "$EXTRA_SERVER_ARGS" "$CV_FOLDS" "$SIGNAL_BINS" "$RHO_GRID_SIZE" "$REPORT_TOP" "$SPLIT_SEED" "$MAX_ROWS_PER_DATASET" "$ALLOW_PARTIAL" "$MAX_INVALID_ROW_RATE" "$VALIDATION_TRACE_RETENTION" "$COMMAND")"
    "$SGLANG_PYTHON" "$REPORTING" init --run-dir "$RUN_DIR" --settings-json "$SETTINGS_JSON"
else
    [[ -d "$RUN_DIR" && -f "$RUN_DIR/settings.json" ]] || { echo "ERROR: invalid --run-dir" >&2; exit 1; }
    mapfile -d '' -t SAVED < <("$SGLANG_PYTHON" -c 'import json,sys
x=json.load(open(sys.argv[1],encoding="utf-8")); keys="model_size model lora_path served_model_name benchmarks concurrencies cost_document tokens context_length temperature top_p max_samples gpu_devices auto_gpu_min_free_gb tp_size trace_batch_size trace_client_concurrency batch_size client_concurrency gpu_memory_reserve_gb mem_fraction search_gpu_hold_gb search_gpu_hold_chunk_gb port proxy_port dataset_max_attempts dataset_retry_delay_s nemo_skills_data_dir sglang_python eval_python sglang_src sglang_work_dir dtype lora_mode extra_server_args cv_folds signal_bins rho_grid_size report_top split_seed max_rows_per_dataset allow_partial max_invalid_row_rate validation_trace_retention".split()
for key in keys: print(str(x.get(key,"")),end="\0")' "$RUN_DIR/settings.json")
    MODEL_SIZE="${SAVED[0]}"; MODEL="${SAVED[1]}"; LORA_PATH="${SAVED[2]}"; SERVED_MODEL_NAME="${SAVED[3]}"; ORDERED_BENCHMARKS="${SAVED[4]}"; CONCURRENCIES="${SAVED[5]}"; COST_DOCUMENT="${SAVED[6]}"
    TOKENS="${SAVED[7]}"; CONTEXT_LENGTH="${SAVED[8]}"; TEMPERATURE="${SAVED[9]}"; TOP_P="${SAVED[10]}"; MAX_SAMPLES="${SAVED[11]}"; GPU_DEVICES="${SAVED[12]}"; AUTO_GPU_MIN_FREE_GB="${SAVED[13]}"; TP_SIZE="${SAVED[14]}"
    TRACE_BATCH_SIZE="${SAVED[15]}"; TRACE_CLIENT_CONCURRENCY="${SAVED[16]}"; BATCH_SIZE="${SAVED[17]}"; CLIENT_CONCURRENCY="${SAVED[18]}"; GPU_MEMORY_RESERVE_GB="${SAVED[19]}"; MEM_FRACTION="${SAVED[20]}"; SEARCH_GPU_HOLD_GB="${SAVED[21]}"; SEARCH_GPU_HOLD_CHUNK_GB="${SAVED[22]}"
    PORT="${SAVED[23]}"; PROXY_PORT="${SAVED[24]}"; DATASET_MAX_ATTEMPTS="${SAVED[25]}"; DATASET_RETRY_DELAY_S="${SAVED[26]}"; NEMO_SKILLS_DATA_DIR="${SAVED[27]}"; SGLANG_PYTHON="${SAVED[28]}"; EVAL_PYTHON="${SAVED[29]}"; SGLANG_SRC="${SAVED[30]}"; SGLANG_WORK_DIR="${SAVED[31]}"; DTYPE="${SAVED[32]}"; LORA_MODE="${SAVED[33]}"; EXTRA_SERVER_ARGS="${SAVED[34]}"
    CV_FOLDS="${SAVED[35]}"; SIGNAL_BINS="${SAVED[36]}"; RHO_GRID_SIZE="${SAVED[37]}"; REPORT_TOP="${SAVED[38]}"; SPLIT_SEED="${SAVED[39]}"; MAX_ROWS_PER_DATASET="${SAVED[40]}"; ALLOW_PARTIAL="${SAVED[41]}"; MAX_INVALID_ROW_RATE="${SAVED[42]}"
    VALIDATION_TRACE_RETENTION="${SAVED[43]:-keep}"; [[ -n "$VALIDATION_TRACE_RETENTION" ]] || VALIDATION_TRACE_RETENTION="keep"
    arg_set --gpu-devices && GPU_DEVICES="$CLI_GPU"
    arg_set --gpu-memory-reserve-gb && GPU_MEMORY_RESERVE_GB="$CLI_RESERVE"
    arg_set --search-gpu-hold-gb && SEARCH_GPU_HOLD_GB="$CLI_HOLD"
    arg_set --trace-batch-size && TRACE_BATCH_SIZE="$CLI_TRACE_BATCH"
    arg_set --trace-client-concurrency && TRACE_CLIENT_CONCURRENCY="$CLI_TRACE_CLIENT"
    if arg_set --batch-size || arg_set --max-running-requests; then BATCH_SIZE="$CLI_BATCH"; fi
    arg_set --client-concurrency && CLIENT_CONCURRENCY="$CLI_CLIENT"
    arg_set --mem-fraction && MEM_FRACTION="$CLI_MEM_FRACTION"
    arg_set --port && PORT="$CLI_PORT"; arg_set --proxy-port && PROXY_PORT="$CLI_PROXY"
    arg_set --dataset-max-attempts && DATASET_MAX_ATTEMPTS="$CLI_ATTEMPTS"
    arg_set --dataset-retry-delay-s && DATASET_RETRY_DELAY_S="$CLI_RETRY"
    arg_set --policy-family && POLICY_FAMILY="$CLI_POLICY_FAMILY"
    if arg_set --benchmarks; then
        ORDERED_BENCHMARKS="$CLI_BENCHMARKS"
        PERSIST_BENCHMARK_OVERRIDE="true"
    fi
    if arg_set --validation-trace-retention; then
        VALIDATION_TRACE_RETENTION="$CLI_VALIDATION_TRACE_RETENTION"
        PERSIST_RETENTION_OVERRIDE="true"
    fi
fi

command -v flock >/dev/null 2>&1 || { echo "ERROR: flock required" >&2; exit 1; }
exec 9>>"$RUN_DIR/runtime/runner.lock"
flock -n 9 || { echo "ERROR: another runner owns $RUN_DIR" >&2; exit 89; }
printf 'pid=%s\nstarted=%s\n' "$$" "$(date --iso-8601=seconds)" >&9
if [[ "$PERSIST_BENCHMARK_OVERRIDE" == true ]]; then
    BENCHMARK_UPDATE="$($SGLANG_PYTHON -c 'import json,sys; print(json.dumps({"benchmarks":sys.argv[1]},ensure_ascii=False))' "$ORDERED_BENCHMARKS")"
    "$SGLANG_PYTHON" "$REPORTING" update-settings --run-dir "$RUN_DIR" --settings-json "$BENCHMARK_UPDATE"
fi
if [[ "$PERSIST_RETENTION_OVERRIDE" == true ]]; then
    RETENTION_UPDATE="$($SGLANG_PYTHON -c 'import json,sys; print(json.dumps({"validation_trace_retention":sys.argv[1]},ensure_ascii=False))' "$VALIDATION_TRACE_RETENTION")"
    "$SGLANG_PYTHON" "$REPORTING" update-settings --run-dir "$RUN_DIR" --settings-json "$RETENTION_UPDATE"
fi

event() {
    local phase="$1" dataset="$2" status="$3" records="$4" message="$5" payload
    payload="$($SGLANG_PYTHON -c 'import json,sys; print(json.dumps(dict(zip(("phase","dataset","status","records","message"),sys.argv[1:])),ensure_ascii=False))' "$phase" "$dataset" "$status" "$records" "$message")"
    "$SGLANG_PYTHON" "$REPORTING" event --run-dir "$RUN_DIR" --event-json "$payload"
}
json_valid() { [[ -s "$1" ]] && "$SGLANG_PYTHON" -c 'import json,sys; json.load(open(sys.argv[1],encoding="utf-8"))' "$1" >/dev/null 2>&1; }
bar() {
    local label="$1" done="$2" total="$3" note="$4" width=28 filled percent output="" i
    (( total > 0 )) || total=1; filled=$((done*width/total)); percent=$((done*100/total))
    for ((i=0;i<filled;i++)); do output+="#"; done; for ((i=filled;i<width;i++)); do output+="-"; done
    printf '[%s] |%s| %d/%d (%3d%%) %s\n' "$label" "$output" "$done" "$total" "$percent" "$note"
}

GUARD_PID=""; GUARD_READY=""; GUARD_RELEASED=""; GUARD_LOG=""
stop_guard() {
    local reason="${1:-CPU搜索结束}" count
    [[ -n "$GUARD_PID" ]] || return 0
    echo "[GPU显存守护] 释放 PID=$GUARD_PID（$reason）"
    kill -TERM "$GUARD_PID" 2>/dev/null || true
    for count in $(seq 1 60); do kill -0 "$GUARD_PID" 2>/dev/null || break; sleep 1; done
    kill -0 "$GUARD_PID" 2>/dev/null && kill -KILL "$GUARD_PID" 2>/dev/null || true
    wait "$GUARD_PID" 2>/dev/null || true; GUARD_PID=""; sleep 2
    event guard "$GPU_DEVICES" released 0 "$reason；显存已释放给在线验证"
}
cleanup() { local status=$?; set +e; stop_guard "主入口退出兜底"; return "$status"; }
trap cleanup EXIT
start_guard() {
    "$SGLANG_PYTHON" -c 'import sys; raise SystemExit(0 if float(sys.argv[1])>0 else 1)' "$SEARCH_GPU_HOLD_GB" || { event guard "$GPU_DEVICES" disabled 0 "预占0GiB"; return 0; }
    local directory="$RUN_DIR/runtime/gpu_memory_guard" suffix waited=0
    suffix="$(date +%Y%m%d_%H%M%S)_pid$$"; mkdir -p "$directory"
    GUARD_READY="$directory/ready_$suffix.json"; GUARD_RELEASED="$directory/released_$suffix.json"; GUARD_LOG="$directory/guard_$suffix.log"
    event guard "$GPU_DEVICES" starting 0 "每卡预占${SEARCH_GPU_HOLD_GB}GiB"
    CUDA_VISIBLE_DEVICES="$GPU_DEVICES" PYTHONPATH="$PROJECT_DIR/observations:${PYTHONPATH:-}" "$SGLANG_PYTHON" "$GPU_GUARD" --hold-gb "$SEARCH_GPU_HOLD_GB" --chunk-gb "$SEARCH_GPU_HOLD_CHUNK_GB" --ready-file "$GUARD_READY" --released-file "$GUARD_RELEASED" >"$GUARD_LOG" 2>&1 & GUARD_PID=$!
    while [[ ! -s "$GUARD_READY" ]]; do
        kill -0 "$GUARD_PID" 2>/dev/null || { wait "$GUARD_PID" || true; tail -80 "$GUARD_LOG" >&2 || true; GUARD_PID=""; return 1; }
        sleep 1; waited=$((waited+1)); (( waited < 300 )) || { echo "ERROR: GPU guard timeout" >&2; return 1; }
    done
    echo "[GPU显存守护] 已就绪：GPU=$GPU_DEVICES，每卡${SEARCH_GPU_HOLD_GB}GiB；开始CPU搜索"
    event guard "$GPU_DEVICES" active "$SEARCH_GPU_HOLD_GB" "CPU检索期间持续持有显存"
}

phase_done() {
    local phase="$1" key="$2" trace="$3"
    [[ -s "$trace" ]] || return 1
    "$SGLANG_PYTHON" -c 'import json,sys
x=json.load(open(sys.argv[1],encoding="utf-8")); phase,key=sys.argv[2:]
raise SystemExit(0 if any(e.get("phase")==phase and e.get("dataset")==key and e.get("status")=="completed" for e in x.get("events",[])) else 1)' "$RUN_DIR/run_state.json" "$phase" "$key"
}

common_eval_cmd() {
    local spec="$1" output="$2" batch="$3" client="$4"
    EVAL_CMD=(bash "$EVAL_SGLANG" --mode linearspec_lora --benchmarks "$spec" --model "$MODEL" --served-model-name "$SERVED_MODEL_NAME" --gpu-devices "$GPU_DEVICES" --tp-size "$TP_SIZE" --batch-size "$batch" --client-concurrency "$client" --gpu-memory-reserve-gb "$GPU_MEMORY_RESERVE_GB" --tokens "$TOKENS" --context-length "$CONTEXT_LENGTH" --temperature "$TEMPERATURE" --top-p "$TOP_P" --mem-fraction "$MEM_FRACTION" --dtype "$DTYPE" --block-size 32 --cuda-graph-bs 1 --output-path "$output" --lora-path "$LORA_PATH" --lora-mode "$LORA_MODE" --sglang-python "$SGLANG_PYTHON" --eval-python "$EVAL_PYTHON" --extra-server-args "--disable-cuda-graph $EXTRA_SERVER_ARGS")
    [[ -n "$PORT" ]] && EVAL_CMD+=(--port "$PORT"); [[ -n "$PROXY_PORT" ]] && EVAL_CMD+=(--proxy-port "$PROXY_PORT")
    [[ -n "$NEMO_SKILLS_DATA_DIR" ]] && EVAL_CMD+=(--nemo-skills-data-dir "$NEMO_SKILLS_DATA_DIR")
    [[ -n "$SGLANG_SRC" ]] && EVAL_CMD+=(--sglang-src "$SGLANG_SRC"); [[ -n "$SGLANG_WORK_DIR" ]] && EVAL_CMD+=(--sglang-work-dir "$SGLANG_WORK_DIR")
    [[ -n "$MAX_SAMPLES" ]] && EVAL_CMD+=(--max-samples "$MAX_SAMPLES")
    # An absent optional argument is normal.  Without an explicit success
    # return, the final false [[ ... ]] above terminates the formal full-data
    # run under `set -e` before the first evaluator is launched.
    return 0
}

run_one_trace() {
    local phase="$1" key="$2" spec="$3" trace="$4" output="$5" policy_mode="$6" concurrency="$7" policy="$8" batch="$9" client="${10}"
    mkdir -p "$(dirname "$trace")" "$output" "$RUN_DIR/runtime/attempt_traces/$phase/$key"
    local attempt attempt_trace records=0 status=1
    for ((attempt=1;attempt<=DATASET_MAX_ATTEMPTS;attempt++)); do
        attempt_trace="$RUN_DIR/runtime/attempt_traces/$phase/$key/attempt_$(date +%Y%m%d_%H%M%S)_pid${BASHPID}_n${attempt}.jsonl"
        : >"$attempt_trace"; common_eval_cmd "$spec" "$output" "$batch" "$client"; status=0
        event "$phase" "$key" running 0 "尝试$attempt/$DATASET_MAX_ATTEMPTS；batch=$batch client=$client"
        NLD_FAIL_ON_BENCHMARK_ERROR=1 NLD_KEEP_FAILED_WORK_DIR=1 NLD_DYNAMIC_BLOCK_ENABLE=0 NLD_B200_LATENCY_POLICY_ENABLE=0 NLD_B200_VERIFY_EFFICIENCY_ENABLE=1 NLD_DYNAMIC_BLOCK_SIZES="8,16,32" NLD_DYNAMIC_BLOCK_POLICY_MODE="$policy_mode" NLD_DYNAMIC_BLOCK_POLICY_PATH="" NLD_DYNAMIC_BLOCK_TRACE_FILE="$attempt_trace" NLD_DYNAMIC_BLOCK_BENCHMARK="${spec%%:*}" NLD_DYNAMIC_BLOCK_SEED="$SPLIT_SEED" NLD_B200_VERIFY_MODEL_SIZE="$MODEL_SIZE" NLD_B200_VERIFY_POLICY_PATH="$policy" NLD_B200_VERIFY_CONCURRENCY="$concurrency" SGLANG_CONFIDENCE_TRACE_FILE="$attempt_trace" PYTHONPATH="$SCRIPT_DIR:$PROJECT_DIR/observations:${PYTHONPATH:-}" "${EVAL_CMD[@]}" || status=$?
        records="$(wc -l <"$attempt_trace")"
        if [[ "$status" == 0 && "$records" -gt 0 ]]; then mv -f "$attempt_trace" "$trace"; break; fi
        [[ "$status" != 0 ]] || status=87
        event "$phase" "$key" retrying "$records" "exit=$status；失败trace保留；下一次用全新server"
        (( attempt < DATASET_MAX_ATTEMPTS )) && sleep "$DATASET_RETRY_DELAY_S"
    done
    if [[ "$status" != 0 || "$records" -eq 0 ]]; then event "$phase" "$key" failed "$records" "重试耗尽 exit=$status"; return "$status"; fi
    event "$phase" "$key" completed "$records" "canonical trace原子提交"
}

run_collection() {
    echo "[阶段1/3] 采集八数据集全量新verifier shadow trace"
    local -a specs; IFS=',' read -ra specs <<<"$ORDERED_BENCHMARKS"; local total=${#specs[@]} done=0 spec name trace
    for spec in "${specs[@]}"; do name="${spec%%:*}"; trace="$RUN_DIR/traces/explore/$name.jsonl"; phase_done collect "$name" "$trace" && done=$((done+1)); done
    bar "新verifier trace" "$done" "$total" "断点扫描"
    for spec in "${specs[@]}"; do
        name="${spec%%:*}"; trace="$RUN_DIR/traces/explore/$name.jsonl"
        if phase_done collect "$name" "$trace"; then bar "新verifier trace" "$done" "$total" "复用 $name"; continue; fi
        run_one_trace collect "$name" "$spec" "$trace" "$RUN_DIR/eval_runs/explore/$name" explore 0 "" "$TRACE_BATCH_SIZE" "$TRACE_CLIENT_CONCURRENCY"
        done=$((done+1)); bar "新verifier trace" "$done" "$total" "完成 $name"
    done
}

run_search() {
    echo "[阶段2/3] 三策略族CPU离线搜索（搜索期间启用GPU显存守护）"
    if json_valid "$RUN_DIR/search/search_results.json" \
        && json_valid "$RUN_DIR/search/policy_local_ratio.json" \
        && json_valid "$RUN_DIR/search/policy_direct_rank.json" \
        && json_valid "$RUN_DIR/search/policy_global_fractional.json" \
        && json_valid "$RUN_DIR/search/policy_winner.json"; then
        event search 八集七C skipped 0 "三策略结果和winner均已存在"
        return
    fi
    local -a specs; IFS=',' read -ra specs <<<"$ORDERED_BENCHMARKS"; local spec name
    for spec in "${specs[@]}"; do name="${spec%%:*}"; phase_done collect "$name" "$RUN_DIR/traces/explore/$name.jsonl" || { echo "ERROR: fresh trace incomplete: $name" >&2; return 1; }; done
    start_guard; event search 八集七C running 0 "三策略族、单/双verify信号、OOF全量等权搜索"
    local -a cmd=("$SGLANG_PYTHON" "$SEARCH" --mode search --trace-root "$RUN_DIR/traces/explore" --output-dir "$RUN_DIR/search" --run-dir "$RUN_DIR" --cost-document "$COST_DOCUMENT" --model-size "$MODEL_SIZE" --concurrencies "$CONCURRENCIES" --split-seed "$SPLIT_SEED" --cv-folds "$CV_FOLDS" --signal-bins "$SIGNAL_BINS" --rho-grid-size "$RHO_GRID_SIZE" --report-top "$REPORT_TOP" --max-invalid-row-rate "$MAX_INVALID_ROW_RATE")
    [[ "$ALLOW_PARTIAL" == true ]] && cmd+=(--allow-partial-datasets); [[ "$MAX_ROWS_PER_DATASET" == 0 ]] || cmd+=(--max-rows-per-dataset "$MAX_ROWS_PER_DATASET")
    if ! "${cmd[@]}"; then event search 八集七C failed 0 "离线搜索失败"; stop_guard "搜索失败"; return 1; fi
    event search 八集七C completed 0 "三类策略已冻结；唯一winner已产生"; stop_guard "CPU搜索完成，即将使用GPU验证"
}

analyze_concurrency() {
    local family="$1" concurrency="$2" partial="$3" policy="$4"
    local -a cmd=("$SGLANG_PYTHON" "$SEARCH" --mode validate --trace-root "$RUN_DIR/traces/validate/$family/c$concurrency" --output-dir "$RUN_DIR/search" --run-dir "$RUN_DIR" --policy "$policy" --eval-root "$RUN_DIR/eval_runs/validate/$family/c$concurrency" --cost-document "$COST_DOCUMENT" --model-size "$MODEL_SIZE" --concurrencies "$CONCURRENCIES" --concurrency "$concurrency" --split-seed "$SPLIT_SEED" --cv-folds "$CV_FOLDS" --signal-bins "$SIGNAL_BINS" --max-invalid-row-rate "$MAX_INVALID_ROW_RATE")
    [[ "$partial" == true ]] && cmd+=(--allow-partial-datasets); "${cmd[@]}"
}

validation_part_path() {
    local family="$1" concurrency="$2" dataset="$3"
    printf '%s\n' "$RUN_DIR/search/validation_parts/$family/c$concurrency/$dataset.json"
}

compact_validation_valid() {
    local path="$1" family="$2" concurrency="$3" dataset="$4"
    [[ -s "$path" ]] && "$SGLANG_PYTHON" -c 'import json,sys
x=json.load(open(sys.argv[1],encoding="utf-8")); family,c,dataset=sys.argv[2],int(sys.argv[3]),sys.argv[4]
p=x.get("protocol") or {}; rows=(x.get("dynamic_policy") or {}).get("datasets") or {}
metrics={"samples","rounds","score_token_ms_req","pure_forward_token_ms_batch","decode_tpf","mean_accept","mean_forward_ms","l8_rate","l16_rate","l32_rate"}
fixed=x.get("fixed_baselines_same_dynamic_state") or {}
ok=(p.get("model_size")==sys.argv[5] and p.get("policy_family")==family and int(p.get("concurrency",-1))==c and int(p.get("policy_replay_mismatches",-1))==0 and set(rows)=={dataset} and metrics<=set(rows[dataset]) and all(set((fixed.get(str(b)) or {}).get("datasets",{}))=={dataset} and metrics<=set(fixed[str(b)]["datasets"][dataset]) for b in (8,16,32)))
raise SystemExit(0 if ok else 1)' "$path" "$family" "$concurrency" "$dataset" "$MODEL_SIZE" >/dev/null 2>&1
}

validation_done() {
    local family="$1" concurrency="$2" dataset="$3" key="$4" trace="$5" part
    if [[ "$VALIDATION_TRACE_RETENTION" == delete-after-analysis ]]; then
        part="$(validation_part_path "$family" "$concurrency" "$dataset")"
        compact_validation_valid "$part" "$family" "$concurrency" "$dataset"
    else
        phase_done validate "$key" "$trace"
    fi
}

analyze_validation_part() {
    local family="$1" concurrency="$2" dataset="$3" trace="$4" policy="$5" eval_root="$6"
    local parts_dir="$RUN_DIR/search/validation_parts/$family/c$concurrency" part
    part="$(validation_part_path "$family" "$concurrency" "$dataset")"
    mkdir -p "$parts_dir"
    local -a cmd=("$SGLANG_PYTHON" "$SEARCH" --mode validate --trace-root "$trace" --output-dir "$RUN_DIR/search" --run-dir "$RUN_DIR" --policy "$policy" --eval-root "$eval_root" --cost-document "$COST_DOCUMENT" --model-size "$MODEL_SIZE" --concurrencies "$CONCURRENCIES" --concurrency "$concurrency" --split-seed "$SPLIT_SEED" --cv-folds "$CV_FOLDS" --signal-bins "$SIGNAL_BINS" --max-invalid-row-rate "$MAX_INVALID_ROW_RATE" --allow-partial-datasets --validation-result "$part" --validation-parts-dir "$parts_dir")
    "${cmd[@]}"
    compact_validation_valid "$part" "$family" "$concurrency" "$dataset" || { echo "ERROR: compact validation result invalid: $part" >&2; return 1; }
    json_valid "$RUN_DIR/search/validation_${family}_c${concurrency}.json" || { echo "ERROR: merged validation result missing for C$concurrency" >&2; return 1; }
}

merge_validation_parts_for_concurrency() {
    local family="$1" concurrency="$2" policy="$3" partial="$4"
    local parts_dir="$RUN_DIR/search/validation_parts/$family/c$concurrency"
    local -a cmd=("$SGLANG_PYTHON" "$SEARCH" --mode merge-validation --output-dir "$RUN_DIR/search" --run-dir "$RUN_DIR" --policy "$policy" --cost-document "$COST_DOCUMENT" --model-size "$MODEL_SIZE" --concurrencies "$CONCURRENCIES" --concurrency "$concurrency" --validation-parts-dir "$parts_dir")
    [[ "$partial" == true ]] && cmd+=(--allow-partial-datasets)
    "${cmd[@]}"
}

run_validation() {
    echo "[阶段3/3] 冻结策略真实SGLang动态轨迹验证（数据集外层、C内层）"
    stop_guard "在线验证前释放"
    local requested_family="$POLICY_FAMILY" family="$POLICY_FAMILY" policy
    if [[ "$family" == winner ]]; then policy="$RUN_DIR/search/policy_winner.json"; family="$($SGLANG_PYTHON -c 'import json,sys; x=json.load(open(sys.argv[1])); print(x.get("policy",x)["family"])' "$policy")"; else policy="$RUN_DIR/search/policy_$family.json"; fi
    json_valid "$policy" || { echo "ERROR: missing frozen policy $policy" >&2; return 1; }
    local -a specs cs; IFS=',' read -ra specs <<<"$ORDERED_BENCHMARKS"; IFS=',' read -ra cs <<<"$CONCURRENCIES"
    local total=$((${#specs[@]}*${#cs[@]})) done=0 spec name concurrency key trace batch client trace_bytes
    for spec in "${specs[@]}"; do name="${spec%%:*}"; for concurrency in "${cs[@]}"; do key="$name/C$concurrency/$family"; trace="$RUN_DIR/traces/validate/$family/c$concurrency/$name.jsonl"; validation_done "$family" "$concurrency" "$name" "$key" "$trace" && done=$((done+1)); done; done
    bar "winner真实验证" "$done" "$total" "策略=$family；数据集外层/C内层"
    for spec in "${specs[@]}"; do
        name="${spec%%:*}"
        for concurrency in "${cs[@]}"; do
            key="$name/C$concurrency/$family"; trace="$RUN_DIR/traces/validate/$family/c$concurrency/$name.jsonl"
            if validation_done "$family" "$concurrency" "$name" "$key" "$trace"; then
                if [[ "$VALIDATION_TRACE_RETENTION" == delete-after-analysis && -f "$trace" ]]; then rm -f -- "$trace"; fi
                bar "winner真实验证" "$done" "$total" "复用 $name/C$concurrency"; continue
            fi
            batch="$concurrency"; client="$concurrency"; [[ "$BATCH_SIZE" != auto ]] && batch="$BATCH_SIZE"; [[ "$CLIENT_CONCURRENCY" != auto ]] && client="$CLIENT_CONCURRENCY"
            if [[ ! -s "$trace" ]]; then
                run_one_trace validate "$key" "$spec" "$trace" "$RUN_DIR/eval_runs/validate/$family/c$concurrency/$name" verify_efficiency_frozen "$concurrency" "$policy" "$batch" "$client"
            fi
            if [[ "$VALIDATION_TRACE_RETENTION" == delete-after-analysis ]]; then
                analyze_validation_part "$family" "$concurrency" "$name" "$trace" "$policy" "$RUN_DIR/eval_runs/validate/$family/c$concurrency/$name"
                trace_bytes="$(stat -c '%s' "$trace")"
                rm -f -- "$trace"
                event validate "$key" completed 0 "紧凑结果已原子保存；删除原始trace ${trace_bytes} bytes"
            else
                analyze_concurrency "$family" "$concurrency" true "$policy"
            fi
            done=$((done+1)); bar "winner真实验证" "$done" "$total" "完成 $name/C$concurrency"
        done
    done
    for concurrency in "${cs[@]}"; do
        if [[ "$VALIDATION_TRACE_RETENTION" == delete-after-analysis ]]; then
            merge_validation_parts_for_concurrency "$family" "$concurrency" "$policy" "$ALLOW_PARTIAL"
        elif [[ "$ALLOW_PARTIAL" == true ]]; then
            analyze_concurrency "$family" "$concurrency" true "$policy"
        else
            analyze_concurrency "$family" "$concurrency" false "$policy"
        fi
    done
    if [[ "$requested_family" == winner ]]; then
        event finalize 八集七C completed 0 "唯一winner=$family全部动态验证完成"
    else
        event finalize-alternative 八集七C completed 0 "附加策略=$family全部动态验证完成"
    fi
}

case "$STAGE" in
    collect) run_collection ;;
    search) run_search ;;
    validate) run_validation ;;
    remaining) run_collection; run_search; run_validation ;;
    all) run_collection; run_search; run_validation ;;
    report) "$SGLANG_PYTHON" "$REPORTING" render --run-dir "$RUN_DIR" ;;
esac
"$SGLANG_PYTHON" "$REPORTING" render --run-dir "$RUN_DIR"
echo "Completed stage=$STAGE"
echo "Run directory: $RUN_DIR"
echo "Progress: $RUN_DIR/progress.md"
echo "Report: $RUN_DIR/report.md"
