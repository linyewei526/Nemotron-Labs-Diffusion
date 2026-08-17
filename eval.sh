#!/bin/bash
# Unified eval driver for the Nemotron-Labs diffusion-LM family.
# One SLURM job per (mode, benchmark). See --help for full usage.

set -euo pipefail

# ─── Paths / defaults ───────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"
export PIPELINE_PRESET_OVERRIDE="${PIPELINE_PRESET_OVERRIDE:-$PROJECT_DIR/xp/examples/run_dlm_eval_pipeline_gpu_only.sh}"

# ─── Default cache + output paths ──────────────────────────────────────────
# All paths below are env-overridable (export HF_HOME=..., OUT_DIR=..., etc.
# to point at cluster-specific locations).
_DEFAULT_CACHE="$HOME/.cache/huggingface"
_DEFAULT_OUT_DIR="$PWD/eval_suit_results"
_DEFAULT_HF_TOKEN_FILE=""

# ─── HF / SLURM secrets ─────────────────────────────────────────────────────
HF_TOKEN_FILE="${HF_TOKEN_FILE:-$_DEFAULT_HF_TOKEN_FILE}"
if [[ -z "${HF_TOKEN:-}" && -n "$HF_TOKEN_FILE" && -f "$HF_TOKEN_FILE" ]]; then
    export HF_TOKEN="$(<"$HF_TOKEN_FILE")"
fi
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-${HF_TOKEN:-}}"

export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$_DEFAULT_CACHE}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$_DEFAULT_CACHE}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$_DEFAULT_CACHE}"
export HF_HOME="${HF_HOME:-$_DEFAULT_CACHE}"
# Note: mkdir -p "$HF_HOME" is deferred until after the --help/--dry-run
# short-circuit so help commands stay side-effect-free.

# Pick a container image. LoRA needs peft, which only PEFT_IMAGE ships;
# eval.sh auto-promotes to it when USE_LORA=true (re-evaluated post-CLI-parse).
# Optional pre-baked container images for srun/enroot. Leave all three unset
# and use evaluate.py instead if you don't already have a NeMo-Skills-ready
# .sqsh on disk. eval.sh's _pick_container_image picks the first one that
# exists; setting CONTAINER_IMAGE directly is the simplest override.
PEFT_IMAGE="${PEFT_IMAGE:-}"       # ships peft — needed for --lora
PREBAKED_IMAGE="${PREBAKED_IMAGE:-}"   # prebuilt NeMo-Skills image
OLD_IMAGE="${OLD_IMAGE:-}"         # any image that can run NeMo-Skills + DLM workers
# Best-effort container-image pick. Non-fatal: --help and --dry-run must
# still work on machines that do not have prebuilt .sqsh images on
# disk. The strict (errors-out) check happens after CLI parsing, only on
# real submission paths.
_pick_container_image() {
    # caller arg $1: true=need PEFT (LoRA path), anything else=default order
    if [[ "${1:-}" == "true" && -f "$PEFT_IMAGE" ]]; then
        export CONTAINER_IMAGE="$PEFT_IMAGE"
    elif [[ -f "$PREBAKED_IMAGE" ]]; then
        export CONTAINER_IMAGE="$PREBAKED_IMAGE"
    elif [[ -f "$OLD_IMAGE" ]]; then
        export CONTAINER_IMAGE="$OLD_IMAGE"
    fi
    # Empty CONTAINER_IMAGE is OK here — _require_container_image gates the
    # actual submission.
}
_require_container_image() {
    if [[ -z "${CONTAINER_IMAGE:-}" || ! -f "${CONTAINER_IMAGE}" ]]; then
        echo "" >&2
        echo "ERROR: no container image available for srun submission." >&2
        echo "Either:" >&2
        echo "  (a) use the no-server path:  python evaluate.py --mode ... --tasks gsm8k" >&2
        echo "                               (pip install torch transformers datasets peft)" >&2
        echo "  (b) export CONTAINER_IMAGE=<path-to-your-image.sqsh> before re-running." >&2
        echo "" >&2
        exit 1
    fi
}
if [[ -z "${CONTAINER_IMAGE:-}" ]]; then
    # Provisional pick (no LoRA info yet); re-evaluated after CLI parse.
    _pick_container_image "false"
fi

# ─── Default model / tokenizer ─────────────────────────────────────────────
DEFAULT_MODEL="/data1/linyewei/models/Nemotron-Labs-Diffusion-8B"
# Empty = use the tokenizer bundled with --model on HF.
DEFAULT_TOKENIZER=""

# Per-mode defaults — only what actually differs per mode. Shared values
# (tokens=8192, steps=8192, max_thinking=6000, batch_size=1, gpus=8,
# temperature=0) are hoisted into DEF_* once.
DEF_TOKENS="8192"; DEF_STEPS="8192"; DEF_MAX_THINKING="6000"
DEF_BATCH_SIZE="1"; DEF_GPUS="8"; DEF_TEMPERATURE="0"

default_mode_settings() {
    case "$1" in
        dlm)
            # DEF_ENGINE="" -> dlm_batch_server defaults to nemotron.
            DEF_ENGINE=""; DEF_GEN_ALGO="nemotron"
            DEF_MAX_MODEL_LEN="20480"; DEF_MAX_POS_EMB="262144"
            DEF_BLOCK_LENGTH="8"; DEF_THRESHOLD="0.9"
            DEF_MODEL_TAG="nemotron-labs-diffusion-8b"
            ;;
        ar)
            DEF_ENGINE="ar_native"; DEF_GEN_ALGO="ar_native"
            DEF_MAX_MODEL_LEN="65536"; DEF_MAX_POS_EMB=""
            DEF_BLOCK_LENGTH="1"; DEF_THRESHOLD=""
            DEF_MODEL_TAG="nemotron-labs-diffusion-8b"
            ;;
        linear_spec)
            DEF_ENGINE=""; DEF_GEN_ALGO="nemotron"
            DEF_MAX_MODEL_LEN="20480"; DEF_MAX_POS_EMB="262144"
            DEF_BLOCK_LENGTH="32"; DEF_THRESHOLD="0"
            DEF_MODEL_TAG="nemotron-labs-diffusion-8b"
            ;;
        *)
            echo "ERROR: unknown mode '$1' (must be dlm|ar|linear_spec)" >&2
            exit 1
            ;;
    esac
}

# ─── CLI parsing ────────────────────────────────────────────────────────────
usage() {
    cat <<EOF
Usage: $0 --mode {dlm|ar|linear_spec} [options]

Required:
  --mode MODE              Generation mode: dlm | ar | linear_spec

Model:
  --model HF_ID            HuggingFace model id (default: $DEFAULT_MODEL)
  --tokenizer ID_OR_PATH   Tokenizer (HF id or directory). Default: model's own tokenizer.
  --exp-name NAME          Experiment name / subdir under EVAL_BASE_DIR
                           (default: auto-derived from mode + model + key knobs)

LoRA (linear_spec only — silently ignored elsewhere):
  --lora                   Attach LoRA adapter (LORA_PATH must be set / passed)
  --no-lora                Run without LoRA (default)
  --lora-path PATH         LoRA adapter directory
                           (default: \$PROJECT_DIR/miscs/linear_spec_lora — the
                           adapter bundled in the public HF model repo)
  --draft-lora-only BOOL   Pass --draft-lora-only to worker (default: false;
                           the refactored public model has a unified
                           LoRA-aware linear_spec_generate, no *_lora variant)


Generation knobs (defaults vary per --mode):
  --benchmarks LIST        Comma-separated "name:reps" pairs
                           (default: "gsm8k:1,human-eval:1,mbpp:1,math-500:1,
                                      aime24:1,aime25:1,gpqa:1,mmlu:1,ifeval:1,
                                      livecodebench-cpp:1")
  --tokens N               SEQ_EVAL_TOKENS_TO_GENERATE
  --steps N                SEQ_EVAL_STEPS
  --block-length N         SEQ_EVAL_BLOCK_LENGTH
  --threshold V            SEQ_EVAL_THRESHOLD
  --temperature V          SEQ_EVAL_TEMPERATURE
  --max-thinking-tokens N  SEQ_EVAL_MAX_THINKING_TOKENS
  --max-model-len N        SERVER_MAX_MODEL_LEN
  --max-pos-emb N          SERVER_MAX_POSITION_EMBEDDINGS
  --batch-size N           SERVER_BATCH_SIZE
  --enable-thinking BOOL   Enable model "thinking" mode (default: false)
  --no-eos-early-stop      Disable SERVER_EOS_EARLY_STOP (default: enabled)

LLM judge (used by Arena-Hard and MT-Bench):
  --judge-model NAME       Override the default GPT-4.1 judge
  --judge-server-address URL
                           Override the default https://api.openai.com/v1 endpoint
  --judge-server-type TYPE Judge server type (currently openai-compatible)
  --judge-concurrency N    Concurrent MT-Bench judge requests (default: 4)
  --mt-bench-max-tokens N  Completion budget per MT-Bench turn (default: 1024)
  --skip-judge-api-key-check
                           Skip the host-side OPENAI_API_KEY preflight check

SLURM:
  --gpus N                 GPUs per job (default per-mode)
  --partition LIST         SLURM partition (default: batch,backfill)
  --account ACCT           SLURM account (required for srun submissions)
  --time HH:MM:SS          SLURM time limit (default: 04:00:00)

Output:
  --out-dir DIR            EVAL_BASE_DIR (default:
                           default: $PWD/eval_suit_results)

Misc:
  --dry-run                Print resolved settings, do not submit jobs
  -h, --help               Show this help

Examples:
  # quick gsm8k sanity check, dlm mode
  bash $0 --mode dlm --benchmarks gsm8k:1

  # AR mode on a different model
  bash $0 --mode ar --model /data1/linyewei/models/Nemotron-Labs-Diffusion-8B --benchmarks gsm8k:1

  # linear self-spec WITH LoRA
  bash $0 --mode linear_spec --lora --benchmarks gsm8k:1

EOF
}

MODE=""
MODEL=""
TOKENIZER=""
EXP_NAME=""
USE_LORA=""
LORA_PATH_ARG=""
DRAFT_LORA_ONLY_ARG="false"
BENCHMARKS_LIST=""
TOKENS_ARG=""
STEPS_ARG=""
BLOCK_LENGTH_ARG=""
THRESHOLD_ARG=""
TEMPERATURE_ARG=""
MAX_THINKING_ARG=""
MAX_MODEL_LEN_ARG=""
MAX_POS_EMB_ARG=""
BATCH_SIZE_ARG=""
ENABLE_THINKING="false"
EOS_EARLY_STOP="true"
JUDGE_MODEL=""
JUDGE_SERVER_ADDRESS=""
JUDGE_SERVER_TYPE=""
JUDGE_CONCURRENCY="4"
MT_BENCH_MAX_TOKENS="1024"
SKIP_JUDGE_API_KEY_CHECK="false"
GPUS_ARG=""
PARTITION="${SERVER_PARTITION:-batch,backfill}"
ACCOUNT_ARG="${ACCOUNT:-}"
TIME_ARG="${SERVER_TIME:-04:00:00}"
OUT_DIR="${OUT_DIR:-}"  # honour env-injected OUT_DIR; --out-dir overrides
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)               MODE="$2"; shift 2 ;;
        --model)              MODEL="$2"; shift 2 ;;
        --tokenizer)          TOKENIZER="$2"; shift 2 ;;
        --exp-name)           EXP_NAME="$2"; shift 2 ;;
        --lora)               USE_LORA="true"; shift 1 ;;
        --no-lora)            USE_LORA="false"; shift 1 ;;
        --lora-path)          LORA_PATH_ARG="$2"; shift 2 ;;
        --draft-lora-only)    DRAFT_LORA_ONLY_ARG="$2"; shift 2 ;;
        --benchmarks)         BENCHMARKS_LIST="$2"; shift 2 ;;
        --tokens)             TOKENS_ARG="$2"; shift 2 ;;
        --steps)              STEPS_ARG="$2"; shift 2 ;;
        --block-length)       BLOCK_LENGTH_ARG="$2"; shift 2 ;;
        --threshold)          THRESHOLD_ARG="$2"; shift 2 ;;
        --temperature)        TEMPERATURE_ARG="$2"; shift 2 ;;
        --max-thinking-tokens) MAX_THINKING_ARG="$2"; shift 2 ;;
        --max-model-len)      MAX_MODEL_LEN_ARG="$2"; shift 2 ;;
        --max-pos-emb)        MAX_POS_EMB_ARG="$2"; shift 2 ;;
        --batch-size)         BATCH_SIZE_ARG="$2"; shift 2 ;;
        --enable-thinking)    ENABLE_THINKING="$2"; shift 2 ;;
        --no-eos-early-stop)  EOS_EARLY_STOP="false"; shift 1 ;;
        --judge-model)        JUDGE_MODEL="$2"; shift 2 ;;
        --judge-server-address) JUDGE_SERVER_ADDRESS="$2"; shift 2 ;;
        --judge-server-type)  JUDGE_SERVER_TYPE="$2"; shift 2 ;;
        --judge-concurrency)  JUDGE_CONCURRENCY="$2"; shift 2 ;;
        --mt-bench-max-tokens) MT_BENCH_MAX_TOKENS="$2"; shift 2 ;;
        --skip-judge-api-key-check) SKIP_JUDGE_API_KEY_CHECK="true"; shift 1 ;;
        --gpus)               GPUS_ARG="$2"; shift 2 ;;
        --partition)          PARTITION="$2"; shift 2 ;;
        --account)            ACCOUNT_ARG="$2"; shift 2 ;;
        --time)               TIME_ARG="$2"; shift 2 ;;
        --out-dir)            OUT_DIR="$2"; shift 2 ;;
        --dry-run)            DRY_RUN=true; shift 1 ;;
        -h|--help)            usage; exit 0 ;;
        *) echo "ERROR: unknown option: $1" >&2; usage; exit 1 ;;
    esac
done

if [[ -z "$MODE" ]]; then
    echo "ERROR: --mode is required" >&2
    usage
    exit 1
fi

default_mode_settings "$MODE"

# ─── Resolve final values (CLI > defaults) ─────────────────────────────────
MODEL="${MODEL:-$DEFAULT_MODEL}"
TOKENIZER="${TOKENIZER:-$DEFAULT_TOKENIZER}"
TOKENS="${TOKENS_ARG:-$DEF_TOKENS}"
STEPS="${STEPS_ARG:-$DEF_STEPS}"
BLOCK_LENGTH="${BLOCK_LENGTH_ARG:-$DEF_BLOCK_LENGTH}"
THRESHOLD="${THRESHOLD_ARG:-$DEF_THRESHOLD}"
TEMPERATURE="${TEMPERATURE_ARG:-$DEF_TEMPERATURE}"
MAX_THINKING="${MAX_THINKING_ARG:-$DEF_MAX_THINKING}"
MAX_MODEL_LEN="${MAX_MODEL_LEN_ARG:-$DEF_MAX_MODEL_LEN}"
MAX_POS_EMB="${MAX_POS_EMB_ARG:-$DEF_MAX_POS_EMB}"
BATCH_SIZE="${BATCH_SIZE_ARG:-$DEF_BATCH_SIZE}"
GPUS="${GPUS_ARG:-$DEF_GPUS}"
OUT_DIR="${OUT_DIR:-$_DEFAULT_OUT_DIR}"
# realpath -m is allow-missing: resolves the canonical path even if the
# directory does not exist yet (so --dry-run stays side-effect-free).
# The actual mkdir is deferred until after the dry-run early exit below.
if command -v realpath >/dev/null; then OUT_DIR="$(realpath -m "$OUT_DIR")"; fi

DEFAULT_BENCHMARKS="gsm8k:1,human-eval:1,mbpp:1,math-500:1,aime24:1,aime25:1,gpqa:1,mmlu:1,ifeval:1,livecodebench-cpp:1"
BENCHMARKS_LIST="${BENCHMARKS_LIST:-$DEFAULT_BENCHMARKS}"
IFS=',' read -ra BENCHMARK_GROUPS <<< "$BENCHMARKS_LIST"

arena_hard_requested() {
    local spec name
    for spec in "${BENCHMARK_GROUPS[@]}"; do
        spec="${spec//[[:space:]]/}"
        name="${spec%%:*}"
        [[ "$name" == "arena-hard" || "$name" == "arena-hard-v2" ]] && return 0
    done
    return 1
}

mt_bench_requested() {
    local spec name
    for spec in "${BENCHMARK_GROUPS[@]}"; do
        spec="${spec//[[:space:]]/}"
        name="${spec%%:*}"
        [[ "$name" == "mt-bench" ]] && return 0
    done
    return 1
}

judge_benchmark_requested() {
    arena_hard_requested || mt_bench_requested
}

[[ "$JUDGE_CONCURRENCY" =~ ^[1-9][0-9]*$ ]] || {
    echo "ERROR: --judge-concurrency must be a positive integer" >&2
    exit 1
}
[[ "$MT_BENCH_MAX_TOKENS" =~ ^[1-9][0-9]*$ ]] || {
    echo "ERROR: --mt-bench-max-tokens must be a positive integer" >&2
    exit 1
}
if mt_bench_requested && [[ -n "$JUDGE_SERVER_TYPE" && "$JUDGE_SERVER_TYPE" != "openai" ]]; then
    echo "ERROR: MT-Bench currently supports only an OpenAI-compatible judge endpoint." >&2
    exit 1
fi

# LoRA
if [[ "$MODE" == "linear_spec" ]]; then
    USE_LORA="${USE_LORA:-false}"
    if [[ "$USE_LORA" == "true" ]]; then
        LORA_PATH_FINAL="${LORA_PATH_ARG:-$PROJECT_DIR/miscs/linear_spec_lora}"
        if [[ ! -d "$LORA_PATH_FINAL" ]]; then
            echo "ERROR: --lora requested but LoRA dir not found: $LORA_PATH_FINAL" >&2
            exit 1
        fi
    else
        LORA_PATH_FINAL=""
    fi
else
    USE_LORA="false"
    LORA_PATH_FINAL=""
fi

# Re-pick container now that USE_LORA is known. LoRA needs PEFT, which only
# the PEFT_IMAGE ships pre-installed; without it the worker errors out on load.
_pick_container_image "$USE_LORA"
# Strict checks: real submissions need a valid image and a SLURM account.
# Dry-run stays permissive so users can preview settings on any host.
if [[ "$DRY_RUN" != true ]]; then
    if judge_benchmark_requested \
        && [[ "$SKIP_JUDGE_API_KEY_CHECK" != "true" ]] \
        && { [[ -z "$JUDGE_SERVER_ADDRESS" ]] || [[ "${JUDGE_SERVER_ADDRESS,,}" == *"api.openai.com"* ]]; } \
        && [[ -z "${OPENAI_API_KEY:-}" ]]; then
        echo "ERROR: The requested judge-based benchmark defaults to GPT-4.1 on OpenAI, but OPENAI_API_KEY is not set." >&2
        echo "       Export OPENAI_API_KEY, pass a custom --judge-server-address, or use" >&2
        echo "       --skip-judge-api-key-check when credentials are injected in the container." >&2
        exit 1
    fi
    _require_container_image
    if [[ -z "$ACCOUNT_ARG" ]]; then
        echo "" >&2
        echo "ERROR: SLURM account required for real submissions." >&2
        echo "Set --account <your-slurm-account> (or export ACCOUNT=...)." >&2
        echo "" >&2
        exit 1
    fi
    mkdir -p "$OUT_DIR" "$HF_HOME"
fi

# Experiment name (auto-derived if not set)
auto_exp_name() {
    local m_safe="${MODEL//\//__}"
    local base="${m_safe}_${MODE}"
    base+="_tok${TOKENS}_blk${BLOCK_LENGTH}_temp${TEMPERATURE}"
    if [[ "$MODE" == "linear_spec" && "$USE_LORA" == "true" ]]; then
        base+="_lora_$(basename "$LORA_PATH_FINAL")"
        if [[ "$DRAFT_LORA_ONLY_ARG" == "true" || "$DRAFT_LORA_ONLY_ARG" == "1" || "$DRAFT_LORA_ONLY_ARG" == "yes" ]]; then
            base+="_draft_only"
        fi
    fi
    if [[ -n "$THRESHOLD" ]]; then
        base+="_thr${THRESHOLD}"
    fi
    echo "$base"
}
if [[ -z "$EXP_NAME" ]]; then
    EXP_NAME="$(auto_exp_name)"
fi

# ─── Print summary ─────────────────────────────────────────────────────────
echo "================================================================"
echo "  Unified eval — mode: $MODE"
echo "================================================================"
echo "  Model:            $MODEL"
echo "  Tokenizer:        ${TOKENIZER:-(model default)}"
echo "  Exp name:         $EXP_NAME"
echo "  Engine:           ${DEF_ENGINE:-(auto)}"
echo "  Gen algorithm:    $DEF_GEN_ALGO"
echo "  Max model len:    $MAX_MODEL_LEN"
echo "  Max pos emb:      ${MAX_POS_EMB:-(unset)}"
echo "  Block length:     $BLOCK_LENGTH"
echo "  Threshold:        $THRESHOLD"
echo "  Temperature:      $TEMPERATURE"
echo "  Tokens to gen:    $TOKENS"
echo "  Diff. steps:      $STEPS"
echo "  Max thinking:     $MAX_THINKING"
echo "  Batch size:       $BATCH_SIZE"
echo "  Enable thinking:  $ENABLE_THINKING"
echo "  EOS early stop:   $EOS_EARLY_STOP"
echo "  Linear spec:      $([[ "$MODE" == linear_spec ]] && echo true || echo "(off)")"
if [[ "$MODE" == "linear_spec" ]]; then
    echo "  LoRA:             $USE_LORA"
    echo "  LoRA path:        ${LORA_PATH_FINAL:-(none)}"
    echo "  Draft LoRA only:  $DRAFT_LORA_ONLY_ARG"
fi
echo "  GPUs / job:       $GPUS"
echo "  Partition:        $PARTITION"
echo "  Account:          $ACCOUNT_ARG"
echo "  Time:             $TIME_ARG"
echo "  Container image:  ${CONTAINER_IMAGE:-(unset — set CONTAINER_IMAGE or use evaluate.py)}"
echo "  Output base:      $OUT_DIR"
echo "  Benchmarks:       ${BENCHMARK_GROUPS[*]}"
if arena_hard_requested; then
    echo "  Arena judge:      ${JUDGE_MODEL:-gpt-4.1 (dataset default)}"
    echo "  Judge endpoint:   ${JUDGE_SERVER_ADDRESS:-https://api.openai.com/v1 (dataset default)}"
    echo "  Judge type:       ${JUDGE_SERVER_TYPE:-openai (dataset default)}"
fi
if mt_bench_requested; then
    echo "  MT-Bench judge:   ${JUDGE_MODEL:-gpt-4.1}"
    echo "  MT max tokens:    $MT_BENCH_MAX_TOKENS per turn"
    echo "  Judge concurrency: $JUDGE_CONCURRENCY"
fi
echo "================================================================"
echo ""

if [[ "$DRY_RUN" == true ]]; then
    echo "[dry-run] Settings printed above. No jobs submitted."
    exit 0
fi

# ─── Common env vars (read by the GPU-only DLM pipeline preset) ────────────
export ACCOUNT="$ACCOUNT_ARG"
export SERVER_PARTITION="$PARTITION"
export SERVER_TIME="$TIME_ARG"
export SERVER_GPUS="$GPUS"
export SERVER_BATCH_SIZE="$BATCH_SIZE"

# HF-direct: SERVER_BASE_MODEL drives chat-template lookup; SERVER_MODEL_PATH
# is forwarded as `--model-path` to the worker (no DCP overlay). Setting both
# to the same HF id mirrors the HF-direct codepath.
export SERVER_BASE_MODEL="$MODEL"
export SERVER_MODEL_PATH="$MODEL"
export SERVER_DCP_PATH=""
export SERVER_TOKENIZER="$TOKENIZER"

if [[ -n "$DEF_ENGINE" ]]; then
    export SERVER_ENGINE="$DEF_ENGINE"
else
    unset SERVER_ENGINE 2>/dev/null || true
fi
if [[ -n "$MAX_POS_EMB" ]]; then
    export SERVER_MAX_POSITION_EMBEDDINGS="$MAX_POS_EMB"
else
    unset SERVER_MAX_POSITION_EMBEDDINGS 2>/dev/null || true
fi
export SERVER_MAX_MODEL_LEN="$MAX_MODEL_LEN"
export SERVER_ENABLE_THINKING="$ENABLE_THINKING"
export SERVER_EOS_EARLY_STOP="$EOS_EARLY_STOP"

# Linear-spec flags (consumed by the worker and the eval client)
if [[ "$MODE" == "linear_spec" ]]; then
    export LINEAR_SPECULATION="true"
else
    unset LINEAR_SPECULATION 2>/dev/null || true
fi
if [[ -n "$LORA_PATH_FINAL" ]]; then
    export LORA_PATH="$LORA_PATH_FINAL"
    export SERVER_LORA_PATH="$LORA_PATH_FINAL"
    export DRAFT_LORA_ONLY="$DRAFT_LORA_ONLY_ARG"
else
    unset LORA_PATH SERVER_LORA_PATH DRAFT_LORA_ONLY 2>/dev/null || true
fi

export SEQ_EVAL_GENERATION_ALGORITHM="$DEF_GEN_ALGO"
export SEQ_EVAL_TOKENS_TO_GENERATE="$TOKENS"
export SEQ_EVAL_STEPS="$STEPS"
export SEQ_EVAL_BLOCK_LENGTH="$BLOCK_LENGTH"
export SEQ_EVAL_TEMPERATURE="$TEMPERATURE"
export SEQ_EVAL_MAX_THINKING_TOKENS="$MAX_THINKING"
export SEQ_EVAL_JUDGE_MODEL="$JUDGE_MODEL"
export SEQ_EVAL_JUDGE_SERVER_ADDRESS="$JUDGE_SERVER_ADDRESS"
export SEQ_EVAL_JUDGE_SERVER_TYPE="$JUDGE_SERVER_TYPE"
export SEQ_EVAL_JUDGE_CONCURRENCY="$JUDGE_CONCURRENCY"
export SEQ_EVAL_MT_BENCH_MAX_TOKENS="$MT_BENCH_MAX_TOKENS"
export SEQ_EVAL_SKIP_JUDGE_API_KEY_CHECK="$SKIP_JUDGE_API_KEY_CHECK"
export SEQ_EVAL_MODEL="$DEF_MODEL_TAG"
export SEQ_EVAL_CLIENT_CONCURRENCY="$BATCH_SIZE"

# `--strip-thinking` is only meaningful when thinking is on (see legacy scripts).
STRIP_THINKING_ARG=""
if [[ "$ENABLE_THINKING" == "true" ]]; then
    STRIP_THINKING_ARG="--strip-thinking"
fi
export SEQ_EVAL_STRIP_THINKING="$([[ -n "$STRIP_THINKING_ARG" ]] && echo true || echo false)"
export SEQ_EVAL_EXTRA_ARGS="--model $DEF_MODEL_TAG ${STRIP_THINKING_ARG}"

# ─── Sweep loop ────────────────────────────────────────────────────────────
EVAL_DIR_NAME="eval_$(date +%Y%m%d_%H%M%S)"
LAUNCH_COUNT=0
PIPELINE_LOG_DIRS=()

submit_one_benchmark() {
    local benchmark="$1"
    local group_idx="$2"
    local exp_name="$3"

    local eval_outdir="$OUT_DIR/${exp_name}"
    local checkpoint_name="hf_base"
    local eval_job_dir="$eval_outdir/$checkpoint_name/$EVAL_DIR_NAME"
    mkdir -p "$eval_job_dir"
    PIPELINE_LOG_DIRS+=("$eval_outdir/$checkpoint_name/")

    local file_tag="_group${group_idx}"
    local server_info_file="$eval_job_dir/server_info${file_tag}.env"
    if command -v realpath &>/dev/null; then
        server_info_file="$(realpath -m "$server_info_file")"
    fi
    local eval_output_dir="$eval_job_dir/results"
    local eval_log_file="$eval_job_dir/pipeline${file_tag}.log"
    local nfe_dir="$eval_job_dir/nfe${file_tag}"
    mkdir -p "$nfe_dir"

    export EVAL_OUTPUT_DIR="$eval_output_dir"
    export SEQ_EVAL_OUTPUT_DIR="$eval_output_dir"
    export SERVER_INFO_FILE="$server_info_file"
    export NFE_LOG_DIR="$nfe_dir"
    export SEQ_EVAL_EXPNAME="$checkpoint_name"
    export SEQ_EVAL_BENCHMARK="$benchmark"
    export PIPELINE_LOG_FILE="$eval_log_file"
    export EVAL_JOB_DIR="$eval_job_dir"
    export PIPELINE_FILE_TAG="$file_tag"

    echo "==> Launching: mode=$MODE  exp=$exp_name  bench=$benchmark"
    if "$PIPELINE_PRESET_OVERRIDE" 2>&1 | tee -a "$eval_log_file"; then
        echo "[ok] submitted: $exp_name :: $benchmark"
    else
        echo "[fail] submission failed: $exp_name :: $benchmark" >&2
    fi
    LAUNCH_COUNT=$((LAUNCH_COUNT + 1))
}

if [[ -n "$THRESHOLD" ]]; then
    export SEQ_EVAL_THRESHOLD="$THRESHOLD"
else
    unset SEQ_EVAL_THRESHOLD 2>/dev/null || true
fi
for g in "${!BENCHMARK_GROUPS[@]}"; do
    submit_one_benchmark "${BENCHMARK_GROUPS[$g]}" "$g" "$EXP_NAME"
done

echo ""
echo "Submitted $LAUNCH_COUNT GPU eval job(s) for mode=$MODE."
echo "Eval dir name (per experiment): $EVAL_DIR_NAME"
echo ""
echo "Pipeline log locations:"
printf '  %s\n' "${PIPELINE_LOG_DIRS[@]}" | awk '!seen[$0]++'
