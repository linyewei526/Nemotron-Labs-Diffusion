#!/bin/bash
#
# GPU-Only DLM server + evaluation pipeline.
#
# Runs the inference server AND evaluation client in a SINGLE GPU SLURM
# allocation, eliminating the need for a separate CPU-node job.  This solves
# the scheduling problem where GPU jobs get killed for idling while waiting
# for CPU-node eval jobs to be allocated.
#
# Workflow inside the single srun:
#   1. Install dependencies (server + eval)
#   2. Convert DCP checkpoint (if needed)
#   3. Start multi-GPU workers + load balancer in the background
#   4. Wait for server to become healthy
#   5. Apply eval patches, prepare benchmark data
#   6. Run eval_dlm.py against localhost
#   7. Clean up server processes
#
# Environment variables: see eval.sh for the canonical caller, which exports
# SERVER_*, SEQ_EVAL_*, CONTAINER_IMAGE, ACCOUNT, etc. before invoking this
# script. The ${VAR:-default} fallbacks below are defensive in case someone
# runs the pipeline directly without eval.sh in front.
#
# Usage (from repo root):
#   bash xp/examples/run_dlm_eval_pipeline_gpu_only.sh
# ---------------------------------------------------------------------------

set -euo pipefail

# === SLURM configuration ====================================================
ACCOUNT="${ACCOUNT:-}"
if [[ -z "$ACCOUNT" ]]; then
    echo "" >&2
    echo "ERROR: ACCOUNT must be set before invoking this pipeline." >&2
    echo "  export ACCOUNT=<your-slurm-account>   # or pass via eval.sh --account" >&2
    echo "" >&2
    exit 1
fi
SERVER_PARTITION="${SERVER_PARTITION:-batch,backfill}"
SERVER_TIME="${SERVER_TIME:-04:00:00}"
SERVER_GPUS="${SERVER_GPUS:-8}"

# === Container ===============================================================
# CONTAINER_IMAGE must already be set by the caller (eval.sh picks one based
# on what's available on the host). We don't bake in an internal fallback
# path here — public users would only get confusing srun errors from a
# missing .sqsh. eval.sh's _pick_container_image handles the failure mode.
if [[ -z "${CONTAINER_IMAGE:-}" ]]; then
    echo "ERROR: CONTAINER_IMAGE must be set before invoking this pipeline." >&2
    echo "       eval.sh normally picks one automatically; set CONTAINER_IMAGE=<path-to-.sqsh>" >&2
    echo "       (or use evaluate.py instead — no container needed)." >&2
    exit 1
fi

# === Server / model configuration ============================================
SERVER_INFO_FILE="${SERVER_INFO_FILE:-}"
SERVER_BATCH_SIZE="${SERVER_BATCH_SIZE:-1}"
SERVER_MODEL_PATH="${SERVER_MODEL_PATH:-}"
SERVER_BASE_MODEL="${SERVER_BASE_MODEL:-nvidia/Nemotron-Labs-Diffusion-8B}"
SERVER_TOKENIZER="${SERVER_TOKENIZER:-}"
SERVER_DCP_PATH="${SERVER_DCP_PATH:-}"
SERVER_ENGINE="${SERVER_ENGINE:-nemotron}"
SERVER_ALGORITHM="${SERVER_ALGORITHM:-}"
SERVER_LORA_PATH="${SERVER_LORA_PATH:-${LORA_PATH:-}}"
SERVER_SHIFT_LOGITS="${SERVER_SHIFT_LOGITS:-}"
SERVER_ENABLE_THINKING="${SERVER_ENABLE_THINKING:-}"
SERVER_EOS_EARLY_STOP="${SERVER_EOS_EARLY_STOP:-}"
SERVER_MAX_MODEL_LEN="${SERVER_MAX_MODEL_LEN:-}"
SERVER_MAX_POSITION_EMBEDDINGS="${SERVER_MAX_POSITION_EMBEDDINGS:-}"
TRANSFORMER_MODEL_PATH="${TRANSFORMER_MODEL_PATH:-}"

# === Evaluation configuration ================================================
SEQ_EVAL_BENCHMARK="${SEQ_EVAL_BENCHMARK:-gsm8k:1}"
SEQ_EVAL_EXPNAME="${SEQ_EVAL_EXPNAME:-}"
SEQ_EVAL_OUTPUT_DIR="${SEQ_EVAL_OUTPUT_DIR:-}"
SEQ_EVAL_GENERATION_ALGORITHM="${SEQ_EVAL_GENERATION_ALGORITHM:-nemotron}"
SEQ_EVAL_THRESHOLD="${SEQ_EVAL_THRESHOLD:-0.9}"
SEQ_EVAL_TOKENS_TO_GENERATE="${SEQ_EVAL_TOKENS_TO_GENERATE:-1024}"
SEQ_EVAL_STEPS="${SEQ_EVAL_STEPS:-1024}"
SEQ_EVAL_BLOCK_LENGTH="${SEQ_EVAL_BLOCK_LENGTH:-32}"
SEQ_EVAL_TEMPERATURE="${SEQ_EVAL_TEMPERATURE:-0}"
SEQ_EVAL_AR_WEIGHT="${SEQ_EVAL_AR_WEIGHT:-}"
SEQ_EVAL_CONF_TEMP="${SEQ_EVAL_CONF_TEMP:-}"
SEQ_EVAL_MAX_THINKING_TOKENS="${SEQ_EVAL_MAX_THINKING_TOKENS:-}"
LINEAR_SPECULATION="${LINEAR_SPECULATION:-}"
# Linear self-speculation is a boolean toggle. Compute the flag once; both the
# worker and the eval client need to receive it.
case "${LINEAR_SPECULATION,,}" in
    ""|false|0|no) _LINEAR_SPEC_FLAG="" ;;
    *)               _LINEAR_SPEC_FLAG="--linear-speculation" ;;
esac
DRAFT_LORA_ONLY="${DRAFT_LORA_ONLY:-}"
SAMPLER="${SAMPLER:-}"
SEQ_EVAL_EXTRA_ARGS="${SEQ_EVAL_EXTRA_ARGS:-}"
GLOBAL_EVAL_FLAGS="${GLOBAL_EVAL_FLAGS:-}"

# === Save text inputs (instead of / in addition to eval) =====================
SAVE_TEXT_INPUTS_DIR="${SAVE_TEXT_INPUTS_DIR:-}"
SAVE_INPUTS_ONLY="${SAVE_INPUTS_ONLY:-}"
# If SAVE_INPUTS_ONLY is set but SAVE_TEXT_INPUTS_DIR is not, default to a
# subdirectory under the eval output dir so each experiment/step is isolated.
if [[ "${SAVE_INPUTS_ONLY,,}" == "true" || "$SAVE_INPUTS_ONLY" == "1" ]] && [[ -z "$SAVE_TEXT_INPUTS_DIR" ]]; then
    SAVE_TEXT_INPUTS_DIR="${SEQ_EVAL_OUTPUT_DIR:-${EVAL_OUTPUT_DIR:-/tmp}}/text_inputs"
fi

# === Multi-GPU server ports ==================================================
# Randomize defaults so multiple eval jobs sharing one node do not collide.
# Override via LOAD_BALANCER_PORT / BASE_WORKER_PORT.
_PORT_OFFSET=$(( (RANDOM % 200) * 20 ))
LOAD_BALANCER_PORT="${LOAD_BALANCER_PORT:-$((8000 + _PORT_OFFSET))}"
BASE_WORKER_PORT="${BASE_WORKER_PORT:-$((LOAD_BALANCER_PORT + 1))}"

# ===========================================================================

export ACCOUNT

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_DIR="$(cd "$ROOT_DIR/.." && pwd)"

WORKER_SCRIPT="$ROOT_DIR/dlm_api/dlm_batch_server.py"
LB_SCRIPT="$ROOT_DIR/dlm_api/dlm_load_balancer.py"
EVAL_SCRIPT="$ROOT_DIR/nemo-skills/eval_dlm.py"
PATCH_SCRIPT="$ROOT_DIR/nemo-skills/patch_openai_extra_body.py"
DICTCONFIG_PATCH="$ROOT_DIR/nemo-skills/patch_dictconfig_serialization.py"

# --- Resolve output / NFE dirs -----------------------------------------------
EVAL_OUTPUT_DIR="${SEQ_EVAL_OUTPUT_DIR:-$PROJECT_DIR/eval-output}"
mkdir -p "$EVAL_OUTPUT_DIR"
EVAL_OUTPUT_DIR="$(realpath "$EVAL_OUTPUT_DIR")"

NFE_LOG_DIR="${NFE_LOG_DIR:-}"

# --- Job completion markers (set by sweep script for sbatch jobs) -------------
EVAL_JOB_DIR="${EVAL_JOB_DIR:-}"
PIPELINE_FILE_TAG="${PIPELINE_FILE_TAG:-}"
NFE_LOG_DIR_ABS=""
if [[ -n "$NFE_LOG_DIR" ]]; then
    mkdir -p "$NFE_LOG_DIR"
    NFE_LOG_DIR_ABS="$(realpath "$NFE_LOG_DIR")"
fi

# --- Server info file (for metadata / monitoring) ----------------------------
if [[ -z "$SERVER_INFO_FILE" ]]; then
    SERVER_INFO_FILE="$EVAL_OUTPUT_DIR/server_info_gpu_only.env"
fi
SERVER_INFO_FILE="$(realpath -m "$SERVER_INFO_FILE")"
mkdir -p "$(dirname "$SERVER_INFO_FILE")"

# --- Build worker args -------------------------------------------------------
WORKER_BASE_ARGS="--host localhost --batch-size $SERVER_BATCH_SIZE --max-wait-time 0.01 --timeout-keep-alive 9000"

if [[ -n "$SERVER_ENGINE" ]]; then
    WORKER_BASE_ARGS="$WORKER_BASE_ARGS --engine $SERVER_ENGINE"
fi
if [[ -n "$SERVER_ALGORITHM" ]]; then
    WORKER_BASE_ARGS="$WORKER_BASE_ARGS --algorithm $SERVER_ALGORITHM"
fi
if [[ -n "${SERVER_REVISION:-}" ]]; then
    WORKER_BASE_ARGS="$WORKER_BASE_ARGS --revision $SERVER_REVISION"
fi
if [[ "$SERVER_ENABLE_THINKING" == "true" || "$SERVER_ENABLE_THINKING" == "1" ]]; then
    WORKER_BASE_ARGS="$WORKER_BASE_ARGS --enable-thinking"
fi
if [[ "$SERVER_SHIFT_LOGITS" == "true" || "$SERVER_SHIFT_LOGITS" == "1" ]]; then
    WORKER_BASE_ARGS="$WORKER_BASE_ARGS --shift-logits"
fi
if [[ "${SERVER_EOS_EARLY_STOP,,}" == "true" || "$SERVER_EOS_EARLY_STOP" == "1" ]]; then
    WORKER_BASE_ARGS="$WORKER_BASE_ARGS --eos-early-stop"
fi
if [[ -n "$NFE_LOG_DIR_ABS" ]]; then
    WORKER_BASE_ARGS="$WORKER_BASE_ARGS --nfe-log-dir $NFE_LOG_DIR_ABS"
fi
if [[ -n "$SERVER_MAX_POSITION_EMBEDDINGS" ]]; then
    WORKER_BASE_ARGS="$WORKER_BASE_ARGS --max-position-embeddings $SERVER_MAX_POSITION_EMBEDDINGS"
fi
if [[ -n "$SEQ_EVAL_MAX_THINKING_TOKENS" ]]; then
    WORKER_BASE_ARGS="$WORKER_BASE_ARGS --max-thinking-tokens $SEQ_EVAL_MAX_THINKING_TOKENS"
fi
WORKER_BASE_ARGS="$WORKER_BASE_ARGS $_LINEAR_SPEC_FLAG"
if [[ "${DRAFT_LORA_ONLY,,}" == "true" || "$DRAFT_LORA_ONLY" == "1" ]]; then
    WORKER_BASE_ARGS="$WORKER_BASE_ARGS --draft-lora-only"
fi
if [[ -n "$SAMPLER" ]]; then
    WORKER_BASE_ARGS="$WORKER_BASE_ARGS --sampler $SAMPLER"
fi
if [[ -n "$SERVER_LORA_PATH" ]]; then
    WORKER_BASE_ARGS="$WORKER_BASE_ARGS --lora-path $SERVER_LORA_PATH"
fi
if [[ -n "$SAVE_TEXT_INPUTS_DIR" ]]; then
    WORKER_BASE_ARGS="$WORKER_BASE_ARGS --save-text-inputs-dir $SAVE_TEXT_INPUTS_DIR"
    if [[ "${SAVE_INPUTS_ONLY,,}" == "true" || "$SAVE_INPUTS_ONLY" == "1" ]]; then
        WORKER_BASE_ARGS="$WORKER_BASE_ARGS --save-inputs-only"
    fi
fi

# If a model path is provided directly (no DCP), add it to worker args now.
# For DCP, workers use the converted path (added inside the container).
if [[ -n "$SERVER_MODEL_PATH" ]] && [[ -z "$SERVER_DCP_PATH" ]]; then
    WORKER_BASE_ARGS="$WORKER_BASE_ARGS --model-path $SERVER_MODEL_PATH"
fi

# --- Build eval args ----------------------------------------------------------
# LB-retry port leakage fix: __LB_PORT__ is rewritten at runtime inside the
# container, after the LB retry chooses its final port. See LB block below.
EVAL_ARGS="--server-address http://localhost:__LB_PORT__/v1"

if [[ -n "$SEQ_EVAL_BENCHMARK" ]]; then
    EVAL_ARGS="$EVAL_ARGS --benchmark $SEQ_EVAL_BENCHMARK"
fi
if [[ -n "$SEQ_EVAL_EXPNAME" ]]; then
    EVAL_ARGS="$EVAL_ARGS --expname $SEQ_EVAL_EXPNAME"
fi
EVAL_ARGS="$EVAL_ARGS --output-dir $EVAL_OUTPUT_DIR"
if [[ -n "$SEQ_EVAL_GENERATION_ALGORITHM" ]]; then
    EVAL_ARGS="$EVAL_ARGS --generation-algorithm $SEQ_EVAL_GENERATION_ALGORITHM"
fi
if [[ -n "$SEQ_EVAL_THRESHOLD" ]]; then
    EVAL_ARGS="$EVAL_ARGS --threshold $SEQ_EVAL_THRESHOLD"
fi
if [[ -n "$SEQ_EVAL_TOKENS_TO_GENERATE" ]]; then
    EVAL_ARGS="$EVAL_ARGS --tokens-to-generate $SEQ_EVAL_TOKENS_TO_GENERATE"
fi
if [[ -n "$SEQ_EVAL_STEPS" ]]; then
    EVAL_ARGS="$EVAL_ARGS --steps $SEQ_EVAL_STEPS"
fi
if [[ -n "$SEQ_EVAL_BLOCK_LENGTH" ]]; then
    EVAL_ARGS="$EVAL_ARGS --block-length $SEQ_EVAL_BLOCK_LENGTH"
fi
if [[ -n "$SEQ_EVAL_TEMPERATURE" ]]; then
    EVAL_ARGS="$EVAL_ARGS --temperature $SEQ_EVAL_TEMPERATURE"
fi
if [[ -n "$SEQ_EVAL_AR_WEIGHT" ]]; then
    EVAL_ARGS="$EVAL_ARGS --ar-weight $SEQ_EVAL_AR_WEIGHT"
fi
if [[ -n "$SEQ_EVAL_CONF_TEMP" ]]; then
    EVAL_ARGS="$EVAL_ARGS --conf-temp $SEQ_EVAL_CONF_TEMP"
fi
if [[ -n "$SEQ_EVAL_MAX_THINKING_TOKENS" ]]; then
    EVAL_ARGS="$EVAL_ARGS --max-thinking-tokens $SEQ_EVAL_MAX_THINKING_TOKENS"
fi
EVAL_ARGS="$EVAL_ARGS $_LINEAR_SPEC_FLAG"
if [[ -n "$DRAFT_LORA_ONLY" ]]; then
    EVAL_ARGS="$EVAL_ARGS --draft-lora-only $DRAFT_LORA_ONLY"
fi
if [[ -n "$SAMPLER" ]]; then
    EVAL_ARGS="$EVAL_ARGS --sampler $SAMPLER"
fi
if [[ -n "$GLOBAL_EVAL_FLAGS" ]]; then
    EVAL_ARGS="$EVAL_ARGS $GLOBAL_EVAL_FLAGS"
fi
if [[ -n "$SEQ_EVAL_EXTRA_ARGS" ]]; then
    EVAL_ARGS="$EVAL_ARGS $SEQ_EVAL_EXTRA_ARGS"
fi

# --- Extract benchmark name for data preparation -----------------------------
BENCHMARK_NAME=""
if [[ -n "$SEQ_EVAL_BENCHMARK" ]]; then
    BENCHMARK_NAME="$(echo "$SEQ_EVAL_BENCHMARK" | sed 's/:[0-9]\+//g')"
fi

# --- Resolve DCP path --------------------------------------------------------
DCP_ABS_PATH=""
if [[ -n "$SERVER_DCP_PATH" ]] && [[ -d "$SERVER_DCP_PATH" ]]; then
    DCP_ABS_PATH="$(realpath "$SERVER_DCP_PATH")"
fi

# SERVER_TOKENIZER may be either a local directory or an HF model id.
# Resolve the path form for container mounts; forward the original (path or
# HF id) verbatim to the worker so HF-only configs work without a mount.
TOKENIZER_ABS_PATH=""
TOKENIZER_ARG="${SERVER_TOKENIZER:-}"
if [[ -n "$SERVER_TOKENIZER" ]] && [[ -d "$SERVER_TOKENIZER" ]]; then
    TOKENIZER_ABS_PATH="$(realpath "$SERVER_TOKENIZER")"
    TOKENIZER_ARG="$TOKENIZER_ABS_PATH"
fi

# --- Build container mounts --------------------------------------------------
CONTAINER_MOUNTS="$PROJECT_DIR:$PROJECT_DIR"

if [[ -n "$EVAL_OUTPUT_DIR" ]] && [[ "$EVAL_OUTPUT_DIR" != "$PROJECT_DIR"* ]]; then
    CONTAINER_MOUNTS="$CONTAINER_MOUNTS,$EVAL_OUTPUT_DIR:$EVAL_OUTPUT_DIR"
fi
if [[ -n "$NFE_LOG_DIR_ABS" ]] && [[ "$NFE_LOG_DIR_ABS" != "$PROJECT_DIR"* ]] && [[ "$NFE_LOG_DIR_ABS" != "$EVAL_OUTPUT_DIR"* ]]; then
    CONTAINER_MOUNTS="$CONTAINER_MOUNTS,$NFE_LOG_DIR_ABS:$NFE_LOG_DIR_ABS"
fi
SERVER_INFO_DIR_ABS="$(dirname "$SERVER_INFO_FILE")"
if [[ "$SERVER_INFO_DIR_ABS" != "$PROJECT_DIR"* ]] && [[ "$SERVER_INFO_DIR_ABS" != "$EVAL_OUTPUT_DIR"* ]]; then
    CONTAINER_MOUNTS="$CONTAINER_MOUNTS,$SERVER_INFO_DIR_ABS:$SERVER_INFO_DIR_ABS"
fi
if [[ -n "$DCP_ABS_PATH" ]] && [[ "$DCP_ABS_PATH" != "$PROJECT_DIR"* ]]; then
    CONTAINER_MOUNTS="$CONTAINER_MOUNTS,$DCP_ABS_PATH:$DCP_ABS_PATH"
fi
if [[ -n "$SERVER_MODEL_PATH" ]] && [[ -d "$SERVER_MODEL_PATH" ]]; then
    MODEL_ABS_PATH="$(realpath "$SERVER_MODEL_PATH")"
    if [[ "$MODEL_ABS_PATH" != "$PROJECT_DIR"* ]]; then
        CONTAINER_MOUNTS="$CONTAINER_MOUNTS,$MODEL_ABS_PATH:$MODEL_ABS_PATH"
    fi
fi
if [[ -n "$TOKENIZER_ABS_PATH" ]] && [[ "$TOKENIZER_ABS_PATH" != "$PROJECT_DIR"* ]]; then
    CONTAINER_MOUNTS="$CONTAINER_MOUNTS,$TOKENIZER_ABS_PATH:$TOKENIZER_ABS_PATH"
fi
# Mount LoRA adapter dir into the container so PEFT can read adapter_config.json
# at the same path the worker was told to load from.
if [[ -n "$SERVER_LORA_PATH" ]] && [[ -d "$SERVER_LORA_PATH" ]]; then
    LORA_ABS_PATH="$(realpath "$SERVER_LORA_PATH")"
    if [[ "$LORA_ABS_PATH" != "$PROJECT_DIR"* ]]; then
        CONTAINER_MOUNTS="$CONTAINER_MOUNTS,$LORA_ABS_PATH:$LORA_ABS_PATH"
    fi
fi
if [[ -n "$SAVE_TEXT_INPUTS_DIR" ]] && [[ "$SAVE_TEXT_INPUTS_DIR" != "$PROJECT_DIR"* ]] && [[ "$SAVE_TEXT_INPUTS_DIR" != "$EVAL_OUTPUT_DIR"* ]]; then
    CONTAINER_MOUNTS="$CONTAINER_MOUNTS,$SAVE_TEXT_INPUTS_DIR:$SAVE_TEXT_INPUTS_DIR"
fi
if [[ -n "${EXTRA_CONTAINER_MOUNTS:-}" ]]; then
    CONTAINER_MOUNTS="$CONTAINER_MOUNTS,$EXTRA_CONTAINER_MOUNTS"
fi

# --- Build worker ports list --------------------------------------------------
WORKER_PORTS=""
for i in $(seq 0 $((SERVER_GPUS - 1))); do
    if [[ -n "$WORKER_PORTS" ]]; then
        WORKER_PORTS="$WORKER_PORTS $((BASE_WORKER_PORT + i))"
    else
        WORKER_PORTS="$((BASE_WORKER_PORT + i))"
    fi
done

# --- Print summary ------------------------------------------------------------
echo "=============================================================="
echo "[gpu-only] GPU-Only Evaluation Pipeline"
echo "=============================================================="
echo "  Account:        $ACCOUNT"
echo "  Partition:       $SERVER_PARTITION"
echo "  Time limit:      $SERVER_TIME"
echo "  GPUs:            $SERVER_GPUS"
echo "  Container:       $CONTAINER_IMAGE"
echo "  Base model:      $SERVER_BASE_MODEL"
echo "  Revision:        ${SERVER_REVISION:-(default)}"
echo "  DCP path:        ${SERVER_DCP_PATH:-(none)}"
echo "  Tokenizer:       ${SERVER_TOKENIZER:-(default)}"
echo "  LoRA path:       ${SERVER_LORA_PATH:-(none)}"
echo "  Engine:          ${SERVER_ENGINE:-(auto)}"
echo "  Shift logits:    ${SERVER_SHIFT_LOGITS:-(default: false)}"
echo "  DLM paradigm:    ${NEMOTRON_DLM_PARADIGM:-(default: bidirectional)}"
echo "  Max pos emb:     ${SERVER_MAX_POSITION_EMBEDDINGS:-(default: from HF config)}"
echo "  Transformer ckpt: ${TRANSFORMER_MODEL_PATH:-(none)}"
echo "  Block length:    ${SEQ_EVAL_BLOCK_LENGTH:-(default)}"
echo "  Draft LoRA only: ${DRAFT_LORA_ONLY:-(unset)}"
echo "  Benchmark:       $SEQ_EVAL_BENCHMARK"
echo "  Output dir:      $EVAL_OUTPUT_DIR"
echo "  Server info:     $SERVER_INFO_FILE"
echo "  NFE log dir:     ${NFE_LOG_DIR_ABS:-(none)}"
echo "  Worker ports:    $WORKER_PORTS"
echo "  Eval args:       $EVAL_ARGS"
if [[ -n "$SAVE_TEXT_INPUTS_DIR" ]]; then
echo "  Save text dir:   $SAVE_TEXT_INPUTS_DIR"
echo "  Save only:       ${SAVE_INPUTS_ONLY:-false}"
fi
echo "=============================================================="
echo ""

# === Build the command block that runs inside the container ===================
# Variables from the pipeline (known at launch time) are expanded now.
# Container-side variables (hostname, PIDs) use \$ to defer expansion.

COMMAND_BLOCK=$(cat <<CMDEOF
set -euo pipefail

unset UV_CACHE_DIR

export PATH="/root/.local/bin:\$PATH"
export PYTHONPATH="$PROJECT_DIR:\${PYTHONPATH:-}"
export HF_TOKEN="${HF_TOKEN:-}"
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-}"
if [[ -n "$TRANSFORMER_MODEL_PATH" ]]; then export TRANSFORMER_MODEL_PATH="$TRANSFORMER_MODEL_PATH"; fi
if [[ -n "$NFE_LOG_DIR_ABS" ]]; then export NFE_LOG_DIR="$NFE_LOG_DIR_ABS"; fi

_MARKER_DIR="$EVAL_JOB_DIR"
_MARKER_TAG="$PIPELINE_FILE_TAG"

VENV_DIR="/opt/nemo_rl_venv"
PYTHON_BIN="\$VENV_DIR/bin/python"

echo "===================================================================="
echo "[gpu-only] GPU-Only Pipeline starting on node: \$(hostname)"
echo "[gpu-only] GPUs: $SERVER_GPUS"
echo "===================================================================="

# --- Phase 1: Environment setup ---
echo "[1/7] Activating container environment..."
if [ -f "\$VENV_DIR/bin/activate" ]; then
    source "\$VENV_DIR/bin/activate"
fi

PREBAKED_MARKER="\$VENV_DIR/.prebaked_marker"
if [ -f "\$PREBAKED_MARKER" ] && \$PYTHON_BIN -c "import vllm, typer, livecodebench, langdetect, immutabledict, nltk" 2>/dev/null; then
    echo "[2/7] Pre-baked container detected — skipping dependency installation."
    echo "  Marker: \$(cat \$PREBAKED_MARKER)"
else
    echo "[2/7] Installing dependencies..."
    uv sync --locked --no-install-project --extra vllm --extra eval || echo "Warning: uv sync failed, continuing..."

    echo "  Installing extra runtime deps (typer, livecodebench)..."
    uv pip install --reinstall typer 2>&1 || echo "Warning: typer install failed"
    if ! \$PYTHON_BIN -c "import livecodebench" 2>/dev/null; then
        echo "  Installing livecodebench..."
        uv pip install --no-deps git+https://github.com/wasiahmad/livecodebench.git@livecodebench 2>&1 || true
    fi
    if ! \$PYTHON_BIN -c "import langdetect, immutabledict, nltk" 2>/dev/null; then
        echo "  Installing IFEval runtime deps (langdetect, immutabledict, nltk)..."
        uv pip install langdetect immutabledict nltk 2>&1 || echo "Warning: IFEval dependency install failed"
    fi
fi

echo "  Applying NeMo-Skills patches..."
\$PYTHON_BIN "$PATCH_SCRIPT" 2>/dev/null || true
\$PYTHON_BIN "$DICTCONFIG_PATCH" 2>/dev/null || true

# --- Phase 2: DCP conversion (if needed) ---
CONVERTED_MODEL_PATH=""
DCP_ABS_PATH_VAR="$DCP_ABS_PATH"
if [[ -n "\$DCP_ABS_PATH_VAR" ]]; then
    echo "[3/7] Converting DCP checkpoint..."
    SHARED_TEMP_DIR="/tmp/model_hf_converted_gpu_only"
    mkdir -p "\$SHARED_TEMP_DIR"
    export SHARED_TEMP_DIR
    export DCP_ABS_PATH_VAR="\$DCP_ABS_PATH_VAR"
    export BASE_MODEL_VAR="$SERVER_BASE_MODEL"

    \$PYTHON_BIN - <<'PYEOF'
import sys, os
sys.path.insert(0, os.environ.get("PYTHONPATH", "").split(":")[0])
from nemo_rl.utils.native_checkpoint import convert_dcp_to_hf, convert_structured_dcp_to_hf

dcp_path = os.environ["DCP_ABS_PATH_VAR"]
temp_dir = os.environ["SHARED_TEMP_DIR"]
base_model = os.environ["BASE_MODEL_VAR"]

weights_dir = os.path.join(dcp_path, "weights")
tokenizer_dir = os.path.join(dcp_path, "tokenizer")

if os.path.exists(weights_dir) and os.path.exists(tokenizer_dir):
    print("Detected structured DCP checkpoint")
    convert_structured_dcp_to_hf(dcp_root_path=dcp_path, hf_ckpt_path=temp_dir,
                                  model_name_or_path=base_model, overwrite=True)
else:
    print("Using legacy DCP checkpoint format")
    convert_dcp_to_hf(dcp_ckpt_path=dcp_path, hf_ckpt_path=temp_dir,
                       model_name_or_path=base_model, tokenizer_name_or_path=base_model,
                       overwrite=True)
print(f"DCP conversion complete: {temp_dir}")
PYEOF

    CONVERTED_MODEL_PATH="\$SHARED_TEMP_DIR"
    echo "  DCP converted to: \$CONVERTED_MODEL_PATH"
else
    echo "[3/7] No DCP conversion needed."
fi

# --- Phase 3: Start GPU workers ---
echo "[4/7] Starting $SERVER_GPUS GPU worker(s)..."
WORKER_PIDS=()
WORKER_LOG_DIR="$EVAL_OUTPUT_DIR/worker_logs"
mkdir -p "\$WORKER_LOG_DIR"
echo "  Worker logs → \$WORKER_LOG_DIR (persistent on Lustre)"

for i in \$(seq 0 $(($SERVER_GPUS - 1))); do
    WORKER_PORT=\$(($BASE_WORKER_PORT + \$i))
    WORKER_CMD="CUDA_VISIBLE_DEVICES=\$i \$PYTHON_BIN -u $WORKER_SCRIPT --port \$WORKER_PORT $WORKER_BASE_ARGS"

    if [[ -n "\$CONVERTED_MODEL_PATH" ]]; then
        WORKER_CMD="\$WORKER_CMD --model-path \$CONVERTED_MODEL_PATH --base-model $SERVER_BASE_MODEL"
    fi

    if [[ -n "" ]]; then
        WORKER_CMD="\$WORKER_CMD --tokenizer-path "
    fi

    echo "  Worker \$i → GPU \$i, port \$WORKER_PORT"
    eval "\$WORKER_CMD" > "\$WORKER_LOG_DIR/worker_\${i}.log" 2>&1 &
    WORKER_PIDS+=(\$!)

    if [ \$i -lt $(($SERVER_GPUS - 1)) ]; then
        sleep 5
    fi
done

# --- Phase 4: Wait for workers + start load balancer ---
echo "[5/7] Waiting for workers to load model (poll up to ~5 min)..."
# Start polling early; typical 8B HF load finishes in 60-90s on H100.
sleep 30

HEALTHY=0
for retry in \$(seq 1 60); do
    HEALTHY=0
    for i in \$(seq 0 $(($SERVER_GPUS - 1))); do
        PORT=\$(($BASE_WORKER_PORT + \$i))
        if curl -sf --max-time 3 "http://localhost:\$PORT/health" > /dev/null 2>&1; then
            HEALTHY=\$((HEALTHY + 1))
        fi
    done
    if [ \$HEALTHY -eq $SERVER_GPUS ]; then
        echo "  All $SERVER_GPUS workers healthy after \$retry checks"
        break
    fi
    if [ \$((retry % 6)) -eq 0 ]; then
        echo "  Health check \$retry/60: \$HEALTHY/$SERVER_GPUS workers healthy"
    fi
    sleep 5
done

if [ \$HEALTHY -eq 0 ]; then
    echo "ERROR: No workers are healthy. Dumping logs:"
    for i in \$(seq 0 $(($SERVER_GPUS - 1))); do
        echo "--- Worker \$i log ---"
        tail -80 "\$WORKER_LOG_DIR/worker_\${i}.log" 2>/dev/null || true
    done
    exit 1
fi

LB_PORTS=""
for i in \$(seq 0 $(($SERVER_GPUS - 1))); do
    PORT=\$(($BASE_WORKER_PORT + \$i))
    if [[ -n "\$LB_PORTS" ]]; then LB_PORTS="\$LB_PORTS \$PORT"; else LB_PORTS="\$PORT"; fi
done

# Load balancer startup with retry+backoff. Most flakes are port-collision
# races (another job on this node grabbed the port between our pre-flight check
# and the bind()). When the first attempt fails we bump the port and try
# again; up to 3 attempts before giving up.
LB_READY=false
LB_RETRIES=3
LB_PORT=$LOAD_BALANCER_PORT
for lb_attempt in \$(seq 1 \$LB_RETRIES); do
    echo "  [LB attempt \$lb_attempt/\$LB_RETRIES] Starting load balancer on port \$LB_PORT..."
    \$PYTHON_BIN "$LB_SCRIPT" --host 0.0.0.0 --port \$LB_PORT --worker-host localhost --worker-ports \$LB_PORTS --batch-size $SERVER_BATCH_SIZE --batch-wait-time 0.01 --timeout-keep-alive 9000 --request-timeout 12000 > "\$WORKER_LOG_DIR/load_balancer.log" 2>&1 &
    LB_PID=\$!

    # Health-poll up to 60 s, bail early if the LB process dies.
    for poll in \$(seq 1 30); do
        if ! kill -0 \$LB_PID 2>/dev/null; then
            echo "  [LB attempt \$lb_attempt] LB process died after \$((poll*2))s. Last log lines:"
            tail -20 "\$WORKER_LOG_DIR/load_balancer.log" 2>/dev/null || true
            break
        fi
        if curl -sf --max-time 2 "http://localhost:\$LB_PORT/health" > /dev/null 2>&1; then
            LB_READY=true
            break
        fi
        sleep 2
    done

    if [[ "\$LB_READY" == "true" ]]; then
        LOAD_BALANCER_PORT=\$LB_PORT
        echo "  [LB attempt \$lb_attempt] healthy after \$((poll*2))s on port \$LB_PORT"
        break
    fi

    # Reap zombie and roll the port. +100 keeps us well clear of the worker ports.
    kill \$LB_PID 2>/dev/null || true
    wait \$LB_PID 2>/dev/null || true
    LB_PORT=\$((LB_PORT + 100))
    sleep 2
done

if [[ "\$LB_READY" != "true" ]]; then
    echo "ERROR: Load balancer did not become healthy after \$LB_RETRIES attempts."
    echo "──────── load_balancer.log (last 80 lines) ────────"
    tail -80 "\$WORKER_LOG_DIR/load_balancer.log" 2>/dev/null || true
    echo "──────── worker logs (tail 20 each) ────────"
    for i in \$(seq 0 $(($SERVER_GPUS - 1))); do
        echo "── worker_\$i.log ──"
        tail -20 "\$WORKER_LOG_DIR/worker_\$i.log" 2>/dev/null || true
    done
    exit 1
fi

echo "  Server ready at http://localhost:\$LOAD_BALANCER_PORT"

# Write server info for monitoring/metadata
COMPUTE_NODE=\$(hostname)
cat > "$SERVER_INFO_FILE" <<SINFO
SERVER_INFO_GENERATED_AT="\$(date -Iseconds)"
SERVER_MODE="gpu_only"
SERVER_STATUS="running"
SERVER_CLIENT_HOST="\$COMPUTE_NODE"
SERVER_PORT="\$LOAD_BALANCER_PORT"
SERVER_ADDRESS="http://\$COMPUTE_NODE:\$LOAD_BALANCER_PORT/v1"
MULTI_GPU="true"
NUM_GPUS="$SERVER_GPUS"
SLURM_JOB_ID="\${SLURM_JOB_ID:-}"
SINFO

# --- Phase 5: Prepare benchmark data ---
echo "[6/7] Preparing benchmark data and running evaluation..."
BENCHMARKS_TO_PREPARE="$BENCHMARK_NAME"
if [[ -n "\$BENCHMARKS_TO_PREPARE" ]]; then
    NS_BIN="\$VENV_DIR/bin/ns"
    if [ -f "\$NS_BIN" ]; then
        IFS=',' read -ra BENCHMARK_ARRAY <<< "\$BENCHMARKS_TO_PREPARE"
        for BENCHMARK in "\${BENCHMARK_ARRAY[@]}"; do
            echo "  Preparing data for \$BENCHMARK..."
            \$NS_BIN prepare_data "\$BENCHMARK" || true
        done
    fi
fi

# --- Phase 6: Run evaluation ---
echo ""
# Resolve the __LB_PORT__ sentinel using the LB retry's final port.
EVAL_ARGS_RESOLVED="${EVAL_ARGS//__LB_PORT__/\$LOAD_BALANCER_PORT}"
echo "  Running: \$PYTHON_BIN $EVAL_SCRIPT \$EVAL_ARGS_RESOLVED"
echo ""
EVAL_EXIT_CODE=0
\$PYTHON_BIN "$EVAL_SCRIPT" \$EVAL_ARGS_RESOLVED || EVAL_EXIT_CODE=\$?

# --- Dump worker debug info to pipeline log before cleanup ---
echo ""
echo "============================================================"
echo "[DEBUG] Worker log excerpts (THINKING DEBUG + transformers version)"
echo "============================================================"
for wlog in "\$WORKER_LOG_DIR"/worker_*.log; do
    [ -f "\$wlog" ] || continue
    echo "--- \$(basename \$wlog) ---"
    grep -E 'THINKING DEBUG|transformers version|enable_thinking' "\$wlog" 2>/dev/null || echo "  (no debug lines found)"
done
echo "============================================================"
echo ""

# --- Phase 7: Cleanup ---
echo "[7/7] Cleaning up server processes..."
kill \$LB_PID 2>/dev/null || true
for PID in "\${WORKER_PIDS[@]}"; do
    kill \$PID 2>/dev/null || true
done

if [[ -n "\$CONVERTED_MODEL_PATH" ]] && [[ -d "\$CONVERTED_MODEL_PATH" ]]; then
    rm -rf "\$CONVERTED_MODEL_PATH"
fi

if [[ -n "\$_MARKER_DIR" ]]; then
    if [ \$EVAL_EXIT_CODE -ne 0 ]; then
        printf "Exit code: %s\n%s\n" "\$EVAL_EXIT_CODE" "\$(date -Iseconds)" > "\${_MARKER_DIR}/FAILED\${_MARKER_TAG}"
    else
        echo "\$(date -Iseconds)" > "\${_MARKER_DIR}/COMPLETED\${_MARKER_TAG}"
    fi
fi

if [ \$EVAL_EXIT_CODE -ne 0 ]; then
    echo "ERROR: Evaluation exited with code \$EVAL_EXIT_CODE"
    exit \$EVAL_EXIT_CODE
fi

echo "[gpu-only] Pipeline completed successfully."
CMDEOF
)

# === Submit via sbatch (non-blocking) =========================================
echo "[gpu-only] Submitting GPU-only eval job via sbatch..."
echo ""

# Write command block to shared Lustre storage (accessible from compute nodes)
CMD_SCRIPT="$EVAL_OUTPUT_DIR/.gpu_only_cmd_$$.sh"
cat > "$CMD_SCRIPT" <<CMDEOF
#!/bin/bash
$COMMAND_BLOCK
CMDEOF
chmod +x "$CMD_SCRIPT"

# Batch script: srun with pyxis container flags wrapping the command script
# (pyxis SPANK options only work with srun, not as sbatch CLI args)
BATCH_SCRIPT="$(mktemp /tmp/gpu_only_eval_XXXXXX.sh)"
cat > "$BATCH_SCRIPT" <<BATCHEOF
#!/bin/bash
srun \\
    --container-image="$CONTAINER_IMAGE" \\
    --container-workdir="$PROJECT_DIR" \\
    --container-mounts="$CONTAINER_MOUNTS" \\
    --no-container-mount-home \\
    bash "$CMD_SCRIPT"
BATCHEOF
chmod +x "$BATCH_SCRIPT"

SBATCH_FLAGS=(
    --job-name="gpu-only-eval"
    --time="$SERVER_TIME"
    --gpus-per-node="$SERVER_GPUS"
    --cpus-per-task=48
    --mem=256G
    --partition="$SERVER_PARTITION"
    --account="$ACCOUNT"
    --comment='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"240","reason":"benchmarking","description":"DiffusionLLM GPU-only eval pipeline: server+eval on same node"}}'
)

PIPELINE_LOG_FILE="${PIPELINE_LOG_FILE:-}"
if [[ -n "$PIPELINE_LOG_FILE" ]]; then
    mkdir -p "$(dirname "$PIPELINE_LOG_FILE")"
    SBATCH_FLAGS+=(--output="$PIPELINE_LOG_FILE" --open-mode=append)
fi

set +e
SBATCH_OUTPUT=$(sbatch "${SBATCH_FLAGS[@]}" "$BATCH_SCRIPT" 2>&1)
SBATCH_EXIT=$?
set -e

if [[ $SBATCH_EXIT -ne 0 ]]; then
    echo "ERROR: sbatch submission failed (exit $SBATCH_EXIT): $SBATCH_OUTPUT"
    rm -f "$BATCH_SCRIPT" "$CMD_SCRIPT"
    exit 1
fi

echo "$SBATCH_OUTPUT"
echo "[gpu-only] Batch script: $BATCH_SCRIPT"
echo "[gpu-only] Command script: $CMD_SCRIPT"
echo "[gpu-only] sbatch submitted successfully."
