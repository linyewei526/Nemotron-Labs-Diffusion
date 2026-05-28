#!/usr/bin/env bash
# Launch SGLang server on a DGX Spark for nvidia/Nemotron-Labs-Diffusion-8B
# with the LoRA-enhanced LinearSpec drafter (HF model card recipe).
#
# Fork branch: hutm/sglang @ upstream/2-dllm-lora-ar
# Container:   lmsysorg/sglang:spark (prebuilt torch + sgl_kernel)
#
# Defaults assume the layout from README.md:
#   ~/sglang_dllm/
#     src/sglang/              # cloned fork (Step 1 + Step 2 patch applied)
#     linear_spec_lora/        # adapter dir (Step 3)
#     hf_cache/                # auto-created HF weight cache
#     logs/                    # server.log
#     flashinfer_cache/        # sm_121a JIT artifacts (cached across restarts)
#
# Override any path with WORK_DIR; override any runtime knob with the env
# vars listed in README.md (ALGO, CTX_LEN, MEM_FRAC, MAX_REQS, PORT,
# LORA_MODE, QUANT, MODEL).
#
# Usage:
#   ./launch_server.sh              # foreground, Ctrl-C to stop
#   ./launch_server.sh detach       # detached, logs to logs/server.log
#   ./launch_server.sh stop         # stop the detached container
#   QUANT=fp8 ./launch_server.sh detach              # FP8 weights (recommended)
#   ALGO=LinearSpec-base ./launch_server.sh detach   # vanilla LinearSpec, no LoRA
#   ALGO=FastDiffuser    ./launch_server.sh detach   # iterative diffusion
#   ALGO=AR              ./launch_server.sh detach   # pure autoregressive
#
# If your user is in the docker group, you can drop the `sudo -n` prefix
# from the docker calls below.
set -euo pipefail

WORK_DIR=${WORK_DIR:-$HOME/sglang_dllm}
FORK_DIR=${WORK_DIR}/src/sglang
HF_CACHE=${WORK_DIR}/hf_cache
LOG_DIR=${WORK_DIR}/logs
LORA_HOST_DIR=${WORK_DIR}/linear_spec_lora
LORA_YAML_HOST=${WORK_DIR}/linearspec_lora.yaml
FLASHINFER_CACHE=${WORK_DIR}/flashinfer_cache

PORT=${PORT:-30000}
ALGO=${ALGO:-LinearSpec}
MODEL=${MODEL:-nvidia/Nemotron-Labs-Diffusion-8B}
CTX_LEN=${CTX_LEN:-2048}
MEM_FRAC=${MEM_FRAC:-0.5}
MAX_REQS=${MAX_REQS:-2}
LORA_MODE=${LORA_MODE:-draft_only}
MODEL_OVERRIDE=${MODEL_OVERRIDE:-}
QUANT=${QUANT:-}                                          # e.g. QUANT=fp8

CONTAINER_NAME=${CONTAINER_NAME:-nemotron_diffusion_sglang}

# Resolve algorithm -> dllm-algorithm-config path inside the container.
CFG_PATH=""
case "${ALGO}" in
  AR)
    # AR mode: HF config gets ar_mode=true forced on; every attention layer
    # becomes causal. Scheduler still runs FastDiffuser (any dllm algo would
    # do — the model itself is fully causal now).
    MODEL_OVERRIDE='{"ar_mode": true}'
    ALGO=FastDiffuser
    CFG_PATH=/opt/sglang_fork/test/registered/dllm/configs/nemotron_labs_fastdiffuser.yaml
    ;;
  LinearSpec)
    # Rewrite the per-launch YAML so lora_mode is fresh.
    cat > "${LORA_YAML_HOST}" <<YML
algorithm: LinearSpec
causal_context: true
lora_path: /opt/linear_spec_lora
lora_mode: ${LORA_MODE}
YML
    CFG_PATH=/opt/linearspec_lora.yaml
    ;;
  LinearSpec-base)
    ALGO=LinearSpec
    CFG_PATH=/opt/sglang_fork/test/registered/dllm/configs/nemotron_labs_linearspec.yaml
    ;;
  FastDiffuser)
    CFG_PATH=/opt/sglang_fork/test/registered/dllm/configs/nemotron_labs_fastdiffuser.yaml
    ;;
  *)
    CFG_PATH=""                                            # algorithm default
    ;;
esac

mkdir -p "${HF_CACHE}" "${LOG_DIR}" "${FLASHINFER_CACHE}"

if [[ "${1:-}" == "stop" ]]; then
    sudo -n docker rm -f "${CONTAINER_NAME}" 2>/dev/null && \
        echo "stopped ${CONTAINER_NAME}" || \
        echo "no container named ${CONTAINER_NAME}"
    exit 0
fi

sudo -n docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true

DOCKER_RUN_FLAGS="--rm -a stdout -a stderr"
if [[ "${1:-}" == "detach" ]]; then
    DOCKER_RUN_FLAGS="--rm -d"
fi

CFG_ARG=""
[[ -n "${CFG_PATH}" ]] && CFG_ARG="--dllm-algorithm-config ${CFG_PATH}"

echo "[launch] algo=${ALGO} model=${MODEL} port=${PORT} ctx=${CTX_LEN} mem-frac=${MEM_FRAC} max_reqs=${MAX_REQS}"
echo "[launch] dllm-algorithm-config: ${CFG_PATH:-<algorithm default>}"
echo "[launch] quantization: ${QUANT:-<bf16 (default)>}"
echo "[launch] model_override: ${MODEL_OVERRIDE:-<none>}"
echo "[launch] logs -> ${LOG_DIR}/server.log"

read -r -d '' INNER_CMD <<INNER_EOF || true
set -e
echo '=== upgrade flashinfer + tvm-ffi + cutlass-dsl ==='
pip install --break-system-packages --quiet --no-deps --upgrade \
    apache-tvm-ffi==0.1.9 \
    nvidia-cutlass-dsl==4.5.0 \
    flashinfer_python==0.6.11.post1 \
    flashinfer_cubin==0.6.11.post1 2>&1 | tail -3
echo '=== shadowed sglang import probe ==='
python3 -c 'import sglang, sglang.srt.dllm as d; print("sglang ver:", sglang.__version__); from sglang.srt.dllm.algorithm import algo_name_to_cls; print("algos    :", sorted(algo_name_to_cls.keys()))'
echo '=== launch ==='
exec python3 -m sglang.launch_server \\
    --model-path ${MODEL} \\
    --trust-remote-code \\
    --tp-size 1 \\
    --mem-fraction-static ${MEM_FRAC} \\
    --max-running-requests ${MAX_REQS} \\
    --attention-backend flashinfer \\
    ${MODEL_OVERRIDE:+--json-model-override-args '${MODEL_OVERRIDE}'} \\
    ${QUANT:+--quantization ${QUANT}} \\
    --dllm-algorithm ${ALGO} \\
    ${CFG_ARG} \\
    --cuda-graph-bs 1 2 3 4 \\
    --context-length ${CTX_LEN} \\
    --host 0.0.0.0 \\
    --port ${PORT} \\
    2>&1 | tee /logs/server.log
INNER_EOF

# shellcheck disable=SC2086
sudo -n docker run ${DOCKER_RUN_FLAGS} \
    --gpus all \
    --name "${CONTAINER_NAME}" \
    -p ${PORT}:${PORT} \
    -v "${FORK_DIR}:/opt/sglang_fork:ro" \
    -v "${HF_CACHE}:/root/.cache/huggingface" \
    -v "${LOG_DIR}:/logs" \
    -v "${FLASHINFER_CACHE}:/root/.cache/flashinfer" \
    -v "${LORA_HOST_DIR}:/opt/linear_spec_lora:ro" \
    -v "${LORA_YAML_HOST}:/opt/linearspec_lora.yaml:ro" \
    -e PYTHONPATH=/opt/sglang_fork/python \
    -e HF_HOME=/root/.cache/huggingface \
    -e PYTHONDONTWRITEBYTECODE=1 \
    -e HF_HUB_ENABLE_HF_TRANSFER=1 \
    -e SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1 \
    lmsysorg/sglang:spark \
    bash -lc "${INNER_CMD}"
