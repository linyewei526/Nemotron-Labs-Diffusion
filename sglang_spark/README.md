# Serving Nemotron-Labs-Diffusion with SGLang on a DGX Spark

Step-by-step guide to deploy
[`nvidia/Nemotron-Labs-Diffusion-8B`](https://huggingface.co/nvidia/Nemotron-Labs-Diffusion-8B)
with the **Linear self-speculation** drafter (LoRA-enhanced) and **FP8**
weight quantization on a DGX Spark, plus a single-file HTML chat client.

The SGLang code lives in the upstream DLLM-onboarding PR stack tracked at
[sgl-project/sglang#25802](https://github.com/sgl-project/sglang/issues/25802).
We use [`hutm/sglang @ upstream/2-dllm-lora-ar`](https://github.com/hutm/sglang/tree/upstream/2-dllm-lora-ar)
(PR #2 in the stack — LoRA-aware LinearSpec execution). No SGLang code
changes are needed beyond a one-line patch noted in Step 2.

## What you get

- OpenAI-compatible chat server on port 30000 serving
  `nvidia/Nemotron-Labs-Diffusion-8B`.
- Default config: **Linear self-spec** drafter + LoRA-enhanced draft pass +
  online BF16 → FP8 weight quantization.
- A black-themed single-file HTML client (`index.html`) with sliders for
  sampling, streaming, token usage + acceptance length. No build step.

## Prerequisites

On the **DGX Spark** (the server host):

- NVIDIA GB10 (or any Blackwell + aarch64 board) with a current driver.
- Docker, with one of:
  - your user in the `docker` group (then drop the `sudo -n` prefix in the
    launch script), **or**
  - `sudo` configured to allow `docker` non-interactively (the launch script
    uses `sudo -n docker` by default).
- NVIDIA Container Toolkit set up so `docker run --gpus all` works.
- ≈ 30 GB free for the SGLang container image.
- ≈ 17 GB free for the BF16 model cache (downloaded once from HuggingFace).
- Outbound HTTPS to `huggingface.co`, `download.pytorch.org`, `pypi.org`,
  and `hub.docker.com`.

On the **client** (the machine running the browser — can be the Spark itself
or any laptop):

- A modern browser.
- Optionally Python 3 to serve the static `index.html` over `http://`.

## Step 1 — Clone the SGLang DLLM branch

```bash
mkdir -p ~/sglang_dllm/src
cd ~/sglang_dllm/src
git clone --depth 1 -b upstream/2-dllm-lora-ar https://github.com/hutm/sglang.git
```

## Step 2 — Apply the one known patch

The DLLM scheduler mixin on this branch has a bare-`self` call that crashes
on the first generate request. One-line fix:

```bash
sed -i 's|self\.report_prefill_stats(|self.metrics_reporter.report_prefill_stats(|' \
  ~/sglang_dllm/src/sglang/python/sglang/srt/dllm/mixin/scheduler.py
```

(No-op once the upstream PR merges the fix.)

## Step 3 — Fetch the bundled LoRA adapter

The HF model repo ships the LoRA-enhanced drafter under `linear_spec_lora/`.
Two files; download both:

```bash
mkdir -p ~/sglang_dllm/linear_spec_lora
cd ~/sglang_dllm/linear_spec_lora
curl -fsSL -O https://huggingface.co/nvidia/Nemotron-Labs-Diffusion-8B/resolve/main/linear_spec_lora/adapter_config.json
curl -fsSL -O https://huggingface.co/nvidia/Nemotron-Labs-Diffusion-8B/resolve/main/linear_spec_lora/adapter_model.safetensors
```

## Step 4 — Pull the SGLang container image

```bash
docker pull lmsysorg/sglang:spark
```

(`sudo docker pull …` if your user isn't in the `docker` group.) This is the
official aarch64 + Blackwell image (Python 3.12, torch 2.9.0+cu130, sgl-kernel
prebuilt). The launch script will additively upgrade a few python packages
inside the container at startup to match the fork's pins, all `--no-deps` and
ephemeral.

## Step 5 — Drop in the launch script

Copy [`launch_server.sh`](./launch_server.sh) from this directory to
`~/sglang_dllm/launch_server.sh` and make it executable:

```bash
cp launch_server.sh ~/sglang_dllm/launch_server.sh
chmod +x ~/sglang_dllm/launch_server.sh
```

The script bind-mounts your cloned fork at `/opt/sglang_fork:ro` and
PYTHONPATH-shadows the prebuilt sglang in the container — no `pip install` of
the fork is needed (its `pyproject.toml` pulls Rust deps that don't yet have
aarch64 wheels published). All host paths default to `~/sglang_dllm/...` and
are overridable via env (see the script header).

## Step 6 — Launch (default: Linear self-spec + LoRA + FP8)

```bash
QUANT=fp8 ~/sglang_dllm/launch_server.sh detach
```

First boot takes ≈ 3 minutes: model download + flashinfer JIT compile
(cached at `~/sglang_dllm/flashinfer_cache/` so subsequent boots skip the
recompile). Watch the log:

```bash
tail -f ~/sglang_dllm/logs/server.log
```

Wait for `INFO: Uvicorn running on http://0.0.0.0:30000`, then check:

```bash
curl -fsS http://localhost:30000/health
# 200, empty body
```

## Step 7 — Smoke test

```bash
curl -sS http://localhost:30000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "nvidia/Nemotron-Labs-Diffusion-8B",
    "messages": [{"role": "user", "content": "What is 15% of 240?"}],
    "max_tokens": 256,
    "temperature": 0
  }'
```

You should see a coherent worked solution with `usage.completion_tokens > 0`.

## Step 8 — Open the HTML client

### Same machine as the server

```bash
# from this directory (sglang_spark/)
python3 -m http.server 8000 --bind 127.0.0.1
# then open http://127.0.0.1:8000 in a browser
```

Or just open `index.html` directly — SGLang sends `Access-Control-Allow-Origin: *`,
so the `file://` origin works fine for `fetch()` calls.

### Different machine (laptop ↔ remote Spark)

On your laptop, open an SSH tunnel to the Spark:

```bash
ssh -L 30000:localhost:30000 user@<spark-host>
# Leave this session open.
```

Then open `index.html` locally; the default Base URL of `http://localhost:30000`
goes through the tunnel. If port 30000 is already taken locally, pick
another (e.g. `-L 31000:localhost:30000`) and update **Base URL** in the
client's ⚙ drawer to `http://localhost:31000`.

## Switching decoding modes

Each algorithm is one server process; switching requires a restart (≈ 3 min
to recapture cuda graphs; flashinfer JIT cache hits).

```bash
# Stop the running server
~/sglang_dllm/launch_server.sh stop

# Default — Linear self-spec + LoRA + FP8 (fastest on templated text)
QUANT=fp8                       ~/sglang_dllm/launch_server.sh detach

# Linear self-spec without the LoRA drafter (baseline)
QUANT=fp8 ALGO=LinearSpec-base  ~/sglang_dllm/launch_server.sh detach

# Pure diffusion via iterative denoising
QUANT=fp8 ALGO=FastDiffuser     ~/sglang_dllm/launch_server.sh detach

# Pure autoregressive (model attention layers forced to causal)
QUANT=fp8 ALGO=AR               ~/sglang_dllm/launch_server.sh detach

# Drop FP8 for a BF16 reference
ALGO=LinearSpec                 ~/sglang_dllm/launch_server.sh detach
```

Other env knobs (see the script header for the full list):

| Var | Default | Notes |
|---|---:|---|
| `CTX_LEN`     | `2048`              | Max sequence length per request. Raise for long outputs. |
| `MEM_FRAC`    | `0.5`               | Fraction of GPU memory for KV cache + weights. |
| `MAX_REQS`    | `2`                 | Max concurrent requests. |
| `PORT`        | `30000`             | OpenAI-compatible server port. |
| `MODEL`       | `nvidia/Nemotron-Labs-Diffusion-8B` | HF id or local path. |
| `LORA_MODE`   | `draft_only`        | `both` applies LoRA on draft + verify. |
| `WORK_DIR`    | `$HOME/sglang_dllm` | Where weights / logs / JIT cache live. |
