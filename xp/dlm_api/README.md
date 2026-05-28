# dlm_api

OpenAI-compatible HTTP server that wraps the Nemotron-Labs diffusion-LM
inference paths. This package backs the inference workers spawned by
`xp/examples/run_dlm_eval_pipeline_gpu_only.sh` and is the runtime that
`xp/nemo-skills/eval_dlm.py` talks to over HTTP.

## Files

- **`dlm_batch_server.py`** — FastAPI worker. Owns model loading, request
  batching, NFE logging, and dispatching into one of the registered
  generation algorithms.
- **`dlm_openai_server.py`** — Pydantic request/response models and chat
  template helpers (no networking).
- **`dlm_load_balancer.py`** — Multi-GPU load balancer that fans requests
  out to N worker processes (one per GPU).
- **`dlm_generate/`** — Generation algorithm registry. The three algorithms
  used by `eval.sh`:
  - `nemotron` — diffusion sampling (used by `--mode dlm` and, with
    `LINEAR_SPECULATION=true`, `--mode linear_spec`).
  - `nemotron_mixed` — mixed AR/dLM (loaded alongside `nemotron` from the
    same engine).
  - `ar_native` — pure autoregressive via the model's own `ar_generate`
    method (`--mode ar`).

The third-party `fast_dllm` / `dinfer` / `dllm_eval` / `huggingface`
algorithm packages from the upstream LLaDA-API tree have been removed in
this slim build because they target LLaDA-family models (e.g.
`GSAI-ML/LLaDA-8B-Instruct`), not the Nemotron diffusion family.

## How `eval.sh` uses this

Each SLURM job runs `dlm_batch_server.py` per GPU and a single
`dlm_load_balancer.py` at the front. The eval client (`eval_dlm.py`) hits
the load balancer over `http://localhost:$LOAD_BALANCER_PORT/v1`. All
flags relevant to the four modes — `--engine`, `--linear-speculation`,
`--draft-lora-only`, `--lora-path`, `--max-thinking-tokens`,
`--eos-early-stop`, etc. — are documented in `dlm_batch_server.py
--help`.
