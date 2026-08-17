#!/usr/bin/env python3
"""Small OpenAI-compatible server for native PyTorch NLD generation.

The server intentionally does not import SGLang.  It loads the model's Hugging
Face remote code and dispatches each request to ar_generate(), generate(), or
linear_spec_generate().  Requests are serialized on the model so native
generation state (attention mode, PEFT adapter toggles, and KV cache) cannot be
mutated concurrently.  A JSONL record is written for every request and later
merged into the NeMo-Skills metrics file.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field
from transformers import AutoModel, AutoTokenizer


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str = "nemotron-labs-diffusion-8b"
    messages: list[dict[str, Any]]
    temperature: Optional[float] = Field(default=0.0, ge=0.0)
    top_p: Optional[float] = Field(default=0.95, ge=0.0, le=1.0)
    top_k: int = -1
    max_completion_tokens: Optional[int] = Field(default=None, gt=0)
    max_tokens: Optional[int] = Field(default=None, gt=0)
    stream: bool = False
    stop: Optional[str | list[str]] = None
    seed: Optional[int] = None


def _round(value: Optional[float], digits: int = 6) -> Optional[float]:
    return None if value is None else round(float(value), digits)


def _normalize_stops(value: Optional[str | list[str]]) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    return [str(item) for item in value if str(item)]


def _truncate_at_stop(text: str, stops: list[str]) -> tuple[str, bool]:
    positions = [text.find(stop) for stop in stops]
    positions = [pos for pos in positions if pos >= 0]
    if not positions:
        return text, False
    return text[: min(positions)], True


class NativePyTorchEngine:
    def __init__(self, args: argparse.Namespace) -> None:
        self.mode = args.mode
        self.model_path = args.model_path
        self.served_model_name = args.served_model_name
        self.block_length = args.block_length
        self.threshold = args.threshold
        self.causal_context = args.causal_context
        self.default_max_new_tokens = args.default_max_new_tokens
        self.context_length = args.context_length
        self.enable_thinking = args.enable_thinking
        self.max_thinking_tokens = args.max_thinking_tokens
        self.stats_file = Path(args.stats_file)
        self.stats_file.parent.mkdir(parents=True, exist_ok=True)
        self.stats_file.touch(exist_ok=True)
        self._stats_lock = threading.Lock()
        self._model_lock = threading.Lock()
        self._peft_wrapper = None

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float16": torch.float16,
            "fp16": torch.float16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        self.dtype = dtype_map[args.dtype]
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for this 8B PyTorch evaluation server")

        self.device = torch.device("cuda:0")
        torch.cuda.set_device(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )
        self.model = AutoModel.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=self.dtype,
        ).to(self.device).eval()

        if self.mode == "linearspec_lora":
            if not args.lora_path:
                raise RuntimeError("linearspec_lora requires --lora-path")
            lora_dir = Path(args.lora_path)
            if not (lora_dir / "adapter_config.json").is_file():
                raise RuntimeError(f"LoRA adapter_config.json not found: {lora_dir}")
            from peft import PeftModel

            self._peft_wrapper = PeftModel.from_pretrained(
                self.model, str(lora_dir)
            ).eval()
            # Native generation methods live on the injected base model.  Keep
            # the wrapper alive so PEFT state and adapter modules remain owned.
            self.model = self._peft_wrapper.model.eval()

        required_method = {
            "ar": "ar_generate",
            "dlm": "generate",
            "linearspec_base": "linear_spec_generate",
            "linearspec_lora": "linear_spec_generate",
        }[self.mode]
        if not hasattr(self.model, required_method):
            raise RuntimeError(
                f"Loaded model does not expose {required_method} for mode={self.mode}"
            )

        self.eos_token_id = self.tokenizer.eos_token_id
        self.end_think_token_id: Optional[int] = None
        if self.max_thinking_tokens is not None:
            ids = self.tokenizer.encode("</think>", add_special_tokens=False)
            if ids:
                self.end_think_token_id = int(ids[-1])

    def _append_stat(self, payload: dict[str, Any]) -> None:
        line = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        with self._stats_lock:
            with self.stats_file.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

    def _format_and_tokenize(self, messages: list[dict[str, Any]]) -> torch.Tensor:
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )
        return self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)

    def _generation_budget(self, requested_tokens: int) -> int:
        if self.mode in {"dlm", "linearspec_base", "linearspec_lora"}:
            return max(
                self.block_length,
                math.ceil(requested_tokens / self.block_length) * self.block_length,
            )
        return requested_tokens

    def _run_native(
        self,
        prompt_ids: torch.Tensor,
        generation_budget: int,
        temperature: float,
    ) -> tuple[torch.Tensor, int]:
        common = {
            "eos_token_id": self.eos_token_id,
            "max_thinking_tokens": self.max_thinking_tokens,
            "end_think_token_id": self.end_think_token_id,
        }
        if self.mode == "ar":
            return self.model.ar_generate(
                prompt_ids=prompt_ids,
                max_new_tokens=generation_budget,
                temperature=temperature,
                **common,
            )
        if self.mode == "dlm":
            return self.model.generate(
                prompt_ids,
                max_new_tokens=generation_budget,
                block_length=self.block_length,
                threshold=self.threshold,
                causal_context=self.causal_context,
                temperature=temperature,
                **common,
            )
        return self.model.linear_spec_generate(
            prompt_ids,
            max_new_tokens=generation_budget,
            block_length=self.block_length,
            threshold=self.threshold or 0.0,
            temperature=temperature,
            **common,
        )

    def generate(
        self,
        request: ChatCompletionRequest,
        request_id: str,
        arrival_time: float,
    ) -> dict[str, Any]:
        requested_tokens = (
            request.max_completion_tokens
            or request.max_tokens
            or self.default_max_new_tokens
        )
        if requested_tokens <= 0:
            raise ValueError("max_completion_tokens must be positive")
        temperature = float(request.temperature or 0.0)
        generation_budget = self._generation_budget(requested_tokens)

        with self._model_lock:
            model_start = time.perf_counter()
            queue_wait_s = model_start - arrival_time
            try:
                if request.seed is not None:
                    torch.manual_seed(request.seed)
                    torch.cuda.manual_seed_all(request.seed)

                prompt_ids = self._format_and_tokenize(request.messages)
                prompt_tokens = int(prompt_ids.shape[1])
                if prompt_tokens + generation_budget > self.context_length:
                    raise ValueError(
                        "Requested sequence exceeds configured context length: "
                        f"prompt_tokens={prompt_tokens}, generation_budget={generation_budget}, "
                        f"context_length={self.context_length}"
                    )

                torch.cuda.synchronize(self.device)
                native_start = time.perf_counter()
                with torch.inference_mode():
                    output_ids, nfe = self._run_native(
                        prompt_ids, generation_budget, temperature
                    )
                torch.cuda.synchronize(self.device)
                native_end = time.perf_counter()

                raw_ids = output_ids[0, prompt_tokens:]
                raw_generated_tokens = int(raw_ids.numel())
                returned_ids = raw_ids[:requested_tokens]
                returned_text = self.tokenizer.decode(
                    returned_ids,
                    skip_special_tokens=True,
                )
                returned_text, stopped_by_string = _truncate_at_stop(
                    returned_text, _normalize_stops(request.stop)
                )

                if stopped_by_string:
                    completion_tokens = len(
                        self.tokenizer.encode(returned_text, add_special_tokens=False)
                    )
                else:
                    completion_tokens = int(returned_ids.numel())

                saw_eos = bool(
                    self.eos_token_id is not None
                    and (returned_ids == self.eos_token_id).any().item()
                )
                finish_reason = (
                    "stop"
                    if stopped_by_string or saw_eos or raw_generated_tokens < requested_tokens
                    else "length"
                )
                model_time_s = native_end - native_start
                request_end = time.perf_counter()
                request_time_s = request_end - arrival_time
                nfe_value = float(nfe)
                stat = {
                    "ok": True,
                    "request_id": request_id,
                    "created": int(time.time()),
                    "mode": self.mode,
                    "model": request.model,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "raw_generated_tokens": raw_generated_tokens,
                    "requested_tokens": requested_tokens,
                    "generation_budget": generation_budget,
                    "nfe": nfe_value,
                    "tokens_per_forward_pass": (
                        completion_tokens / nfe_value if nfe_value > 0 else None
                    ),
                    "model_time_s": _round(model_time_s),
                    "queue_wait_s": _round(queue_wait_s),
                    "request_time_s": _round(request_time_s),
                    "model_output_tokens_per_s": (
                        _round(completion_tokens / model_time_s, 4)
                        if model_time_s > 0
                        else None
                    ),
                    "temperature": temperature,
                    "top_p_requested": request.top_p,
                    "top_p_applied": False,
                    "top_k_requested": request.top_k,
                    "top_k_applied": False,
                    "block_length": self.block_length,
                    "threshold": self.threshold,
                    "finish_reason": finish_reason,
                }
                self._append_stat(stat)
                return {
                    "text": returned_text,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "nfe": nfe_value,
                    "finish_reason": finish_reason,
                }
            except Exception as exc:
                failed_at = time.perf_counter()
                self._append_stat(
                    {
                        "ok": False,
                        "request_id": request_id,
                        "created": int(time.time()),
                        "mode": self.mode,
                        "model": request.model,
                        "queue_wait_s": _round(queue_wait_s),
                        "request_time_s": _round(failed_at - arrival_time),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback_tail": traceback.format_exc()[-2000:],
                    }
                )
                raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=32000)
    parser.add_argument("--model-path", required=True)
    parser.add_argument(
        "--served-model-name", default="nemotron-labs-diffusion-8b"
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["ar", "dlm", "linearspec_base", "linearspec_lora"],
    )
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "bf16", "float16", "fp16", "float32", "fp32"])
    parser.add_argument("--block-length", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--causal-context", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--default-max-new-tokens", type=int, default=8192)
    parser.add_argument("--context-length", type=int, default=10240)
    parser.add_argument("--lora-path", default="")
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--max-thinking-tokens", type=int, default=None)
    parser.add_argument("--stats-file", required=True)
    parser.add_argument("--timeout-keep-alive", type=int, default=300)
    return parser.parse_args()


def create_app(engine: NativePyTorchEngine) -> FastAPI:
    app = FastAPI(title="NLD Native PyTorch Evaluation Server", version="1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "backend": "native_pytorch",
            "model_loaded": True,
            "mode": engine.mode,
            "device": str(engine.device),
            "dtype": str(engine.dtype),
        }

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": engine.served_model_name,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "local-pytorch",
                }
            ],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest) -> dict[str, Any]:
        if request.stream:
            raise HTTPException(
                status_code=400,
                detail="Streaming is not supported by the native PyTorch evaluation server",
            )
        request_id = f"chatcmpl-pytorch-{uuid.uuid4().hex}"
        arrival = time.perf_counter()
        try:
            result = await asyncio.to_thread(
                engine.generate, request, request_id, arrival
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except torch.cuda.OutOfMemoryError as exc:
            raise HTTPException(status_code=507, detail=f"CUDA out of memory: {exc}") from exc
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Native PyTorch generation failed: {type(exc).__name__}: {exc}",
            ) from exc

        return {
            "id": request_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": result["text"]},
                    "finish_reason": result["finish_reason"],
                }
            ],
            "usage": {
                "prompt_tokens": result["prompt_tokens"],
                "completion_tokens": result["completion_tokens"],
                "total_tokens": result["prompt_tokens"] + result["completion_tokens"],
                "nfe": result["nfe"],
            },
        }

    return app


def main() -> int:
    args = parse_args()
    engine = NativePyTorchEngine(args)
    app = create_app(engine)
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        timeout_keep_alive=args.timeout_keep_alive,
        timeout_graceful_shutdown=30,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
