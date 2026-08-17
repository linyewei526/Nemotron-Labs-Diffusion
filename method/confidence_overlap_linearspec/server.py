#!/usr/bin/env python3
"""Isolated OpenAI-compatible server for confidence-overlap LinearSpec."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import socket
import sys
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Optional

METHOD_DIR = Path(__file__).resolve().parent
PROJECT_DIR = METHOD_DIR.parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field
from transformers import AutoModel, AutoTokenizer

from method.confidence_overlap_linearspec.generation import overlap_linear_spec_generate
from method.confidence_overlap_linearspec.segmented_lora import install_segmented_lora


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


def _rounded(value: Optional[float], digits: int = 6) -> Optional[float]:
    return None if value is None else round(float(value), digits)


def _normalize_stops(value: Optional[str | list[str]]) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    return [str(item) for item in value if str(item)]


def _truncate_at_stop(text: str, stops: list[str]) -> tuple[str, bool]:
    positions = [text.find(stop) for stop in stops]
    positions = [position for position in positions if position >= 0]
    if not positions:
        return text, False
    return text[: min(positions)], True


class ConfidenceOverlapEngine:
    def __init__(self, args: argparse.Namespace) -> None:
        self.mode = args.mode
        self.model_path = str(Path(args.model_path).resolve())
        self.served_model_name = args.served_model_name
        self.block_length = int(args.block_length)
        self.draft_threshold = float(args.draft_threshold)
        self.drop_pct_threshold = float(args.drop_pct_threshold)
        self.default_max_new_tokens = int(args.default_max_new_tokens)
        self.context_length = int(args.context_length)
        self.enable_thinking = bool(args.enable_thinking)
        self.max_thinking_tokens = args.max_thinking_tokens
        self.stats_file = Path(args.stats_file).resolve()
        self.stats_file.parent.mkdir(parents=True, exist_ok=True)
        self.stats_file.touch(exist_ok=True)
        self._stats_lock = threading.Lock()
        self._model_lock = threading.Lock()

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
            raise RuntimeError("CUDA is required for the 8B overlap experiment")
        self.device = torch.device("cuda:0")
        torch.cuda.set_device(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=self.dtype,
        ).to(self.device).eval()
        self.lora_controller = None
        if self.mode == "overlap_lora":
            if not args.lora_path:
                raise RuntimeError("overlap_lora requires --lora-path")
            self.lora_controller = install_segmented_lora(self.model, args.lora_path)

        self.eos_token_id = self.tokenizer.eos_token_id
        self.end_think_token_id: Optional[int] = None
        if self.max_thinking_tokens is not None:
            ids = self.tokenizer.encode("</think>", add_special_tokens=False)
            if ids:
                self.end_think_token_id = int(ids[-1])

    def _append_stat(self, payload: dict[str, Any]) -> None:
        line = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        with self._stats_lock:
            with self.stats_file.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    def _format_and_tokenize(self, messages: list[dict[str, Any]]) -> torch.Tensor:
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )
        return self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)

    def _generation_budget(self, requested_tokens: int) -> int:
        return max(
            self.block_length,
            math.ceil(requested_tokens / self.block_length) * self.block_length,
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
        if temperature != 0.0:
            raise ValueError("confidence-overlap experiment currently requires temperature=0")
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

                torch.cuda.reset_peak_memory_stats(self.device)
                torch.cuda.synchronize(self.device)
                native_start = time.perf_counter()
                with torch.inference_mode():
                    output_ids, decode_stats = overlap_linear_spec_generate(
                        self.model,
                        prompt_ids,
                        max_new_tokens=generation_budget,
                        block_length=self.block_length,
                        drop_pct_threshold=self.drop_pct_threshold,
                        temperature=temperature,
                        draft_threshold=self.draft_threshold,
                        eos_token_id=self.eos_token_id,
                        max_thinking_tokens=self.max_thinking_tokens,
                        end_think_token_id=self.end_think_token_id,
                        lora_controller=self.lora_controller,
                    )
                torch.cuda.synchronize(self.device)
                native_end = time.perf_counter()

                raw_ids = output_ids[0, prompt_tokens:]
                raw_generated_tokens = int(raw_ids.numel())
                returned_ids = raw_ids[:requested_tokens]
                returned_text = self.tokenizer.decode(returned_ids, skip_special_tokens=True)
                returned_text, stopped_by_string = _truncate_at_stop(
                    returned_text, _normalize_stops(request.stop)
                )
                completion_tokens = (
                    len(self.tokenizer.encode(returned_text, add_special_tokens=False))
                    if stopped_by_string
                    else int(returned_ids.numel())
                )
                saw_eos = bool(
                    self.eos_token_id is not None
                    and (returned_ids == int(self.eos_token_id)).any().item()
                )
                finish_reason = (
                    "stop"
                    if stopped_by_string or saw_eos or raw_generated_tokens < requested_tokens
                    else "length"
                )
                model_time_s = native_end - native_start
                request_time_s = time.perf_counter() - arrival_time
                nfe = float(decode_stats.physical_nfe)
                overlap = decode_stats.to_dict()
                stat = {
                    "ok": True,
                    "request_id": request_id,
                    "created": int(time.time()),
                    "backend": "native_pytorch_confidence_overlap",
                    "mode": self.mode,
                    "model": request.model,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "raw_generated_tokens": raw_generated_tokens,
                    "requested_tokens": requested_tokens,
                    "generation_budget": generation_budget,
                    "nfe": nfe,
                    "tokens_per_forward_pass": completion_tokens / nfe if nfe else None,
                    "model_time_s": _rounded(model_time_s),
                    "queue_wait_s": _rounded(queue_wait_s),
                    "request_time_s": _rounded(request_time_s),
                    "model_output_tokens_per_s": _rounded(
                        completion_tokens / model_time_s if model_time_s > 0 else None, 4
                    ),
                    "peak_gpu_memory_gib": _rounded(
                        torch.cuda.max_memory_allocated(self.device) / 1024**3, 4
                    ),
                    "temperature": temperature,
                    "top_p_requested": request.top_p,
                    "top_p_applied": False,
                    "top_k_requested": request.top_k,
                    "top_k_applied": False,
                    "block_length": self.block_length,
                    "draft_threshold": self.draft_threshold,
                    "drop_pct_threshold": self.drop_pct_threshold,
                    "finish_reason": finish_reason,
                    "overlap": overlap,
                }
                self._append_stat(stat)
                return {
                    "text": returned_text,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "nfe": nfe,
                    "finish_reason": finish_reason,
                }
            except Exception as exc:
                self._append_stat(
                    {
                        "ok": False,
                        "request_id": request_id,
                        "created": int(time.time()),
                        "backend": "native_pytorch_confidence_overlap",
                        "mode": self.mode,
                        "model": request.model,
                        "queue_wait_s": _rounded(queue_wait_s),
                        "request_time_s": _rounded(time.perf_counter() - arrival_time),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback_tail": traceback.format_exc()[-4000:],
                    }
                )
                raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0, help="0 asks the OS for a free port")
    parser.add_argument("--port-file", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--served-model-name", default="nemotron-labs-diffusion-8b-overlap")
    parser.add_argument("--mode", choices=["overlap_base", "overlap_lora"], required=True)
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "bf16", "float16", "fp16", "float32", "fp32"],
    )
    parser.add_argument("--block-length", type=int, default=16)
    parser.add_argument("--draft-threshold", type=float, default=0.0)
    parser.add_argument("--drop-pct-threshold", type=float, default=0.15)
    parser.add_argument("--default-max-new-tokens", type=int, default=8192)
    parser.add_argument("--context-length", type=int, default=10240)
    parser.add_argument("--lora-path", default="")
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--max-thinking-tokens", type=int, default=None)
    parser.add_argument("--stats-file", required=True)
    parser.add_argument("--timeout-keep-alive", type=int, default=300)
    args = parser.parse_args()
    if not 0 <= args.port <= 65535:
        parser.error("--port must be in [0,65535]")
    if args.block_length < 2:
        parser.error("--block-length must be at least 2")
    if args.draft_threshold != 0.0:
        parser.error("this experiment currently requires --draft-threshold 0")
    if not 0 <= args.drop_pct_threshold < 1:
        parser.error("--drop-pct-threshold must be in [0,1)")
    return args


def create_app(engine: ConfidenceOverlapEngine) -> FastAPI:
    app = FastAPI(title="NLD Confidence-Overlap LinearSpec Server", version="1.0")
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
            "backend": "native_pytorch_confidence_overlap",
            "model_loaded": True,
            "mode": engine.mode,
            "device": str(engine.device),
            "dtype": str(engine.dtype),
            "block_length": engine.block_length,
            "drop_pct_threshold": engine.drop_pct_threshold,
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
                    "owned_by": "local-confidence-overlap",
                }
            ],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest) -> dict[str, Any]:
        if request.stream:
            raise HTTPException(status_code=400, detail="Streaming is not supported")
        request_id = f"chatcmpl-overlap-{uuid.uuid4().hex}"
        arrival = time.perf_counter()
        try:
            result = await asyncio.to_thread(engine.generate, request, request_id, arrival)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except torch.cuda.OutOfMemoryError as exc:
            raise HTTPException(status_code=507, detail=f"CUDA out of memory: {exc}") from exc
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Confidence-overlap generation failed: {type(exc).__name__}: {exc}",
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


def _write_port_file(path: Path, host: str, port: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(
            {"pid": os.getpid(), "host": host, "port": int(port)},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> int:
    args = parse_args()
    engine = ConfidenceOverlapEngine(args)
    app = create_app(engine)
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((args.host, args.port))
    listener.listen(2048)
    actual_port = int(listener.getsockname()[1])
    _write_port_file(Path(args.port_file).resolve(), args.host, actual_port)
    config = uvicorn.Config(
        app,
        host=args.host,
        port=actual_port,
        timeout_keep_alive=args.timeout_keep_alive,
        timeout_graceful_shutdown=30,
        log_level="info",
    )
    uvicorn.Server(config).run(sockets=[listener])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
