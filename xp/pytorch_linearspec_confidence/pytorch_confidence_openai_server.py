#!/usr/bin/env python3
"""OpenAI-compatible native PyTorch server with LinearSpec confidence tracing."""

from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path

import uvicorn


PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from xp.pytorch_nemo_eval.pytorch_openai_server import (  # noqa: E402
    NativePyTorchEngine,
    create_app,
)
from xp.pytorch_linearspec_confidence.confidence_trace import (  # noqa: E402
    NativeLinearSpecConfidenceTracer,
)
from xp.pytorch_linearspec_confidence.native_linearspec_confidence import (  # noqa: E402
    linear_spec_generate_with_confidence,
)


class NativeConfidenceEngine(NativePyTorchEngine):
    def __init__(self, args: argparse.Namespace) -> None:
        if args.mode not in {"linearspec_base", "linearspec_lora"}:
            raise ValueError("confidence diagnostics support only LinearSpec modes")
        self._trace_context = threading.local()
        self.confidence_tracer = NativeLinearSpecConfidenceTracer(
            args.confidence_trace_file
        )
        super().__init__(args)

    def generate(self, request, request_id: str, arrival_time: float):
        self._trace_context.request_id = request_id
        try:
            return super().generate(request, request_id, arrival_time)
        finally:
            self._trace_context.request_id = ""

    def _run_native(self, prompt_ids, generation_budget: int, temperature: float):
        return linear_spec_generate_with_confidence(
            self.model,
            prompt_ids,
            max_new_tokens=generation_budget,
            block_length=self.block_length,
            threshold=self.threshold or 0.0,
            temperature=temperature,
            eos_token_id=self.eos_token_id,
            max_thinking_tokens=self.max_thinking_tokens,
            end_think_token_id=self.end_think_token_id,
            tracer=self.confidence_tracer,
            request_id=getattr(self._trace_context, "request_id", ""),
            mode=self.mode,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=33000)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--served-model-name", default="nemotron-labs-diffusion-8b")
    parser.add_argument(
        "--mode",
        required=True,
        choices=["linearspec_base", "linearspec_lora"],
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "bf16", "float16", "fp16", "float32", "fp32"],
    )
    parser.add_argument("--block-length", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument(
        "--causal-context", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--default-max-new-tokens", type=int, default=8192)
    parser.add_argument("--context-length", type=int, default=10240)
    parser.add_argument("--lora-path", default="")
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--max-thinking-tokens", type=int, default=None)
    parser.add_argument("--stats-file", required=True)
    parser.add_argument("--confidence-trace-file", required=True)
    parser.add_argument("--timeout-keep-alive", type=int, default=300)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    engine = NativeConfidenceEngine(args)
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
