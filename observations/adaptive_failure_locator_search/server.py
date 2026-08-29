#!/usr/bin/env python3
"""OpenAI-compatible server for the independent failure-locator observation."""

from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path

import uvicorn


PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from observations.adaptive_failure_locator_search.locator_generation import (  # noqa: E402
    linear_spec_generate_with_locator_trace,
)
from observations.adaptive_failure_locator_search.trace_writer import (  # noqa: E402
    LocatorTraceWriter,
)
from xp.pytorch_nemo_eval.pytorch_openai_server import (  # noqa: E402
    NativePyTorchEngine,
    create_app,
)


def parse_int_list(value: str, *, minimum: int = 1) -> tuple[int, ...]:
    try:
        parsed = tuple(sorted({int(item.strip()) for item in value.split(",") if item.strip()}))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not parsed or any(item < minimum for item in parsed):
        raise argparse.ArgumentTypeError(f"all values must be >= {minimum}")
    return parsed


class FailureLocatorEngine(NativePyTorchEngine):
    """Ordinary LinearSpec decoding with a non-intervening round tracer."""

    def __init__(self, args: argparse.Namespace) -> None:
        if args.mode not in {"linearspec_base", "linearspec_lora"}:
            raise ValueError("failure locator supports LinearSpec only")
        self.history_windows = parse_int_list(args.history_windows)
        self._trace_context = threading.local()
        self.locator_tracer = LocatorTraceWriter(
            args.trace_file, benchmark=args.benchmark, detail=args.trace_detail
        )
        args.block_length = int(args.block_size)
        super().__init__(args)

    def generate(self, request, request_id: str, arrival_time: float):
        self._trace_context.request_id = request_id
        try:
            return super().generate(request, request_id, arrival_time)
        finally:
            self._trace_context.request_id = ""

    def _append_stat(self, payload):
        # Timing is intentionally not interpreted by this observational study.
        clean = dict(payload)
        for field in (
            "model_time_s",
            "queue_wait_s",
            "request_time_s",
            "model_output_tokens_per_s",
            "tokens_per_forward_pass",
        ):
            clean.pop(field, None)
        super()._append_stat(clean)

    def _run_native(self, prompt_ids, generation_budget: int, temperature: float):
        return linear_spec_generate_with_locator_trace(
            self.model,
            prompt_ids,
            max_new_tokens=generation_budget,
            block_size=self.block_length,
            threshold=self.threshold or 0.0,
            temperature=temperature,
            eos_token_id=self.eos_token_id,
            max_thinking_tokens=self.max_thinking_tokens,
            end_think_token_id=self.end_think_token_id,
            tracer=self.locator_tracer,
            request_id=getattr(self._trace_context, "request_id", ""),
            mode=self.mode,
            history_windows=self.history_windows,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=36000)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--served-model-name", default="nemotron-labs-diffusion-8b")
    parser.add_argument(
        "--mode", default="linearspec_lora", choices=["linearspec_base", "linearspec_lora"]
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "bf16", "float16", "fp16", "float32", "fp32"],
    )
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--history-windows", default="1,2,4")
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument(
        "--causal-context", action=argparse.BooleanOptionalAction, default=True, help=argparse.SUPPRESS
    )
    parser.add_argument("--default-max-new-tokens", type=int, default=8192)
    parser.add_argument("--context-length", type=int, default=10240)
    parser.add_argument("--lora-path", default="")
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--max-thinking-tokens", type=int, default=None)
    parser.add_argument("--stats-file", required=True)
    parser.add_argument("--trace-file", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--trace-detail", default="position", choices=["position", "tokens"])
    parser.add_argument("--timeout-keep-alive", type=int, default=300)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.block_size < 2:
        raise ValueError("--block-size must be >=2")
    engine = FailureLocatorEngine(args)
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
