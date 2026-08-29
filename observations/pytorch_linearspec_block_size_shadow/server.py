#!/usr/bin/env python3
"""OpenAI-compatible server for paired LinearSpec block-size observation."""

from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path

import uvicorn


PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from observations.pytorch_linearspec_block_size_shadow.shadow_generation import (  # noqa: E402
    linear_spec_generate_with_block_size_shadows,
)
from observations.pytorch_linearspec_block_size_shadow.trace_writer import (  # noqa: E402
    ShadowTraceWriter,
)
from xp.pytorch_nemo_eval.pytorch_openai_server import (  # noqa: E402
    NativePyTorchEngine,
    create_app,
)


def parse_int_list(value: str, *, minimum: int, name: str) -> tuple[int, ...]:
    try:
        parsed = tuple(sorted({int(item.strip()) for item in value.split(",") if item.strip()}))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{name} must be comma-separated integers") from exc
    if not parsed or any(item < minimum for item in parsed):
        raise argparse.ArgumentTypeError(f"{name} values must be >= {minimum}")
    return parsed


class BlockSizeShadowEngine(NativePyTorchEngine):
    """Native engine whose sole committed trajectory is anchor LinearSpec."""

    def __init__(self, args: argparse.Namespace) -> None:
        if args.mode not in {"linearspec_base", "linearspec_lora"}:
            raise ValueError("block-size shadow observation supports LinearSpec only")
        self.block_sizes = parse_int_list(
            args.block_sizes, minimum=2, name="--block-sizes"
        )
        self.anchor_block_size = int(args.anchor_block_size)
        if self.anchor_block_size not in self.block_sizes:
            raise ValueError("--anchor-block-size must occur in --block-sizes")
        self.history_windows = parse_int_list(
            args.history_windows, minimum=1, name="--history-windows"
        )
        if args.context_length <= max(self.block_sizes):
            raise ValueError("--context-length must exceed the largest shadow block")
        self.configured_context_length = int(args.context_length)
        self._trace_context = threading.local()
        self.shadow_tracer = ShadowTraceWriter(
            args.trace_file,
            benchmark=args.benchmark,
            trace_detail=args.trace_detail,
        )
        args.block_length = self.anchor_block_size
        # Parent validation checks prompt+anchor budget.  Reserve one largest
        # shadow block as an explicit context guard for the final paired round.
        args.context_length = self.configured_context_length - max(self.block_sizes)
        super().__init__(args)

    def generate(self, request, request_id: str, arrival_time: float):
        self._trace_context.request_id = request_id
        try:
            return super().generate(request, request_id, arrival_time)
        finally:
            self._trace_context.request_id = ""

    def _append_stat(self, payload):
        # Request stats remain an audit of success/token counts and logical
        # anchor NFE, not an end-to-end efficiency result for this intentionally
        # expensive shadow experiment.
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
        return linear_spec_generate_with_block_size_shadows(
            self.model,
            prompt_ids,
            max_new_tokens=generation_budget,
            block_sizes=self.block_sizes,
            anchor_block_size=self.anchor_block_size,
            threshold=self.threshold or 0.0,
            temperature=temperature,
            eos_token_id=self.eos_token_id,
            max_thinking_tokens=self.max_thinking_tokens,
            end_think_token_id=self.end_think_token_id,
            tracer=self.shadow_tracer,
            request_id=getattr(self._trace_context, "request_id", ""),
            mode=self.mode,
            history_windows=self.history_windows,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=34000)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--served-model-name", default="nemotron-labs-diffusion-8b")
    parser.add_argument(
        "--mode",
        default="linearspec_lora",
        choices=["linearspec_base", "linearspec_lora"],
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "bf16", "float16", "fp16", "float32", "fp32"],
    )
    parser.add_argument("--block-sizes", default="4,8,16,32")
    parser.add_argument("--anchor-block-size", type=int, default=16)
    parser.add_argument("--history-windows", default="1,2,4,8")
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument(
        "--causal-context",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--default-max-new-tokens", type=int, default=8192)
    parser.add_argument("--context-length", type=int, default=10240)
    parser.add_argument("--lora-path", default="")
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--max-thinking-tokens", type=int, default=None)
    parser.add_argument("--stats-file", required=True)
    parser.add_argument("--trace-file", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument(
        "--trace-detail",
        default="position",
        choices=["scalar", "position", "tokens"],
    )
    parser.add_argument("--timeout-keep-alive", type=int, default=300)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    engine = BlockSizeShadowEngine(args)
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
