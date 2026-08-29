#!/usr/bin/env python3
"""Real-model smoke: traced L16 output and logical NFE must equal native L16."""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer


PROJECT_DIR = Path(__file__).resolve().parents[3]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from observations.adaptive_failure_locator_search.locator_generation import (  # noqa: E402
    linear_spec_generate_with_locator_trace,
)
from observations.adaptive_failure_locator_search.trace_writer import (  # noqa: E402
    LocatorTraceWriter,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/data1/linyewei/models/Nemotron-Labs-Diffusion-8B")
    parser.add_argument("--lora-path", default="")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()
    lora_path = args.lora_path or str(Path(args.model) / "linear_spec_lora")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    base = AutoModel.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to("cuda:0").eval()
    from peft import PeftModel

    wrapper = PeftModel.from_pretrained(base, lora_path).eval()
    model = wrapper.model.eval()
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Compute 17 + 25 and give a short answer."}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    prompt_ids = tokenizer(text, return_tensors="pt").input_ids.to("cuda:0")
    with tempfile.TemporaryDirectory() as temporary:
        tracer = LocatorTraceWriter(Path(temporary) / "trace.jsonl", benchmark="smoke")
        with torch.inference_mode():
            native_ids, native_nfe = model.linear_spec_generate(
                prompt_ids,
                max_new_tokens=args.max_new_tokens,
                block_length=args.block_size,
                threshold=0.0,
                temperature=0.0,
                eos_token_id=tokenizer.eos_token_id,
            )
            traced_ids, traced_nfe = linear_spec_generate_with_locator_trace(
                model,
                prompt_ids,
                max_new_tokens=args.max_new_tokens,
                block_size=args.block_size,
                threshold=0.0,
                temperature=0.0,
                eos_token_id=tokenizer.eos_token_id,
                tracer=tracer,
                request_id="native-equivalence",
            )
        trace_rows = sum(1 for line in tracer.path.read_text().splitlines() if line.strip())
    result = {
        "token_equal": bool(torch.equal(native_ids, traced_ids)),
        "nfe_equal": int(native_nfe) == int(traced_nfe),
        "native_nfe": int(native_nfe),
        "traced_nfe": int(traced_nfe),
        "trace_rounds": trace_rows,
    }
    print(result)
    return 0 if result["token_equal"] and result["nfe_equal"] and trace_rows > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
