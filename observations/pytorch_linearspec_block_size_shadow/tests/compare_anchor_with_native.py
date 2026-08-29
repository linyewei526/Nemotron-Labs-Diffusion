#!/usr/bin/env python3
"""Real-model smoke: prove shadow L16 output/NFE equals native L16 at greedy decode."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer


PROJECT_DIR = Path(__file__).resolve().parents[3]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from observations.pytorch_linearspec_block_size_shadow.shadow_generation import (  # noqa: E402
    linear_spec_generate_with_block_size_shadows,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="/data1/linyewei/models/Nemotron-Labs-Diffusion-8B",
    )
    parser.add_argument("--lora-path", default="")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()
    lora_path = args.lora_path or str(Path(args.model) / "linear_spec_lora")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    base = AutoModel.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
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
    with torch.inference_mode():
        native_ids, native_nfe = model.linear_spec_generate(
            prompt_ids,
            max_new_tokens=args.max_new_tokens,
            block_length=16,
            threshold=0.0,
            temperature=0.0,
            eos_token_id=tokenizer.eos_token_id,
        )
        shadow_ids, shadow_nfe = linear_spec_generate_with_block_size_shadows(
            model,
            prompt_ids,
            max_new_tokens=args.max_new_tokens,
            block_sizes=(4, 8, 16, 32),
            anchor_block_size=16,
            threshold=0.0,
            temperature=0.0,
            eos_token_id=tokenizer.eos_token_id,
        )
    equal = bool(torch.equal(native_ids, shadow_ids))
    nfe_equal = int(native_nfe) == int(shadow_nfe)
    print(
        {
            "token_equal": equal,
            "nfe_equal": nfe_equal,
            "native_nfe": int(native_nfe),
            "shadow_anchor_nfe": int(shadow_nfe),
            "native_tokens": int(native_ids.shape[1] - prompt_ids.shape[1]),
            "shadow_tokens": int(shadow_ids.shape[1] - prompt_ids.shape[1]),
        }
    )
    if not equal:
        common = min(native_ids.shape[1], shadow_ids.shape[1])
        mismatch = (native_ids[:, :common] != shadow_ids[:, :common]).nonzero()
        print({"first_mismatch": mismatch[0].tolist() if len(mismatch) else None})
    return 0 if equal and nfe_equal else 2


if __name__ == "__main__":
    raise SystemExit(main())
