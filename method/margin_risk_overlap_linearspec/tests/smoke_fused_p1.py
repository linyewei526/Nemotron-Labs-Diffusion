#!/usr/bin/env python3
"""Real-model smoke for the newly reachable candidate_position=1 fused path."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer
from transformers.cache_utils import DynamicCache

PROJECT_DIR = Path(__file__).resolve().parents[3]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from method.margin_risk_overlap_linearspec.generation import (
    DraftState,
    OverlapGenerationStats,
    _fused_verify_and_draft,
    _normal_draft,
    _normal_verify,
    _set_diffusion_lm,
)
from method.margin_risk_overlap_linearspec.hybrid import repeat_dynamic_cache
from method.margin_risk_overlap_linearspec.segmented_lora import install_segmented_lora


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/data1/linyewei/models/Nemotron-Labs-Diffusion-8B")
    parser.add_argument("--lora-path", default="")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=dtype
    ).to("cuda").eval()
    lora_path = args.lora_path or f"{args.model}/linear_spec_lora"
    controller = install_segmented_lora(model, lora_path)
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Return the number one."}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
    _set_diffusion_lm(model, False)
    with controller.disabled():
        prefill = model.encoder(
            input_ids=prompt_ids,
            past_key_values=DynamicCache(),
            use_cache=True,
            use_causal_mask=True,
        )
    cache = prefill.past_key_values
    cache_length = int(cache.get_seq_length())
    seed = model.diffusion_head(prefill.last_hidden_state[:, -1:, :]).argmax(dim=-1)
    mask_id = int(model.config.mask_token_id)
    eos_id = tokenizer.eos_token_id
    excluded = tuple(token for token in (mask_id, eos_id) if token is not None)
    draft_stats = OverlapGenerationStats()
    draft = _normal_draft(
        model,
        seed=seed,
        past_key_values=cache,
        block_length=args.block_size,
        mask_token_id=mask_id,
        excluded_alternative_token_ids=excluded,
        margin_risk_threshold=0.5,
        controller=controller,
        stats=draft_stats,
    )
    original = int(draft.tokens[1].item())
    alternative = (original + 1) % int(model.config.vocab_size)
    while alternative in set(excluded) | {original}:
        alternative = (alternative + 1) % int(model.config.vocab_size)
    forced = DraftState(
        tokens=draft.tokens,
        confidences=draft.confidences,
        top1_top2_margins=draft.top1_top2_margins,
        candidate_position=1,
        alternative_token=alternative,
        candidate_margin_risk=1.0,
        source="forced_p1_smoke",
    )
    fused_stats = OverlapGenerationStats()
    fused_logits, _, prospective = _fused_verify_and_draft(
        model,
        draft=forced,
        past_key_values=cache,
        block_length=args.block_size,
        mask_token_id=mask_id,
        excluded_alternative_token_ids=excluded,
        margin_risk_threshold=0.5,
        controller=controller,
        stats=fused_stats,
    )
    normal_stats = OverlapGenerationStats()
    normal_logits, _ = _normal_verify(
        model,
        draft=forced,
        past_key_values=repeat_dynamic_cache(cache, 1),
        controller=controller,
        stats=normal_stats,
    )
    fused_tokens = fused_logits.argmax(dim=-1)
    normal_tokens = normal_logits.argmax(dim=-1)
    if not torch.equal(fused_tokens, normal_tokens):
        raise RuntimeError("p=1 fused verifier tokens differ from normal causal verify")
    if int(cache.get_seq_length()) != cache_length:
        raise RuntimeError("fused path mutated canonical prefill cache")
    if int(prospective.tokens[0].item()) != alternative:
        raise RuntimeError("prospective p=1 seed is not forced alternative B")
    print(
        json.dumps(
            {
                "ok": True,
                "candidate_position": 1,
                "block_size": args.block_size,
                "cache_length_unchanged": cache_length,
                "verifier_argmax_equal": True,
                "prospective_seed": alternative,
                "fused_processed_rows": fused_stats.processed_rows,
                "fused_query_tokens": fused_stats.processed_query_tokens,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
