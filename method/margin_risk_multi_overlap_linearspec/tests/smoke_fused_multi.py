#!/usr/bin/env python3
"""Real-model smoke for four-row candidate and continuation fused paths."""

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

from method.margin_risk_multi_overlap_linearspec.generation import (
    CandidateSpec,
    DraftState,
    OverlapGenerationStats,
    _fused_verify_and_drafts,
    _normal_draft,
    _normal_verify,
    _set_diffusion_lm,
)
from method.margin_risk_multi_overlap_linearspec.hybrid import repeat_dynamic_cache
from method.margin_risk_multi_overlap_linearspec.segmented_lora import install_segmented_lora


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", default="/data1/linyewei/models/Nemotron-Labs-Diffusion-8B"
    )
    parser.add_argument("--lora-path", default="")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    return parser.parse_args()


def choose_alternative(
    original: int, vocab_size: int, excluded: tuple[int, ...]
) -> int:
    alternative = (original + 1) % vocab_size
    while alternative in set(excluded) | {original}:
        alternative = (alternative + 1) % vocab_size
    return alternative


def forced_spec(
    draft: DraftState,
    rank: int,
    position: int,
    vocab_size: int,
    excluded: tuple[int, ...],
) -> CandidateSpec:
    return CandidateSpec(
        rank=rank,
        position=position,
        alternative_token=choose_alternative(
            int(draft.tokens[position].item()), vocab_size, excluded
        ),
        margin_risk=1.0,
    )


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.block_size < 4:
        raise ValueError("--block-size must be at least 4 for this smoke")
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
    seed = model.diffusion_head(
        prefill.last_hidden_state[:, -1:, :]
    ).argmax(dim=-1)
    mask_id = int(model.config.mask_token_id)
    eos_id = tokenizer.eos_token_id
    excluded = tuple(token for token in (mask_id, eos_id) if token is not None)
    draft = _normal_draft(
        model,
        seed=seed,
        past_key_values=cache,
        block_length=args.block_size,
        mask_token_id=mask_id,
        excluded_alternative_token_ids=excluded,
        margin_risk_threshold=0.5,
        controller=controller,
        stats=OverlapGenerationStats(),
    )
    vocab_size = int(model.config.vocab_size)
    positions = (1, max(2, args.block_size // 2), args.block_size - 1)
    specs = tuple(
        forced_spec(draft, rank, position, vocab_size, excluded)
        for rank, position in enumerate(positions, start=1)
    )
    forced = DraftState(
        tokens=draft.tokens,
        confidences=draft.confidences,
        top1_top2_margins=draft.top1_top2_margins,
        candidates=specs,
        total_crossing_count=3,
        source="forced_multi_smoke",
    )

    normal_logits, _ = _normal_verify(
        model,
        draft=forced,
        past_key_values=repeat_dynamic_cache(cache, 1),
        controller=controller,
        stats=OverlapGenerationStats(),
    )
    expected = normal_logits.argmax(dim=-1)

    candidate_stats = OverlapGenerationStats()
    candidate_logits, _, candidate_drafts, candidate_new = (
        _fused_verify_and_drafts(
            model,
            draft=forced,
            candidate_specs=specs,
            include_continuation=False,
            past_key_values=cache,
            block_length=args.block_size,
            mask_token_id=mask_id,
            excluded_alternative_token_ids=excluded,
            margin_risk_threshold=0.5,
            controller=controller,
            stats=candidate_stats,
        )
    )
    if not torch.equal(candidate_logits.argmax(dim=-1), expected):
        raise RuntimeError("four-row candidate verifier differs from normal causal verify")
    if candidate_new is not None or set(candidate_drafts) != {1, 2, 3}:
        raise RuntimeError("candidate-only fused path returned incorrect branches")
    for spec in specs:
        if int(candidate_drafts[spec.rank].tokens[0].item()) != spec.alternative_token:
            raise RuntimeError(f"candidate P{spec.rank} prospective seed is not B")

    continuation_stats = OverlapGenerationStats()
    continuation_logits, _, continuation_drafts, continuation_new = (
        _fused_verify_and_drafts(
            model,
            draft=forced,
            candidate_specs=specs[:2],
            include_continuation=True,
            past_key_values=cache,
            block_length=args.block_size,
            mask_token_id=mask_id,
            excluded_alternative_token_ids=excluded,
            margin_risk_threshold=0.5,
            controller=controller,
            stats=continuation_stats,
        )
    )
    if not torch.equal(continuation_logits.argmax(dim=-1), expected):
        raise RuntimeError("four-row continuation verifier differs from normal causal verify")
    if set(continuation_drafts) != {1, 2} or continuation_new is None:
        raise RuntimeError("candidate+continuation fused path returned incorrect branches")
    if int(cache.get_seq_length()) != cache_length:
        raise RuntimeError("fused paths mutated canonical prefill cache")

    print(
        json.dumps(
            {
                "ok": True,
                "block_size": args.block_size,
                "cache_length_unchanged": cache_length,
                "verifier_argmax_equal": True,
                "candidate_positions": list(positions),
                "candidate_only": {
                    "rows": candidate_stats.processed_rows,
                    "computed_query_tokens": candidate_stats.processed_query_tokens,
                    "valid_query_tokens": candidate_stats.valid_query_tokens,
                    "padding_query_tokens": candidate_stats.padding_query_tokens,
                },
                "candidate_plus_continuation": {
                    "rows": continuation_stats.processed_rows,
                    "computed_query_tokens": continuation_stats.processed_query_tokens,
                    "valid_query_tokens": continuation_stats.valid_query_tokens,
                    "padding_query_tokens": continuation_stats.padding_query_tokens,
                    "continuation_seed": int(continuation_new.tokens[0].item()),
                },
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
