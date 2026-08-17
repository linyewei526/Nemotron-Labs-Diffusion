#!/usr/bin/env python3
"""Trace-enabled equivalent of the model remote code's linear_spec_generate()."""

from __future__ import annotations

import logging
from typing import Optional

import torch
from transformers.cache_utils import DynamicCache

from .confidence_trace import NativeLinearSpecConfidenceTracer


logger = logging.getLogger(__name__)


def _crop_dynamic_cache(past_key_values: DynamicCache, max_length: int) -> None:
    if hasattr(past_key_values, "crop"):
        past_key_values.crop(max_length)
        return
    for layer_idx in range(len(past_key_values)):
        past_key_values.key_cache[layer_idx] = past_key_values.key_cache[layer_idx][
            :, :, :max_length
        ]
        past_key_values.value_cache[layer_idx] = past_key_values.value_cache[
            layer_idx
        ][:, :, :max_length]
    past_key_values._seen_tokens = max_length


@torch.no_grad()
def linear_spec_generate_with_confidence(
    model,
    prompt_ids: torch.Tensor,
    *,
    max_new_tokens: int = 128,
    block_length: int = 32,
    temperature: float = 0.0,
    mask_token_id: Optional[int] = None,
    eos_token_id: Optional[int] = None,
    max_thinking_tokens: Optional[int] = None,
    end_think_token_id: Optional[int] = None,
    threshold: float = 0.0,
    tracer: Optional[NativeLinearSpecConfidenceTracer] = None,
    request_id: str = "",
    mode: str = "linearspec_lora",
):
    """Run native LinearSpec while tracing logits at token commit time.

    Token selection, adapter toggling, cache cropping, EOS handling and NFE
    accounting intentionally mirror the model repository implementation.
    """
    if prompt_ids.shape[0] != 1:
        raise ValueError("Linear speculative decoding requires batch_size == 1")
    if block_length <= 0 or max_new_tokens <= 0:
        raise ValueError("block_length and max_new_tokens must be positive")

    token_mask_id = (
        int(mask_token_id)
        if mask_token_id is not None
        else int(model.config.mask_token_id)
    )
    if eos_token_id is None:
        eos_token_id = getattr(model.config, "eos_token_id", None)
    device = prompt_ids.device

    def set_diffusion_lm(value: bool) -> None:
        for layer in model.encoder.layers:
            if hasattr(layer.self_attn, "diffusion_lm"):
                layer.self_attn.diffusion_lm = value

    def toggle_adapters(enable: bool) -> None:
        for module in model.modules():
            if hasattr(module, "_disable_adapters"):
                module._disable_adapters = not enable

    set_diffusion_lm(False)
    toggle_adapters(False)
    enc_out = model.encoder(
        input_ids=prompt_ids,
        past_key_values=DynamicCache(),
        use_cache=True,
        use_causal_mask=True,
    )
    past_key_values = enc_out.past_key_values
    last_logit = model.diffusion_head(enc_out.last_hidden_state[:, -1:, :]).squeeze(1)
    nfe = 1
    if temperature > 0:
        next_token = torch.multinomial(
            torch.softmax(last_logit / temperature, dim=-1), num_samples=1
        )
    else:
        next_token = torch.argmax(last_logit, dim=-1, keepdim=True)

    if eos_token_id is not None and next_token.item() == eos_token_id:
        return torch.cat([prompt_ids, next_token], dim=1), nfe

    generated = [next_token]
    total_gen = 1
    round_index = 0
    tracing_enabled = tracer is not None

    while total_gen < max_new_tokens:
        generation_offset = total_gen
        cache_len = past_key_values.get_seq_length()
        block = torch.full(
            (1, block_length),
            token_mask_id,
            dtype=torch.long,
            device=device,
        )
        block[0, 0] = next_token.item()

        set_diffusion_lm(True)
        toggle_adapters(True)
        committed_logits: Optional[torch.Tensor] = None
        draft_forward_passes = 0
        while True:
            is_mask = block == token_mask_id
            if not is_mask.any():
                break
            enc_out = model.encoder(
                input_ids=block,
                past_key_values=past_key_values,
                use_cache=False,
            )
            nfe += 1
            draft_forward_passes += 1
            draft_logits = model.diffusion_head(enc_out.last_hidden_state)
            if committed_logits is None:
                committed_logits = torch.zeros_like(draft_logits)

            if temperature > 0:
                draft_probs = torch.softmax(draft_logits / temperature, dim=-1)
                draft_tokens = torch.multinomial(
                    draft_probs.view(-1, draft_probs.shape[-1]), num_samples=1
                ).view(1, block_length)
            else:
                draft_tokens = draft_logits.argmax(dim=-1)
                draft_probs = torch.softmax(draft_logits, dim=-1)

            if threshold > 0:
                draft_conf = torch.gather(
                    draft_probs, -1, draft_tokens.unsqueeze(-1)
                ).squeeze(-1)
                draft_conf = torch.where(is_mask, draft_conf, -torch.inf)
                unmask = draft_conf >= threshold
                if not unmask.any():
                    best_idx = draft_conf.view(-1).argmax()
                    unmask = torch.zeros_like(is_mask, dtype=torch.bool)
                    unmask.view(-1)[best_idx] = True
                committed_logits[unmask] = draft_logits.detach()[unmask]
                block[unmask] = draft_tokens[unmask]
            else:
                committed_logits[is_mask] = draft_logits.detach()[is_mask]
                block[is_mask] = draft_tokens[is_mask]
                break

        if committed_logits is None:
            raise RuntimeError("LinearSpec draft produced no logits")

        set_diffusion_lm(False)
        toggle_adapters(False)
        enc_out = model.encoder(
            input_ids=block,
            past_key_values=past_key_values,
            use_cache=True,
            use_causal_mask=True,
        )
        past_key_values = enc_out.past_key_values
        nfe += 1
        verify_logits = model.diffusion_head(enc_out.last_hidden_state)
        if temperature > 0:
            ar_tokens = torch.multinomial(
                torch.softmax(verify_logits / temperature, dim=-1).view(
                    -1, verify_logits.shape[-1]
                ),
                num_samples=1,
            ).view(1, block_length)
        else:
            ar_tokens = verify_logits.argmax(dim=-1)

        matched = 0
        for index in range(block_length - 1):
            if ar_tokens[0, index].item() == block[0, index + 1].item():
                matched += 1
            else:
                break
        emitted = matched + 1
        accepted_toks = ar_tokens[:, :emitted]
        generated.append(accepted_toks)
        total_gen += emitted

        _crop_dynamic_cache(past_key_values, cache_len + emitted)
        next_token = ar_tokens[:, emitted - 1 : emitted]

        eos_hit = False
        accepted_draft_tokens = matched
        if eos_token_id is not None:
            eos_positions = (accepted_toks[0] == eos_token_id).nonzero(as_tuple=True)[0]
            if len(eos_positions) > 0:
                eos_hit = True
                first_eos = int(eos_positions[0].item())
                actual_emitted = first_eos + 1
                generated[-1] = accepted_toks[:, :actual_emitted]
                total_gen = total_gen - emitted + actual_emitted
                emitted = actual_emitted
                accepted_draft_tokens = min(matched, actual_emitted)

        if tracing_enabled and tracer is not None:
            try:
                tracer.record_round(
                    draft_logits_by_position=committed_logits[0],
                    draft_tokens=block[0],
                    ar_tokens=ar_tokens[0],
                    request_id=request_id,
                    round_index=round_index,
                    generation_offset=generation_offset,
                    block_length=block_length,
                    matched_draft_tokens=matched,
                    accepted_draft_tokens=accepted_draft_tokens,
                    emitted_tokens=emitted,
                    eos_hit=eos_hit,
                    mask_id=token_mask_id,
                    draft_forward_passes=draft_forward_passes,
                    nfe_after_round=nfe,
                    threshold=float(threshold),
                    temperature=float(temperature),
                    mode=mode,
                )
            except Exception:
                logger.exception("Native LinearSpec confidence trace failed; disabling it")
                tracing_enabled = False

        round_index += 1
        if eos_hit:
            break

        if end_think_token_id is not None and max_thinking_tokens is not None:
            if total_gen > max_thinking_tokens:
                all_gen = torch.cat(generated, dim=1)
                if not (all_gen == end_think_token_id).any():
                    next_token = torch.tensor([[end_think_token_id]], device=device)

        if total_gen >= max_new_tokens:
            break

    all_generated = torch.cat(generated, dim=1)
    output_ids = torch.cat([prompt_ids, all_generated], dim=1)
    return output_ids, nfe
