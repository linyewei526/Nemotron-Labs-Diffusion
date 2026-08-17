#!/usr/bin/env python3
"""Confidence-triggered all-MASK redraft LinearSpec generation.

The causal verifier row is fused with a second row of the form
``verified_prefix_before_trigger + MASK * L``.  The second row autonomously
redrafts the suspected position and L-1 following positions.  Its output is
only reusable when it matches every trustworthy verifier token from the
trigger through the verifier correction/bonus token.  Output tokens and the
canonical KV cache always come exclusively from the causal verifier row.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass
from typing import Optional

import torch
from transformers.cache_utils import DynamicCache

from .hybrid import (
    build_mask_redraft_attention_mask,
    repeat_dynamic_cache,
    select_and_crop_cache,
)
from .segmented_lora import SegmentedLoraController


@dataclass
class DraftState:
    """A verifier-ready draft whose first token is the uncommitted AR seed."""

    tokens: torch.Tensor
    logits: torch.Tensor
    confidences: torch.Tensor
    candidate_position: Optional[int]
    candidate_drop_pct: Optional[float]
    source: str


@dataclass
class RedraftProposal:
    """Full L-token autonomous proposal beginning at the trigger position."""

    tokens: torch.Tensor
    logits: torch.Tensor
    trigger_position: int


@dataclass
class ReuseDecision:
    reusable: bool
    reason: str
    retained_offset: Optional[int] = None


@dataclass
class MaskRedraftGenerationStats:
    physical_nfe: int = 0
    processed_rows: int = 0
    processed_query_tokens: int = 0
    rounds: int = 0
    draft_length_sum: int = 0
    partial_draft_rounds: int = 0
    normal_draft_forwards: int = 0
    normal_verify_forwards: int = 0
    fused_verify_redraft_forwards: int = 0
    rounds_without_candidate: int = 0
    redraft_attempts: int = 0
    redraft_verified_hits: int = 0
    redraft_reuse_hits: int = 0
    redraft_saved_draft_forwards: int = 0
    redraft_direct_trigger_hits: int = 0
    redraft_downstream_correction_hits: int = 0
    redraft_full_block_bonus_hits: int = 0
    redraft_discarded_before_trigger: int = 0
    redraft_discarded_trigger_token_mismatch: int = 0
    redraft_discarded_before_correction: int = 0
    redraft_discarded_correction_mismatch: int = 0
    redraft_discarded_eos: int = 0
    redraft_discarded_thinking_budget: int = 0
    redraft_discarded_generation_end: int = 0
    redraft_skipped_no_future_round: int = 0
    redraft_skipped_context_limit: int = 0
    candidate_position_sum: int = 0
    retained_draft_tokens_sum: int = 0
    retained_draft_tokens_min: Optional[int] = None
    retained_draft_tokens_max: int = 0
    full_length_reuses: int = 0
    partial_length_reuses: int = 0
    prospective_query_tokens: int = 0

    def record_forward(self, *, rows: int, query_tokens_per_row: int) -> None:
        self.physical_nfe += 1
        self.processed_rows += int(rows)
        self.processed_query_tokens += int(rows) * int(query_tokens_per_row)

    def record_draft_length(self, length: int, configured_length: int) -> None:
        self.draft_length_sum += int(length)
        if length < configured_length:
            self.partial_draft_rounds += 1

    def record_retained_length(self, length: int, configured_length: int) -> None:
        value = int(length)
        self.retained_draft_tokens_sum += value
        self.retained_draft_tokens_min = (
            value
            if self.retained_draft_tokens_min is None
            else min(self.retained_draft_tokens_min, value)
        )
        self.retained_draft_tokens_max = max(self.retained_draft_tokens_max, value)
        if value == configured_length:
            self.full_length_reuses += 1
        else:
            self.partial_length_reuses += 1

    def to_dict(self) -> dict[str, int | float | None]:
        payload: dict[str, int | float | None] = asdict(self)
        payload["redraft_hit_rate"] = (
            self.redraft_reuse_hits / self.redraft_attempts
            if self.redraft_attempts
            else None
        )
        payload["redraft_verified_hit_rate"] = (
            self.redraft_verified_hits / self.redraft_attempts
            if self.redraft_attempts
            else None
        )
        payload["average_candidate_position"] = (
            self.candidate_position_sum / self.redraft_attempts
            if self.redraft_attempts
            else None
        )
        payload["average_draft_length"] = (
            self.draft_length_sum / self.rounds if self.rounds else None
        )
        payload["average_retained_draft_length"] = (
            self.retained_draft_tokens_sum / self.redraft_reuse_hits
            if self.redraft_reuse_hits
            else None
        )
        payload["saved_draft_fraction_of_rounds"] = (
            self.redraft_saved_draft_forwards / self.rounds
            if self.rounds
            else None
        )
        return payload


def _set_diffusion_lm(model, enabled: bool) -> None:
    for layer in model.encoder.layers:
        if hasattr(layer.self_attn, "diffusion_lm"):
            layer.self_attn.diffusion_lm = bool(enabled)


def _controller_context(
    controller: Optional[SegmentedLoraController],
    *,
    all_tokens: bool = False,
    mask: Optional[torch.Tensor] = None,
):
    if controller is None:
        return nullcontext()
    if all_tokens:
        return controller.use_all()
    if mask is not None:
        return controller.use_mask(mask)
    return controller.disabled()


def _analyze_draft(
    *,
    logits: torch.Tensor,
    tokens: torch.Tensor,
    mask_token_id: int,
    drop_pct_threshold: float,
    source: str,
) -> DraftState:
    """Analyze one draft, resetting confidence history at its seed."""

    if logits.ndim != 2 or tokens.ndim != 1 or logits.shape[0] != tokens.shape[0]:
        raise ValueError("Draft logits/tokens must have shapes [length,vocab] and [length]")
    clean = logits.detach().float().clone()
    if 0 <= mask_token_id < clean.shape[-1]:
        clean[:, mask_token_id] = -torch.inf
    selected = tokens.long()
    selected_logits = clean.gather(-1, selected.unsqueeze(-1)).squeeze(-1)
    confidences = torch.exp(selected_logits - torch.logsumexp(clean, dim=-1))
    confidences = torch.where(
        selected == mask_token_id, torch.zeros_like(confidences), confidences
    )

    candidate_position: Optional[int] = None
    candidate_drop: Optional[float] = None
    prefix_sum = 0.0
    prefix_count = 0
    values = confidences.detach().cpu().tolist()
    # Position zero is the verifier-provided seed. Position one has no earlier
    # draft-token confidence and therefore cannot define a prefix drop.
    for position in range(1, tokens.shape[0]):
        confidence = float(values[position])
        if prefix_count:
            prefix_mean = prefix_sum / prefix_count
            drop_pct = 1.0 - confidence / prefix_mean if prefix_mean != 0 else float("-inf")
            if drop_pct > drop_pct_threshold:
                candidate_position = position
                candidate_drop = drop_pct
                break
        prefix_sum += confidence
        prefix_count += 1

    return DraftState(
        tokens=tokens.detach(),
        logits=logits.detach(),
        confidences=confidences.detach(),
        candidate_position=candidate_position,
        candidate_drop_pct=candidate_drop,
        source=source,
    )


def _normal_draft(
    model,
    *,
    seed: torch.Tensor,
    past_key_values: DynamicCache,
    block_length: int,
    mask_token_id: int,
    drop_pct_threshold: float,
    controller: Optional[SegmentedLoraController],
    stats: MaskRedraftGenerationStats,
) -> DraftState:
    block = torch.full(
        (1, block_length),
        mask_token_id,
        dtype=torch.long,
        device=seed.device,
    )
    block[0, 0] = int(seed.item())
    _set_diffusion_lm(model, True)
    with _controller_context(controller, all_tokens=True):
        output = model.encoder(
            input_ids=block,
            past_key_values=past_key_values,
            use_cache=False,
        )
    stats.record_forward(rows=1, query_tokens_per_row=block_length)
    stats.normal_draft_forwards += 1
    logits = model.diffusion_head(output.last_hidden_state)[0]
    proposed = logits.argmax(dim=-1)
    draft_tokens = block[0].clone()
    is_mask = draft_tokens == mask_token_id
    draft_tokens[is_mask] = proposed[is_mask]
    return _analyze_draft(
        logits=logits,
        tokens=draft_tokens,
        mask_token_id=mask_token_id,
        drop_pct_threshold=drop_pct_threshold,
        source="normal",
    )


def _normal_verify(
    model,
    *,
    draft: DraftState,
    past_key_values: DynamicCache,
    controller: Optional[SegmentedLoraController],
    stats: MaskRedraftGenerationStats,
) -> tuple[torch.Tensor, DynamicCache]:
    verify_length = int(draft.tokens.shape[0])
    _set_diffusion_lm(model, False)
    with _controller_context(controller):
        output = model.encoder(
            input_ids=draft.tokens.unsqueeze(0),
            past_key_values=past_key_values,
            use_cache=True,
            use_causal_mask=True,
        )
    stats.record_forward(rows=1, query_tokens_per_row=verify_length)
    stats.normal_verify_forwards += 1
    return model.diffusion_head(output.last_hidden_state)[0], output.past_key_values


def _fused_verify_and_redraft(
    model,
    *,
    draft: DraftState,
    past_key_values: DynamicCache,
    prospective_length: int,
    mask_token_id: int,
    controller: Optional[SegmentedLoraController],
    stats: MaskRedraftGenerationStats,
) -> tuple[torch.Tensor, DynamicCache, RedraftProposal]:
    trigger = draft.candidate_position
    if trigger is None:
        raise ValueError("Fused redraft requires a confidence-drop trigger")
    verify_length = int(draft.tokens.shape[0])
    cache_length = int(past_key_values.get_seq_length())
    query_length = trigger + prospective_length
    device = draft.tokens.device

    fused_ids = torch.full(
        (2, query_length), mask_token_id, dtype=torch.long, device=device
    )
    fused_ids[0, :verify_length] = draft.tokens
    fused_ids[1, :trigger] = draft.tokens[:trigger]

    attention_mask = build_mask_redraft_attention_mask(
        cache_length=cache_length,
        verify_length=verify_length,
        prospective_length=prospective_length,
        trigger_position=trigger,
        device=device,
    )
    route = torch.zeros((2, query_length, 1), dtype=torch.bool, device=device)
    route[1, trigger:] = True
    fused_cache = repeat_dynamic_cache(past_key_values, 2)
    cache_position = torch.arange(
        cache_length, cache_length + query_length, dtype=torch.long, device=device
    )
    position_ids = cache_position.unsqueeze(0).expand(2, -1)

    # A single global diffusion_lm value cannot differ by batch row. Keep it
    # disabled and express both attention regimes in the explicit 4D mask.
    _set_diffusion_lm(model, False)
    with _controller_context(controller, mask=route):
        output = model.encoder(
            input_ids=fused_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=fused_cache,
            use_cache=True,
            cache_position=cache_position,
            use_causal_mask=True,
        )
    stats.record_forward(rows=2, query_tokens_per_row=query_length)
    stats.fused_verify_redraft_forwards += 1
    stats.prospective_query_tokens += prospective_length
    logits = model.diffusion_head(output.last_hidden_state)
    verify_logits = logits[0, :verify_length]
    redraft_logits = logits[1, trigger : trigger + prospective_length]
    redraft_tokens = redraft_logits.argmax(dim=-1)
    proposal = RedraftProposal(
        tokens=redraft_tokens.detach(),
        logits=redraft_logits.detach(),
        trigger_position=int(trigger),
    )
    return verify_logits, output.past_key_values, proposal


def _greedy_tokens(logits: torch.Tensor) -> torch.Tensor:
    return logits.argmax(dim=-1)


def _count_matched_draft_tokens(ar_tokens: torch.Tensor, draft_tokens: torch.Tensor) -> int:
    """Count the longest draft prefix accepted under LinearSpec's shift."""

    if ar_tokens.ndim != 1 or draft_tokens.ndim != 1:
        raise ValueError("AR and draft tokens must be one-dimensional")
    if ar_tokens.shape[0] != draft_tokens.shape[0]:
        raise ValueError("AR and draft blocks must have the same length")
    matched = 0
    for index in range(draft_tokens.shape[0] - 1):
        if int(ar_tokens[index].item()) == int(draft_tokens[index + 1].item()):
            matched += 1
        else:
            break
    return matched


def _decide_redraft_reuse(
    *,
    proposal_tokens: torch.Tensor,
    ar_tokens: torch.Tensor,
    trigger_position: int,
    emitted_tokens: int,
) -> ReuseDecision:
    """Apply the unified trustworthy-segment match rule.

    ``emitted_tokens`` is ``matched + 1``.  It is the first row-0 mismatch
    position when a mismatch exists, and the verifier bonus position when the
    complete variable-length draft matched.
    """

    if proposal_tokens.ndim != 1 or ar_tokens.ndim != 1:
        raise ValueError("proposal_tokens and ar_tokens must be one-dimensional")
    p = int(trigger_position)
    m = int(emitted_tokens)
    if not 1 <= p < ar_tokens.shape[0]:
        raise ValueError("trigger_position must be inside the verifier draft")
    if not 1 <= m <= ar_tokens.shape[0]:
        raise ValueError("emitted_tokens must be in [1, verify_length]")
    if m < p:
        return ReuseDecision(False, "before_trigger")

    retained_offset = m - p
    target = ar_tokens[p - 1 : m]
    proposed = proposal_tokens[: target.shape[0]]
    if proposed.shape[0] != target.shape[0]:
        raise ValueError("Autonomous proposal is too short for verifier comparison")
    mismatches = (proposed != target).nonzero(as_tuple=True)[0]
    if len(mismatches) == 0:
        return ReuseDecision(True, "match", retained_offset)
    first = int(mismatches[0].item())
    if first == 0:
        return ReuseDecision(False, "trigger_token_mismatch")
    if first < retained_offset:
        return ReuseDecision(False, "before_correction")
    return ReuseDecision(False, "correction_mismatch")


def _record_reuse_rejection(
    stats: MaskRedraftGenerationStats, decision: ReuseDecision
) -> None:
    field = {
        "before_trigger": "redraft_discarded_before_trigger",
        "trigger_token_mismatch": "redraft_discarded_trigger_token_mismatch",
        "before_correction": "redraft_discarded_before_correction",
        "correction_mismatch": "redraft_discarded_correction_mismatch",
    }.get(decision.reason)
    if field is None:
        raise ValueError(f"Unexpected rejection reason: {decision.reason}")
    setattr(stats, field, int(getattr(stats, field)) + 1)


@torch.no_grad()
def mask_redraft_linear_spec_generate(
    model,
    prompt_ids: torch.Tensor,
    *,
    max_new_tokens: int = 128,
    block_length: int = 16,
    drop_pct_threshold: float = 0.15,
    temperature: float = 0.0,
    draft_threshold: float = 0.0,
    mask_token_id: Optional[int] = None,
    eos_token_id: Optional[int] = None,
    max_thinking_tokens: Optional[int] = None,
    end_think_token_id: Optional[int] = None,
    lora_controller: Optional[SegmentedLoraController] = None,
) -> tuple[torch.Tensor, MaskRedraftGenerationStats]:
    """Generate with autonomous all-MASK redraft overlap."""

    if prompt_ids.shape[0] != 1:
        raise ValueError("Mask-redraft LinearSpec requires request batch_size == 1")
    if block_length < 2 or max_new_tokens <= 0:
        raise ValueError("block_length must be >=2 and max_new_tokens must be positive")
    if not 0.0 <= drop_pct_threshold < 1.0:
        raise ValueError("drop_pct_threshold must be in [0,1)")
    if temperature != 0.0:
        raise ValueError("The first isolated mask-redraft experiment supports temperature=0 only")
    if draft_threshold != 0.0:
        raise ValueError("The first isolated mask-redraft experiment requires draft_threshold=0")

    token_mask_id = (
        int(mask_token_id) if mask_token_id is not None else int(model.config.mask_token_id)
    )
    if eos_token_id is None:
        eos_token_id = getattr(model.config, "eos_token_id", None)
    device = prompt_ids.device
    stats = MaskRedraftGenerationStats()

    _set_diffusion_lm(model, False)
    with _controller_context(lora_controller):
        output = model.encoder(
            input_ids=prompt_ids,
            past_key_values=DynamicCache(),
            use_cache=True,
            use_causal_mask=True,
        )
    stats.record_forward(rows=1, query_tokens_per_row=int(prompt_ids.shape[1]))
    past_key_values = output.past_key_values
    next_token = _greedy_tokens(
        model.diffusion_head(output.last_hidden_state[:, -1:, :]).squeeze(1)
    ).unsqueeze(1)
    if eos_token_id is not None and int(next_token.item()) == int(eos_token_id):
        return torch.cat([prompt_ids, next_token], dim=1), stats

    generated = [next_token]
    total_gen = 1
    prefetched: Optional[DraftState] = None
    max_positions = int(getattr(model.config, "max_position_embeddings", 2**31 - 1))

    while total_gen < max_new_tokens:
        stats.rounds += 1
        if prefetched is not None:
            if int(prefetched.tokens[0].item()) != int(next_token.item()):
                raise RuntimeError("Retained redraft seed does not match verifier seed")
            draft = prefetched
            prefetched = None
            stats.redraft_saved_draft_forwards += 1
        else:
            draft = _normal_draft(
                model,
                seed=next_token,
                past_key_values=past_key_values,
                block_length=block_length,
                mask_token_id=token_mask_id,
                drop_pct_threshold=drop_pct_threshold,
                controller=lora_controller,
                stats=stats,
            )

        verify_length = int(draft.tokens.shape[0])
        if not 2 <= verify_length <= block_length:
            raise RuntimeError(
                f"Invalid variable draft length {verify_length}; expected [2,{block_length}]"
            )
        stats.record_draft_length(verify_length, block_length)
        cache_length = int(past_key_values.get_seq_length())
        trigger = draft.candidate_position
        can_redraft = trigger is not None
        if not can_redraft:
            stats.rounds_without_candidate += 1
        elif total_gen + int(trigger) >= max_new_tokens:
            can_redraft = False
            stats.redraft_skipped_no_future_round += 1
        elif cache_length + int(trigger) + block_length > max_positions:
            can_redraft = False
            stats.redraft_skipped_context_limit += 1

        fused_cache: Optional[DynamicCache] = None
        proposal: Optional[RedraftProposal] = None
        if can_redraft:
            stats.redraft_attempts += 1
            stats.candidate_position_sum += int(trigger)
            verify_logits, fused_cache, proposal = _fused_verify_and_redraft(
                model,
                draft=draft,
                past_key_values=past_key_values,
                prospective_length=block_length,
                mask_token_id=token_mask_id,
                controller=lora_controller,
                stats=stats,
            )
            ar_tokens = _greedy_tokens(verify_logits)
        else:
            verify_logits, past_key_values = _normal_verify(
                model,
                draft=draft,
                past_key_values=past_key_values,
                controller=lora_controller,
                stats=stats,
            )
            ar_tokens = _greedy_tokens(verify_logits)

        matched = _count_matched_draft_tokens(ar_tokens, draft.tokens)
        emitted = matched + 1
        if fused_cache is not None:
            past_key_values = select_and_crop_cache(
                fused_cache,
                batch_index=0,
                max_length=cache_length + emitted,
            )
        else:
            past_key_values.crop(cache_length + emitted)

        emitted_tokens = ar_tokens[:emitted].unsqueeze(0)
        generated.append(emitted_tokens)
        total_gen += emitted
        next_token = ar_tokens[emitted - 1 : emitted].view(1, 1)

        eos_hit = False
        if eos_token_id is not None:
            eos_positions = (emitted_tokens[0] == int(eos_token_id)).nonzero(as_tuple=True)[0]
            if len(eos_positions) > 0:
                eos_hit = True
                actual = int(eos_positions[0].item()) + 1
                generated[-1] = emitted_tokens[:, :actual]
                total_gen = total_gen - emitted + actual

        decision: Optional[ReuseDecision] = None
        if can_redraft and proposal is not None and trigger is not None:
            decision = _decide_redraft_reuse(
                proposal_tokens=proposal.tokens,
                ar_tokens=ar_tokens,
                trigger_position=int(trigger),
                emitted_tokens=emitted,
            )
            if decision.reusable:
                stats.redraft_verified_hits += 1
                if emitted == int(trigger):
                    stats.redraft_direct_trigger_hits += 1
                elif emitted == verify_length and matched == verify_length - 1:
                    stats.redraft_full_block_bonus_hits += 1
                else:
                    stats.redraft_downstream_correction_hits += 1
            else:
                _record_reuse_rejection(stats, decision)

        if eos_hit:
            if decision is not None and decision.reusable:
                stats.redraft_discarded_eos += 1
            break

        forced_thinking_seed = False
        if end_think_token_id is not None and max_thinking_tokens is not None:
            if total_gen > max_thinking_tokens:
                all_gen = torch.cat(generated, dim=1)
                if not bool((all_gen == int(end_think_token_id)).any().item()):
                    next_token = torch.tensor([[int(end_think_token_id)]], device=device)
                    forced_thinking_seed = True

        if decision is not None and decision.reusable and proposal is not None:
            if forced_thinking_seed:
                stats.redraft_discarded_thinking_budget += 1
            elif total_gen >= max_new_tokens:
                stats.redraft_discarded_generation_end += 1
            else:
                offset = decision.retained_offset
                if offset is None:
                    raise RuntimeError("Reusable decision is missing retained_offset")
                retained_tokens = proposal.tokens[offset:].clone()
                retained_logits = proposal.logits[offset:].clone()
                if int(retained_tokens[0].item()) != int(next_token.item()):
                    raise RuntimeError("Verifier correction does not match retained redraft seed")
                prefetched = _analyze_draft(
                    logits=retained_logits,
                    tokens=retained_tokens,
                    mask_token_id=token_mask_id,
                    drop_pct_threshold=drop_pct_threshold,
                    source="mask_redraft_retained",
                )
                retained_length = int(retained_tokens.shape[0])
                stats.redraft_reuse_hits += 1
                stats.record_retained_length(retained_length, block_length)

        if total_gen >= max_new_tokens:
            break

    all_generated = torch.cat(generated, dim=1)
    output_ids = torch.cat([prompt_ids, all_generated], dim=1)
    if stats.physical_nfe <= 0:
        raise RuntimeError("Invalid physical NFE accounting")
    return output_ids, stats
