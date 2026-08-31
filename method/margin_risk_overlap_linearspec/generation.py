#!/usr/bin/env python3
"""Fixed-margin-risk full-block overlap LinearSpec generation.

This implementation is deliberately independent from the model repository's
``linear_spec_generate`` method.  A fused call contains a padded causal
verifier row and a hybrid causal-prefix/bidirectional-draft row.  Only causal
verification commits output or canonical KV state.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from typing import Optional

import torch
from transformers.cache_utils import DynamicCache

from .hybrid import (
    build_hybrid_attention_mask,
    repeat_dynamic_cache,
    select_and_crop_cache,
)
from .segmented_lora import SegmentedLoraController


@dataclass
class DraftState:
    tokens: torch.Tensor
    confidences: torch.Tensor
    top1_top2_margins: torch.Tensor
    candidate_position: Optional[int]
    alternative_token: Optional[int]
    candidate_margin_risk: Optional[float]
    source: str


OUTCOME_STATES = (
    "before_candidate_error",
    "candidate_fixed_by_alternative",
    "candidate_wrong_alternative",
    "after_candidate_error",
    "full_block_bonus",
)


@dataclass
class OutcomeTransitionStats:
    """Per-attempt outcome and its paired next-verify acceptance statistics."""

    count: int = 0
    current_accept_sum: int = 0
    next_count: int = 0
    paired_current_accept_sum: int = 0
    next_accept_sum: int = 0
    next_minus_current_sum: int = 0

    def record_current(self, accept_length: int) -> None:
        self.count += 1
        self.current_accept_sum += int(accept_length)

    def record_next(self, current_accept_length: int, next_accept_length: int) -> None:
        self.next_count += 1
        self.paired_current_accept_sum += int(current_accept_length)
        self.next_accept_sum += int(next_accept_length)
        self.next_minus_current_sum += int(next_accept_length) - int(current_accept_length)

    def to_dict(self, attempts: int) -> dict[str, int | float | None]:
        return {
            **asdict(self),
            "share_of_attempts": self.count / attempts if attempts else None,
            "current_accept_avg": self.current_accept_sum / self.count if self.count else None,
            "next_coverage": self.next_count / self.count if self.count else None,
            "paired_current_accept_avg": (
                self.paired_current_accept_sum / self.next_count if self.next_count else None
            ),
            "next_accept_avg": self.next_accept_sum / self.next_count if self.next_count else None,
            "next_minus_current_avg": (
                self.next_minus_current_sum / self.next_count if self.next_count else None
            ),
        }


@dataclass
class OverlapGenerationStats:
    physical_nfe: int = 0
    processed_rows: int = 0
    processed_query_tokens: int = 0
    rounds: int = 0
    normal_draft_forwards: int = 0
    normal_verify_forwards: int = 0
    fused_verify_draft_forwards: int = 0
    rounds_without_candidate: int = 0
    prefetch_attempts: int = 0
    prefetch_verified_hits: int = 0
    prefetch_hits: int = 0
    prefetch_saved_draft_forwards: int = 0
    prefetch_discarded_before_candidate: int = 0
    prefetch_discarded_candidate_accepted: int = 0
    prefetch_discarded_wrong_b: int = 0
    prefetch_discarded_eos: int = 0
    prefetch_discarded_thinking_budget: int = 0
    prefetch_skipped_no_future_round: int = 0
    prefetch_skipped_context_limit: int = 0
    prefetch_skipped_thinking_budget: int = 0
    candidate_position_sum: int = 0
    prospective_query_tokens: int = 0
    outcome_states: dict[str, OutcomeTransitionStats] = field(
        default_factory=lambda: {name: OutcomeTransitionStats() for name in OUTCOME_STATES}
    )

    def record_forward(self, *, rows: int, query_tokens_per_row: int) -> None:
        self.physical_nfe += 1
        self.processed_rows += int(rows)
        self.processed_query_tokens += int(rows) * int(query_tokens_per_row)

    def record_outcome(self, state: str, accept_length: int) -> None:
        if state not in self.outcome_states:
            raise ValueError(f"unknown overlap outcome state: {state}")
        self.outcome_states[state].record_current(accept_length)

    def record_next_accept(
        self, state: str, current_accept_length: int, next_accept_length: int
    ) -> None:
        if state not in self.outcome_states:
            raise ValueError(f"unknown overlap outcome state: {state}")
        self.outcome_states[state].record_next(current_accept_length, next_accept_length)

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = asdict(self)
        payload["outcome_states"] = {
            name: self.outcome_states[name].to_dict(self.prefetch_attempts)
            for name in OUTCOME_STATES
        }
        payload["prefetch_hit_rate"] = (
            self.prefetch_hits / self.prefetch_attempts if self.prefetch_attempts else None
        )
        payload["prefetch_verified_hit_rate"] = (
            self.prefetch_verified_hits / self.prefetch_attempts
            if self.prefetch_attempts
            else None
        )
        payload["average_candidate_position"] = (
            self.candidate_position_sum / self.prefetch_attempts
            if self.prefetch_attempts
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
    excluded_alternative_token_ids: tuple[int, ...] = (),
    margin_risk_threshold: float,
    source: str,
) -> DraftState:
    """Find the leftmost strict margin-risk crossing after excluding MASK.

    Position zero is the known seed.  Positions 1..L-1 are draft candidates.
    ``margin_risk = 1 - (P_top1 - P_top2)`` is evaluated independently at
    every candidate position, exactly as in the failure-locator observation.
    """

    if logits.ndim != 2 or tokens.ndim != 1 or logits.shape[0] != tokens.shape[0]:
        raise ValueError("Draft logits/tokens must have shapes [length,vocab] and [length]")
    clean = logits.detach().float().clone()
    if 0 <= mask_token_id < clean.shape[-1]:
        clean[:, mask_token_id] = -torch.inf
    selected = tokens.long()
    selected_logits = clean.gather(-1, selected.unsqueeze(-1)).squeeze(-1)
    log_z = torch.logsumexp(clean, dim=-1)
    confidences = torch.exp(selected_logits - log_z)
    confidences = torch.where(selected == mask_token_id, torch.zeros_like(confidences), confidences)
    top_values = clean.topk(k=2, dim=-1).values
    top_probs = torch.exp(top_values - log_z.unsqueeze(-1))
    margins = top_probs[:, 0] - top_probs[:, 1]
    risks = 1.0 - margins

    candidate_position: Optional[int] = None
    candidate_risk: Optional[float] = None
    risk_values = risks.detach().cpu().tolist()
    for position in range(1, tokens.shape[0]):
        margin_risk = float(risk_values[position])
        if margin_risk > margin_risk_threshold:
            candidate_position = position
            candidate_risk = margin_risk
            break

    alternative_token: Optional[int] = None
    if candidate_position is not None:
        alternative_logits = clean[candidate_position].clone()
        alternative_logits[int(tokens[candidate_position].item())] = -torch.inf
        for token_id in excluded_alternative_token_ids:
            if 0 <= int(token_id) < alternative_logits.shape[-1]:
                alternative_logits[int(token_id)] = -torch.inf
        alternative_token = int(torch.argmax(alternative_logits).item())
        if not torch.isfinite(alternative_logits[alternative_token]):
            alternative_token = None
        elif alternative_token == mask_token_id:
            raise RuntimeError("MASK survived exclusion while selecting alternative token")
    return DraftState(
        tokens=tokens.detach(),
        confidences=confidences.detach(),
        top1_top2_margins=margins.detach(),
        candidate_position=candidate_position,
        alternative_token=alternative_token,
        candidate_margin_risk=candidate_risk,
        source=source,
    )


def _normal_draft(
    model,
    *,
    seed: torch.Tensor,
    past_key_values: DynamicCache,
    block_length: int,
    mask_token_id: int,
    excluded_alternative_token_ids: tuple[int, ...],
    margin_risk_threshold: float,
    controller: Optional[SegmentedLoraController],
    stats: OverlapGenerationStats,
) -> DraftState:
    device = seed.device
    block = torch.full((1, block_length), mask_token_id, dtype=torch.long, device=device)
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
    logits = model.diffusion_head(output.last_hidden_state)
    draft_tokens = logits.argmax(dim=-1)
    is_mask = block == mask_token_id
    block[is_mask] = draft_tokens[is_mask]
    return _analyze_draft(
        logits=logits[0],
        tokens=block[0],
        mask_token_id=mask_token_id,
        excluded_alternative_token_ids=excluded_alternative_token_ids,
        margin_risk_threshold=margin_risk_threshold,
        source="normal",
    )


def _normal_verify(
    model,
    *,
    draft: DraftState,
    past_key_values: DynamicCache,
    controller: Optional[SegmentedLoraController],
    stats: OverlapGenerationStats,
) -> tuple[torch.Tensor, DynamicCache]:
    block_length = int(draft.tokens.shape[0])
    _set_diffusion_lm(model, False)
    with _controller_context(controller):
        output = model.encoder(
            input_ids=draft.tokens.unsqueeze(0),
            past_key_values=past_key_values,
            use_cache=True,
            use_causal_mask=True,
        )
    stats.record_forward(rows=1, query_tokens_per_row=block_length)
    stats.normal_verify_forwards += 1
    return model.diffusion_head(output.last_hidden_state)[0], output.past_key_values


def _fused_verify_and_draft(
    model,
    *,
    draft: DraftState,
    past_key_values: DynamicCache,
    block_length: int,
    mask_token_id: int,
    excluded_alternative_token_ids: tuple[int, ...],
    margin_risk_threshold: float,
    controller: Optional[SegmentedLoraController],
    stats: OverlapGenerationStats,
) -> tuple[torch.Tensor, DynamicCache, DraftState]:
    position = draft.candidate_position
    alternative = draft.alternative_token
    if position is None or alternative is None:
        raise ValueError("Fused path requires a margin-risk candidate and alternative")
    device = draft.tokens.device
    cache_length = int(past_key_values.get_seq_length())
    query_length = position + block_length

    fused_ids = torch.full(
        (2, query_length), mask_token_id, dtype=torch.long, device=device
    )
    fused_ids[0, :block_length] = draft.tokens
    fused_ids[1, :position] = draft.tokens[:position]
    fused_ids[1, position] = alternative

    attention_mask = build_hybrid_attention_mask(
        cache_length=cache_length,
        block_length=block_length,
        candidate_position=position,
        device=device,
    )
    route = torch.zeros((2, query_length, 1), dtype=torch.bool, device=device)
    route[1, position:] = True
    fused_cache = repeat_dynamic_cache(past_key_values, 2)
    cache_position = torch.arange(
        cache_length, cache_length + query_length, dtype=torch.long, device=device
    )
    position_ids = cache_position.unsqueeze(0).expand(2, -1)

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
    stats.fused_verify_draft_forwards += 1
    stats.prospective_query_tokens += block_length
    logits = model.diffusion_head(output.last_hidden_state)
    verify_logits = logits[0, :block_length]

    prospective_ids = fused_ids[1, position : position + block_length].clone()
    prospective_logits = logits[1, position : position + block_length]
    proposed = prospective_logits.argmax(dim=-1)
    prospective_mask = prospective_ids == mask_token_id
    prospective_ids[prospective_mask] = proposed[prospective_mask]
    prospective = _analyze_draft(
        logits=prospective_logits,
        tokens=prospective_ids,
        mask_token_id=mask_token_id,
        excluded_alternative_token_ids=excluded_alternative_token_ids,
        margin_risk_threshold=margin_risk_threshold,
        source="prefetched",
    )
    return verify_logits, output.past_key_values, prospective


def _greedy_tokens(logits: torch.Tensor) -> torch.Tensor:
    return logits.argmax(dim=-1)


def _count_matched_draft_tokens(ar_tokens: torch.Tensor, draft_tokens: torch.Tensor) -> int:
    """Count the longest verified draft prefix under the model's one-token shift."""

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


def _classify_prefetch_outcome(
    *,
    mismatch_position: Optional[int],
    candidate_position: int,
    alternative_token: int,
    ar_tokens: torch.Tensor,
) -> str:
    """Return one of five mutually-exclusive outcomes for an actual attempt."""

    if mismatch_position is None:
        return "full_block_bonus"
    if mismatch_position < candidate_position:
        return "before_candidate_error"
    if mismatch_position > candidate_position:
        return "after_candidate_error"
    alternative_correct = (
        int(ar_tokens[candidate_position - 1].item()) == int(alternative_token)
    )
    return (
        "candidate_fixed_by_alternative"
        if alternative_correct
        else "candidate_wrong_alternative"
    )


@torch.no_grad()
def overlap_linear_spec_generate(
    model,
    prompt_ids: torch.Tensor,
    *,
    max_new_tokens: int = 128,
    block_length: int = 16,
    margin_risk_threshold: float = 0.5,
    temperature: float = 0.0,
    draft_threshold: float = 0.0,
    mask_token_id: Optional[int] = None,
    eos_token_id: Optional[int] = None,
    max_thinking_tokens: Optional[int] = None,
    end_think_token_id: Optional[int] = None,
    lora_controller: Optional[SegmentedLoraController] = None,
) -> tuple[torch.Tensor, OverlapGenerationStats]:
    """Generate with full-block margin-risk-guided verify/draft overlap."""

    if prompt_ids.shape[0] != 1:
        raise ValueError("Overlap LinearSpec currently requires request batch_size == 1")
    if block_length < 2 or max_new_tokens <= 0:
        raise ValueError("block_length must be >=2 and max_new_tokens must be positive")
    if not 0.0 <= margin_risk_threshold <= 1.0:
        raise ValueError("margin_risk_threshold must be in [0,1]")
    if temperature != 0.0:
        raise ValueError("The first isolated overlap experiment supports temperature=0 only")
    if draft_threshold != 0.0:
        raise ValueError("The first isolated overlap experiment requires draft_threshold=0")

    token_mask_id = (
        int(mask_token_id) if mask_token_id is not None else int(model.config.mask_token_id)
    )
    if eos_token_id is None:
        eos_token_id = getattr(model.config, "eos_token_id", None)
    excluded_alternative_token_ids = tuple(
        int(token_id)
        for token_id in (token_mask_id, eos_token_id)
        if token_id is not None
    )
    device = prompt_ids.device
    stats = OverlapGenerationStats()

    # Causal prefill, always base weights.
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
    pending_transition: Optional[tuple[str, int]] = None
    max_positions = int(getattr(model.config, "max_position_embeddings", 2**31 - 1))

    while total_gen < max_new_tokens:
        stats.rounds += 1
        if prefetched is not None:
            if int(prefetched.tokens[0].item()) != int(next_token.item()):
                raise RuntimeError("Prefetched draft seed does not match committed next token")
            draft = prefetched
            prefetched = None
            stats.prefetch_saved_draft_forwards += 1
        else:
            draft = _normal_draft(
                model,
                seed=next_token,
                past_key_values=past_key_values,
                block_length=block_length,
                mask_token_id=token_mask_id,
                excluded_alternative_token_ids=excluded_alternative_token_ids,
                margin_risk_threshold=margin_risk_threshold,
                controller=lora_controller,
                stats=stats,
            )

        cache_length = int(past_key_values.get_seq_length())
        position = draft.candidate_position
        alternative = draft.alternative_token
        can_prefetch = position is not None and alternative is not None
        if not can_prefetch:
            stats.rounds_without_candidate += 1
        elif total_gen + int(position) >= max_new_tokens:
            can_prefetch = False
            stats.prefetch_skipped_no_future_round += 1
        elif cache_length + int(position) + block_length > max_positions:
            can_prefetch = False
            stats.prefetch_skipped_context_limit += 1
        elif end_think_token_id is not None and max_thinking_tokens is not None:
            all_gen = torch.cat(generated, dim=1)
            thinking_already_closed = bool(
                (all_gen == int(end_think_token_id)).any().item()
            )
            if total_gen + int(position) > max_thinking_tokens and not thinking_already_closed:
                can_prefetch = False
                stats.prefetch_skipped_thinking_budget += 1

        fused_cache: Optional[DynamicCache] = None
        prospective: Optional[DraftState] = None
        if can_prefetch:
            stats.prefetch_attempts += 1
            stats.candidate_position_sum += int(position)
            verify_logits, fused_cache, prospective = _fused_verify_and_draft(
                model,
                draft=draft,
                past_key_values=past_key_values,
                block_length=block_length,
                mask_token_id=token_mask_id,
                excluded_alternative_token_ids=excluded_alternative_token_ids,
                margin_risk_threshold=margin_risk_threshold,
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
        accepted = matched + 1
        mismatch_position = matched + 1 if matched < block_length - 1 else None

        # The present verifier is the immediate next verifier for the previous
        # actual overlap attempt, even if this round itself has no candidate.
        if pending_transition is not None:
            previous_state, previous_accept = pending_transition
            stats.record_next_accept(previous_state, previous_accept, accepted)
            pending_transition = None

        if fused_cache is not None:
            past_key_values = select_and_crop_cache(
                fused_cache,
                batch_index=0,
                max_length=cache_length + accepted,
            )
        else:
            past_key_values.crop(cache_length + accepted)

        accepted_toks = ar_tokens[:accepted].unsqueeze(0)
        generated.append(accepted_toks)
        total_gen += accepted
        next_token = ar_tokens[accepted - 1 : accepted].view(1, 1)

        eos_hit = False
        if eos_token_id is not None:
            eos_positions = (accepted_toks[0] == int(eos_token_id)).nonzero(as_tuple=True)[0]
            if len(eos_positions) > 0:
                eos_hit = True
                first_eos = int(eos_positions[0].item())
                actual = first_eos + 1
                generated[-1] = accepted_toks[:, :actual]
                total_gen = total_gen - accepted + actual

        verified_hit = bool(
            can_prefetch
            and mismatch_position == position
            and alternative is not None
            and int(ar_tokens[int(position) - 1].item()) == int(alternative)
        )
        if can_prefetch:
            outcome_state = _classify_prefetch_outcome(
                mismatch_position=mismatch_position,
                candidate_position=int(position),
                alternative_token=int(alternative),
                ar_tokens=ar_tokens,
            )
            stats.record_outcome(outcome_state, accepted)
            pending_transition = (outcome_state, accepted)
            if outcome_state == "before_candidate_error":
                stats.prefetch_discarded_before_candidate += 1
            elif outcome_state in {"after_candidate_error", "full_block_bonus"}:
                stats.prefetch_discarded_candidate_accepted += 1
            elif outcome_state == "candidate_wrong_alternative":
                stats.prefetch_discarded_wrong_b += 1
            else:
                stats.prefetch_verified_hits += 1

        if eos_hit:
            if verified_hit:
                stats.prefetch_discarded_eos += 1
            break

        forced_thinking_seed = False
        if end_think_token_id is not None and max_thinking_tokens is not None:
            if total_gen > max_thinking_tokens:
                all_gen = torch.cat(generated, dim=1)
                if not bool((all_gen == int(end_think_token_id)).any().item()):
                    next_token = torch.tensor([[int(end_think_token_id)]], device=device)
                    forced_thinking_seed = True

        if verified_hit and prospective is not None:
            if forced_thinking_seed:
                stats.prefetch_discarded_thinking_budget += 1
            elif total_gen < max_new_tokens:
                if int(prospective.tokens[0].item()) != int(next_token.item()):
                    raise RuntimeError("Verified B does not match prospective draft seed")
                prefetched = prospective
                stats.prefetch_hits += 1

        if total_gen >= max_new_tokens:
            break

    all_generated = torch.cat(generated, dim=1)
    output_ids = torch.cat([prompt_ids, all_generated], dim=1)
    if stats.physical_nfe <= 0:
        raise RuntimeError("Invalid physical NFE accounting")
    return output_ids, stats
