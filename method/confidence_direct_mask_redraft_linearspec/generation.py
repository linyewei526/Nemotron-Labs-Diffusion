#!/usr/bin/env python3
"""Strict direct-hit, confidence-triggered all-MASK redraft generation.

The causal verifier row is fused with a second row of the form
``verified_prefix_before_trigger + MASK * L``.  The second row autonomously
redrafts the suspected position and L-1 following positions.  It is reusable
only when the causal verifier rejects exactly at the trigger and row 1 predicts
that correction token.  Accepting A at the trigger, downstream corrections,
and full-block bonus matches are deliberately discarded.  Every reused draft
therefore has the configured full length L.  Output tokens and canonical KV
state always come exclusively from the causal verifier row.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from typing import Optional

import torch
from transformers.cache_utils import DynamicCache

from .hybrid import (
    build_direct_mask_redraft_attention_mask,
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


STATE_NO_CANDIDATE = "no_candidate"
STATE_SKIP_NO_FUTURE = "skip_no_future"
STATE_SKIP_CONTEXT = "skip_context"
STATE_M_LT_P = "m_lt_p"
STATE_DIRECT_HIT = "direct_hit"
STATE_REPEAT_A = "repeat_a"
STATE_WRONG_NON_A = "wrong_non_a"
STATE_A_OK_LATER = "a_ok_later_reject"
STATE_FULL_BONUS = "full_bonus"

ALL_STATES = (
    STATE_NO_CANDIDATE,
    STATE_SKIP_NO_FUTURE,
    STATE_SKIP_CONTEXT,
    STATE_M_LT_P,
    STATE_DIRECT_HIT,
    STATE_REPEAT_A,
    STATE_WRONG_NON_A,
    STATE_A_OK_LATER,
    STATE_FULL_BONUS,
)

ATTEMPT_STATES = (
    STATE_M_LT_P,
    STATE_DIRECT_HIT,
    STATE_REPEAT_A,
    STATE_WRONG_NON_A,
    STATE_A_OK_LATER,
    STATE_FULL_BONUS,
)


@dataclass
class StateTransitionStats:
    """Current-round and paired next-round verifier work for one state."""

    count: int = 0
    current_matched_sum: int = 0
    current_emitted_sum: int = 0
    next_observed_count: int = 0
    no_next_round_count: int = 0
    next_matched_sum: int = 0
    next_emitted_sum: int = 0
    next_minus_current_matched_sum: int = 0
    next_minus_current_emitted_sum: int = 0

    def record_current(self, matched: int, emitted: int) -> None:
        self.count += 1
        self.current_matched_sum += int(matched)
        self.current_emitted_sum += int(emitted)

    def record_next(
        self,
        *,
        current_matched: int,
        current_emitted: int,
        next_matched: int,
        next_emitted: int,
    ) -> None:
        self.next_observed_count += 1
        self.next_matched_sum += int(next_matched)
        self.next_emitted_sum += int(next_emitted)
        self.next_minus_current_matched_sum += int(next_matched) - int(current_matched)
        self.next_minus_current_emitted_sum += int(next_emitted) - int(current_emitted)

    def record_no_next(self) -> None:
        self.no_next_round_count += 1

    def to_dict(self, *, rounds: int, attempts: int) -> dict[str, int | float | None]:
        current = self.count
        paired = self.next_observed_count
        return {
            **asdict(self),
            "round_share": self.count / rounds if rounds else None,
            "attempt_share": self.count / attempts if attempts else None,
            "current_matched_mean": self.current_matched_sum / current if current else None,
            "current_emitted_mean": self.current_emitted_sum / current if current else None,
            "next_matched_mean": self.next_matched_sum / paired if paired else None,
            "next_emitted_mean": self.next_emitted_sum / paired if paired else None,
            "next_minus_current_matched_mean": (
                self.next_minus_current_matched_sum / paired if paired else None
            ),
            "next_minus_current_emitted_mean": (
                self.next_minus_current_emitted_sum / paired if paired else None
            ),
        }


@dataclass
class DirectMaskRedraftGenerationStats:
    physical_nfe: int = 0
    processed_rows: int = 0
    processed_query_tokens: int = 0
    rounds: int = 0
    draft_length_sum: int = 0
    normal_draft_forwards: int = 0
    normal_verify_forwards: int = 0
    fused_verify_redraft_forwards: int = 0
    rounds_without_candidate: int = 0
    redraft_attempts: int = 0
    redraft_direct_hits: int = 0
    redraft_reuse_hits: int = 0
    redraft_saved_draft_forwards: int = 0
    redraft_m_lt_p: int = 0
    redraft_repeat_a: int = 0
    redraft_wrong_non_a: int = 0
    redraft_a_ok_later_reject: int = 0
    redraft_full_bonus_discard: int = 0
    redraft_a_ok_r0_a: int = 0
    redraft_a_ok_r0_changed: int = 0
    redraft_bonus_r0_a: int = 0
    redraft_bonus_r0_changed: int = 0
    redraft_discarded_eos: int = 0
    redraft_discarded_thinking_budget: int = 0
    redraft_discarded_generation_end: int = 0
    redraft_skipped_no_future_round: int = 0
    redraft_skipped_context_limit: int = 0
    candidate_position_sum: int = 0
    full_length_reuses: int = 0
    prospective_query_tokens: int = 0
    state_stats: dict[str, StateTransitionStats] = field(
        default_factory=lambda: {name: StateTransitionStats() for name in ALL_STATES}
    )

    def record_forward(self, *, rows: int, query_tokens_per_row: int) -> None:
        self.physical_nfe += 1
        self.processed_rows += int(rows)
        self.processed_query_tokens += int(rows) * int(query_tokens_per_row)

    def record_draft_length(self, length: int, configured_length: int) -> None:
        if int(length) != int(configured_length):
            raise RuntimeError(
                f"Strict direct redraft requires full drafts: {length} != {configured_length}"
            )
        self.draft_length_sum += int(length)

    def record_state(self, name: str, *, matched: int, emitted: int) -> None:
        self.state_stats[name].record_current(matched, emitted)

    def record_next_state(
        self,
        name: str,
        *,
        current_matched: int,
        current_emitted: int,
        next_matched: int,
        next_emitted: int,
    ) -> None:
        self.state_stats[name].record_next(
            current_matched=current_matched,
            current_emitted=current_emitted,
            next_matched=next_matched,
            next_emitted=next_emitted,
        )

    def record_no_next(self, name: str) -> None:
        self.state_stats[name].record_no_next()

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            key: value
            for key, value in asdict(self).items()
            if key != "state_stats"
        }
        payload["redraft_hit_rate"] = (
            self.redraft_reuse_hits / self.redraft_attempts
            if self.redraft_attempts
            else None
        )
        payload["redraft_direct_hit_rate"] = (
            self.redraft_direct_hits / self.redraft_attempts
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
        payload["saved_draft_fraction_of_rounds"] = (
            self.redraft_saved_draft_forwards / self.rounds
            if self.rounds
            else None
        )
        payload["state_stats"] = {
            name: self.state_stats[name].to_dict(
                rounds=self.rounds, attempts=self.redraft_attempts
            )
            for name in ALL_STATES
        }
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
    stats: DirectMaskRedraftGenerationStats,
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
    stats: DirectMaskRedraftGenerationStats,
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
    stats: DirectMaskRedraftGenerationStats,
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

    attention_mask = build_direct_mask_redraft_attention_mask(
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
    draft_tokens: torch.Tensor,
    trigger_position: int,
    matched_tokens: int,
    emitted_tokens: int,
) -> ReuseDecision:
    """Classify one attempt and allow only an exact direct-trigger correction."""

    if proposal_tokens.ndim != 1 or ar_tokens.ndim != 1 or draft_tokens.ndim != 1:
        raise ValueError("proposal, verifier, and draft tokens must be one-dimensional")
    if ar_tokens.shape != draft_tokens.shape:
        raise ValueError("verifier and draft token shapes must match")
    p = int(trigger_position)
    m = int(emitted_tokens)
    if not 1 <= p < ar_tokens.shape[0]:
        raise ValueError("trigger_position must be inside the verifier draft")
    if not 1 <= m <= ar_tokens.shape[0]:
        raise ValueError("emitted_tokens must be in [1, verify_length]")
    if int(matched_tokens) + 1 != m:
        raise ValueError("emitted_tokens must equal matched_tokens + 1")
    if m < p:
        return ReuseDecision(False, STATE_M_LT_P)

    original_a = int(draft_tokens[p].item())
    redraft_r0 = int(proposal_tokens[0].item())
    if m == p:
        verifier_c = int(ar_tokens[p - 1].item())
        if verifier_c == original_a:
            raise RuntimeError("Direct rejection token unexpectedly equals original A")
        if redraft_r0 == verifier_c:
            return ReuseDecision(True, STATE_DIRECT_HIT)
        if redraft_r0 == original_a:
            return ReuseDecision(False, STATE_REPEAT_A)
        return ReuseDecision(False, STATE_WRONG_NON_A)

    if m == int(draft_tokens.shape[0]) and int(matched_tokens) == m - 1:
        return ReuseDecision(False, STATE_FULL_BONUS)
    return ReuseDecision(False, STATE_A_OK_LATER)


def _record_reuse_rejection(
    stats: DirectMaskRedraftGenerationStats, decision: ReuseDecision
) -> None:
    field = {
        STATE_M_LT_P: "redraft_m_lt_p",
        STATE_REPEAT_A: "redraft_repeat_a",
        STATE_WRONG_NON_A: "redraft_wrong_non_a",
        STATE_A_OK_LATER: "redraft_a_ok_later_reject",
        STATE_FULL_BONUS: "redraft_full_bonus_discard",
    }.get(decision.reason)
    if field is None:
        raise ValueError(f"Unexpected rejection reason: {decision.reason}")
    setattr(stats, field, int(getattr(stats, field)) + 1)


@torch.no_grad()
def direct_mask_redraft_linear_spec_generate(
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
) -> tuple[torch.Tensor, DirectMaskRedraftGenerationStats]:
    """Generate with strict direct-hit, full-length all-MASK redraft overlap."""

    if prompt_ids.shape[0] != 1:
        raise ValueError("Direct MASK-redraft LinearSpec requires request batch_size == 1")
    if block_length < 2 or max_new_tokens <= 0:
        raise ValueError("block_length must be >=2 and max_new_tokens must be positive")
    if not 0.0 <= drop_pct_threshold < 1.0:
        raise ValueError("drop_pct_threshold must be in [0,1)")
    if temperature != 0.0:
        raise ValueError("The strict direct MASK-redraft experiment supports temperature=0 only")
    if draft_threshold != 0.0:
        raise ValueError("The strict direct MASK-redraft experiment requires draft_threshold=0")

    token_mask_id = (
        int(mask_token_id) if mask_token_id is not None else int(model.config.mask_token_id)
    )
    if eos_token_id is None:
        eos_token_id = getattr(model.config, "eos_token_id", None)
    device = prompt_ids.device
    stats = DirectMaskRedraftGenerationStats()

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
    pending_transition: Optional[tuple[str, int, int]] = None
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
        if verify_length != block_length:
            raise RuntimeError(
                f"Strict direct redraft requires Q=L={block_length}, got Q={verify_length}"
            )
        stats.record_draft_length(verify_length, block_length)
        cache_length = int(past_key_values.get_seq_length())
        trigger = draft.candidate_position
        can_redraft = trigger is not None
        if not can_redraft:
            stats.rounds_without_candidate += 1
            decision_state = STATE_NO_CANDIDATE
        elif total_gen + int(trigger) >= max_new_tokens:
            can_redraft = False
            stats.redraft_skipped_no_future_round += 1
            decision_state = STATE_SKIP_NO_FUTURE
        elif cache_length + int(trigger) + block_length > max_positions:
            can_redraft = False
            stats.redraft_skipped_context_limit += 1
            decision_state = STATE_SKIP_CONTEXT
        else:
            decision_state = ""

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

        if pending_transition is not None:
            previous_state, previous_matched, previous_emitted = pending_transition
            stats.record_next_state(
                previous_state,
                current_matched=previous_matched,
                current_emitted=previous_emitted,
                next_matched=matched,
                next_emitted=emitted,
            )
            pending_transition = None

        decision: Optional[ReuseDecision] = None
        if can_redraft and proposal is not None and trigger is not None:
            decision = _decide_redraft_reuse(
                proposal_tokens=proposal.tokens,
                ar_tokens=ar_tokens,
                draft_tokens=draft.tokens,
                trigger_position=int(trigger),
                matched_tokens=matched,
                emitted_tokens=emitted,
            )
            decision_state = decision.reason
            if decision.reusable:
                stats.redraft_direct_hits += 1
            else:
                _record_reuse_rejection(stats, decision)
                original_a = int(draft.tokens[int(trigger)].item())
                row1_r0 = int(proposal.tokens[0].item())
                if decision.reason == STATE_A_OK_LATER:
                    if row1_r0 == original_a:
                        stats.redraft_a_ok_r0_a += 1
                    else:
                        stats.redraft_a_ok_r0_changed += 1
                elif decision.reason == STATE_FULL_BONUS:
                    if row1_r0 == original_a:
                        stats.redraft_bonus_r0_a += 1
                    else:
                        stats.redraft_bonus_r0_changed += 1

        if not decision_state:
            raise RuntimeError("Every verify round must have a decision state")
        stats.record_state(decision_state, matched=matched, emitted=emitted)

        if eos_hit:
            if decision is not None and decision.reusable:
                stats.redraft_discarded_eos += 1
            stats.record_no_next(decision_state)
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
                retained_tokens = proposal.tokens.clone()
                retained_logits = proposal.logits.clone()
                if int(retained_tokens.shape[0]) != block_length:
                    raise RuntimeError("Direct hit must retain the complete L-token proposal")
                if int(retained_tokens[0].item()) != int(next_token.item()):
                    raise RuntimeError("Verifier correction does not match retained redraft seed")
                prefetched = _analyze_draft(
                    logits=retained_logits,
                    tokens=retained_tokens,
                    mask_token_id=token_mask_id,
                    drop_pct_threshold=drop_pct_threshold,
                    source="direct_mask_redraft_full",
                )
                stats.redraft_reuse_hits += 1
                stats.full_length_reuses += 1

        if total_gen >= max_new_tokens:
            stats.record_no_next(decision_state)
            break
        pending_transition = (decision_state, matched, emitted)

    all_generated = torch.cat(generated, dim=1)
    output_ids = torch.cat([prompt_ids, all_generated], dim=1)
    if stats.physical_nfe <= 0:
        raise RuntimeError("Invalid physical NFE accounting")
    return output_ids, stats
