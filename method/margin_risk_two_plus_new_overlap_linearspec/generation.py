#!/usr/bin/env python3
"""Fixed-margin-risk two-candidate plus always-New overlap generation.

Each fused call contains one padded causal verifier row and up to three
prospective rows: corrections for the first two strict margin-risk crossings
and, whenever sequence boundaries allow it, one full-block continuation.  Only
the causal verifier commits output and canonical KV state.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from typing import Optional

import torch
from transformers.cache_utils import DynamicCache

from .hybrid import (
    build_multi_hybrid_attention_mask,
    repeat_dynamic_cache,
    select_and_crop_cache,
)
from .segmented_lora import SegmentedLoraController


MAX_RISK_CANDIDATES = 2
MAX_PROSPECTIVE_BRANCHES = 3
FORWARD_KINDS = ("prefill", "normal_draft", "normal_verify", "multi_fused")
OUTCOME_STATES = (
    "miss_no_candidate_error",
    "miss_before_first",
    "miss_between_candidates",
    "miss_after_last",
    "candidate_1_fixed",
    "candidate_1_wrong",
    "candidate_2_fixed",
    "candidate_2_wrong",
    "full_continuation_hit",
    "full_continuation_miss",
    "full_continuation_absent",
)


@dataclass(frozen=True)
class CandidateSpec:
    rank: int
    position: int
    alternative_token: int
    margin_risk: float


@dataclass
class DraftState:
    tokens: torch.Tensor
    confidences: torch.Tensor
    top1_top2_margins: torch.Tensor
    candidates: tuple[CandidateSpec, ...]
    total_crossing_count: int
    source: str


@dataclass
class OutcomeTransitionStats:
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


def _hist_percentile(histogram: dict[str, int], quantile: float) -> Optional[float]:
    total = sum(int(count) for count in histogram.values())
    if total <= 0:
        return None
    target = (total - 1) * quantile
    low_rank = int(target)
    high_rank = low_rank if target == low_rank else low_rank + 1

    def value_at(rank: int) -> int:
        seen = 0
        for value_text, count in sorted(histogram.items(), key=lambda item: int(item[0])):
            seen += int(count)
            if rank < seen:
                return int(value_text)
        raise RuntimeError("invalid forward-token histogram")

    low = value_at(low_rank)
    high = value_at(high_rank)
    return low + (high - low) * (target - low_rank)


@dataclass
class ForwardDistributionStats:
    count: int = 0
    computed_token_sum: int = 0
    valid_token_sum: int = 0
    padding_token_sum: int = 0
    row_sum: int = 0
    query_length_sum: int = 0
    computed_token_histogram: dict[str, int] = field(default_factory=dict)

    def record(self, *, rows: int, query_length: int, valid_lengths: tuple[int, ...]) -> None:
        if rows <= 0 or query_length <= 0 or len(valid_lengths) != rows:
            raise ValueError("invalid forward geometry")
        if any(length <= 0 or length > query_length for length in valid_lengths):
            raise ValueError("valid row lengths must be in [1, query_length]")
        computed = int(rows) * int(query_length)
        valid = sum(int(length) for length in valid_lengths)
        padding = computed - valid
        self.count += 1
        self.computed_token_sum += computed
        self.valid_token_sum += valid
        self.padding_token_sum += padding
        self.row_sum += int(rows)
        self.query_length_sum += int(query_length)
        key = str(computed)
        self.computed_token_histogram[key] = self.computed_token_histogram.get(key, 0) + 1

    def to_dict(self) -> dict[str, object]:
        count = self.count
        hist = self.computed_token_histogram
        values = [int(key) for key in hist]
        return {
            **asdict(self),
            "computed_token_avg": self.computed_token_sum / count if count else None,
            "valid_token_avg": self.valid_token_sum / count if count else None,
            "padding_token_avg": self.padding_token_sum / count if count else None,
            "padding_ratio": (
                self.padding_token_sum / self.computed_token_sum
                if self.computed_token_sum
                else None
            ),
            "rows_avg": self.row_sum / count if count else None,
            "query_length_avg": self.query_length_sum / count if count else None,
            "computed_token_min": min(values) if values else None,
            "computed_token_p50": _hist_percentile(hist, 0.50),
            "computed_token_p90": _hist_percentile(hist, 0.90),
            "computed_token_p95": _hist_percentile(hist, 0.95),
            "computed_token_p99": _hist_percentile(hist, 0.99),
            "computed_token_max": max(values) if values else None,
        }


@dataclass
class OverlapGenerationStats:
    physical_nfe: int = 0
    processed_rows: int = 0
    processed_query_tokens: int = 0
    valid_query_tokens: int = 0
    padding_query_tokens: int = 0
    rounds: int = 0
    normal_draft_forwards: int = 0
    normal_verify_forwards: int = 0
    fused_verify_draft_forwards: int = 0
    rounds_without_crossing: int = 0
    rounds_without_speculative_branch: int = 0
    prefetch_attempts: int = 0
    prefetch_verified_hits: int = 0
    prefetch_hits: int = 0
    prefetch_saved_draft_forwards: int = 0
    candidate_branches_executed: int = 0
    continuation_branches_executed: int = 0
    continuation_verified_hits: int = 0
    candidate_position_sum: int = 0
    candidate_position_count: int = 0
    prospective_query_tokens: int = 0
    candidate_skipped_no_future_round: int = 0
    candidate_skipped_context_limit: int = 0
    candidate_skipped_thinking_budget: int = 0
    continuation_skipped_no_future_round: int = 0
    continuation_skipped_context_limit: int = 0
    continuation_skipped_thinking_budget: int = 0
    risk_candidates_discarded_after_p2: int = 0
    rounds_with_3plus_crossings: int = 0
    continuation_attempts_3plus_crossings: int = 0
    continuation_verified_hits_3plus_crossings: int = 0
    continuation_prefetch_hits_3plus_crossings: int = 0
    prefetch_discarded_eos: int = 0
    prefetch_discarded_thinking_budget: int = 0
    crossing_count_rounds: dict[str, int] = field(
        default_factory=lambda: {"0": 0, "1": 0, "2": 0, "3+": 0}
    )
    fused_row_count: dict[str, int] = field(
        default_factory=lambda: {"2": 0, "3": 0, "4": 0}
    )
    forward_kinds: dict[str, ForwardDistributionStats] = field(
        default_factory=lambda: {name: ForwardDistributionStats() for name in FORWARD_KINDS}
    )
    outcome_states: dict[str, OutcomeTransitionStats] = field(
        default_factory=lambda: {name: OutcomeTransitionStats() for name in OUTCOME_STATES}
    )

    def record_forward(
        self,
        *,
        kind: str,
        rows: int,
        query_length: int,
        valid_lengths: tuple[int, ...],
    ) -> None:
        if kind not in self.forward_kinds:
            raise ValueError(f"unknown forward kind: {kind}")
        self.physical_nfe += 1
        self.processed_rows += int(rows)
        computed = int(rows) * int(query_length)
        valid = sum(int(length) for length in valid_lengths)
        self.processed_query_tokens += computed
        self.valid_query_tokens += valid
        self.padding_query_tokens += computed - valid
        self.forward_kinds[kind].record(
            rows=rows, query_length=query_length, valid_lengths=valid_lengths
        )

    def record_crossing_count(self, count: int) -> None:
        key = str(count) if count <= 2 else "3+"
        self.crossing_count_rounds[key] += 1
        if count == 0:
            self.rounds_without_crossing += 1

    def record_outcome(self, state: str, accept_length: int) -> None:
        if state not in self.outcome_states:
            raise ValueError(f"unknown two-plus-new outcome state: {state}")
        self.outcome_states[state].record_current(accept_length)

    def record_next_accept(
        self, state: str, current_accept_length: int, next_accept_length: int
    ) -> None:
        if state not in self.outcome_states:
            raise ValueError(f"unknown two-plus-new outcome state: {state}")
        self.outcome_states[state].record_next(current_accept_length, next_accept_length)

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = asdict(self)
        payload["forward_kinds"] = {
            name: self.forward_kinds[name].to_dict() for name in FORWARD_KINDS
        }
        all_forward = ForwardDistributionStats()
        for item in self.forward_kinds.values():
            all_forward.count += item.count
            all_forward.computed_token_sum += item.computed_token_sum
            all_forward.valid_token_sum += item.valid_token_sum
            all_forward.padding_token_sum += item.padding_token_sum
            all_forward.row_sum += item.row_sum
            all_forward.query_length_sum += item.query_length_sum
            for key, count in item.computed_token_histogram.items():
                all_forward.computed_token_histogram[key] = (
                    all_forward.computed_token_histogram.get(key, 0) + count
                )
        decode_forward = ForwardDistributionStats()
        for name in ("normal_draft", "normal_verify", "multi_fused"):
            item = self.forward_kinds[name]
            decode_forward.count += item.count
            decode_forward.computed_token_sum += item.computed_token_sum
            decode_forward.valid_token_sum += item.valid_token_sum
            decode_forward.padding_token_sum += item.padding_token_sum
            decode_forward.row_sum += item.row_sum
            decode_forward.query_length_sum += item.query_length_sum
            for key, count in item.computed_token_histogram.items():
                decode_forward.computed_token_histogram[key] = (
                    decode_forward.computed_token_histogram.get(key, 0) + count
                )
        payload["forward_distribution_all"] = all_forward.to_dict()
        payload["forward_distribution_decode"] = decode_forward.to_dict()
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
            self.candidate_position_sum / self.candidate_position_count
            if self.candidate_position_count
            else None
        )
        payload["padding_query_ratio"] = (
            self.padding_query_tokens / self.processed_query_tokens
            if self.processed_query_tokens
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


def _select_alternative(
    clean_logits: torch.Tensor,
    selected_token: int,
    excluded_token_ids: tuple[int, ...],
) -> Optional[int]:
    alternative_logits = clean_logits.clone()
    alternative_logits[int(selected_token)] = -torch.inf
    for token_id in excluded_token_ids:
        if 0 <= int(token_id) < alternative_logits.shape[-1]:
            alternative_logits[int(token_id)] = -torch.inf
    alternative = int(torch.argmax(alternative_logits).item())
    return alternative if torch.isfinite(alternative_logits[alternative]) else None


def _analyze_draft(
    *,
    logits: torch.Tensor,
    tokens: torch.Tensor,
    mask_token_id: int,
    excluded_alternative_token_ids: tuple[int, ...] = (),
    margin_risk_threshold: float,
    source: str,
) -> DraftState:
    """Find the first two strict margin-risk crossings after MASK exclusion."""

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
    risk_values = (1.0 - margins).detach().cpu().tolist()

    candidates: list[CandidateSpec] = []
    total_crossings = 0
    for position in range(1, tokens.shape[0]):
        risk = float(risk_values[position])
        if risk <= margin_risk_threshold:
            continue
        total_crossings += 1
        if total_crossings > MAX_RISK_CANDIDATES:
            continue
        alternative = _select_alternative(
            clean[position],
            int(tokens[position].item()),
            excluded_alternative_token_ids,
        )
        if alternative is None:
            continue
        candidates.append(
            CandidateSpec(
                rank=total_crossings,
                position=position,
                alternative_token=alternative,
                margin_risk=risk,
            )
        )
    return DraftState(
        tokens=tokens.detach(),
        confidences=confidences.detach(),
        top1_top2_margins=margins.detach(),
        candidates=tuple(candidates),
        total_crossing_count=total_crossings,
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
    block = torch.full(
        (1, block_length), mask_token_id, dtype=torch.long, device=seed.device
    )
    block[0, 0] = int(seed.item())
    _set_diffusion_lm(model, True)
    with _controller_context(controller, all_tokens=True):
        output = model.encoder(
            input_ids=block,
            past_key_values=past_key_values,
            use_cache=False,
        )
    stats.record_forward(
        kind="normal_draft",
        rows=1,
        query_length=block_length,
        valid_lengths=(block_length,),
    )
    stats.normal_draft_forwards += 1
    logits = model.diffusion_head(output.last_hidden_state)
    proposed = logits.argmax(dim=-1)
    is_mask = block == mask_token_id
    block[is_mask] = proposed[is_mask]
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
    stats.record_forward(
        kind="normal_verify",
        rows=1,
        query_length=block_length,
        valid_lengths=(block_length,),
    )
    stats.normal_verify_forwards += 1
    return model.diffusion_head(output.last_hidden_state)[0], output.past_key_values


def _fused_verify_and_drafts(
    model,
    *,
    draft: DraftState,
    candidate_specs: tuple[CandidateSpec, ...],
    include_continuation: bool,
    past_key_values: DynamicCache,
    block_length: int,
    mask_token_id: int,
    excluded_alternative_token_ids: tuple[int, ...],
    margin_risk_threshold: float,
    controller: Optional[SegmentedLoraController],
    stats: OverlapGenerationStats,
) -> tuple[
    torch.Tensor,
    DynamicCache,
    dict[int, DraftState],
    Optional[DraftState],
]:
    if not candidate_specs and not include_continuation:
        raise ValueError("fused path requires at least one speculative branch")
    if len(candidate_specs) + int(include_continuation) > MAX_PROSPECTIVE_BRANCHES:
        raise ValueError("speculative branch count exceeds three")
    device = draft.tokens.device
    cache_length = int(past_key_values.get_seq_length())
    branch_prefixes = [spec.position for spec in candidate_specs]
    if include_continuation:
        branch_prefixes.append(block_length)
    valid_lengths = (block_length,) + tuple(
        prefix + block_length for prefix in branch_prefixes
    )
    query_length = max(valid_lengths)
    rows = len(valid_lengths)
    if rows > 4:
        raise RuntimeError("two-plus-new fused batch exceeds four rows")

    fused_ids = torch.full(
        (rows, query_length), mask_token_id, dtype=torch.long, device=device
    )
    fused_ids[0, :block_length] = draft.tokens
    for row_index, spec in enumerate(candidate_specs, start=1):
        fused_ids[row_index, : spec.position] = draft.tokens[: spec.position]
        fused_ids[row_index, spec.position] = spec.alternative_token
    if include_continuation:
        continuation_row = rows - 1
        fused_ids[continuation_row, :block_length] = draft.tokens

    attention_mask = build_multi_hybrid_attention_mask(
        cache_length=cache_length,
        verifier_length=block_length,
        branch_prefix_lengths=tuple(branch_prefixes),
        prospective_length=block_length,
        query_length=query_length,
        device=device,
    )
    route = torch.zeros((rows, query_length, 1), dtype=torch.bool, device=device)
    for row_index, prefix in enumerate(branch_prefixes, start=1):
        route[row_index, prefix : prefix + block_length] = True
    fused_cache = repeat_dynamic_cache(past_key_values, rows)
    cache_position = torch.arange(
        cache_length, cache_length + query_length, dtype=torch.long, device=device
    )
    position_ids = cache_position.unsqueeze(0).expand(rows, -1)

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
    stats.record_forward(
        kind="multi_fused",
        rows=rows,
        query_length=query_length,
        valid_lengths=valid_lengths,
    )
    stats.fused_verify_draft_forwards += 1
    stats.fused_row_count[str(rows)] += 1
    stats.prospective_query_tokens += block_length * (rows - 1)
    logits = model.diffusion_head(output.last_hidden_state)
    verify_logits = logits[0, :block_length]

    prospective_by_rank: dict[int, DraftState] = {}
    for row_index, spec in enumerate(candidate_specs, start=1):
        start = spec.position
        prospective_ids = fused_ids[row_index, start : start + block_length].clone()
        prospective_logits = logits[row_index, start : start + block_length]
        proposed = prospective_logits.argmax(dim=-1)
        masks = prospective_ids == mask_token_id
        prospective_ids[masks] = proposed[masks]
        prospective_by_rank[spec.rank] = _analyze_draft(
            logits=prospective_logits,
            tokens=prospective_ids,
            mask_token_id=mask_token_id,
            excluded_alternative_token_ids=excluded_alternative_token_ids,
            margin_risk_threshold=margin_risk_threshold,
            source=f"candidate_{spec.rank}",
        )

    continuation: Optional[DraftState] = None
    if include_continuation:
        row_index = rows - 1
        start = block_length
        prospective_logits = logits[row_index, start : start + block_length]
        prospective_ids = prospective_logits.argmax(dim=-1)
        continuation = _analyze_draft(
            logits=prospective_logits,
            tokens=prospective_ids,
            mask_token_id=mask_token_id,
            excluded_alternative_token_ids=excluded_alternative_token_ids,
            margin_risk_threshold=margin_risk_threshold,
            source="continuation",
        )
    return verify_logits, output.past_key_values, prospective_by_rank, continuation


def _greedy_tokens(logits: torch.Tensor) -> torch.Tensor:
    return logits.argmax(dim=-1)


def _count_matched_draft_tokens(ar_tokens: torch.Tensor, draft_tokens: torch.Tensor) -> int:
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


def _classify_multi_outcome(
    *,
    mismatch_position: Optional[int],
    candidate_specs: tuple[CandidateSpec, ...],
    ar_tokens: torch.Tensor,
    continuation: Optional[DraftState],
) -> tuple[str, Optional[int]]:
    """Return a mutually-exclusive state and selected candidate rank.

    Selected rank zero denotes the continuation branch.
    """

    if mismatch_position is None:
        if continuation is None:
            return "full_continuation_absent", None
        bonus = int(ar_tokens[-1].item())
        if int(continuation.tokens[0].item()) == bonus:
            return "full_continuation_hit", 0
        return "full_continuation_miss", None

    for spec in candidate_specs:
        if mismatch_position != spec.position:
            continue
        correct = (
            int(ar_tokens[spec.position - 1].item()) == spec.alternative_token
        )
        return (
            f"candidate_{spec.rank}_{'fixed' if correct else 'wrong'}",
            spec.rank if correct else None,
        )

    if not candidate_specs:
        return "miss_no_candidate_error", None
    positions = [spec.position for spec in candidate_specs]
    if mismatch_position < positions[0]:
        return "miss_before_first", None
    if mismatch_position > positions[-1]:
        return "miss_after_last", None
    return "miss_between_candidates", None


def _thinking_open(
    generated: list[torch.Tensor],
    end_think_token_id: Optional[int],
) -> bool:
    if end_think_token_id is None:
        return False
    all_generated = torch.cat(generated, dim=1)
    return not bool((all_generated == int(end_think_token_id)).any().item())


def _continuation_skip_reason(
    *,
    total_gen: int,
    block_length: int,
    max_new_tokens: int,
    cache_length: int,
    max_positions: int,
    thinking_is_open: bool,
    max_thinking_tokens: Optional[int],
) -> Optional[str]:
    """Return the boundary that suppresses New, independent of risk count."""

    if total_gen + block_length >= max_new_tokens:
        return "no_future_round"
    if cache_length + 2 * block_length > max_positions:
        return "context_limit"
    if (
        thinking_is_open
        and max_thinking_tokens is not None
        and total_gen + block_length > max_thinking_tokens
    ):
        return "thinking_budget"
    return None


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
    """Generate with at most four padded rows per fused overlap forward."""

    if prompt_ids.shape[0] != 1:
        raise ValueError("Multi-overlap LinearSpec requires request batch_size == 1")
    if block_length < 2 or max_new_tokens <= 0:
        raise ValueError("block_length must be >=2 and max_new_tokens must be positive")
    if not 0.0 <= margin_risk_threshold <= 1.0:
        raise ValueError("margin_risk_threshold must be in [0,1]")
    if temperature != 0.0:
        raise ValueError("The isolated two-plus-new experiment supports temperature=0 only")
    if draft_threshold != 0.0:
        raise ValueError("The isolated two-plus-new experiment requires draft_threshold=0")

    token_mask_id = int(
        mask_token_id if mask_token_id is not None else model.config.mask_token_id
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

    _set_diffusion_lm(model, False)
    with _controller_context(lora_controller):
        output = model.encoder(
            input_ids=prompt_ids,
            past_key_values=DynamicCache(),
            use_cache=True,
            use_causal_mask=True,
        )
    prompt_length = int(prompt_ids.shape[1])
    stats.record_forward(
        kind="prefill",
        rows=1,
        query_length=prompt_length,
        valid_lengths=(prompt_length,),
    )
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

        stats.record_crossing_count(draft.total_crossing_count)
        has_3plus_crossings = draft.total_crossing_count >= 3
        if has_3plus_crossings:
            stats.rounds_with_3plus_crossings += 1
            stats.risk_candidates_discarded_after_p2 += (
                draft.total_crossing_count - MAX_RISK_CANDIDATES
            )
        cache_length = int(past_key_values.get_seq_length())
        thinking_is_open = (
            max_thinking_tokens is not None
            and _thinking_open(generated, end_think_token_id)
        )

        executable_candidates: list[CandidateSpec] = []
        for spec in draft.candidates:
            if total_gen + spec.position >= max_new_tokens:
                stats.candidate_skipped_no_future_round += 1
            elif cache_length + spec.position + block_length > max_positions:
                stats.candidate_skipped_context_limit += 1
            elif (
                thinking_is_open
                and max_thinking_tokens is not None
                and total_gen + spec.position > max_thinking_tokens
            ):
                stats.candidate_skipped_thinking_budget += 1
            else:
                executable_candidates.append(spec)

        # Always reserve the third prospective slot for New.  It may be absent
        # only when generation/thinking/context boundaries make a future block
        # impossible, never because the current draft has many risk crossings.
        continuation_skip_reason = _continuation_skip_reason(
            total_gen=total_gen,
            block_length=block_length,
            max_new_tokens=max_new_tokens,
            cache_length=cache_length,
            max_positions=max_positions,
            thinking_is_open=thinking_is_open,
            max_thinking_tokens=max_thinking_tokens,
        )
        include_continuation = continuation_skip_reason is None
        if continuation_skip_reason == "no_future_round":
            stats.continuation_skipped_no_future_round += 1
        elif continuation_skip_reason == "context_limit":
            stats.continuation_skipped_context_limit += 1
        elif continuation_skip_reason == "thinking_budget":
            stats.continuation_skipped_thinking_budget += 1

        candidate_specs = tuple(executable_candidates)
        speculative_rows = len(candidate_specs) + int(include_continuation)
        can_overlap = speculative_rows > 0
        fused_cache: Optional[DynamicCache] = None
        prospective_by_rank: dict[int, DraftState] = {}
        continuation: Optional[DraftState] = None
        if can_overlap:
            stats.prefetch_attempts += 1
            stats.candidate_branches_executed += len(candidate_specs)
            stats.continuation_branches_executed += int(include_continuation)
            if include_continuation and has_3plus_crossings:
                stats.continuation_attempts_3plus_crossings += 1
            for spec in candidate_specs:
                stats.candidate_position_sum += spec.position
                stats.candidate_position_count += 1
            verify_logits, fused_cache, prospective_by_rank, continuation = (
                _fused_verify_and_drafts(
                    model,
                    draft=draft,
                    candidate_specs=candidate_specs,
                    include_continuation=include_continuation,
                    past_key_values=past_key_values,
                    block_length=block_length,
                    mask_token_id=token_mask_id,
                    excluded_alternative_token_ids=excluded_alternative_token_ids,
                    margin_risk_threshold=margin_risk_threshold,
                    controller=lora_controller,
                    stats=stats,
                )
            )
            ar_tokens = _greedy_tokens(verify_logits)
        else:
            stats.rounds_without_speculative_branch += 1
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
                actual = int(eos_positions[0].item()) + 1
                generated[-1] = accepted_toks[:, :actual]
                total_gen = total_gen - accepted + actual

        selected_prospective: Optional[DraftState] = None
        verified_hit = False
        if can_overlap:
            outcome_state, selected_rank = _classify_multi_outcome(
                mismatch_position=mismatch_position,
                candidate_specs=candidate_specs,
                ar_tokens=ar_tokens,
                continuation=continuation,
            )
            stats.record_outcome(outcome_state, accepted)
            pending_transition = (outcome_state, accepted)
            if selected_rank is not None:
                verified_hit = True
                stats.prefetch_verified_hits += 1
                if selected_rank == 0:
                    stats.continuation_verified_hits += 1
                    if has_3plus_crossings:
                        stats.continuation_verified_hits_3plus_crossings += 1
                    selected_prospective = continuation
                else:
                    selected_prospective = prospective_by_rank.get(selected_rank)

        if eos_hit:
            if verified_hit:
                stats.prefetch_discarded_eos += 1
            break

        forced_thinking_seed = False
        if max_thinking_tokens is not None and total_gen > max_thinking_tokens:
            if _thinking_open(generated, end_think_token_id):
                if end_think_token_id is None:
                    raise RuntimeError("thinking budget requires end_think_token_id")
                next_token = torch.tensor([[int(end_think_token_id)]], device=device)
                forced_thinking_seed = True

        if verified_hit and selected_prospective is not None:
            if forced_thinking_seed:
                stats.prefetch_discarded_thinking_budget += 1
            elif total_gen < max_new_tokens:
                if int(selected_prospective.tokens[0].item()) != int(next_token.item()):
                    raise RuntimeError("Verified branch seed does not match committed next token")
                prefetched = selected_prospective
                stats.prefetch_hits += 1
                if selected_rank == 0 and has_3plus_crossings:
                    stats.continuation_prefetch_hits_3plus_crossings += 1

        if total_gen >= max_new_tokens:
            break

    all_generated = torch.cat(generated, dim=1)
    output_ids = torch.cat([prompt_ids, all_generated], dim=1)
    if stats.physical_nfe <= 0:
        raise RuntimeError("Invalid physical NFE accounting")
    return output_ids, stats
