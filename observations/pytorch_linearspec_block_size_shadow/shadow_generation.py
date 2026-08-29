#!/usr/bin/env python3
"""Native LinearSpec with same-state L4/L8/L16/L32 draft/verify shadows.

Only the configured anchor branch advances tokens and KV state.  Every other
branch starts from an independent clone of the exact same causal cache and is
discarded after verification.  The anchor branch therefore remains a native
LinearSpec decoding trajectory while the other branches are paired
counterfactual observations.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import torch
from transformers.cache_utils import DynamicCache

from .trace_writer import ShadowTraceWriter


@dataclass
class BranchResult:
    block_size: int
    cache: Optional[DynamicCache]
    block_tokens: torch.Tensor
    verifier_tokens: torch.Tensor
    matched: int
    accept_length: int
    effective_accept_length: int
    eos_hit: bool
    draft_forward_passes: int
    position_metrics: dict[str, list[Any]]
    accepted_confidence_mean: Optional[float]
    accepted_confidence_min: Optional[float]
    accepted_confidence_last: Optional[float]
    rejected_confidence: Optional[float]
    rejected_margin: Optional[float]
    rejected_entropy: Optional[float]


def decompose_pair(a_small: int, a_large: int, small_size: int) -> dict[str, int]:
    """Split the large-block gain into beyond-capacity tail and in-cap decay.

    delta = A_large - A_small
    tail  = max(A_large - L_small, 0)
    decay = min(A_large, L_small) - A_small
    so delta == tail + decay for every paired round.
    """

    if small_size <= 0:
        raise ValueError("small_size must be positive")
    if not 1 <= a_small <= small_size:
        raise ValueError("a_small must be in [1, small_size]")
    if a_large < 1:
        raise ValueError("a_large must be positive")
    tail = max(int(a_large) - int(small_size), 0)
    decay = min(int(a_large), int(small_size)) - int(a_small)
    delta = int(a_large) - int(a_small)
    if delta != tail + decay:
        raise AssertionError("pair decomposition identity failed")
    return {"delta_a": delta, "tail": tail, "decay": decay}


def clone_dynamic_cache(cache: DynamicCache) -> DynamicCache:
    """Deep-clone a DynamicCache so verifier updates cannot leak across arms."""

    data: list[tuple[torch.Tensor, torch.Tensor]] = []
    if hasattr(cache, "layers"):
        for layer in cache.layers:
            keys = layer.keys
            values = layer.values
            if keys is None or values is None:
                raise ValueError("Cannot clone an uninitialized DynamicCache layer")
            data.append((keys.clone(), values.clone()))
    else:
        for keys, values in zip(cache.key_cache, cache.value_cache):
            data.append((keys.clone(), values.clone()))
    return DynamicCache(ddp_cache_data=data)


def _crop_dynamic_cache(cache: DynamicCache, length: int) -> None:
    if hasattr(cache, "crop"):
        cache.crop(length)
        return
    for layer_index in range(len(cache)):
        cache.key_cache[layer_index] = cache.key_cache[layer_index][:, :, :length]
        cache.value_cache[layer_index] = cache.value_cache[layer_index][:, :, :length]
    cache._seen_tokens = length


def _set_diffusion_lm(model: Any, value: bool) -> None:
    for layer in model.encoder.layers:
        if hasattr(layer.self_attn, "diffusion_lm"):
            layer.self_attn.diffusion_lm = value


def _toggle_adapters(model: Any, enabled: bool) -> None:
    for module in model.modules():
        if hasattr(module, "_disable_adapters"):
            module._disable_adapters = not enabled


def _sample_tokens(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature > 0:
        probabilities = torch.softmax(logits / temperature, dim=-1)
        return torch.multinomial(
            probabilities.reshape(-1, probabilities.shape[-1]),
            num_samples=1,
        ).reshape(logits.shape[:-1])
    return logits.argmax(dim=-1)


def _position_diagnostics(
    logits: torch.Tensor,
    selected_tokens: torch.Tensor,
    *,
    mask_token_id: int,
) -> dict[str, list[Any]]:
    """Return scalar diagnostics for draft positions 1..L-1, excluding MASK."""

    rows = logits[1:].detach().float()
    tokens = selected_tokens[1:].detach()
    if rows.numel() == 0:
        return {
            "selected_confidence": [],
            "top1_top2_margin": [],
            "entropy": [],
            "selected_is_top1": [],
        }
    clean = rows.clone()
    if 0 <= mask_token_id < clean.shape[-1]:
        clean[:, mask_token_id] = -torch.inf
    log_z = torch.logsumexp(clean, dim=-1)
    selected_logits = clean.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
    confidence = torch.exp(selected_logits - log_z)
    top_values, top_indices = clean.topk(k=2, dim=-1)
    top_probs = torch.exp(top_values - log_z.unsqueeze(-1))
    margin = top_probs[:, 0] - top_probs[:, 1]
    probabilities = torch.softmax(clean, dim=-1)
    finite_clean = torch.where(torch.isfinite(clean), clean, torch.zeros_like(clean))
    entropy = log_z - (probabilities * finite_clean).sum(dim=-1)
    return {
        "selected_confidence": confidence.cpu().tolist(),
        "top1_top2_margin": margin.cpu().tolist(),
        "entropy": entropy.cpu().tolist(),
        "selected_is_top1": (top_indices[:, 0] == tokens).cpu().tolist(),
    }


def _optional_mean(values: list[float]) -> Optional[float]:
    return float(sum(values) / len(values)) if values else None


def _run_branch(
    model: Any,
    *,
    seed_token: torch.Tensor,
    canonical_cache: DynamicCache,
    block_size: int,
    mask_token_id: int,
    eos_token_id: Optional[int],
    threshold: float,
    temperature: float,
) -> BranchResult:
    device = seed_token.device
    branch_cache = clone_dynamic_cache(canonical_cache)
    block = torch.full(
        (1, block_size),
        int(mask_token_id),
        dtype=torch.long,
        device=device,
    )
    block[0, 0] = int(seed_token.item())

    _set_diffusion_lm(model, True)
    _toggle_adapters(model, True)
    committed_logits: Optional[torch.Tensor] = None
    draft_forward_passes = 0
    while True:
        is_mask = block == int(mask_token_id)
        if not bool(is_mask.any().item()):
            break
        output = model.encoder(
            input_ids=block,
            past_key_values=branch_cache,
            use_cache=False,
        )
        draft_forward_passes += 1
        draft_logits = model.diffusion_head(output.last_hidden_state)
        if committed_logits is None:
            committed_logits = torch.zeros_like(draft_logits)
        draft_tokens = _sample_tokens(draft_logits, temperature)
        draft_probabilities = torch.softmax(
            draft_logits / temperature if temperature > 0 else draft_logits,
            dim=-1,
        )
        if threshold > 0:
            confidence = draft_probabilities.gather(
                -1, draft_tokens.unsqueeze(-1)
            ).squeeze(-1)
            confidence = torch.where(is_mask, confidence, -torch.inf)
            unmask = confidence >= threshold
            if not bool(unmask.any().item()):
                best = int(confidence.reshape(-1).argmax().item())
                unmask = torch.zeros_like(is_mask, dtype=torch.bool)
                unmask.reshape(-1)[best] = True
            committed_logits[unmask] = draft_logits.detach()[unmask]
            block[unmask] = draft_tokens[unmask]
        else:
            committed_logits[is_mask] = draft_logits.detach()[is_mask]
            block[is_mask] = draft_tokens[is_mask]
            break
    if committed_logits is None:
        raise RuntimeError("LinearSpec draft produced no logits")

    _set_diffusion_lm(model, False)
    _toggle_adapters(model, False)
    output = model.encoder(
        input_ids=block,
        past_key_values=branch_cache,
        use_cache=True,
        use_causal_mask=True,
    )
    branch_cache = output.past_key_values
    verify_logits = model.diffusion_head(output.last_hidden_state)
    verifier_tokens = _sample_tokens(verify_logits, temperature)[0]

    matched = 0
    for index in range(block_size - 1):
        if int(verifier_tokens[index].item()) != int(block[0, index + 1].item()):
            break
        matched += 1
    accept_length = matched + 1
    eos_hit = False
    effective_accept_length = accept_length
    if eos_token_id is not None:
        eos_positions = (
            verifier_tokens[:accept_length] == int(eos_token_id)
        ).nonzero(as_tuple=True)[0]
        if len(eos_positions) > 0:
            eos_hit = True
            effective_accept_length = int(eos_positions[0].item()) + 1

    metrics = _position_diagnostics(
        committed_logits[0],
        block[0],
        mask_token_id=mask_token_id,
    )
    accepted_confidence = [
        float(value) for value in metrics["selected_confidence"][:matched]
    ]
    rejected_index = matched if matched < block_size - 1 else None
    return BranchResult(
        block_size=block_size,
        cache=branch_cache,
        block_tokens=block[0].detach(),
        verifier_tokens=verifier_tokens.detach(),
        matched=matched,
        accept_length=accept_length,
        effective_accept_length=effective_accept_length,
        eos_hit=eos_hit,
        draft_forward_passes=draft_forward_passes,
        position_metrics=metrics,
        accepted_confidence_mean=_optional_mean(accepted_confidence),
        accepted_confidence_min=(min(accepted_confidence) if accepted_confidence else None),
        accepted_confidence_last=(accepted_confidence[-1] if accepted_confidence else None),
        rejected_confidence=(
            float(metrics["selected_confidence"][rejected_index])
            if rejected_index is not None
            else None
        ),
        rejected_margin=(
            float(metrics["top1_top2_margin"][rejected_index])
            if rejected_index is not None
            else None
        ),
        rejected_entropy=(
            float(metrics["entropy"][rejected_index])
            if rejected_index is not None
            else None
        ),
    )


def _agreement(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    common = min(int(left.numel()), int(right.numel()))
    if common <= 0:
        return {
            "common_positions": 0,
            "agree_positions": 0,
            "agreement_rate": None,
            "all_common_equal": True,
            "first_divergence_position": None,
        }
    equal = left[:common] == right[:common]
    agree = int(equal.sum().item())
    divergence = (~equal).nonzero(as_tuple=True)[0]
    return {
        "common_positions": common,
        "agree_positions": agree,
        "agreement_rate": agree / common,
        "all_common_equal": agree == common,
        "first_divergence_position": (
            int(divergence[0].item()) + 1 if len(divergence) else None
        ),
    }


def _branch_record(branch: BranchResult, detail: str) -> dict[str, Any]:
    record: dict[str, Any] = {
        "block_size": branch.block_size,
        "matched_draft_tokens": branch.matched,
        "accept_length": branch.accept_length,
        "effective_accept_length": branch.effective_accept_length,
        "accept_rate": branch.accept_length / branch.block_size,
        "full_accept": branch.accept_length == branch.block_size,
        "zero_draft_match": branch.matched == 0,
        "mismatch_draft_position": (
            branch.matched + 1 if branch.matched < branch.block_size - 1 else None
        ),
        "eos_hit": branch.eos_hit,
        "draft_forward_passes": branch.draft_forward_passes,
        "accepted_confidence_mean": branch.accepted_confidence_mean,
        "accepted_confidence_min": branch.accepted_confidence_min,
        "accepted_confidence_last": branch.accepted_confidence_last,
        "rejected_confidence": branch.rejected_confidence,
        "rejected_margin": branch.rejected_margin,
        "rejected_entropy": branch.rejected_entropy,
    }
    if detail in {"position", "tokens"}:
        record["position"] = branch.position_metrics
        record["position"]["draft_accepted"] = [
            index < branch.matched for index in range(branch.block_size - 1)
        ]
    if detail == "tokens":
        record["draft_token_ids"] = [
            int(value) for value in branch.block_tokens[1:].cpu().tolist()
        ]
        record["verifier_token_ids"] = [
            int(value) for value in branch.verifier_tokens.cpu().tolist()
        ]
    return record


def _history_features(
    accept_history: list[int],
    confidence_history: list[Optional[float]],
    windows: Sequence[int],
) -> dict[str, Optional[float]]:
    result: dict[str, Optional[float]] = {
        "prev_anchor_a": float(accept_history[-1]) if accept_history else None,
        "prev_anchor_conf": (
            float(confidence_history[-1])
            if confidence_history and confidence_history[-1] is not None
            else None
        ),
    }
    for window in windows:
        a_values = accept_history[-window:]
        c_values = [
            float(value)
            for value in confidence_history[-window:]
            if value is not None
        ]
        result[f"a_ma{window}"] = _optional_mean([float(value) for value in a_values])
        result[f"conf_ma{window}"] = _optional_mean(c_values)
    if accept_history:
        alpha = 0.5
        ewma = float(accept_history[0])
        for value in accept_history[1:]:
            ewma = alpha * float(value) + (1 - alpha) * ewma
        result["a_ewma05"] = ewma
    else:
        result["a_ewma05"] = None
    return result


def _capture_rng(device: torch.device) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    cpu = torch.random.get_rng_state()
    cuda = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    return cpu, cuda


def _restore_rng(
    state: tuple[torch.Tensor, Optional[torch.Tensor]], device: torch.device
) -> None:
    torch.random.set_rng_state(state[0])
    if device.type == "cuda" and state[1] is not None:
        torch.cuda.set_rng_state(state[1], device)


@torch.no_grad()
def linear_spec_generate_with_block_size_shadows(
    model: Any,
    prompt_ids: torch.Tensor,
    *,
    max_new_tokens: int = 128,
    block_sizes: Sequence[int] = (4, 8, 16, 32),
    anchor_block_size: int = 16,
    temperature: float = 0.0,
    mask_token_id: Optional[int] = None,
    eos_token_id: Optional[int] = None,
    max_thinking_tokens: Optional[int] = None,
    end_think_token_id: Optional[int] = None,
    threshold: float = 0.0,
    tracer: Optional[ShadowTraceWriter] = None,
    request_id: str = "",
    mode: str = "linearspec_lora",
    history_windows: Sequence[int] = (1, 2, 4, 8),
) -> tuple[torch.Tensor, int]:
    """Generate on the anchor trajectory and record paired shadow outcomes."""

    if prompt_ids.ndim != 2 or prompt_ids.shape[0] != 1:
        raise ValueError("LinearSpec shadow observation requires batch_size == 1")
    normalized_sizes = tuple(sorted({int(value) for value in block_sizes}))
    if not normalized_sizes or any(value < 2 for value in normalized_sizes):
        raise ValueError("block_sizes must contain unique integers >= 2")
    if int(anchor_block_size) not in normalized_sizes:
        raise ValueError("anchor_block_size must be present in block_sizes")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if threshold < 0 or temperature < 0:
        raise ValueError("threshold and temperature must be non-negative")
    windows = tuple(sorted({int(value) for value in history_windows}))
    if not windows or any(value <= 0 for value in windows):
        raise ValueError("history_windows must be positive")

    token_mask_id = (
        int(mask_token_id)
        if mask_token_id is not None
        else int(model.config.mask_token_id)
    )
    if eos_token_id is None:
        eos_token_id = getattr(model.config, "eos_token_id", None)
    device = prompt_ids.device

    _set_diffusion_lm(model, False)
    _toggle_adapters(model, False)
    output = model.encoder(
        input_ids=prompt_ids,
        past_key_values=DynamicCache(),
        use_cache=True,
        use_causal_mask=True,
    )
    canonical_cache = output.past_key_values
    last_logits = model.diffusion_head(output.last_hidden_state[:, -1:, :]).squeeze(1)
    next_token = _sample_tokens(last_logits, temperature).reshape(1, 1)
    anchor_nfe = 1
    if eos_token_id is not None and int(next_token.item()) == int(eos_token_id):
        return torch.cat([prompt_ids, next_token], dim=1), anchor_nfe

    generated = [next_token]
    total_generated = 1
    round_index = 0
    accept_history: list[int] = []
    confidence_history: list[Optional[float]] = []
    max_block = max(normalized_sizes)
    branch_order = tuple(
        value for value in normalized_sizes if value != int(anchor_block_size)
    ) + (int(anchor_block_size),)

    while total_generated < max_new_tokens:
        generation_offset = total_generated
        cache_length = int(canonical_cache.get_seq_length())
        remaining_budget = max_new_tokens - generation_offset
        history = _history_features(accept_history, confidence_history, windows)
        round_rng = _capture_rng(device)
        anchor_end_rng = round_rng
        branches: dict[int, BranchResult] = {}
        for block_size in branch_order:
            _restore_rng(round_rng, device)
            branch = _run_branch(
                model,
                seed_token=next_token,
                canonical_cache=canonical_cache,
                block_size=block_size,
                mask_token_id=token_mask_id,
                eos_token_id=eos_token_id,
                threshold=float(threshold),
                temperature=float(temperature),
            )
            branches[block_size] = branch
            if block_size == int(anchor_block_size):
                anchor_end_rng = _capture_rng(device)
            else:
                # Retain only scalar/token diagnostics.  Releasing each shadow
                # cache here keeps peak memory near canonical+one arm instead
                # of canonical+all arms.
                branch.cache = None
        _restore_rng(anchor_end_rng, device)

        anchor = branches[int(anchor_block_size)]
        if anchor.cache is None:
            raise AssertionError("anchor branch lost its committed cache")
        anchor_nfe += anchor.draft_forward_passes + 1
        _crop_dynamic_cache(
            anchor.cache,
            cache_length + anchor.accept_length,
        )
        canonical_cache = anchor.cache
        accepted_tokens = anchor.verifier_tokens[: anchor.accept_length].reshape(1, -1)
        if anchor.eos_hit:
            accepted_tokens = accepted_tokens[:, : anchor.effective_accept_length]
        generated.append(accepted_tokens)
        total_generated += int(accepted_tokens.shape[1])
        next_token = anchor.verifier_tokens[
            anchor.accept_length - 1 : anchor.accept_length
        ].reshape(1, 1)

        any_eos = any(branch.eos_hit for branch in branches.values())
        budget_boundary = remaining_budget < max_block
        analysis_valid = (not any_eos) and (not budget_boundary)
        pair_records: dict[str, dict[str, Any]] = {}
        for small_index, small_size in enumerate(normalized_sizes):
            for large_size in normalized_sizes[small_index + 1 :]:
                small = branches[small_size]
                large = branches[large_size]
                pair = decompose_pair(
                    small.accept_length,
                    large.accept_length,
                    small_size,
                )
                pair["small_size"] = small_size
                pair["large_size"] = large_size
                pair["small_ge_large"] = small.accept_length >= large.accept_length
                pair["small_gt_large"] = small.accept_length > large.accept_length
                pair["equal_accept_length"] = (
                    small.accept_length == large.accept_length
                )
                pair["draft_agreement"] = _agreement(
                    small.block_tokens[1:], large.block_tokens[1:]
                )
                pair["verifier_agreement"] = _agreement(
                    small.verifier_tokens, large.verifier_tokens
                )
                pair_records[f"{small_size}_{large_size}"] = pair

        if tracer is not None:
            branch_records = {
                str(size): _branch_record(branches[size], tracer.trace_detail)
                for size in normalized_sizes
            }
            tracer.write(
                {
                    "schema_version": 1,
                    "event": "linearspec_block_size_shadow_round",
                    "backend": "native_pytorch",
                    "created_at_unix": time.time(),
                    "request_id": str(request_id),
                    "mode": mode,
                    "round_index": round_index,
                    "generation_offset": generation_offset,
                    "cache_length": cache_length,
                    "remaining_generation_budget": remaining_budget,
                    "block_sizes": list(normalized_sizes),
                    "anchor_block_size": int(anchor_block_size),
                    "threshold": float(threshold),
                    "temperature": float(temperature),
                    "paired_rng": temperature > 0,
                    "budget_boundary": budget_boundary,
                    "any_branch_eos": any_eos,
                    "analysis_valid": analysis_valid,
                    "history_before_round": history,
                    "branches": branch_records,
                    "pairs": pair_records,
                }
            )

        accept_history.append(anchor.accept_length)
        confidence_history.append(anchor.accepted_confidence_mean)
        round_index += 1
        if anchor.eos_hit:
            break

        if end_think_token_id is not None and max_thinking_tokens is not None:
            if total_generated > max_thinking_tokens:
                all_generated = torch.cat(generated, dim=1)
                if not bool((all_generated == int(end_think_token_id)).any().item()):
                    next_token = torch.tensor(
                        [[int(end_think_token_id)]],
                        dtype=torch.long,
                        device=device,
                    )
        if total_generated >= max_new_tokens:
            break

    all_generated = torch.cat(generated, dim=1)
    return torch.cat([prompt_ids, all_generated], dim=1), anchor_nfe
