#!/usr/bin/env python3
"""Native LinearSpec generation with observational first-mismatch traces.

The locator is never consulted by decoding.  This function therefore follows
the ordinary LinearSpec trajectory and records only information available
before the current verifier plus the verifier label used by offline analysis.
"""

from __future__ import annotations

import math
import statistics
import time
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import torch
from transformers.cache_utils import DynamicCache

from .trace_writer import LocatorTraceWriter


@dataclass
class RoundHistory:
    accept_length: int
    mismatch_position: Optional[int]
    full_accept: bool
    good_confidences: list[float]
    error_confidence: Optional[float]


def _crop_dynamic_cache(cache: DynamicCache, length: int) -> None:
    if hasattr(cache, "crop"):
        cache.crop(length)
        return
    for layer_index in range(len(cache)):
        cache.key_cache[layer_index] = cache.key_cache[layer_index][:, :, :length]
        cache.value_cache[layer_index] = cache.value_cache[layer_index][:, :, :length]
    cache._seen_tokens = length


def _set_diffusion_lm(model: Any, enabled: bool) -> None:
    for layer in model.encoder.layers:
        if hasattr(layer.self_attn, "diffusion_lm"):
            layer.self_attn.diffusion_lm = bool(enabled)


def _toggle_adapters(model: Any, enabled: bool) -> None:
    for module in model.modules():
        if hasattr(module, "_disable_adapters"):
            module._disable_adapters = not enabled


def _sample_tokens(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature > 0:
        probabilities = torch.softmax(logits / temperature, dim=-1)
        return torch.multinomial(
            probabilities.reshape(-1, probabilities.shape[-1]), num_samples=1
        ).reshape(logits.shape[:-1])
    return logits.argmax(dim=-1)


def _safe_drop(current: float, reference: Optional[float]) -> Optional[float]:
    if reference is None or not math.isfinite(reference) or reference <= 0:
        return None
    return 1.0 - current / reference


def position_diagnostics(
    logits: torch.Tensor,
    selected_tokens: torch.Tensor,
    *,
    mask_token_id: int,
) -> dict[str, list[Any]]:
    """Diagnostics for draft positions 1..L-1; MASK is excluded from softmax."""

    rows = logits[1:].detach().float()
    tokens = selected_tokens[1:].detach()
    if rows.numel() == 0:
        return {key: [] for key in (
            "selected_confidence", "top1_top2_margin", "entropy",
            "selected_is_top1", "prefix_mean_before", "prefix_median_before",
            "prefix_min_before", "prefix_drop_pct", "local_drop_pct",
        )}
    clean = rows.clone()
    if 0 <= mask_token_id < clean.shape[-1]:
        clean[:, mask_token_id] = -torch.inf
    log_z = torch.logsumexp(clean, dim=-1)
    selected_logits = clean.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
    confidence = torch.exp(selected_logits - log_z)
    top_values, top_indices = clean.topk(k=2, dim=-1)
    top_probs = torch.exp(top_values - log_z.unsqueeze(-1))
    probabilities = torch.softmax(clean, dim=-1)
    finite_clean = torch.where(torch.isfinite(clean), clean, torch.zeros_like(clean))
    entropy = log_z - (probabilities * finite_clean).sum(dim=-1)

    confidence_values = [float(value) for value in confidence.cpu().tolist()]
    prefix_mean: list[Optional[float]] = []
    prefix_median: list[Optional[float]] = []
    prefix_min: list[Optional[float]] = []
    prefix_drop: list[Optional[float]] = []
    local_drop: list[Optional[float]] = []
    for index, value in enumerate(confidence_values):
        preceding = confidence_values[:index]
        mean_value = statistics.fmean(preceding) if preceding else None
        median_value = statistics.median(preceding) if preceding else None
        minimum_value = min(preceding) if preceding else None
        prefix_mean.append(mean_value)
        prefix_median.append(median_value)
        prefix_min.append(minimum_value)
        prefix_drop.append(_safe_drop(value, mean_value))
        local_drop.append(_safe_drop(value, preceding[-1] if preceding else None))
    return {
        "selected_confidence": confidence_values,
        "top1_top2_margin": [float(value) for value in (top_probs[:, 0] - top_probs[:, 1]).cpu().tolist()],
        "entropy": [float(value) for value in entropy.cpu().tolist()],
        "selected_is_top1": [bool(value) for value in (top_indices[:, 0] == tokens).cpu().tolist()],
        "prefix_mean_before": prefix_mean,
        "prefix_median_before": prefix_median,
        "prefix_min_before": prefix_min,
        "prefix_drop_pct": prefix_drop,
        "local_drop_pct": local_drop,
    }


def summarize_history(history: list[RoundHistory], windows: Sequence[int]) -> dict[str, Any]:
    payload: dict[str, Any] = {"rounds_available": len(history)}
    if history:
        last = history[-1]
        payload["last"] = {
            "accept_length": last.accept_length,
            "mismatch_position": last.mismatch_position,
            "full_accept": last.full_accept,
            "good_confidence_mean": statistics.fmean(last.good_confidences) if last.good_confidences else None,
            "error_confidence": last.error_confidence,
        }
    else:
        payload["last"] = None
    for window in sorted({int(value) for value in windows}):
        recent = history[-window:]
        good = [value for item in recent for value in item.good_confidences]
        errors = [item.error_confidence for item in recent if item.error_confidence is not None]
        positions = [item.mismatch_position for item in recent if item.mismatch_position is not None]
        good_mean = statistics.fmean(good) if good else None
        error_mean = statistics.fmean(errors) if errors else None
        payload[str(window)] = {
            "round_count": len(recent),
            "accept_mean": statistics.fmean(item.accept_length for item in recent) if recent else None,
            "full_accept_rate": statistics.fmean(float(item.full_accept) for item in recent) if recent else None,
            "mismatch_position_mean": statistics.fmean(positions) if positions else None,
            "good_confidence_mean": good_mean,
            "error_confidence_mean": error_mean,
            "good_error_gap": good_mean - error_mean if good_mean is not None and error_mean is not None else None,
            "good_error_ratio": good_mean / error_mean if good_mean is not None and error_mean not in {None, 0.0} else None,
        }
    return payload


def _draft_verify_round(
    model: Any,
    *,
    seed_token: torch.Tensor,
    cache: DynamicCache,
    block_size: int,
    mask_token_id: int,
    temperature: float,
    threshold: float,
) -> tuple[DynamicCache, torch.Tensor, torch.Tensor, int, int, dict[str, list[Any]]]:
    block = torch.full(
        (1, block_size), int(mask_token_id), dtype=torch.long, device=seed_token.device
    )
    block[0, 0] = int(seed_token.item())
    _set_diffusion_lm(model, True)
    _toggle_adapters(model, True)
    committed_logits: Optional[torch.Tensor] = None
    draft_forwards = 0
    while bool((block == int(mask_token_id)).any().item()):
        is_mask = block == int(mask_token_id)
        output = model.encoder(input_ids=block, past_key_values=cache, use_cache=False)
        draft_forwards += 1
        draft_logits = model.diffusion_head(output.last_hidden_state)
        if committed_logits is None:
            committed_logits = torch.zeros_like(draft_logits)
        draft_tokens = _sample_tokens(draft_logits, temperature)
        probabilities = torch.softmax(
            draft_logits / temperature if temperature > 0 else draft_logits, dim=-1
        )
        if threshold > 0:
            confidence = probabilities.gather(-1, draft_tokens.unsqueeze(-1)).squeeze(-1)
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
    if committed_logits is None:
        raise RuntimeError("draft produced no logits")

    _set_diffusion_lm(model, False)
    _toggle_adapters(model, False)
    output = model.encoder(
        input_ids=block,
        past_key_values=cache,
        use_cache=True,
        use_causal_mask=True,
    )
    verify_tokens = _sample_tokens(model.diffusion_head(output.last_hidden_state), temperature)[0]
    matched = 0
    for index in range(block_size - 1):
        if int(verify_tokens[index].item()) != int(block[0, index + 1].item()):
            break
        matched += 1
    diagnostics = position_diagnostics(
        committed_logits[0], block[0], mask_token_id=mask_token_id
    )
    return output.past_key_values, block[0].detach(), verify_tokens.detach(), matched, draft_forwards, diagnostics


@torch.no_grad()
def linear_spec_generate_with_locator_trace(
    model: Any,
    prompt_ids: torch.Tensor,
    *,
    max_new_tokens: int = 128,
    block_size: int = 16,
    temperature: float = 0.0,
    threshold: float = 0.0,
    mask_token_id: Optional[int] = None,
    eos_token_id: Optional[int] = None,
    max_thinking_tokens: Optional[int] = None,
    end_think_token_id: Optional[int] = None,
    tracer: Optional[LocatorTraceWriter] = None,
    request_id: str = "",
    mode: str = "linearspec_lora",
    history_windows: Sequence[int] = (1, 2, 4),
) -> tuple[torch.Tensor, int]:
    if prompt_ids.ndim != 2 or prompt_ids.shape[0] != 1:
        raise ValueError("failure-locator observation requires batch_size == 1")
    if block_size < 2 or max_new_tokens <= 0:
        raise ValueError("block_size must be >=2 and max_new_tokens positive")
    if temperature < 0 or threshold < 0:
        raise ValueError("temperature and threshold must be non-negative")
    windows = tuple(sorted({int(value) for value in history_windows}))
    if not windows or any(value <= 0 for value in windows):
        raise ValueError("history_windows must contain positive integers")
    token_mask_id = int(mask_token_id if mask_token_id is not None else model.config.mask_token_id)
    if eos_token_id is None:
        eos_token_id = getattr(model.config, "eos_token_id", None)

    _set_diffusion_lm(model, False)
    _toggle_adapters(model, False)
    output = model.encoder(
        input_ids=prompt_ids,
        past_key_values=DynamicCache(),
        use_cache=True,
        use_causal_mask=True,
    )
    cache = output.past_key_values
    next_token = _sample_tokens(
        model.diffusion_head(output.last_hidden_state[:, -1:, :]).squeeze(1), temperature
    ).reshape(1, 1)
    nfe = 1
    if eos_token_id is not None and int(next_token.item()) == int(eos_token_id):
        return torch.cat([prompt_ids, next_token], dim=1), nfe

    generated = [next_token]
    total_generated = 1
    round_index = 0
    history: list[RoundHistory] = []
    while total_generated < max_new_tokens:
        generation_offset = total_generated
        remaining_budget = max_new_tokens - generation_offset
        cache_length = int(cache.get_seq_length())
        history_before = summarize_history(history, windows)
        cache, draft_tokens, verifier_tokens, matched, draft_forwards, diagnostics = _draft_verify_round(
            model,
            seed_token=next_token,
            cache=cache,
            block_size=block_size,
            mask_token_id=token_mask_id,
            temperature=float(temperature),
            threshold=float(threshold),
        )
        nfe += draft_forwards + 1
        accept_length = matched + 1
        mismatch_position = matched + 1 if matched < block_size - 1 else None
        _crop_dynamic_cache(cache, cache_length + accept_length)
        accepted_tokens = verifier_tokens[:accept_length].reshape(1, -1)
        eos_hit = False
        effective_accept_length = accept_length
        if eos_token_id is not None:
            eos_positions = (accepted_tokens[0] == int(eos_token_id)).nonzero(as_tuple=True)[0]
            if len(eos_positions) > 0:
                eos_hit = True
                effective_accept_length = int(eos_positions[0].item()) + 1
                accepted_tokens = accepted_tokens[:, :effective_accept_length]
        generated.append(accepted_tokens)
        total_generated += int(accepted_tokens.shape[1])
        next_token = verifier_tokens[accept_length - 1 : accept_length].reshape(1, 1)

        confidences = [float(value) for value in diagnostics["selected_confidence"]]
        good_confidences = confidences[:matched]
        error_confidence = confidences[matched] if mismatch_position is not None else None
        diagnostics["draft_accepted"] = [position < matched for position in range(block_size - 1)]
        record: dict[str, Any] = {
            "schema_version": 1,
            "event": "linearspec_failure_locator_round",
            "backend": "native_pytorch",
            "created_at_unix": time.time(),
            "request_id": str(request_id),
            "mode": mode,
            "round_index": round_index,
            "generation_offset": generation_offset,
            "cache_length": cache_length,
            "remaining_generation_budget": remaining_budget,
            "block_size": block_size,
            "threshold": float(threshold),
            "temperature": float(temperature),
            "draft_forward_passes": draft_forwards,
            "matched_draft_tokens": matched,
            "accept_length": accept_length,
            "effective_accept_length": effective_accept_length,
            "mismatch_position": mismatch_position,
            "full_accept": mismatch_position is None,
            "eos_hit": eos_hit,
            "budget_boundary": remaining_budget < block_size,
            "analysis_valid": (not eos_hit) and remaining_budget >= block_size,
            "accepted_confidence_mean": statistics.fmean(good_confidences) if good_confidences else None,
            "accepted_confidence_min": min(good_confidences) if good_confidences else None,
            "error_confidence": error_confidence,
            "history_before_round": history_before,
            "position": diagnostics,
        }
        if tracer is not None and tracer.detail == "tokens":
            record["draft_token_ids"] = [int(value) for value in draft_tokens[1:].cpu().tolist()]
            record["verifier_token_ids"] = [int(value) for value in verifier_tokens.cpu().tolist()]
        if tracer is not None:
            tracer.write(record)

        history.append(
            RoundHistory(
                accept_length=accept_length,
                mismatch_position=mismatch_position,
                full_accept=mismatch_position is None,
                good_confidences=good_confidences,
                error_confidence=error_confidence,
            )
        )
        round_index += 1
        if eos_hit:
            break
        if end_think_token_id is not None and max_thinking_tokens is not None:
            if total_generated > max_thinking_tokens:
                all_generated = torch.cat(generated, dim=1)
                if not bool((all_generated == int(end_think_token_id)).any().item()):
                    next_token = torch.tensor(
                        [[int(end_think_token_id)]], dtype=torch.long, device=prompt_ids.device
                    )
        if total_generated >= max_new_tokens:
            break

    return torch.cat([prompt_ids, torch.cat(generated, dim=1)], dim=1), nfe
