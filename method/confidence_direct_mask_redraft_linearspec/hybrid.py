#!/usr/bin/env python3
"""Fixed-length hybrid mask and DynamicCache helpers for direct MASK redraft."""

from __future__ import annotations

import torch
from transformers.cache_utils import DynamicCache


def build_direct_mask_redraft_attention_mask(
    *,
    cache_length: int,
    verify_length: int,
    prospective_length: int,
    trigger_position: int,
    device: torch.device,
) -> torch.Tensor:
    """Build a bool SDPA allow-mask for causal verify and all-MASK redraft.

    Both fused rows have the same fixed draft length ``L``.  Row 0 contains
    the causal verifier tokens followed by ignored padding in the extra
    reconstruction-prefix slots.
    Row 1 contains a causal reconstruction prefix of ``trigger_position``
    tokens followed by a bidirectional all-MASK segment of fixed length
    ``prospective_length``.
    """

    if cache_length < 0:
        raise ValueError("cache_length must be non-negative")
    if prospective_length < 2:
        raise ValueError("prospective_length must be at least 2")
    if verify_length != prospective_length:
        raise ValueError("strict direct MASK-redraft requires verify_length == prospective_length")
    if not 1 <= trigger_position < verify_length:
        raise ValueError("trigger_position must be inside the verifier draft")
    query_length = trigger_position + prospective_length
    if verify_length > query_length:
        raise ValueError("Verifier does not fit in the fused query length")
    key_length = cache_length + query_length
    mask = torch.zeros((2, 1, query_length, key_length), dtype=torch.bool, device=device)
    if cache_length:
        mask[:, :, :, :cache_length] = True

    # Row 0 is causal for all slots. Outputs beyond verify_length are ignored;
    # allowing a valid causal row avoids fully-masked SDPA padding queries and
    # cannot affect any earlier verifier query.
    causal = torch.ones((query_length, query_length), dtype=torch.bool, device=device).tril()
    mask[0, 0, :, cache_length:] = causal

    # Row 1 prefix is a causal, base-weight reconstruction of draft[:p].
    prefix = trigger_position
    prefix_causal = torch.ones((prefix, prefix), dtype=torch.bool, device=device).tril()
    mask[1, 0, :prefix, cache_length : cache_length + prefix] = prefix_causal

    # Every MASK query sees the full reconstructed prefix and all L MASK slots.
    mask[1, 0, prefix:, cache_length:] = True
    return mask


def repeat_dynamic_cache(cache: DynamicCache, repeats: int) -> DynamicCache:
    """Create an independent batch-repeat without mutating canonical cache."""

    if repeats <= 0:
        raise ValueError("repeats must be positive")
    data: list[tuple[torch.Tensor, torch.Tensor]] = []
    if hasattr(cache, "layers"):
        for layer in cache.layers:
            keys = layer.keys
            values = layer.values
            if keys is None or values is None:
                raise ValueError("Cannot repeat an uninitialized DynamicCache layer")
            data.append(
                (
                    keys.repeat_interleave(repeats, dim=0),
                    values.repeat_interleave(repeats, dim=0),
                )
            )
    else:
        for keys, values in zip(cache.key_cache, cache.value_cache):
            data.append(
                (
                    keys.repeat_interleave(repeats, dim=0),
                    values.repeat_interleave(repeats, dim=0),
                )
            )
    return DynamicCache(ddp_cache_data=data)


def select_and_crop_cache(
    cache: DynamicCache,
    *,
    batch_index: int,
    max_length: int,
) -> DynamicCache:
    """Keep a causal verifier row and crop to its committed input prefix."""

    device = None
    if hasattr(cache, "layers") and cache.layers:
        device = cache.layers[0].keys.device
    elif getattr(cache, "key_cache", None):
        device = cache.key_cache[0].device
    indices = torch.tensor([batch_index], dtype=torch.long, device=device)
    cache.batch_select_indices(indices)
    cache.crop(max_length)
    return cache
