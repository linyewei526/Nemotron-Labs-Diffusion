#!/usr/bin/env python3
"""Multi-row hybrid attention masks and DynamicCache helpers."""

from __future__ import annotations

import torch
from transformers.cache_utils import DynamicCache


def build_multi_hybrid_attention_mask(
    *,
    cache_length: int,
    verifier_length: int,
    branch_prefix_lengths: tuple[int, ...],
    prospective_length: int,
    query_length: int,
    device: torch.device,
) -> torch.Tensor:
    """Build a dense allow-mask for one verifier and up to three branches.

    Every branch has a causal prefix followed by one bidirectional prospective
    block.  All rows share ``query_length``; positions after a row's valid
    length are padding queries with a safe causal mask, and valid queries never
    attend to padding keys.
    """

    if cache_length < 0:
        raise ValueError("cache_length must be non-negative")
    if verifier_length < 2 or prospective_length < 2:
        raise ValueError("verifier/prospective lengths must be at least 2")
    if not 1 <= len(branch_prefix_lengths) <= 3:
        raise ValueError("multi-row fused path requires one to three branches")
    valid_lengths = (verifier_length,) + tuple(
        prefix + prospective_length for prefix in branch_prefix_lengths
    )
    if query_length != max(valid_lengths):
        raise ValueError("query_length must equal the longest valid row")
    if any(prefix < 1 for prefix in branch_prefix_lengths):
        raise ValueError("branch prefixes must be positive")

    rows = len(valid_lengths)
    key_length = cache_length + query_length
    mask = torch.zeros(
        (rows, 1, query_length, key_length), dtype=torch.bool, device=device
    )
    if cache_length:
        mask[:, :, :, :cache_length] = True
    causal = torch.ones(
        (query_length, query_length), dtype=torch.bool, device=device
    ).tril()

    # Verifier valid queries are causal. Padding queries are also given a safe
    # causal row but their outputs are ignored.
    mask[0, 0, :, cache_length:] = causal

    for row_index, prefix in enumerate(branch_prefix_lengths, start=1):
        valid_length = prefix + prospective_length
        prefix_causal = torch.ones(
            (prefix, prefix), dtype=torch.bool, device=device
        ).tril()
        mask[
            row_index,
            0,
            :prefix,
            cache_length : cache_length + prefix,
        ] = prefix_causal
        mask[
            row_index,
            0,
            prefix:valid_length,
            cache_length : cache_length + valid_length,
        ] = True
        if valid_length < query_length:
            mask[
                row_index,
                0,
                valid_length:,
                cache_length:,
            ] = causal[valid_length:]
    return mask


def build_hybrid_attention_mask(
    *,
    cache_length: int,
    block_length: int,
    candidate_position: int,
    device: torch.device,
) -> torch.Tensor:
    """Return an SDPA boolean allow-mask for verifier and prospective rows.

    Row 0 is a padded causal verifier.  Row 1 contains a causal reconstruction
    prefix of length ``candidate_position`` followed by a full bidirectional
    draft block of length ``block_length``.
    """

    if cache_length < 0:
        raise ValueError("cache_length must be non-negative")
    if block_length < 2:
        raise ValueError("block_length must be at least 2")
    if not 1 <= candidate_position < block_length:
        raise ValueError("candidate_position must be in [1, block_length)")
    query_length = candidate_position + block_length
    key_length = cache_length + query_length
    mask = torch.zeros((2, 1, query_length, key_length), dtype=torch.bool, device=device)
    if cache_length:
        mask[:, :, :, :cache_length] = True

    # Row 0: all query slots are causal. Slots after block_length are padding;
    # their outputs are ignored, but giving them a valid causal row avoids an
    # all-masked SDPA query and cannot affect earlier valid queries.
    causal = torch.ones((query_length, query_length), dtype=torch.bool, device=device).tril()
    mask[0, 0, :, cache_length:] = causal

    # Row 1 prefix P: causal reconstruction of [seed, accepted-before-A].
    prefix = candidate_position
    prefix_causal = torch.ones((prefix, prefix), dtype=torch.bool, device=device).tril()
    mask[1, 0, :prefix, cache_length : cache_length + prefix] = prefix_causal

    # Row 1 draft N: every query sees the whole causal P and the whole N block.
    mask[1, 0, prefix:, cache_length:] = True
    return mask


def repeat_dynamic_cache(cache: DynamicCache, repeats: int) -> DynamicCache:
    """Create an independent repeated cache without mutating the canonical one."""

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
    """Keep one batch row and crop it to the committed causal prefix."""

    device = None
    if hasattr(cache, "layers") and cache.layers:
        device = cache.layers[0].keys.device
    elif getattr(cache, "key_cache", None):
        device = cache.key_cache[0].device
    indices = torch.tensor([batch_index], dtype=torch.long, device=device)
    cache.batch_select_indices(indices)
    cache.crop(max_length)
    return cache
