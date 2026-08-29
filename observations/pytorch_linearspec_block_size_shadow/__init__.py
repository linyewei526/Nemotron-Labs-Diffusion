"""Paired multi-block-size observation for native PyTorch LinearSpec."""

from .shadow_generation import (
    decompose_pair,
    linear_spec_generate_with_block_size_shadows,
)

__all__ = ["decompose_pair", "linear_spec_generate_with_block_size_shadows"]
