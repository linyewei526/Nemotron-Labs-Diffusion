#!/usr/bin/env python3
"""Method-local LoRA routing with token-segment masks.

The bundled LinearSpec adapter targets only attention ``o_proj`` modules.  This
module loads those tensors without installing PEFT wrappers and applies the
LoRA delta only to positions selected by ``SegmentedLoraController``.  Nothing
in the model repository or the existing evaluation paths is modified.
"""

from __future__ import annotations

import json
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

import torch
import torch.nn.functional as F
from torch import nn
from safetensors.torch import safe_open


class SegmentedLoraController:
    """Process-local routing state; generation is serialized by the server."""

    def __init__(self) -> None:
        self._mode: str = "off"
        self._mask: Optional[torch.Tensor] = None

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def mask(self) -> Optional[torch.Tensor]:
        return self._mask

    @contextmanager
    def use_all(self) -> Iterator[None]:
        previous = (self._mode, self._mask)
        self._mode, self._mask = "all", None
        try:
            yield
        finally:
            self._mode, self._mask = previous

    @contextmanager
    def use_mask(self, mask: torch.Tensor) -> Iterator[None]:
        if mask.ndim != 3 or mask.shape[-1] != 1:
            raise ValueError("LoRA route mask must have shape [batch, sequence, 1]")
        previous = (self._mode, self._mask)
        self._mode, self._mask = "mask", mask
        try:
            yield
        finally:
            self._mode, self._mask = previous

    @contextmanager
    def disabled(self) -> Iterator[None]:
        previous = (self._mode, self._mask)
        self._mode, self._mask = "off", None
        try:
            yield
        finally:
            self._mode, self._mask = previous


class SegmentedLoraLinear(nn.Module):
    """A frozen base Linear plus a conditionally routed frozen LoRA delta."""

    def __init__(
        self,
        base_layer: nn.Linear,
        lora_a: torch.Tensor,
        lora_b: torch.Tensor,
        scaling: float,
        controller: SegmentedLoraController,
    ) -> None:
        super().__init__()
        if base_layer.bias is not None:
            raise ValueError("Bundled o_proj is expected to be bias-free")
        if tuple(lora_a.shape)[1] != base_layer.in_features:
            raise ValueError("LoRA A input dimension does not match base layer")
        if tuple(lora_b.shape)[0] != base_layer.out_features:
            raise ValueError("LoRA B output dimension does not match base layer")
        if tuple(lora_b.shape)[1] != tuple(lora_a.shape)[0]:
            raise ValueError("LoRA A/B rank mismatch")
        self.base_layer = base_layer
        target_device = base_layer.weight.device
        if lora_a.dtype != lora_b.dtype:
            raise ValueError("LoRA A/B dtype mismatch")
        # PEFT keeps fp32 adapter weights by default even on a BF16 base model.
        # Preserve that dtype so this method-local implementation has the same
        # draft logits as the established linearspec_lora path.
        self.register_buffer("lora_a", lora_a.to(device=target_device))
        self.register_buffer("lora_b", lora_b.to(device=target_device))
        self.scaling = float(scaling)
        self.controller = controller

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.base_layer(x)
        result_dtype = result.dtype
        mode = self.controller.mode
        if mode == "off":
            return result
        delta = F.linear(F.linear(x.to(self.lora_a.dtype), self.lora_a), self.lora_b)
        delta = delta * self.scaling
        if mode == "all":
            return (result + delta).to(result_dtype)
        route = self.controller.mask
        if route is None or tuple(route.shape[:2]) != tuple(x.shape[:2]):
            raise RuntimeError(
                f"LoRA route shape {None if route is None else tuple(route.shape)} "
                f"does not match activation shape {tuple(x.shape)}"
            )
        routed = delta * route.to(device=result.device, dtype=delta.dtype)
        return (result + routed).to(result_dtype)


_LAYER_KEY = re.compile(
    r"^base_model\.model\.encoder\.layers\.(\d+)\.self_attn\.o_proj\.lora_([AB])\.weight$"
)


def install_segmented_lora(
    model: nn.Module,
    adapter_dir: str | Path,
) -> SegmentedLoraController:
    """Install method-private wrappers and return their shared controller."""

    adapter_path = Path(adapter_dir).resolve()
    config_path = adapter_path / "adapter_config.json"
    weights_path = adapter_path / "adapter_model.safetensors"
    if not config_path.is_file() or not weights_path.is_file():
        raise FileNotFoundError(f"Incomplete LoRA adapter directory: {adapter_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    targets = set(config.get("target_modules") or [])
    if targets != {"o_proj"}:
        raise ValueError(f"This isolated implementation requires target_modules=['o_proj'], got {targets}")
    if float(config.get("lora_dropout", 0.0)) != 0.0:
        raise ValueError("Non-zero LoRA dropout is not supported for deterministic inference")
    rank = int(config["r"])
    scaling = float(config["lora_alpha"]) / rank

    by_layer: dict[int, dict[str, torch.Tensor]] = {}
    with safe_open(str(weights_path), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            match = _LAYER_KEY.match(key)
            if match is None:
                raise ValueError(f"Unexpected tensor in bundled adapter: {key}")
            layer_idx, side = int(match.group(1)), match.group(2)
            by_layer.setdefault(layer_idx, {})[side] = handle.get_tensor(key)

    layers = list(model.encoder.layers)
    if set(by_layer) != set(range(len(layers))):
        raise ValueError(
            f"Adapter layers {sorted(by_layer)} do not match model layers 0..{len(layers) - 1}"
        )
    controller = SegmentedLoraController()
    for layer_idx, layer in enumerate(layers):
        pair = by_layer[layer_idx]
        if set(pair) != {"A", "B"}:
            raise ValueError(f"Missing LoRA tensor for layer {layer_idx}: {sorted(pair)}")
        base = layer.self_attn.o_proj
        if not isinstance(base, nn.Linear):
            raise TypeError(f"Expected nn.Linear o_proj at layer {layer_idx}, got {type(base).__name__}")
        layer.self_attn.o_proj = SegmentedLoraLinear(
            base, pair["A"], pair["B"], scaling, controller
        )
    return controller

