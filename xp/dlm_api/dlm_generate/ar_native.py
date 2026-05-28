#!/usr/bin/env python3
"""
AR Native generation algorithm using the model's ar_generate() method.

This calls model.ar_generate() which implements efficient autoregressive
generation with causal attention and shifted logits, using 1 forward pass
per token with EOS early stopping.

Requires the model to expose ar_generate(), defined in modeling.py.
"""

import logging
from typing import Optional, Tuple

import torch
from transformers import AutoModel, PreTrainedModel

from .base import GenerationAlgorithm

logger = logging.getLogger(__name__)


class ArNativeGeneration(GenerationAlgorithm):
    """AR generation using the model's ar_generate() method."""

    def __init__(self):
        super().__init__(
            name="ar_native",
            description="Autoregressive generation via model.ar_generate()",
            engine="ar_native",
        )

    def load_model_class(self, model_path: str, **kwargs) -> PreTrainedModel:
        kwargs.pop("batch_size", None)
        logger.info("Loading AR-enabled model with AutoModel (trust_remote_code=True)")
        model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            **kwargs,
        )
        if not hasattr(model, "ar_generate"):
            raise RuntimeError(
                f"Loaded model ({type(model).__name__}) does not expose ar_generate(). "
                "Make sure you are loading from a model repository whose modeling.py "
                "defines MinistralDiffEncoderModel with ar_generate."
            )
        logger.info("Verified model exposes ar_generate()")
        return model

    def generate(
        self,
        model: PreTrainedModel,
        prompt: torch.Tensor,
        steps: int,
        gen_length: int,
        block_length: int,
        temperature: float = 0.0,
        remasking: bool = True,
        threshold: Optional[float] = None,
        factor: float = 1.0,
        **kwargs,
    ) -> Tuple[torch.Tensor, int]:
        """Generate text autoregressively via model.ar_generate()."""
        if not hasattr(model, "ar_generate"):
            raise RuntimeError("Model does not have an ar_generate() method")

        eos_token_id = None
        if self.eos_early_stop and self.tokenizer is not None:
            eos_token_id = getattr(self.tokenizer, "eos_token_id", None)

        max_thinking_tokens = kwargs.get("max_thinking_tokens", None) or self.max_thinking_tokens
        end_think_token_id = None
        if max_thinking_tokens is not None and self.tokenizer is not None:
            ids = self.tokenizer.encode("</think>", add_special_tokens=False)
            if ids:
                end_think_token_id = ids[-1]
                logger.info(
                    "Thinking budget enabled: max_thinking_tokens=%d, "
                    "end_think_token_id=%d (from tokenizer)",
                    max_thinking_tokens, end_think_token_id,
                )
            else:
                logger.warning(
                    "max_thinking_tokens=%d set but tokenizer cannot encode '</think>'; "
                    "thinking budget will be ignored",
                    max_thinking_tokens,
                )
                max_thinking_tokens = None

        logger.info(
            "AR native generate: max_new_tokens=%d, temperature=%s, eos_token_id=%s, "
            "eos_early_stop=%s, max_thinking_tokens=%s",
            gen_length, temperature, eos_token_id, self.eos_early_stop,
            max_thinking_tokens,
        )

        with torch.no_grad():
            generate_kwargs = dict(
                prompt_ids=prompt,
                max_new_tokens=gen_length,
                temperature=temperature,
            )
            if eos_token_id is not None:
                generate_kwargs['eos_token_id'] = eos_token_id
            if end_think_token_id is not None:
                generate_kwargs['end_think_token_id'] = end_think_token_id
            if max_thinking_tokens is not None:
                generate_kwargs['max_thinking_tokens'] = max_thinking_tokens

            output_ids, nfe = model.ar_generate(**generate_kwargs)

        logger.info("AR native generation complete: output shape %s, nfe=%d", output_ids.shape, nfe)
        return output_ids, nfe

    def is_available(self) -> bool:
        return True

    def get_required_args(self):
        return {
            "steps": 0,
            "gen_length": 128,
            "block_length": 1,
            "temperature": 0.0,
            "remasking": False,
            "threshold": None,
            "factor": 1.0,
            "shift_logits": False,
        }
