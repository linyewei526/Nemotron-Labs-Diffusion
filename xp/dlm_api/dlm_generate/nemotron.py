#!/usr/bin/env python3
"""
Nemotron generation algorithm using the model's built-in generate method.
"""

import logging
import inspect
import os
from typing import Optional, Tuple, Union
import torch
from transformers import AutoConfig, AutoModel, PreTrainedModel

from .base import GenerationAlgorithm

logger = logging.getLogger(__name__)


def resolve_linear_speculation_mode(value: Optional[Union[bool, str]]) -> Optional[str]:
    """Normalize a truthy linear-speculation setting into the native method name.

    Linear speculation only has one supported native method, linear_spec_generate
    (the linear_spec_generate_mp variant was removed for simplicity). Returns
    None when speculation should be disabled.
    """
    if value is None or value is False:
        return None
    if value is True:
        return "linear_spec_generate"

    normalized = str(value).strip().lower()
    if normalized in {"", "0", "false", "no", "off", "none"}:
        return None
    if normalized in {"1", "true", "yes", "on", "linear_spec_generate"}:
        return "linear_spec_generate"

    raise ValueError(
        "Unsupported linear_speculation value "
        f"{value!r}. Expected: true, false, or linear_spec_generate."
    )


_SIGNATURE_CACHE = {}  # id(fn) -> (param_names_or_None, accepts_var_kw)


def _filter_kwargs_for_signature(fn, kwargs):
    """Drop kwargs the callable does not declare.

    The upstream HF repo for Nemotron-Labs-Diffusion-8B iterates its generate()
    signature between releases (steps, shift_logits, causal_context come and
    go); this lets eval_suit forward unconditional kwargs without per-revision
    branching at the call sites.
    """
    cache_key = id(fn)
    cached = _SIGNATURE_CACHE.get(cache_key)
    if cached is None:
        try:
            params = inspect.signature(fn).parameters
        except (TypeError, ValueError):
            cached = (None, True)  # treat as accepting everything
        else:
            accepts_var = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
            cached = (None if accepts_var else frozenset(params), accepts_var)
        _SIGNATURE_CACHE[cache_key] = cached
    param_names, accepts_var = cached
    if accepts_var:
        return kwargs
    return {k: v for k, v in kwargs.items() if k in param_names}


class NemotronGeneration(GenerationAlgorithm):
    """Nemotron generation using the model's built-in generate method."""
    
    def __init__(self):
        super().__init__(
            name="nemotron",
            description="Native Nemotron diffusion generation using model's built-in generate method",
            engine="nemotron"
        )
    
    def load_model_class(self, model_path: str, **kwargs) -> PreTrainedModel:
        """
        Load model class for Nemotron generation.
        Always uses standard AutoModel for Nemotron.

        Set NEMOTRON_DLM_PARADIGM=autoregressive to load with causal attention
        (default: bidirectional).
        """
        dlm_paradigm = os.environ.get("NEMOTRON_DLM_PARADIGM", "bidirectional")
        logger.info(f"Loading Nemotron model with standard AutoModel (dlm_paradigm={dlm_paradigm})")
        kwargs.pop("batch_size", None)
        revision = kwargs.get("revision")

        config = AutoConfig.from_pretrained(
            model_path,
            trust_remote_code=True,
            revision=revision,
        )
        if self.max_position_embeddings_override is not None:
            original = config.max_position_embeddings
            config.max_position_embeddings = self.max_position_embeddings_override
            logger.info(
                f"Overriding max_position_embeddings: {original} -> {self.max_position_embeddings_override}"
            )
        else:
            logger.info(
                f"Using default max_position_embeddings={config.max_position_embeddings} "
                f"(pass --max-position-embeddings to override)"
            )

        config.dlm_paradigm = dlm_paradigm
        return AutoModel.from_pretrained(
            model_path,
            config=config,
            trust_remote_code=True,
            **kwargs
        )
    
    def generate(
        self,
        model: PreTrainedModel,
        prompt: torch.Tensor,
        steps: int,
        gen_length: int,
        block_length: int,
        temperature: float = 1.0,
        remasking: bool = True,
        threshold: float = 0.9,
        factor: float = 1.0,
        **kwargs
    ) -> Tuple[torch.Tensor, int]:
        """Generate text using Nemotron's native generate method."""
        if not self.is_available():
            raise RuntimeError("Nemotron generation is not available - model does not have native generate method")

        native_model = self._resolve_native_model(model)

        # Skip the .generate signature gate on the linear-spec path — that
        # dispatches into linear_spec_generate* and never touches .generate,
        # so a stricter signature on .generate is irrelevant.
        _requested_linear_spec = kwargs.get("linear_speculation", None) or getattr(self, "linear_speculation", None)
        _linear_spec_active = resolve_linear_speculation_mode(_requested_linear_spec) is not None
        if not _linear_spec_active and not self._is_nemotron_model_cached(native_model):
            raise RuntimeError("Model does not appear to be a Nemotron model with native generate method")
        
        # Validate and adjust parameters for Nemotron
        validated_args = self.validate_args(
            steps=steps,
            gen_length=gen_length,
            block_length=block_length,
            temperature=temperature,
            remasking=remasking,
            threshold=threshold,
            factor=factor,
            **kwargs
        )
        
        causal_context = kwargs.get("causal_context", True)
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

        logger.debug(
            f"Using Nemotron native generation with args: {validated_args}, "
            f"causal_context={causal_context}, eos_token_id={eos_token_id}, "
            f"eos_early_stop={self.eos_early_stop}, "
            f"max_thinking_tokens={max_thinking_tokens}"
        )
        
        try:
            linear_speculation = kwargs.get("linear_speculation", None)
            linear_speculation_mode = resolve_linear_speculation_mode(
                linear_speculation if linear_speculation is not None else self.linear_speculation
            )

            if linear_speculation_mode:
                draft_lora_only = bool(
                    kwargs.get("draft_lora_only", getattr(self, "draft_lora_only", False))
                )
                # Prefer the dedicated *_lora method when the model exposes it;
                # newer revisions of the public HF repo fold LoRA-awareness into
                # plain linear_spec_generate and ship no _lora variant.
                if linear_speculation_mode == "linear_spec_generate" and draft_lora_only \
                        and hasattr(native_model, "linear_spec_generate_lora"):
                    linear_speculation_method = "linear_spec_generate_lora"
                else:
                    linear_speculation_method = linear_speculation_mode

                if not hasattr(native_model, linear_speculation_method):
                    raise RuntimeError(
                        f"Model does not support {linear_speculation_method} for "
                        f"linear_speculation={linear_speculation!r}"
                    )

                self.last_generate_path = f"native_model.{linear_speculation_method}"
                target_fn = getattr(native_model, linear_speculation_method)
                candidate_kwargs = dict(
                    max_new_tokens=validated_args['gen_length'],
                    block_length=validated_args['block_length'],
                    temperature=validated_args['temperature'],
                    threshold=validated_args['threshold'],
                    end_think_token_id=end_think_token_id,
                    max_thinking_tokens=max_thinking_tokens,
                )
                mask_token_id = getattr(native_model, "mask_token_id",
                                        getattr(native_model.config, "mask_token_id", None))
                if mask_token_id is not None:
                    candidate_kwargs['mask_token_id'] = mask_token_id
                if eos_token_id is not None:
                    candidate_kwargs['eos_token_id'] = eos_token_id

                logger.info(
                    "Using %s (LINEAR_SPECULATION=%s, draft_lora_only=%s)",
                    linear_speculation_method,
                    linear_speculation if linear_speculation is not None else self.linear_speculation,
                    draft_lora_only,
                )
                generate_kwargs = _filter_kwargs_for_signature(target_fn, candidate_kwargs)
                output_ids, nfe = target_fn(prompt, **generate_kwargs)
            else:
                self.last_generate_path = "native_model.generate"
                candidate_kwargs = dict(
                    max_new_tokens=validated_args['gen_length'],
                    steps=validated_args['steps'],
                    block_length=validated_args['block_length'],
                    threshold=validated_args['threshold'],
                    shift_logits=validated_args['shift_logits'],
                    temperature=validated_args['temperature'],
                    causal_context=causal_context,
                    end_think_token_id=end_think_token_id,
                    max_thinking_tokens=max_thinking_tokens,
                )
                if eos_token_id is not None:
                    candidate_kwargs['eos_token_id'] = eos_token_id

                sampler = kwargs.get("sampler", None) or self.sampler
                if sampler is not None:
                    candidate_kwargs['sampler'] = sampler

                generate_kwargs = _filter_kwargs_for_signature(native_model.generate, candidate_kwargs)
                output_ids, nfe = native_model.generate(prompt, **generate_kwargs)
            
            return output_ids, nfe
            
        except Exception as e:
            logger.error(f"Nemotron generation failed: {e}")
            raise RuntimeError(f"Nemotron generation failed: {e}")
    
    def is_available(self) -> bool:
        """
        Check if Nemotron generation is available.
        This requires checking if the currently loaded model has a native generate method.
        Since we don't have access to the model here, we return True and let the generate
        method do the actual validation.
        """
        return True
    
    def _is_nemotron_model(self, model: PreTrainedModel) -> bool:
        """True iff model.generate has max_new_tokens + block_length params.

        That pair distinguishes the Nemotron block-diffusion sampler from
        HF's standard .generate. Other diffusion-specific params (steps,
        shift_logits, …) drop in and out across HF revisions, so we don't
        require them.
        """
        if not hasattr(model, "generate"):
            return False
        try:
            params = inspect.signature(model.generate).parameters
        except (TypeError, ValueError):
            return False
        return "max_new_tokens" in params and "block_length" in params

    def _resolve_native_model(self, model: PreTrainedModel) -> PreTrainedModel:
        """Cached PEFT unwrap. PEFT wrapping doesn't change post-load, so we
        resolve once and stash the result on the algorithm instance."""
        cached = getattr(self, "_cached_native_model", None)
        if cached is not None and cached[0] is model:
            return cached[1]
        native = self._get_native_generation_model(model)
        self._cached_native_model = (model, native)
        return native

    def _is_nemotron_model_cached(self, native_model: PreTrainedModel) -> bool:
        cached = getattr(self, "_cached_is_nemotron", None)
        if cached is not None and cached[0] is native_model:
            return cached[1]
        result = self._is_nemotron_model(native_model)
        self._cached_is_nemotron = (native_model, result)
        return result

    def _get_native_generation_model(self, model: PreTrainedModel) -> PreTrainedModel:
        """Unwrap PEFT-style wrappers and return the underlying native generation model."""
        current = model

        get_base_model = getattr(current, "get_base_model", None)
        if callable(get_base_model):
            try:
                base_model = get_base_model()
                if base_model is not None:
                    current = base_model
            except Exception as e:
                logger.debug(f"Could not unwrap model via get_base_model(): {e}")

        for attr in ("model", "base_model"):
            candidate = getattr(current, attr, None)
            if candidate is not None and candidate is not current and hasattr(candidate, "generate"):
                current = candidate
                break

        return current
    
    def get_required_args(self):
        """Get the required arguments with Nemotron-specific defaults."""
        return {
            'steps': 128,
            'gen_length': 128,
            'block_length': 32,
            'temperature': 1.0,  # Not used by Nemotron but kept for compatibility
            'remasking': True,   # Not used by Nemotron but kept for compatibility
            'threshold': 0.9,    # Nemotron default
            'factor': 1.0,       # Not used by Nemotron but kept for compatibility
            # shift_logits is a model/checkpoint property set via --shift-logits CLI flag.
            # Default: False (diffusion eval). Set True for AR eval.
            'shift_logits': getattr(self, 'shift_logits', False),
        }
