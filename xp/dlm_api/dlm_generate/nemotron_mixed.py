#!/usr/bin/env python3
"""
Nemotron Mixed AR+dLM generation algorithm.

Loads model code from a separate HF code repo (NEMOTRON_CODE_REPO) that
contains the ``generate_mixed`` method, while weights come from the standard
model path.  When ``ar_weight > 0``, generation is dispatched to
``model.generate_mixed()``; otherwise it falls back to ``model.generate()``.
"""

import inspect
import json
import logging
import os
from typing import Tuple

import torch
from transformers import PreTrainedModel

from .nemotron import NemotronGeneration

logger = logging.getLogger(__name__)


class NemotronMixedGeneration(NemotronGeneration):
    """Nemotron generation with mixed AR + dLM logit support."""

    def __init__(self):
        # Bypass NemotronGeneration.__init__ to set our own name/description,
        # but keep engine="nemotron" so model/tokenizer sharing works.
        super(NemotronGeneration, self).__init__(
            name="nemotron_mixed",
            description="Nemotron mixed AR+dLM generation via model.generate_mixed()",
            engine="nemotron",
        )
        self.ar_weight: float = 0.0

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------
    def load_model_class(self, model_path: str, **kwargs) -> PreTrainedModel:
        """
        Load the model class.

        When the ``NEMOTRON_CODE_REPO`` env var is set, the model class is
        loaded from that repo (which contains ``generate_mixed``), while the
        checkpoint weights are loaded from *model_path*.  This mirrors the
        two-source pattern used in the interactive chat script.

        Falls back to the standard ``NemotronGeneration`` loader when the
        env var is absent.
        """
        code_repo = os.environ.get("NEMOTRON_CODE_REPO", "")
        if not code_repo:
            logger.info("NEMOTRON_CODE_REPO not set – falling back to standard Nemotron loader")
            return super().load_model_class(model_path, **kwargs)

        kwargs.pop("batch_size", None)
        torch_dtype = kwargs.pop("torch_dtype", torch.bfloat16)
        hf_token = os.environ.get("HF_TOKEN")

        # 1. Resolve code source (download if remote)
        code_source = code_repo
        if not os.path.isdir(code_source):
            from huggingface_hub import snapshot_download
            logger.info(f"Downloading code repo: {code_source}")
            code_source = snapshot_download(code_source, token=hf_token)

        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        logger.info(f"Loading model class from code repo: {code_source}")
        config_class = get_class_from_dynamic_module(
            "configuration_ministral_dlm.MinistralDLMConfig",
            code_source,
            token=hf_token,
        )
        model_class = get_class_from_dynamic_module(
            "modeling_ministral_dlm.MinistralDiffEncoderModel",
            code_source,
            token=hf_token,
        )

        # 2. Build config from the WEIGHTS source so dimensions match
        if os.path.isdir(model_path):
            cfg_path = os.path.join(model_path, "config.json")
        else:
            from huggingface_hub import hf_hub_download
            logger.info(f"Fetching config.json from weights repo: {model_path}")
            cfg_path = hf_hub_download(model_path, "config.json", token=hf_token)

        with open(cfg_path) as f:
            weight_config_dict = json.load(f)
        config = config_class.from_dict(weight_config_dict)
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
        logger.info(
            f"Config: hidden_size={config.hidden_size}, "
            f"num_hidden_layers={config.num_hidden_layers}"
        )

        # 3. Load pretrained weights using the code repo's model class
        logger.info(f"Loading weights from: {model_path}")
        model = model_class.from_pretrained(
            model_path,
            config=config,
            torch_dtype=torch_dtype,
            token=hf_token,
        )
        return model

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
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
        **kwargs,
    ) -> Tuple[torch.Tensor, int]:
        """Generate text, dispatching to generate_mixed when ar_weight > 0."""

        ar_weight = float(kwargs.pop("ar_weight", self.ar_weight))

        if ar_weight > 0.0 and hasattr(model, "generate_mixed"):
            return self._generate_mixed(
                model=model,
                prompt=prompt,
                steps=steps,
                gen_length=gen_length,
                block_length=block_length,
                temperature=temperature,
                remasking=remasking,
                threshold=threshold,
                ar_weight=ar_weight,
                **kwargs,
            )

        return super().generate(
            model=model,
            prompt=prompt,
            steps=steps,
            gen_length=gen_length,
            block_length=block_length,
            temperature=temperature,
            remasking=remasking,
            threshold=threshold,
            factor=factor,
            **kwargs,
        )

    def _generate_mixed(
        self,
        model: PreTrainedModel,
        prompt: torch.Tensor,
        steps: int,
        gen_length: int,
        block_length: int,
        temperature: float,
        remasking: bool,
        threshold: float,
        ar_weight: float,
        **kwargs,
    ) -> Tuple[torch.Tensor, int]:
        """Call model.generate_mixed() with the correct arguments."""

        validated_args = self.validate_args(
            steps=steps,
            gen_length=gen_length,
            block_length=block_length,
            temperature=temperature,
            remasking=remasking,
            threshold=threshold,
            **kwargs,
        )

        causal_context = kwargs.get("causal_context", True)
        eos_token_id = None
        if self.eos_early_stop and self.tokenizer is not None:
            eos_token_id = getattr(self.tokenizer, "eos_token_id", None)

        logger.info(
            f"Using Nemotron mixed generation: ar_weight={ar_weight}, "
            f"args={validated_args}, causal_context={causal_context}, "
            f"eos_token_id={eos_token_id}"
        )

        self.last_generate_path = "model.generate_mixed"

        try:
            output_ids, nfe = model.generate_mixed(
                prompt_ids=prompt,
                max_new_tokens=validated_args["gen_length"],
                steps=validated_args["steps"],
                block_length=validated_args["block_length"],
                threshold=validated_args["threshold"],
                ar_weight=ar_weight,
                temperature=validated_args["temperature"],
                eos_token_id=eos_token_id,
                remasking="low_confidence",
                neg_entropy=False,
            )
            return output_ids, nfe
        except Exception as e:
            logger.error(f"Nemotron mixed generation failed: {e}")
            raise RuntimeError(f"Nemotron mixed generation failed: {e}")

    # ------------------------------------------------------------------
    # Model validation
    # ------------------------------------------------------------------
    def _is_nemotron_model(self, model: PreTrainedModel) -> bool:
        """Accept models with either generate or generate_mixed."""
        for method_name in ("generate_mixed", "generate"):
            if not hasattr(model, method_name):
                continue
            try:
                sig = inspect.signature(getattr(model, method_name))
                params = list(sig.parameters.keys())
                if method_name == "generate_mixed":
                    expected = ["prompt_ids", "max_new_tokens", "steps", "block_length", "ar_weight"]
                else:
                    expected = ["max_new_tokens", "steps", "block_length", "threshold", "shift_logits"]
                if all(p in params for p in expected):
                    return True
            except Exception:
                continue
        return False

    # ------------------------------------------------------------------
    # Required args
    # ------------------------------------------------------------------
    def get_required_args(self):
        args = super().get_required_args()
        args["ar_weight"] = getattr(self, "ar_weight", 0.0)
        return args
