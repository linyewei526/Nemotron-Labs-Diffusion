#!/usr/bin/env python3
"""Process-local SGLang LinearSpec using one unified scalar block policy."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Tuple

from sglang_dynamic_block_history_signal.dynamic_shadow_algorithm import (
    DynamicBlockShadowLinearSpec,
)

from sglang_unified_scalar_block_policy.policy_runtime import choose_action


class UnifiedScalarBlockLinearSpec(DynamicBlockShadowLinearSpec):
    """Reuse the validated same-bucket L8/L16/L32 shadow implementation.

    Only the pre-outcome action rule changes.  The parent still owns causal KV
    submission, chosen-last shadow ordering, trace schema v2, and canonical-only
    decode statistics.
    """

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        policy_path = os.environ.get("NLD_UNIFIED_BLOCK_POLICY_PATH", "")
        if not policy_path:
            raise RuntimeError(
                "NLD_UNIFIED_BLOCK_POLICY_PATH is required for unified validation"
            )
        with open(policy_path, "r", encoding="utf-8") as file:
            payload = json.load(file)
        self._unified_policy: Dict[str, Any] = payload.get("policy", payload)
        if self._unified_policy.get("policy_type") != "unified_scalar_value":
            raise ValueError("unexpected unified block policy type")
        lambda_text = os.environ.get("NLD_UNIFIED_BLOCK_POLICY_LAMBDA", "").strip()
        self._lambda_override = float(lambda_text) if lambda_text else None
        self.policy_mode = "unified_scalar_frozen"
        self.policy_target = "unified"

    def _choose_action(
        self, rid: str, features: Dict[str, Any]
    ) -> Tuple[int, str, Dict[str, float]]:
        if int(features.get("history_rounds", 0)) == 0:
            return 16, "unified_cold_start_l16", {}
        action, scores = choose_action(
            self._unified_policy, features, self._lambda_override
        )
        return action, f"unified_scalar_l{action}", scores

