#!/usr/bin/env python3
"""Process-local SGLang LinearSpec for the frozen B200 latency policy."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Tuple

from sglang_dynamic_block_history_signal.dynamic_shadow_algorithm import (
    DynamicBlockShadowLinearSpec,
)

from sglang_b200_latency_dynamic_block_policy.policy_runtime import choose_action


class B200LatencyDynamicBlockLinearSpec(DynamicBlockShadowLinearSpec):
    """Reuse tested variable-block mechanics and replace only the action rule."""

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        policy_path = os.environ.get("NLD_B200_LATENCY_POLICY_PATH", "")
        if not policy_path:
            raise RuntimeError("NLD_B200_LATENCY_POLICY_PATH is required")
        with open(policy_path, "r", encoding="utf-8") as file:
            payload = json.load(file)
        self._b200_policy: Dict[str, Any] = payload.get("policy", payload)
        if self._b200_policy.get("policy_type") != "b200_latency_scalar_value":
            raise ValueError("unexpected B200 latency policy type")
        if self._b200_policy.get("schema_version") != 2:
            raise ValueError(
                "B200 latency policy predates the concurrency-specific "
                "cold-start protocol; rerun offline search"
            )
        self._virtual_concurrency = int(
            os.environ.get("NLD_B200_LATENCY_CONCURRENCY", "0")
        )
        if str(self._virtual_concurrency) not in self._b200_policy.get(
            "concurrency_profiles", {}
        ):
            raise ValueError(
                f"policy has no profile for C={self._virtual_concurrency}"
            )
        lambda_text = os.environ.get("NLD_B200_LATENCY_LAMBDA", "").strip()
        self._lambda_override = float(lambda_text) if lambda_text else None
        self.policy_mode = "b200_latency_frozen"
        self.policy_target = f"c{self._virtual_concurrency}"

    def _choose_action(
        self, rid: str, features: Dict[str, Any]
    ) -> Tuple[int, str, Dict[str, float]]:
        if int(features.get("history_rounds", 0)) == 0:
            profile = self._b200_policy["concurrency_profiles"][
                str(self._virtual_concurrency)
            ]
            block = int(profile["cold_start_block"])
            return block, f"b200_latency_cold_l{block}", {
                "virtual_concurrency": float(self._virtual_concurrency)
            }
        action, scores = choose_action(
            self._b200_policy,
            features,
            self._virtual_concurrency,
            self._lambda_override,
        )
        return action, f"b200_latency_c{self._virtual_concurrency}_l{action}", scores
