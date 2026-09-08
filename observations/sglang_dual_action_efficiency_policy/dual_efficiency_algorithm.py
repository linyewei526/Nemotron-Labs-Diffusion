#!/usr/bin/env python3
"""Process-local SGLang LinearSpec for the S8/S16 efficiency profiles."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Tuple

from sglang_dynamic_block_history_signal.dynamic_shadow_algorithm import (
    DynamicBlockShadowLinearSpec,
)
from sglang_dual_action_efficiency_policy.policy_runtime import (
    choose_action,
    profile_spec,
)


class DualActionEfficiencyLinearSpec(DynamicBlockShadowLinearSpec):
    """Keep validated three-shadow mechanics and replace only the action rule."""

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        policy_path = os.environ.get("NLD_DUAL_EFF_POLICY_PATH", "")
        if not policy_path:
            raise RuntimeError("NLD_DUAL_EFF_POLICY_PATH is required")
        with open(policy_path, "r", encoding="utf-8") as file:
            payload = json.load(file)
        self._dual_policy: Dict[str, Any] = payload.get("policy", payload)
        if self._dual_policy.get("policy_type") != "dual_action_space_scalar_efficiency":
            raise ValueError("unexpected dual-action efficiency policy type")
        self._dual_profile = os.environ.get("NLD_DUAL_EFF_PROFILE", "").strip().lower()
        configured = profile_spec(self._dual_policy, self._dual_profile)
        self._cold_start_block = int(configured["cold_start_block"])
        lambda_text = os.environ.get("NLD_DUAL_EFF_POLICY_LAMBDA", "").strip()
        self._lambda_override = float(lambda_text) if lambda_text else None
        self.policy_mode = "dual_action_efficiency_frozen"
        self.policy_target = self._dual_profile

    def _choose_action(
        self, rid: str, features: Dict[str, Any]
    ) -> Tuple[int, str, Dict[str, float]]:
        if int(features.get("history_rounds", 0)) == 0:
            return (
                self._cold_start_block,
                f"dual_eff_{self._dual_profile}_cold_l{self._cold_start_block}",
                {},
            )
        action, scores = choose_action(
            self._dual_policy,
            features,
            self._dual_profile,
            self._lambda_override,
        )
        return action, f"dual_eff_{self._dual_profile}_l{action}", scores

