#!/usr/bin/env python3
"""SGLang LinearSpec shadow trace with genuine causal-verifier history.

The tested dynamic shadow implementation remains untouched.  This subclass
changes only branch instrumentation, history construction, and the frozen
action rule.  Draft logits are retained for trace auditing but are never a
legal policy input in this experiment.
"""

from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Sequence, Tuple

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner

from sglang_dynamic_block_history_signal.dynamic_shadow_algorithm import (
    DynamicBlockShadowLinearSpec,
    _mean,
)
from sglang_b200_verify_efficiency_policy.policy_runtime import choose_action


def _rolling_mean(rows: List[Dict[str, Any]], key: str) -> float | None:
    return _mean(row.get(key) for row in rows)


class VerifyEfficiencyShadowLinearSpec(DynamicBlockShadowLinearSpec):
    """Collect B8/B16/B32 outcomes and only expose verifier history to policy."""

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        self.model_size = os.environ.get("NLD_B200_VERIFY_MODEL_SIZE", "8b").lower()
        self._verify_policy: Dict[str, Any] | None = None
        policy_path = os.environ.get("NLD_B200_VERIFY_POLICY_PATH", "")
        if policy_path:
            import json

            with open(policy_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            self._verify_policy = payload.get("policy", payload)
        self._verify_concurrency = int(
            os.environ.get("NLD_B200_VERIFY_CONCURRENCY", "0")
        )

    @staticmethod
    def _verifier_position_metrics(
        logits: torch.Tensor,
        verifier_tokens: torch.Tensor,
        draft_tokens: torch.Tensor,
        mask_id: int,
    ) -> Dict[str, List[float]]:
        """Metrics from causal-verify logits aligned to each checked draft token."""
        rows = logits.float().clone()
        rows[:, mask_id] = -torch.inf
        log_z = torch.logsumexp(rows, dim=-1)
        verifier_selected = rows.gather(
            1, verifier_tokens.long().unsqueeze(1)
        ).squeeze(1)
        draft_selected = rows.gather(1, draft_tokens.long().unsqueeze(1)).squeeze(1)
        verifier_confidence = torch.exp(verifier_selected - log_z)
        draft_probability = torch.exp(draft_selected - log_z)
        top_values = torch.topk(rows, k=2, dim=-1).values
        top_probs = torch.exp(top_values - log_z.unsqueeze(1))
        margin = top_probs[:, 0] - top_probs[:, 1]
        probabilities = torch.softmax(rows, dim=-1)
        entropy = -(
            probabilities * torch.log(probabilities.clamp_min(1e-30))
        ).sum(-1)
        gap = verifier_confidence - draft_probability
        return {
            "confidence": [float(v) for v in verifier_confidence.detach().cpu().tolist()],
            "margin": [float(v) for v in margin.detach().cpu().tolist()],
            "entropy": [float(v) for v in entropy.detach().cpu().tolist()],
            "draft_probability": [float(v) for v in draft_probability.detach().cpu().tolist()],
            "top1_draft_gap": [float(v) for v in gap.detach().cpu().tolist()],
        }

    def _run_branch(
        self,
        model_runner: ModelRunner,
        master: ForwardBatch,
        request_indices: Sequence[int],
        size: int,
    ) -> Tuple[Any, List[Dict[str, Any]], bool]:
        """Run one real draft+verify branch and retain genuine verifier summaries."""
        fb = self._view(master, request_indices, size)
        starts: List[int] = []
        for local, rid in enumerate(fb.rids):
            begin = local * size
            end = begin + size
            masks = int((fb.input_ids[begin:end] == self.mask_id).sum().item())
            generation_start = size - masks
            starts.append(generation_start)
            seed = self._seed_tokens.get(rid)
            if seed is not None and generation_start < size:
                fb.input_ids[begin + generation_start] = int(seed)

        self._load_lora_deltas(model_runner)
        need_swap = self._lora_deltas is not None and not self._graphs_baked
        if need_swap:
            for parameter, delta, _module in self._lora_deltas:
                parameter.data.add_(delta)
        draft_out = model_runner.forward(fb, pp_proxy_tensors=None)
        self._stats_forward_passes += 1
        if need_swap and self._lora_mode == "draft_only":
            for parameter, delta, _module in self._lora_deltas:
                parameter.data.sub_(delta)

        draft_logits_raw = draft_out.logits_output.full_logits
        draft_logits = draft_logits_raw.clone()
        draft_logits[:, self.mask_id] = -1e9
        draft_tokens = torch.argmax(draft_logits, dim=-1)
        mask_positions = fb.input_ids == self.mask_id
        fb.input_ids[mask_positions] = draft_tokens[mask_positions]

        fb.dllm_causal_kv_update = True
        verify_out = model_runner.forward(fb, pp_proxy_tensors=None)
        fb.dllm_causal_kv_update = False
        self._stats_forward_passes += 1
        if need_swap and self._lora_mode == "both":
            for parameter, delta, _module in self._lora_deltas:
                parameter.data.sub_(delta)

        verify_logits_raw = verify_out.logits_output.full_logits
        verify_logits = verify_logits_raw.clone()
        verify_logits[:, self.mask_id] = -1e9
        autoregressive_tokens = torch.argmax(verify_logits, dim=-1)
        eos_id = self._get_eos_id(model_runner)
        records: List[Dict[str, Any]] = []

        for local, original_index in enumerate(request_indices):
            begin = local * size
            generation_start = starts[local]
            generation_length = size - generation_start
            offset = begin + generation_start
            if generation_length > 1:
                matches = (
                    fb.input_ids[offset + 1 : offset + generation_length]
                    == autoregressive_tokens[offset : offset + generation_length - 1]
                )
                matched = int(matches.cumprod(0).sum().item())
            else:
                matched = 0
            accepted = matched + 1
            output = fb.input_ids[offset : offset + 1]
            if matched:
                output = torch.cat(
                    [output, autoregressive_tokens[offset : offset + matched]]
                )
            eos_hit = False
            if eos_id is not None and (output == eos_id).any():
                eos_position = int(
                    (output == eos_id).to(torch.int32).argmax().item()
                ) + 1
                output = output[:eos_position]
                accepted = eos_position
                eos_hit = True

            draft_positions = torch.arange(
                offset + 1, offset + generation_length, device=fb.input_ids.device
            )
            if draft_positions.numel():
                draft_position = self._position_metrics(
                    draft_logits_raw.index_select(0, draft_positions),
                    draft_tokens.index_select(0, draft_positions),
                )
                verify_positions = torch.arange(
                    offset, offset + generation_length - 1, device=fb.input_ids.device
                )
                verifier_position = self._verifier_position_metrics(
                    verify_logits_raw.index_select(0, verify_positions),
                    autoregressive_tokens.index_select(0, verify_positions),
                    fb.input_ids.index_select(0, draft_positions),
                    self.mask_id,
                )
            else:
                draft_position = {
                    key: [] for key in ("confidence", "margin", "margin_risk", "entropy")
                }
                verifier_position = {
                    key: []
                    for key in (
                        "confidence",
                        "margin",
                        "entropy",
                        "draft_probability",
                        "top1_draft_gap",
                    )
                }

            accepted_draft = max(0, min(matched, accepted - 1))
            reject_index = (
                matched
                if not eos_hit and matched < max(generation_length - 1, 0)
                else None
            )
            accepted_verify_confidence = verifier_position["confidence"][:accepted_draft]
            accepted_verify_margin = verifier_position["margin"][:accepted_draft]
            next_seed_position = min(offset + matched, begin + size - 1)
            full = bool(accepted >= generation_length)
            bonus_position = None
            if full and not eos_hit:
                bonus_position = self._verifier_position_metrics(
                    verify_logits_raw[next_seed_position : next_seed_position + 1],
                    autoregressive_tokens[next_seed_position : next_seed_position + 1],
                    autoregressive_tokens[next_seed_position : next_seed_position + 1],
                    self.mask_id,
                )
            records.append(
                {
                    "block_size": size,
                    "accept_length": accepted,
                    "matched_draft_tokens": matched,
                    "full": full,
                    "eos_hit": eos_hit,
                    "output_token_ids": [int(v) for v in output.detach().cpu().tolist()],
                    "next_seed_token_id": int(
                        autoregressive_tokens[next_seed_position].item()
                    ),
                    "position": draft_position,
                    "verifier_position": verifier_position,
                    "verifier_metrics_source": "causal_verify_logits",
                    "verifier_metrics_schema": 2,
                    "verifier_accepted_confidence_mean": _mean(
                        accepted_verify_confidence
                    ),
                    "verifier_accepted_confidence_min": (
                        min(accepted_verify_confidence)
                        if accepted_verify_confidence
                        else None
                    ),
                    "verifier_accepted_margin_mean": _mean(accepted_verify_margin),
                    "verifier_bonus_confidence": (
                        bonus_position["confidence"][0]
                        if bonus_position is not None
                        else None
                    ),
                    "verifier_bonus_margin": (
                        bonus_position["margin"][0]
                        if bonus_position is not None
                        else None
                    ),
                    "verifier_bonus_entropy": (
                        bonus_position["entropy"][0]
                        if bonus_position is not None
                        else None
                    ),
                    "verifier_rejected_confidence": (
                        verifier_position["confidence"][reject_index]
                        if reject_index is not None
                        else None
                    ),
                    "verifier_rejected_margin": (
                        verifier_position["margin"][reject_index]
                        if reject_index is not None
                        else None
                    ),
                    "verifier_rejected_entropy": (
                        verifier_position["entropy"][reject_index]
                        if reject_index is not None
                        else None
                    ),
                    "verifier_rejected_draft_probability": (
                        verifier_position["draft_probability"][reject_index]
                        if reject_index is not None
                        else None
                    ),
                    "verifier_rejected_top1_draft_gap": (
                        verifier_position["top1_draft_gap"][reject_index]
                        if reject_index is not None
                        else None
                    ),
                    "request_batch_index": int(original_index),
                    "model_size": self.model_size,
                }
            )

        # Shadow branches reuse KV slots.  Correctness is more important than
        # wall time because B200 latency is supplied by the external sweep.
        torch.cuda.synchronize(fb.input_ids.device)
        return verify_out.logits_output, records, verify_out.can_run_graph

    def _canonical_history_row(self, branch: Dict[str, Any]) -> Dict[str, Any]:
        # Keep legacy draft fields so the inherited integrity path remains
        # compatible, but no legal signal specification references them.
        draft = branch["position"]
        verifier = branch["verifier_position"]
        matched = int(branch["matched_draft_tokens"])
        # Causal logits after the first mismatch condition on an already wrong
        # draft prefix and are not valid history for the committed trajectory.
        # Keep accepted decisions plus the first rejected decision only.
        if branch["eos_hit"]:
            valid_verify_decisions = min(
                max(int(branch["accept_length"]) - 1, 0),
                len(verifier["confidence"]),
            )
        elif branch["full"]:
            valid_verify_decisions = min(matched, len(verifier["confidence"]))
        else:
            valid_verify_decisions = min(matched + 1, len(verifier["confidence"]))
        prefix = slice(0, min(7, valid_verify_decisions))
        draft_prefix = slice(0, min(7, len(draft["confidence"])))
        return {
            "block_size": int(branch["block_size"]),
            "accept_length": int(branch["accept_length"]),
            "full": bool(branch["full"]),
            "head_conf_mean": _mean(draft["confidence"][draft_prefix]),
            "head_margin_mean": _mean(draft["margin"][draft_prefix]),
            "head_entropy_mean": _mean(draft["entropy"][draft_prefix]),
            "rejected_confidence": branch.get("rejected_confidence"),
            "rejected_margin": branch.get("rejected_margin"),
            "verify_head_conf_mean": _mean(verifier["confidence"][prefix]),
            "verify_head_margin_mean": _mean(verifier["margin"][prefix]),
            "verify_head_entropy_mean": _mean(verifier["entropy"][prefix]),
            "verify_accepted_conf_mean": branch.get(
                "verifier_accepted_confidence_mean"
            ),
            "verify_accepted_margin_mean": branch.get(
                "verifier_accepted_margin_mean"
            ),
            "verify_bonus_conf": branch.get("verifier_bonus_confidence"),
            "verify_bonus_margin": branch.get("verifier_bonus_margin"),
            "verify_bonus_entropy": branch.get("verifier_bonus_entropy"),
            "verify_rejected_conf": branch.get("verifier_rejected_confidence"),
            "verify_rejected_margin": branch.get("verifier_rejected_margin"),
            "verify_rejected_entropy": branch.get("verifier_rejected_entropy"),
            "verify_rejected_gap": branch.get(
                "verifier_rejected_top1_draft_gap"
            ),
        }

    def _history_features(self, rid: str) -> Dict[str, Any]:
        features = super()._history_features(rid)
        history = self._history.get(rid, [])
        if not history:
            return features
        previous = history[-1]
        features.update(
            {
                "prev_verify_head_conf": previous.get("verify_head_conf_mean"),
                "prev_verify_head_margin": previous.get("verify_head_margin_mean"),
                "prev_verify_head_entropy": previous.get("verify_head_entropy_mean"),
                "prev_verify_accepted_conf": previous.get(
                    "verify_accepted_conf_mean"
                ),
                "prev_verify_accepted_margin": previous.get(
                    "verify_accepted_margin_mean"
                ),
                "prev_verify_bonus_conf": previous.get("verify_bonus_conf"),
                "prev_verify_bonus_margin": previous.get("verify_bonus_margin"),
                "prev_verify_bonus_entropy": previous.get("verify_bonus_entropy"),
                "prev_verify_rejected_conf": previous.get("verify_rejected_conf"),
                "prev_verify_rejected_margin": previous.get(
                    "verify_rejected_margin"
                ),
                "prev_verify_rejected_entropy": previous.get(
                    "verify_rejected_entropy"
                ),
                "prev_verify_rejected_gap": previous.get("verify_rejected_gap"),
            }
        )
        for window in (1, 2, 4, 8):
            rows = history[-window:]
            features[f"verify_conf_ma{window}"] = _rolling_mean(
                rows, "verify_head_conf_mean"
            )
            features[f"verify_margin_ma{window}"] = _rolling_mean(
                rows, "verify_head_margin_mean"
            )
            features[f"verify_entropy_ma{window}"] = _rolling_mean(
                rows, "verify_head_entropy_mean"
            )
            features[f"verify_accept_conf_ma{window}"] = _rolling_mean(
                rows, "verify_accepted_conf_mean"
            )
            features[f"verify_accept_margin_ma{window}"] = _rolling_mean(
                rows, "verify_accepted_margin_mean"
            )
            features[f"verify_bonus_conf_ma{window}"] = _rolling_mean(
                rows, "verify_bonus_conf"
            )
            features[f"verify_bonus_margin_ma{window}"] = _rolling_mean(
                rows, "verify_bonus_margin"
            )
            features[f"verify_bonus_entropy_ma{window}"] = _rolling_mean(
                rows, "verify_bonus_entropy"
            )
        return features

    def _choose_action(
        self, rid: str, features: Dict[str, Any]
    ) -> Tuple[int, str, Dict[str, float]]:
        if self.policy_mode == "verify_efficiency_frozen":
            if self._verify_policy is None:
                raise RuntimeError(
                    "verify_efficiency_frozen requires NLD_B200_VERIFY_POLICY_PATH"
                )
            action, scores = choose_action(
                self._verify_policy, features, self._verify_concurrency
            )
            family = self._verify_policy["family"]
            return action, f"verify_efficiency_{family}_c{self._verify_concurrency}_l{action}", scores
        return super()._choose_action(rid, features)
