from __future__ import annotations

import unittest

import torch
from torch import nn
from transformers.cache_utils import DynamicCache

from method.confidence_mask_redraft_linearspec.generation import (
    MaskRedraftGenerationStats,
    _analyze_draft,
    _count_matched_draft_tokens,
    _decide_redraft_reuse,
)
from method.confidence_mask_redraft_linearspec.hybrid import (
    build_mask_redraft_attention_mask,
    repeat_dynamic_cache,
    select_and_crop_cache,
)
from method.confidence_mask_redraft_linearspec.segmented_lora import (
    SegmentedLoraController,
    SegmentedLoraLinear,
)


class CoreTests(unittest.TestCase):
    def test_mask_redraft_attention_exact_visibility(self) -> None:
        mask = build_mask_redraft_attention_mask(
            cache_length=2,
            verify_length=4,
            prospective_length=4,
            trigger_position=2,
            device=torch.device("cpu"),
        )
        self.assertEqual(tuple(mask.shape), (2, 1, 6, 8))
        self.assertEqual(
            mask[0, 0, 0].tolist(),
            [True, True, True, False, False, False, False, False],
        )
        self.assertEqual(
            mask[1, 0, 1].tolist(),
            [True, True, True, True, False, False, False, False],
        )
        self.assertTrue(mask[1, 0, 2].all().item())
        self.assertTrue(mask[1, 0, 5].all().item())

    def test_variable_verify_length_is_padded_without_changing_redraft_length(self) -> None:
        mask = build_mask_redraft_attention_mask(
            cache_length=1,
            verify_length=3,
            prospective_length=5,
            trigger_position=2,
            device=torch.device("cpu"),
        )
        self.assertEqual(tuple(mask.shape), (2, 1, 7, 8))
        self.assertTrue(mask[0, 0, 2, :4].all().item())
        self.assertFalse(mask[0, 0, 2, 4:].any().item())
        self.assertTrue(mask[1, 0, 2].all().item())

    def test_segmented_lora_routes_only_redraft_suffix(self) -> None:
        base = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            base.weight.copy_(torch.eye(2))
        controller = SegmentedLoraController()
        layer = SegmentedLoraLinear(base, torch.eye(2), torch.eye(2), 1.0, controller)
        x = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
        route = torch.zeros((2, 2, 1), dtype=torch.bool)
        route[1, 1] = True
        with controller.use_mask(route):
            actual = layer(x)
        expected = x.clone()
        expected[1, 1] *= 2
        self.assertTrue(torch.equal(actual, expected))

    def test_segmented_lora_matches_peft_fp32_adapter_math(self) -> None:
        base = nn.Linear(3, 2, bias=False, dtype=torch.bfloat16)
        lora_a = torch.tensor([[0.25, -0.5, 1.0], [1.0, 0.5, -0.25]])
        lora_b = torch.tensor([[0.75, -1.0], [0.5, 0.125]])
        controller = SegmentedLoraController()
        layer = SegmentedLoraLinear(base, lora_a, lora_b, 4.0, controller)
        x = torch.tensor([[[1.0, 2.0, -1.0], [0.5, -0.5, 3.0]]], dtype=torch.bfloat16)
        expected = (
            base(x)
            + torch.nn.functional.linear(
                torch.nn.functional.linear(x.float(), lora_a), lora_b
            )
            * 4.0
        ).to(torch.bfloat16)
        with controller.use_all():
            actual = layer(x)
        self.assertTrue(torch.equal(actual, expected))

    def test_confidence_history_resets_for_retained_suffix(self) -> None:
        mask_id = 4
        logits = torch.tensor(
            [
                [6.0, 0.0, 0.0, 0.0, 99.0],
                [6.0, 0.0, 0.0, 0.0, 99.0],
                [0.0, 1.2, 1.1, 0.0, 99.0],
                [6.0, 0.0, 0.0, 0.0, 99.0],
                [0.0, 1.2, 1.1, 0.0, 99.0],
            ]
        )
        tokens = torch.tensor([0, 0, 1, 0, 1])
        full = _analyze_draft(
            logits=logits,
            tokens=tokens,
            mask_token_id=mask_id,
            drop_pct_threshold=0.15,
            source="full",
        )
        retained = _analyze_draft(
            logits=logits[2:],
            tokens=tokens[2:],
            mask_token_id=mask_id,
            drop_pct_threshold=0.15,
            source="retained",
        )
        self.assertEqual(full.candidate_position, 2)
        # The old trigger becomes the new seed and is not used in the mean.
        self.assertEqual(retained.candidate_position, 2)

    def test_shifted_verification_indexing(self) -> None:
        draft = torch.tensor([10, 11, 12, 13])
        self.assertEqual(
            _count_matched_draft_tokens(torch.tensor([11, 99, 13, 0]), draft), 1
        )
        self.assertEqual(
            _count_matched_draft_tokens(torch.tensor([11, 12, 13, 0]), draft), 3
        )

    def test_reuse_rejects_when_verifier_fails_before_trigger(self) -> None:
        decision = _decide_redraft_reuse(
            proposal_tokens=torch.tensor([20, 21, 22, 23]),
            ar_tokens=torch.tensor([11, 99, 0, 0]),
            trigger_position=2,
            emitted_tokens=1,
        )
        self.assertFalse(decision.reusable)
        self.assertEqual(decision.reason, "before_trigger")

    def test_direct_trigger_correction_reuses_full_redraft(self) -> None:
        proposal = torch.tensor([99, 21, 22, 23])
        decision = _decide_redraft_reuse(
            proposal_tokens=proposal,
            ar_tokens=torch.tensor([11, 99, 0, 0]),
            trigger_position=2,
            emitted_tokens=2,
        )
        self.assertTrue(decision.reusable)
        self.assertEqual(decision.retained_offset, 0)
        self.assertTrue(torch.equal(proposal[decision.retained_offset :], proposal))

    def test_same_a_then_downstream_correction_reuses_partial_suffix(self) -> None:
        # p=2, m=4: row-1 must match trusted absolute positions 2,3,4.
        ar = torch.tensor([11, 12, 13, 99, 0, 0])
        proposal = torch.tensor([12, 13, 99, 30, 31, 32])
        decision = _decide_redraft_reuse(
            proposal_tokens=proposal,
            ar_tokens=ar,
            trigger_position=2,
            emitted_tokens=4,
        )
        self.assertTrue(decision.reusable)
        self.assertEqual(decision.retained_offset, 2)
        self.assertEqual(proposal[decision.retained_offset :].tolist(), [99, 30, 31, 32])

    def test_all_rejection_reasons_are_distinguished(self) -> None:
        ar = torch.tensor([11, 12, 13, 99, 0, 0])
        trigger = _decide_redraft_reuse(
            proposal_tokens=torch.tensor([77, 13, 99, 0, 0, 0]),
            ar_tokens=ar,
            trigger_position=2,
            emitted_tokens=4,
        )
        early = _decide_redraft_reuse(
            proposal_tokens=torch.tensor([12, 77, 99, 0, 0, 0]),
            ar_tokens=ar,
            trigger_position=2,
            emitted_tokens=4,
        )
        correction = _decide_redraft_reuse(
            proposal_tokens=torch.tensor([12, 13, 77, 0, 0, 0]),
            ar_tokens=ar,
            trigger_position=2,
            emitted_tokens=4,
        )
        self.assertEqual(trigger.reason, "trigger_token_mismatch")
        self.assertEqual(early.reason, "before_correction")
        self.assertEqual(correction.reason, "correction_mismatch")

    def test_full_block_bonus_can_retain_tail(self) -> None:
        ar = torch.tensor([11, 12, 13, 14, 99])
        proposal = torch.tensor([12, 13, 14, 99, 30])
        decision = _decide_redraft_reuse(
            proposal_tokens=proposal,
            ar_tokens=ar,
            trigger_position=2,
            emitted_tokens=5,
        )
        self.assertTrue(decision.reusable)
        self.assertEqual(decision.retained_offset, 3)
        self.assertEqual(proposal[decision.retained_offset :].tolist(), [99, 30])

    def test_dynamic_cache_repeat_select_and_crop_isolated(self) -> None:
        key = torch.arange(1 * 1 * 3 * 2, dtype=torch.float32).reshape(1, 1, 3, 2)
        value = key + 100
        original = DynamicCache(ddp_cache_data=[(key, value)])
        repeated = repeat_dynamic_cache(original, 2)
        repeated.layers[0].keys[1].add_(1000)
        selected = select_and_crop_cache(repeated, batch_index=0, max_length=2)
        self.assertEqual(selected.layers[0].keys.shape, (1, 1, 2, 2))
        self.assertEqual(original.layers[0].keys.shape, (1, 1, 3, 2))

    def test_stats_report_variable_length_and_reuse_averages(self) -> None:
        stats = MaskRedraftGenerationStats(rounds=2, redraft_attempts=2, redraft_reuse_hits=1)
        stats.record_draft_length(16, 16)
        stats.record_draft_length(10, 16)
        stats.record_retained_length(10, 16)
        payload = stats.to_dict()
        self.assertEqual(payload["average_draft_length"], 13.0)
        self.assertEqual(payload["average_retained_draft_length"], 10.0)
        self.assertEqual(payload["partial_draft_rounds"], 1)


if __name__ == "__main__":
    unittest.main()
