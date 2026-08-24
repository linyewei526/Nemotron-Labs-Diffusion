from __future__ import annotations

import unittest

import torch
from torch import nn
from transformers.cache_utils import DynamicCache

from method.confidence_direct_mask_redraft_linearspec.generation import (
    STATE_A_OK_LATER,
    STATE_DIRECT_HIT,
    STATE_FULL_BONUS,
    STATE_M_LT_P,
    STATE_REPEAT_A,
    STATE_WRONG_NON_A,
    DirectMaskRedraftGenerationStats,
    _analyze_draft,
    _count_matched_draft_tokens,
    _decide_redraft_reuse,
)
from method.confidence_direct_mask_redraft_linearspec.hybrid import (
    build_direct_mask_redraft_attention_mask,
    repeat_dynamic_cache,
    select_and_crop_cache,
)
from method.confidence_direct_mask_redraft_linearspec.segmented_lora import (
    SegmentedLoraController,
    SegmentedLoraLinear,
)


class CoreTests(unittest.TestCase):
    def test_attention_visibility(self) -> None:
        mask = build_direct_mask_redraft_attention_mask(
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
        self.assertTrue(mask[1, 0, 2].all().item())
        self.assertTrue(mask[1, 0, 5].all().item())

    def test_segmented_lora_routes_only_row1_suffix(self) -> None:
        base = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            base.weight.copy_(torch.eye(2))
        controller = SegmentedLoraController()
        layer = SegmentedLoraLinear(base, torch.eye(2), torch.eye(2), 1.0, controller)
        x = torch.tensor([[[1.0, 2.0]], [[5.0, 6.0]]])
        route = torch.zeros((2, 1, 1), dtype=torch.bool)
        route[1, 0] = True
        with controller.use_mask(route):
            actual = layer(x)
        expected = x.clone()
        expected[1, 0] *= 2
        self.assertTrue(torch.equal(actual, expected))

    def test_confidence_history_resets_on_full_reuse(self) -> None:
        logits = torch.tensor(
            [
                [6.0, 0.0, 0.0, 0.0, 99.0],
                [6.0, 0.0, 0.0, 0.0, 99.0],
                [0.0, 1.2, 1.1, 0.0, 99.0],
                [6.0, 0.0, 0.0, 0.0, 99.0],
            ]
        )
        draft = _analyze_draft(
            logits=logits,
            tokens=torch.tensor([0, 0, 1, 0]),
            mask_token_id=4,
            drop_pct_threshold=0.15,
            source="direct_full",
        )
        self.assertEqual(draft.candidate_position, 2)

    def test_shifted_verification_indexing(self) -> None:
        draft = torch.tensor([10, 11, 12, 13])
        self.assertEqual(
            _count_matched_draft_tokens(torch.tensor([11, 99, 13, 0]), draft), 1
        )
        self.assertEqual(
            _count_matched_draft_tokens(torch.tensor([11, 12, 13, 0]), draft), 3
        )

    @staticmethod
    def decide(proposal: list[int], ar: list[int], *, matched: int, p: int = 2):
        draft = torch.tensor([10, 11, 12, 13, 14, 15])
        return _decide_redraft_reuse(
            proposal_tokens=torch.tensor(proposal),
            ar_tokens=torch.tensor(ar),
            draft_tokens=draft,
            trigger_position=p,
            matched_tokens=matched,
            emitted_tokens=matched + 1,
        )

    def test_m_lt_p_is_discarded(self) -> None:
        decision = self.decide([99] * 6, [90] * 6, matched=0)
        self.assertFalse(decision.reusable)
        self.assertEqual(decision.reason, STATE_M_LT_P)

    def test_direct_correction_is_only_reusable_state(self) -> None:
        decision = self.decide([99, 20, 21, 22, 23, 24], [11, 99, 0, 0, 0, 0], matched=1)
        self.assertTrue(decision.reusable)
        self.assertEqual(decision.reason, STATE_DIRECT_HIT)

    def test_direct_failure_repeating_a_is_distinct(self) -> None:
        decision = self.decide([12, 20, 21, 22, 23, 24], [11, 99, 0, 0, 0, 0], matched=1)
        self.assertFalse(decision.reusable)
        self.assertEqual(decision.reason, STATE_REPEAT_A)

    def test_direct_failure_wrong_non_a_is_distinct(self) -> None:
        decision = self.decide([77, 20, 21, 22, 23, 24], [11, 99, 0, 0, 0, 0], matched=1)
        self.assertFalse(decision.reusable)
        self.assertEqual(decision.reason, STATE_WRONG_NON_A)

    def test_a_correct_then_later_rejection_is_discarded(self) -> None:
        decision = self.decide([12, 13, 99, 20, 21, 22], [11, 12, 13, 99, 0, 0], matched=3)
        self.assertFalse(decision.reusable)
        self.assertEqual(decision.reason, STATE_A_OK_LATER)

    def test_full_block_bonus_is_discarded(self) -> None:
        decision = self.decide([12, 13, 14, 15, 99, 20], [11, 12, 13, 14, 15, 99], matched=5)
        self.assertFalse(decision.reusable)
        self.assertEqual(decision.reason, STATE_FULL_BONUS)

    def test_cache_repeat_select_crop_keeps_row0(self) -> None:
        key = torch.arange(6, dtype=torch.float32).reshape(1, 1, 3, 2)
        original = DynamicCache(ddp_cache_data=[(key, key + 100)])
        repeated = repeat_dynamic_cache(original, 2)
        repeated.layers[0].keys[1].add_(1000)
        selected = select_and_crop_cache(repeated, batch_index=0, max_length=2)
        self.assertEqual(selected.layers[0].keys.shape, (1, 1, 2, 2))
        self.assertEqual(original.layers[0].keys.shape, (1, 1, 3, 2))

    def test_transition_averages_and_full_length_invariant(self) -> None:
        stats = DirectMaskRedraftGenerationStats(rounds=2, redraft_attempts=2)
        stats.record_draft_length(16, 16)
        stats.record_draft_length(16, 16)
        stats.record_state(STATE_DIRECT_HIT, matched=4, emitted=5)
        stats.record_next_state(
            STATE_DIRECT_HIT,
            current_matched=4,
            current_emitted=5,
            next_matched=9,
            next_emitted=10,
        )
        stats.record_state(STATE_M_LT_P, matched=1, emitted=2)
        stats.record_no_next(STATE_M_LT_P)
        payload = stats.to_dict()
        direct = payload["state_stats"][STATE_DIRECT_HIT]
        self.assertEqual(payload["average_draft_length"], 16.0)
        self.assertEqual(direct["next_matched_mean"], 9.0)
        self.assertEqual(direct["next_minus_current_matched_mean"], 5.0)
        self.assertEqual(payload["state_stats"][STATE_M_LT_P]["no_next_round_count"], 1)
        with self.assertRaises(RuntimeError):
            stats.record_draft_length(15, 16)


if __name__ == "__main__":
    unittest.main()
