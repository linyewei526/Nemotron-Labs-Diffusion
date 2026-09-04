from __future__ import annotations

import unittest

import torch
from torch import nn
from transformers.cache_utils import DynamicCache

from method.margin_risk_conditional_rank_overlap_linearspec.generation import (
    OUTCOME_STATES,
    CandidateSpec,
    DraftState,
    OverlapGenerationStats,
    _analyze_draft,
    _classify_conditional_outcome,
    _count_matched_draft_tokens,
)
from method.margin_risk_conditional_rank_overlap_linearspec.hybrid import (
    build_multi_hybrid_attention_mask,
    repeat_dynamic_cache,
    select_and_crop_cache,
)
from method.margin_risk_conditional_rank_overlap_linearspec.segmented_lora import (
    SegmentedLoraController,
    SegmentedLoraLinear,
)


def candidate(
    rank: int, position: int, alternative: int, confidence_rank: int = 2
) -> CandidateSpec:
    return CandidateSpec(rank, position, alternative, 0.8, confidence_rank)


def continuation(seed: int, length: int = 4) -> DraftState:
    return DraftState(
        tokens=torch.tensor([seed] + [1] * (length - 1)),
        confidences=torch.ones(length),
        top1_top2_margins=torch.zeros(length),
        candidates=(),
        total_crossing_count=0,
        source="test_continuation",
    )


class CoreTests(unittest.TestCase):
    def test_four_row_mask_visibility_and_padding(self) -> None:
        mask = build_multi_hybrid_attention_mask(
            cache_length=2,
            verifier_length=4,
            branch_prefix_lengths=(1, 3, 4),
            prospective_length=4,
            query_length=8,
            device=torch.device("cpu"),
        )
        self.assertEqual(tuple(mask.shape), (4, 1, 8, 10))
        # Verifier is causal; its valid queries cannot see its padding keys.
        self.assertEqual(mask[0, 0, 0].tolist(), [True, True, True, False, False, False, False, False, False, False])
        self.assertEqual(mask[0, 0, 3].tolist(), [True, True, True, True, True, True, False, False, False, False])
        # p=1 branch suffix sees cache, its prefix, and its four-token suffix,
        # but not the row's three padding keys.
        self.assertTrue(mask[1, 0, 1, :7].all().item())
        self.assertFalse(mask[1, 0, 1, 7:].any().item())
        # Continuation has no padding and its suffix sees the full 2L row.
        self.assertTrue(mask[3, 0, 4].all().item())

    def test_segmented_lora_routes_only_selected_tokens(self) -> None:
        base = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            base.weight.copy_(torch.eye(2))
        controller = SegmentedLoraController()
        layer = SegmentedLoraLinear(base, torch.eye(2), torch.eye(2), 1.0, controller)
        x = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
        self.assertTrue(torch.equal(layer(x), x))
        with controller.use_all():
            self.assertTrue(torch.equal(layer(x), 2 * x))
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
        base_result = base(x)
        delta = torch.nn.functional.linear(
            torch.nn.functional.linear(x.float(), lora_a), lora_b
        ) * 4.0
        expected = (base_result + delta).to(torch.bfloat16)
        with controller.use_all():
            actual = layer(x)
        self.assertEqual(actual.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(actual, expected))

    def test_analyze_keeps_first_three_crossings_and_counts_all(self) -> None:
        mask_id = 3
        logits = torch.tensor(
            [
                [6.0, 0.0, 0.0, 99.0],
                [0.0, 0.0, 0.0, 99.0],
                [0.0, 0.0, 0.0, 99.0],
                [0.0, 0.0, 0.0, 99.0],
                [0.0, 0.0, 0.0, 99.0],
                [6.0, 0.0, 0.0, 99.0],
            ]
        )
        state = _analyze_draft(
            logits=logits,
            tokens=torch.tensor([0, 0, 1, 2, 0, 0]),
            mask_token_id=mask_id,
            excluded_alternative_token_ids=(mask_id,),
            margin_risk_threshold=0.5,
            source="test",
        )
        self.assertEqual(state.total_crossing_count, 4)
        self.assertEqual(state.risk_positions, (1, 2, 3))
        self.assertEqual([item.rank for item in state.candidates], [1, 1, 2])
        self.assertEqual([item.confidence_rank for item in state.candidates], [2, 3, 2])
        self.assertEqual([item.position for item in state.candidates], [1, 1, 2])
        self.assertEqual([item.branch_key for item in state.candidates], [
            "p1_rank2", "p1_rank3", "p2_rank2",
        ])
        self.assertNotEqual(
            state.candidates[0].alternative_token,
            state.candidates[1].alternative_token,
        )

    def test_at_most_two_crossings_use_rank2_per_position(self) -> None:
        mask_id = 4
        logits = torch.tensor(
            [
                [8.0, 0.0, -1.0, -2.0, 99.0],
                [4.0, 3.9, 2.0, 1.0, 99.0],
                [8.0, 0.0, -1.0, -2.0, 99.0],
                [4.0, 3.9, 2.0, 1.0, 99.0],
            ]
        )
        state = _analyze_draft(
            logits=logits,
            tokens=torch.tensor([0, 0, 0, 0]),
            mask_token_id=mask_id,
            excluded_alternative_token_ids=(mask_id,),
            margin_risk_threshold=0.5,
            source="test",
        )
        self.assertEqual(state.total_crossing_count, 2)
        self.assertEqual(state.risk_positions, (1, 3))
        self.assertEqual(
            [(item.rank, item.confidence_rank, item.position) for item in state.candidates],
            [(1, 2, 1), (2, 2, 3)],
        )

    def test_margin_risk_threshold_is_strict_and_mask_excluded(self) -> None:
        logits = torch.tensor([[5.0, 0.0, 99.0], [0.0, 0.0, 99.0]])
        tokens = torch.tensor([0, 0])
        crossing = _analyze_draft(
            logits=logits,
            tokens=tokens,
            mask_token_id=2,
            excluded_alternative_token_ids=(2,),
            margin_risk_threshold=0.0,
            source="test",
        )
        self.assertEqual(crossing.candidates[0].position, 1)
        risk = crossing.candidates[0].margin_risk
        equal = _analyze_draft(
            logits=logits,
            tokens=tokens,
            mask_token_id=2,
            excluded_alternative_token_ids=(2,),
            margin_risk_threshold=risk,
            source="test",
        )
        self.assertEqual(equal.total_crossing_count, 0)

    def test_rank_specific_candidate_outcomes(self) -> None:
        specs = (
            candidate(1, 2, 20, 2),
            candidate(1, 2, 21, 3),
            candidate(2, 4, 40, 2),
        )
        ar = torch.tensor([10, 20, 30, 40, 50, 60, 70])
        state, selected = _classify_conditional_outcome(
            mismatch_position=2,
            risk_positions=(2, 4, 6),
            candidate_specs=specs,
            ar_tokens=ar,
            continuation=None,
        )
        self.assertEqual((state, selected), ("p1_rank2_fixed", "p1_rank2"))

        rank3_specs = (
            candidate(1, 2, 99, 2),
            candidate(1, 2, 20, 3),
            candidate(2, 4, 40, 2),
        )
        state, selected = _classify_conditional_outcome(
            mismatch_position=2,
            risk_positions=(2, 4, 6),
            candidate_specs=rank3_specs,
            ar_tokens=ar,
            continuation=None,
        )
        self.assertEqual((state, selected), ("p1_rank3_fixed", "p1_rank3"))

        wrong_specs = tuple(
            candidate(item.rank, item.position, 99, item.confidence_rank)
            for item in specs
        )
        state, selected = _classify_conditional_outcome(
            mismatch_position=2,
            risk_positions=(2, 4, 6),
            candidate_specs=wrong_specs,
            ar_tokens=ar,
            continuation=None,
        )
        self.assertEqual(state, "p1_all_candidates_wrong")
        self.assertIsNone(selected)

        state, selected = _classify_conditional_outcome(
            mismatch_position=4,
            risk_positions=(2, 4, 6),
            candidate_specs=specs,
            ar_tokens=ar,
            continuation=None,
        )
        self.assertEqual((state, selected), ("p2_rank2_fixed", "p2_rank2"))

        state, selected = _classify_conditional_outcome(
            mismatch_position=6,
            risk_positions=(2, 4, 6),
            candidate_specs=specs,
            ar_tokens=ar,
            continuation=None,
        )
        self.assertEqual(state, "p3_detected_no_candidate")
        self.assertIsNone(selected)

    def test_branch_checks_count_rank2_and_rank3_independently(self) -> None:
        stats = OverlapGenerationStats()
        specs = (
            candidate(1, 2, 99, 2),
            candidate(1, 2, 20, 3),
            candidate(2, 4, 40, 2),
        )
        stats.record_candidate_checks(
            mismatch_position=2,
            candidate_specs=specs,
            ar_tokens=torch.tensor([10, 20, 30, 40, 50]),
        )
        payload = stats.to_dict()["candidate_correction"]
        self.assertEqual(payload["p1_rank2"]["wrong"], 1)
        self.assertEqual(payload["p1_rank3"]["fixed"], 1)
        self.assertEqual(payload["p2_rank2"]["checked"], 0)

    def test_detected_position_without_executable_branch_is_not_a_miss(self) -> None:
        state, selected = _classify_conditional_outcome(
            mismatch_position=2,
            risk_positions=(2, 5),
            candidate_specs=(candidate(2, 5, 50),),
            ar_tokens=torch.tensor([10, 20, 30, 40, 50, 60]),
            continuation=None,
        )
        self.assertEqual(state, "p1_no_executable_candidate")
        self.assertIsNone(selected)

    def test_miss_and_full_outcomes_are_mutually_exclusive(self) -> None:
        specs = (candidate(1, 2, 20), candidate(2, 5, 50))
        ar = torch.tensor([10, 20, 30, 40, 50, 60])
        cases = {
            1: "miss_before_first",
            3: "miss_between_positions",
            6: "miss_after_last",
        }
        observed = set()
        for mismatch, expected in cases.items():
            state, selected = _classify_conditional_outcome(
                mismatch_position=mismatch,
                risk_positions=(2, 5),
                candidate_specs=specs,
                ar_tokens=ar,
                continuation=None,
            )
            self.assertEqual(state, expected)
            self.assertIsNone(selected)
            observed.add(state)
        state, _ = _classify_conditional_outcome(
            mismatch_position=2,
            risk_positions=(),
            candidate_specs=(),
            ar_tokens=ar,
            continuation=None,
        )
        self.assertEqual(state, "miss_no_risk_position")
        observed.add(state)
        state, selected = _classify_conditional_outcome(
            mismatch_position=None,
            risk_positions=(2, 5),
            candidate_specs=specs,
            ar_tokens=ar,
            continuation=continuation(seed=60, length=6),
        )
        self.assertEqual((state, selected), ("full_continuation_hit", "continuation"))
        observed.add(state)
        state, _ = _classify_conditional_outcome(
            mismatch_position=None,
            risk_positions=(2, 5),
            candidate_specs=specs,
            ar_tokens=ar,
            continuation=continuation(seed=99, length=6),
        )
        self.assertEqual(state, "full_continuation_miss")
        observed.add(state)
        state, _ = _classify_conditional_outcome(
            mismatch_position=None,
            risk_positions=(2, 5),
            candidate_specs=specs,
            ar_tokens=ar,
            continuation=None,
        )
        self.assertEqual(state, "full_continuation_absent")
        observed.add(state)
        self.assertTrue(observed.issubset(set(OUTCOME_STATES)))

    def test_forward_token_distribution_includes_dense_padding(self) -> None:
        stats = OverlapGenerationStats()
        stats.record_forward(
            kind="prefill", rows=1, query_length=10, valid_lengths=(10,)
        )
        stats.record_forward(
            kind="multi_fused",
            rows=4,
            query_length=32,
            valid_lengths=(16, 21, 26, 32),
        )
        payload = stats.to_dict()
        self.assertEqual(payload["processed_query_tokens"], 138)
        self.assertEqual(payload["valid_query_tokens"], 105)
        self.assertEqual(payload["padding_query_tokens"], 33)
        fused = payload["forward_kinds"]["multi_fused"]
        self.assertEqual(fused["computed_token_avg"], 128)
        self.assertEqual(fused["padding_token_sum"], 33)
        self.assertEqual(fused["computed_token_p50"], 128)

    def test_transition_stats_do_not_impute_missing_next_round(self) -> None:
        stats = OverlapGenerationStats(prefetch_attempts=2)
        state = "p2_rank2_fixed"
        stats.record_outcome(state, 3)
        stats.record_next_accept(state, 3, 5)
        stats.record_outcome(state, 7)
        payload = stats.to_dict()["outcome_states"][state]
        self.assertEqual(payload["count"], 2)
        self.assertEqual(payload["current_accept_avg"], 5)
        self.assertEqual(payload["next_count"], 1)
        self.assertEqual(payload["next_coverage"], 0.5)
        self.assertEqual(payload["next_minus_current_avg"], 2)

    def test_shifted_verification_indexing(self) -> None:
        draft = torch.tensor([10, 11, 12, 13])
        self.assertEqual(_count_matched_draft_tokens(torch.tensor([11, 99, 13, 0]), draft), 1)
        self.assertEqual(_count_matched_draft_tokens(torch.tensor([11, 12, 13, 0]), draft), 3)

    def test_dynamic_cache_repeat_select_and_crop_isolated(self) -> None:
        key = torch.arange(1 * 1 * 3 * 2, dtype=torch.float32).reshape(1, 1, 3, 2)
        value = key + 100
        original = DynamicCache(ddp_cache_data=[(key, value)])
        repeated = repeat_dynamic_cache(original, 4)
        self.assertEqual(original.get_seq_length(), 3)
        self.assertEqual(repeated.layers[0].keys.shape[0], 4)
        repeated.layers[0].keys[3].add_(1000)
        selected = select_and_crop_cache(repeated, batch_index=0, max_length=2)
        self.assertEqual(selected.layers[0].keys.shape, (1, 1, 2, 2))
        self.assertEqual(original.layers[0].keys.shape, (1, 1, 3, 2))


if __name__ == "__main__":
    unittest.main()
