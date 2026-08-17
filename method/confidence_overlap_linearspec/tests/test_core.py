from __future__ import annotations

import unittest

import torch
from torch import nn
from transformers.cache_utils import DynamicCache

from method.confidence_overlap_linearspec.generation import (
    _analyze_draft,
    _count_matched_draft_tokens,
)
from method.confidence_overlap_linearspec.hybrid import (
    build_hybrid_attention_mask,
    repeat_dynamic_cache,
    select_and_crop_cache,
)
from method.confidence_overlap_linearspec.segmented_lora import (
    SegmentedLoraController,
    SegmentedLoraLinear,
)


class CoreTests(unittest.TestCase):
    def test_hybrid_mask_exact_visibility(self) -> None:
        cache, block, position = 2, 4, 2
        mask = build_hybrid_attention_mask(
            cache_length=cache,
            block_length=block,
            candidate_position=position,
            device=torch.device("cpu"),
        )
        self.assertEqual(tuple(mask.shape), (2, 1, 6, 8))
        self.assertEqual(mask[0, 0, 0].tolist(), [True, True, True, False, False, False, False, False])
        self.assertEqual(mask[0, 0, 3].tolist(), [True, True, True, True, True, True, False, False])
        self.assertEqual(mask[1, 0, 0].tolist(), [True, True, True, False, False, False, False, False])
        self.assertEqual(mask[1, 0, 1].tolist(), [True, True, True, True, False, False, False, False])
        self.assertTrue(mask[1, 0, 2].all().item())
        self.assertTrue(mask[1, 0, 5].all().item())

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

    def test_drop_crossing_and_mask_excluded_alternative(self) -> None:
        mask_id = 4
        logits = torch.tensor(
            [
                [3.0, 0.0, 0.0, 0.0, 99.0],
                [6.0, 0.0, 0.0, 0.0, 99.0],
                [0.0, 1.2, 1.1, 0.0, 99.0],
                [5.0, 0.0, 0.0, 0.0, 99.0],
            ]
        )
        state = _analyze_draft(
            logits=logits,
            tokens=torch.tensor([0, 0, 1, 0]),
            mask_token_id=mask_id,
            excluded_alternative_token_ids=(mask_id, 2),
            drop_pct_threshold=0.15,
            source="test",
        )
        self.assertEqual(state.candidate_position, 2)
        self.assertEqual(state.alternative_token, 0)
        self.assertNotEqual(state.alternative_token, mask_id)
        self.assertNotEqual(state.alternative_token, 2)
        self.assertIsNotNone(state.candidate_drop_pct)
        self.assertGreater(state.candidate_drop_pct, 0.15)

    def test_shifted_verification_indexing(self) -> None:
        draft = torch.tensor([10, 11, 12, 13])
        self.assertEqual(_count_matched_draft_tokens(torch.tensor([11, 99, 13, 0]), draft), 1)
        self.assertEqual(_count_matched_draft_tokens(torch.tensor([11, 12, 13, 0]), draft), 3)

    def test_dynamic_cache_repeat_select_and_crop_isolated(self) -> None:
        key = torch.arange(1 * 1 * 3 * 2, dtype=torch.float32).reshape(1, 1, 3, 2)
        value = key + 100
        original = DynamicCache(ddp_cache_data=[(key, value)])
        repeated = repeat_dynamic_cache(original, 2)
        self.assertEqual(original.get_seq_length(), 3)
        self.assertEqual(repeated.layers[0].keys.shape[0], 2)
        repeated.layers[0].keys[1].add_(1000)
        selected = select_and_crop_cache(repeated, batch_index=0, max_length=2)
        self.assertEqual(selected.layers[0].keys.shape, (1, 1, 2, 2))
        self.assertEqual(original.layers[0].keys.shape, (1, 1, 3, 2))


if __name__ == "__main__":
    unittest.main()
