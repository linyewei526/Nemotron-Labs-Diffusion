from __future__ import annotations

import unittest

from method.confidence_mask_redraft_linearspec.merge_metrics import summarize


class MetricsTests(unittest.TestCase):
    def test_variable_redraft_metrics_are_aggregated(self) -> None:
        rows = [
            {
                "ok": True,
                "mode": "mask_redraft_lora",
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "raw_generated_tokens": 20,
                "nfe": 4,
                "model_time_s": 1.0,
                "request_time_s": 1.1,
                "queue_wait_s": 0.1,
                "model_output_tokens_per_s": 20,
                "finish_reason": "stop",
                "drop_pct_threshold": 0.15,
                "mask_redraft": {
                    "physical_nfe": 4,
                    "rounds": 2,
                    "draft_length_sum": 26,
                    "partial_draft_rounds": 1,
                    "redraft_attempts": 2,
                    "redraft_verified_hits": 1,
                    "redraft_reuse_hits": 1,
                    "redraft_saved_draft_forwards": 1,
                    "candidate_position_sum": 8,
                    "retained_draft_tokens_sum": 10,
                    "retained_draft_tokens_min": 10,
                    "retained_draft_tokens_max": 10,
                    "partial_length_reuses": 1,
                },
            },
            {
                "ok": True,
                "mode": "mask_redraft_lora",
                "prompt_tokens": 10,
                "completion_tokens": 10,
                "raw_generated_tokens": 10,
                "nfe": 2,
                "model_time_s": 1.0,
                "request_time_s": 1.0,
                "queue_wait_s": 0.0,
                "model_output_tokens_per_s": 10,
                "finish_reason": "stop",
                "drop_pct_threshold": 0.15,
                "mask_redraft": {
                    "physical_nfe": 2,
                    "rounds": 1,
                    "draft_length_sum": 16,
                    "redraft_attempts": 1,
                    "candidate_position_sum": 4,
                    "retained_draft_tokens_min": None,
                    "retained_draft_tokens_max": 0,
                },
            },
        ]
        result = summarize(rows, 3.0)
        redraft = result["mask_redraft"]
        self.assertEqual(result["tokens_per_forward_pass"], 7.5)
        self.assertEqual(result["decode_forward_passes"], 4.0)
        self.assertEqual(result["prefill_forward_passes"], 2.0)
        self.assertEqual(result["total_forward_passes"], 6.0)
        self.assertEqual(result["end_to_end_tokens_per_forward_pass"], 5.0)
        self.assertEqual(redraft["redraft_attempts"], 3)
        self.assertAlmostEqual(redraft["redraft_hit_rate"], 1 / 3, places=6)
        self.assertEqual(redraft["average_draft_length"], 14.0)
        self.assertEqual(redraft["average_retained_draft_length"], 10.0)
        self.assertEqual(redraft["retained_draft_tokens_min"], 10)
        self.assertEqual(redraft["retained_draft_tokens_max"], 10)


if __name__ == "__main__":
    unittest.main()
