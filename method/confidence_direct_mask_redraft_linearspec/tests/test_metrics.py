from __future__ import annotations

import unittest

from method.confidence_direct_mask_redraft_linearspec.generation import ALL_STATES
from method.confidence_direct_mask_redraft_linearspec.merge_metrics import summarize


def state_payload(**overrides):
    payload = {
        "count": 0,
        "current_matched_sum": 0,
        "current_emitted_sum": 0,
        "next_observed_count": 0,
        "no_next_round_count": 0,
        "next_matched_sum": 0,
        "next_emitted_sum": 0,
        "next_minus_current_matched_sum": 0,
        "next_minus_current_emitted_sum": 0,
    }
    payload.update(overrides)
    return payload


class MetricsTests(unittest.TestCase):
    def test_direct_state_metrics_are_aggregated_and_recomputed(self) -> None:
        states = {name: state_payload() for name in ALL_STATES}
        states["direct_hit"] = state_payload(
            count=1,
            current_matched_sum=4,
            current_emitted_sum=5,
            next_observed_count=1,
            next_matched_sum=9,
            next_emitted_sum=10,
            next_minus_current_matched_sum=5,
            next_minus_current_emitted_sum=5,
        )
        states["repeat_a"] = state_payload(
            count=1,
            current_matched_sum=4,
            current_emitted_sum=5,
            no_next_round_count=1,
        )
        rows = [
            {
                "ok": True,
                "mode": "direct_mask_redraft_lora",
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
                "direct_mask_redraft": {
                    "physical_nfe": 4,
                    "rounds": 2,
                    "draft_length_sum": 32,
                    "normal_draft_forwards": 1,
                    "fused_verify_redraft_forwards": 2,
                    "redraft_attempts": 2,
                    "redraft_direct_hits": 1,
                    "redraft_reuse_hits": 1,
                    "redraft_saved_draft_forwards": 1,
                    "redraft_repeat_a": 1,
                    "candidate_position_sum": 8,
                    "full_length_reuses": 1,
                    "state_stats": states,
                },
            }
        ]
        result = summarize(rows, 2.0)
        redraft = result["direct_mask_redraft"]
        self.assertEqual(result["tokens_per_forward_pass"], 6.6667)
        self.assertEqual(result["decode_forward_passes"], 3.0)
        self.assertEqual(result["prefill_forward_passes"], 1.0)
        self.assertEqual(result["total_forward_passes"], 4.0)
        self.assertEqual(result["end_to_end_tokens_per_forward_pass"], 5.0)
        self.assertEqual(redraft["average_draft_length"], 16.0)
        self.assertEqual(redraft["redraft_direct_hit_rate"], 0.5)
        self.assertEqual(redraft["state_stats"]["direct_hit"]["next_matched_mean"], 9.0)
        self.assertEqual(
            redraft["state_stats"]["direct_hit"]["next_minus_current_matched_mean"],
            5.0,
        )
        self.assertEqual(redraft["state_stats"]["repeat_a"]["no_next_round_count"], 1)


if __name__ == "__main__":
    unittest.main()
