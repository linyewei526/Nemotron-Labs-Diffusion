from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from method.confidence_direct_mask_redraft_linearspec.generation import ALL_STATES
from method.confidence_direct_mask_redraft_linearspec.merge_metrics import summarize
from method.confidence_direct_mask_redraft_linearspec.report_results import generate


def state_payload(**overrides):
    value = {
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
    value.update(overrides)
    return value


def current_metric(name: str, accuracy: float) -> dict:
    states = {state: state_payload() for state in ALL_STATES}
    states["direct_hit"] = state_payload(
        count=1,
        current_matched_sum=2,
        current_emitted_sum=3,
        no_next_round_count=1,
    )
    row = {
        "ok": True,
        "mode": "direct_mask_redraft_lora",
        "prompt_tokens": 10,
        "completion_tokens": 10,
        "raw_generated_tokens": 10,
        "nfe": 3,
        "model_time_s": 1.0,
        "request_time_s": 1.1,
        "queue_wait_s": 0.1,
        "finish_reason": "stop",
        "drop_pct_threshold": 0.15,
        "direct_mask_redraft": {
            "physical_nfe": 3,
            "rounds": 1,
            "draft_length_sum": 16,
            "normal_draft_forwards": 1,
            "normal_verify_forwards": 0,
            "fused_verify_redraft_forwards": 1,
            "redraft_attempts": 1,
            "redraft_direct_hits": 1,
            "redraft_reuse_hits": 0,
            "redraft_saved_draft_forwards": 0,
            "redraft_discarded_generation_end": 1,
            "candidate_position_sum": 2,
            "full_length_reuses": 0,
            "state_stats": states,
        },
    }
    decode = summarize([row], 1.2)
    return {
        "tpf": decode["tokens_per_forward_pass"],
        name: {"pass@1": {"symbolic_correct": accuracy}},
        "pytorch_confidence_direct_mask_redraft": {"decode": decode},
    }


class ReportTests(unittest.TestCase):
    def test_aime_row_is_kept_but_excluded_from_macro(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            result = root / "result"
            b16 = root / "b16"
            b32 = root / "b32"
            for directory in (result, b16, b32):
                directory.mkdir()
            (result / "metrics_gsm8k.json").write_text(
                json.dumps(current_metric("gsm8k", 80.0)), encoding="utf-8"
            )
            (result / "metrics_aime24.json").write_text(
                json.dumps(current_metric("aime24", 10.0)), encoding="utf-8"
            )
            for directory, tpf, acc in ((b16, 2.0, 70.0), (b32, 4.0, 75.0)):
                for name in ("gsm8k", "aime24"):
                    metric = {
                        "tpf": tpf,
                        name: {"pass@1": {"symbolic_correct": acc}},
                    }
                    (directory / f"metrics_{name}.json").write_text(
                        json.dumps(metric), encoding="utf-8"
                    )
            report = generate(result, b16, b32)
            self.assertIn("AIME24", report)
            self.assertIn("可用集等权平均(1)", report)
            self.assertIn("已对 2 个 metrics 文件", report)


if __name__ == "__main__":
    unittest.main()
