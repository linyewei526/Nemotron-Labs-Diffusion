from __future__ import annotations

import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from method.margin_risk_overlap_linearspec.merge_metrics import main, summarize
from method.margin_risk_overlap_linearspec.report import (
    DEFAULT_BASELINE_16,
    DEFAULT_BASELINE_32,
    render,
)
from method.margin_risk_overlap_linearspec.server import (
    ChatCompletionRequest,
    MarginRiskOverlapEngine,
)


def outcome(count: int, current: int, next_count: int, paired: int, nxt: int) -> dict:
    return {
        "count": count,
        "current_accept_sum": current,
        "next_count": next_count,
        "paired_current_accept_sum": paired,
        "next_accept_sum": nxt,
        "next_minus_current_sum": nxt - paired,
    }


class MetricsAndReportTests(unittest.TestCase):
    def test_merge_aggregates_outcomes_and_preserves_dataset_level_denominator(self) -> None:
        empty = outcome(0, 0, 0, 0, 0)
        rows = [
            {
                "ok": True,
                "prompt_tokens": 10,
                "completion_tokens": 8,
                "raw_generated_tokens": 8,
                "nfe": 2,
                "model_time_s": 1,
                "request_time_s": 1,
                "margin_risk_threshold": 0.5,
                "overlap": {
                    "physical_nfe": 2,
                    "rounds": 2,
                    "prefetch_attempts": 2,
                    "outcome_states": {
                        "before_candidate_error": outcome(1, 2, 1, 2, 4),
                        "candidate_fixed_by_alternative": outcome(1, 3, 0, 0, 0),
                        "candidate_wrong_alternative": empty,
                        "after_candidate_error": empty,
                        "full_block_bonus": empty,
                    },
                },
            }
        ]
        merged = summarize(rows, 1.5)
        overlap = merged["overlap"]
        self.assertTrue(overlap["outcome_partition_valid"])
        self.assertEqual(overlap["outcome_state_count_sum"], 2)
        before = overlap["outcome_states"]["before_candidate_error"]
        self.assertEqual(before["share_of_attempts"], 0.5)
        self.assertEqual(before["current_accept_avg"], 2)
        self.assertEqual(before["next_accept_avg"], 4)
        self.assertEqual(before["next_minus_current_avg"], 2)
        fixed = overlap["outcome_states"]["candidate_fixed_by_alternative"]
        self.assertEqual(fixed["next_count"], 0)
        self.assertIsNone(fixed["next_accept_avg"])
        self.assertEqual(merged["margin_risk_thresholds"], [0.5])

    def test_failed_oom_is_disclosed_but_excluded_from_efficiency_means(self) -> None:
        rows = [
            {
                "ok": True,
                "prompt_tokens": 10,
                "completion_tokens": 8,
                "raw_generated_tokens": 8,
                "nfe": 2,
                "model_time_s": 1,
                "request_time_s": 1,
                "overlap": {},
            },
            {
                "ok": False,
                "prompt_tokens": 100,
                "error_type": "OutOfMemoryError",
                "oom_skipped_for_efficiency": True,
            },
        ]
        merged = summarize(rows, 2)
        self.assertEqual(merged["attempted_request_count"], 2)
        self.assertEqual(merged["request_count"], 1)
        self.assertEqual(merged["failed_request_count"], 1)
        self.assertEqual(merged["oom_skipped_request_count"], 1)
        self.assertEqual(merged["successful_request_rate"], 0.5)
        self.assertEqual(merged["tokens_per_forward_pass"], 4)

    def test_merge_can_create_efficiency_only_metrics_when_scorer_failed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metrics = root / "metrics.json"
            stats = root / "stats.jsonl"
            stats.write_text(
                json.dumps(
                    {
                        "ok": True,
                        "prompt_tokens": 10,
                        "completion_tokens": 8,
                        "raw_generated_tokens": 8,
                        "nfe": 2,
                        "model_time_s": 1,
                        "request_time_s": 1,
                        "overlap": {},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            argv = [
                "merge_metrics.py",
                "--metrics-json",
                str(metrics),
                "--request-stats-file",
                str(stats),
                "--benchmark",
                "gsm8k",
                "--create-metrics-if-missing",
                "--accuracy-status",
                "skipped_after_eval_failure",
            ]
            with patch.object(sys, "argv", argv):
                self.assertEqual(main(), 0)
            payload = json.loads(metrics.read_text(encoding="utf-8"))
            wrapper = payload["pytorch_margin_risk_overlap"]
            self.assertEqual(wrapper["accuracy_status"], "skipped_after_eval_failure")
            self.assertEqual(wrapper["decode"]["request_count"], 1)

    def test_engine_turns_oom_into_failed_stat_and_empty_placeholder(self) -> None:
        engine = MarginRiskOverlapEngine.__new__(MarginRiskOverlapEngine)
        engine.mode = "overlap_lora"
        engine.block_length = 16
        engine.draft_threshold = 0.0
        engine.margin_risk_threshold = 0.5
        engine.default_max_new_tokens = 16
        engine.context_length = 128
        engine.enable_thinking = False
        engine.efficiency_only = True
        engine.max_thinking_tokens = None
        engine.end_think_token_id = None
        engine.eos_token_id = None
        engine.lora_controller = None
        engine.model = object()
        engine.device = torch.device("cuda:0")
        engine._model_lock = threading.Lock()
        engine._format_and_tokenize = lambda _messages: torch.tensor([[1, 2]])
        captured: list[dict] = []
        engine._append_stat = captured.append
        request = ChatCompletionRequest(
            model="test",
            messages=[{"role": "user", "content": "x"}],
            max_completion_tokens=16,
            temperature=0,
        )
        oom = torch.cuda.OutOfMemoryError("synthetic OOM")
        with (
            patch(
                "method.margin_risk_overlap_linearspec.server.overlap_linear_spec_generate",
                side_effect=oom,
            ),
            patch("torch.cuda.reset_peak_memory_stats"),
            patch("torch.cuda.synchronize"),
            patch("torch.cuda.empty_cache"),
        ):
            result = engine.generate(request, "req-test", 0.0)
        self.assertEqual(result["text"], "")
        self.assertEqual(result["completion_tokens"], 0)
        self.assertFalse(captured[0]["ok"])
        self.assertTrue(captured[0]["oom_skipped_for_efficiency"])

    def test_initial_report_is_valid_before_any_dataset_finishes(self) -> None:
        content = render(Path("/path/that/does/not/exist"), DEFAULT_BASELINE_16, DEFAULT_BASELINE_32)
        self.assertIn("0/9", content)
        self.assertIn("尚无已完成", content)
        self.assertIn("AIME24", content)
        self.assertIn("九数据集等权", content)

    def test_report_uses_coverage_and_omits_accuracy_columns(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "Settings.json").write_text(
                json.dumps(
                    {
                        "status": "running",
                        "benchmark": {"tokens": 64, "temperature": 0},
                        "pytorch": {
                            "mode": "overlap_lora",
                            "block_length": 16,
                            "margin_risk_threshold": 0.5,
                        },
                    }
                ),
                encoding="utf-8",
            )
            (root / "metrics_gsm8k.json").write_text(
                json.dumps(
                    {
                        "gsm8k": {"pass@1": {"symbolic_correct": 99}},
                        "pytorch_margin_risk_overlap": {
                            "accuracy_status": "ignored_efficiency_only",
                            "decode": {
                                "attempted_request_count": 2,
                                "request_count": 1,
                                "failed_request_count": 1,
                                "oom_skipped_request_count": 1,
                                "successful_request_rate": 0.5,
                                "tokens_per_forward_pass": 4,
                                "average_forward_passes_per_sample": 2,
                                "model_output_tokens_per_s": 20,
                                "overlap": {},
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            content = render(root, DEFAULT_BASELINE_16, DEFAULT_BASELINE_32)
            self.assertIn("|gsm8k|2|1|1|1|50.00%|", content)
            self.assertNotIn("新质量", content)
            self.assertNotIn("Acc状态", content)
            self.assertNotIn("99.00", content)


if __name__ == "__main__":
    unittest.main()
