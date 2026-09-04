from __future__ import annotations

import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from method.margin_risk_conditional_rank_overlap_linearspec.generation import (
    FORWARD_KINDS,
    OUTCOME_STATES,
)
from method.margin_risk_conditional_rank_overlap_linearspec.merge_metrics import main, summarize
from method.margin_risk_conditional_rank_overlap_linearspec.report import (
    DEFAULT_BASELINE_16,
    DEFAULT_BASELINE_32,
    decode,
    render,
)
from method.margin_risk_conditional_rank_overlap_linearspec.server import (
    ChatCompletionRequest,
    MarginRiskConditionalRankOverlapEngine,
)


def outcome(
    count: int = 0,
    current: int = 0,
    next_count: int = 0,
    paired: int = 0,
    nxt: int = 0,
) -> dict:
    return {
        "count": count,
        "current_accept_sum": current,
        "next_count": next_count,
        "paired_current_accept_sum": paired,
        "next_accept_sum": nxt,
        "next_minus_current_sum": nxt - paired,
    }


def forward_kind(
    *,
    count: int = 0,
    computed: int = 0,
    valid: int = 0,
    padding: int = 0,
    rows: int = 0,
    query: int = 0,
) -> dict:
    histogram = {str(computed // count): count} if count else {}
    return {
        "count": count,
        "computed_token_sum": computed,
        "valid_token_sum": valid,
        "padding_token_sum": padding,
        "row_sum": rows,
        "query_length_sum": query,
        "computed_token_histogram": histogram,
    }


class MetricsAndReportTests(unittest.TestCase):
    def test_legacy_baseline_tpf_is_rebased_to_decode_only(self) -> None:
        payload = {
            "pytorch_native": {
                "decode": {
                    "request_count": 2,
                    "completion_tokens": 100,
                    "forward_passes": 21,
                    "tokens_per_forward_pass": 100 / 21,
                    "average_forward_passes_per_sample": 10.5,
                }
            }
        }
        normalized = decode(payload, new_method=False)
        self.assertEqual(normalized["decode_forward_passes"], 19)
        self.assertEqual(normalized["prefill_forward_passes"], 2)
        self.assertAlmostEqual(normalized["tokens_per_forward_pass"], 100 / 19)
        self.assertAlmostEqual(normalized["average_forward_passes_per_sample"], 9.5)

    def test_merge_keeps_ranked_states_and_dense_forward_distribution(self) -> None:
        states = {state: outcome() for state in OUTCOME_STATES}
        states["miss_before_first"] = outcome(1, 2, 1, 2, 4)
        states["p2_rank2_fixed"] = outcome(1, 5, 0, 0, 0)
        kinds = {kind: forward_kind() for kind in FORWARD_KINDS}
        kinds["prefill"] = forward_kind(
            count=1, computed=10, valid=10, padding=0, rows=1, query=10
        )
        kinds["multi_fused"] = forward_kind(
            count=2, computed=256, valid=210, padding=46, rows=8, query=64
        )
        rows = [
            {
                "ok": True,
                "prompt_tokens": 10,
                "completion_tokens": 8,
                "raw_generated_tokens": 8,
                "nfe": 3,
                "model_time_s": 1,
                "request_time_s": 1,
                "margin_risk_threshold": 0.5,
                "overlap": {
                    "physical_nfe": 3,
                    "processed_rows": 9,
                    "processed_query_tokens": 266,
                    "valid_query_tokens": 220,
                    "padding_query_tokens": 46,
                    "rounds": 2,
                    "prefetch_attempts": 2,
                    "candidate_branches_executed": 4,
                    "continuation_branches_executed": 1,
                    "crossing_count_rounds": {"0": 0, "1": 1, "2": 0, "3+": 1},
                    "policy_case_rounds": {
                        "at_most_two_plus_new": 1,
                        "three_plus_ranked": 1,
                    },
                    "fused_row_count": {"2": 0, "3": 0, "4": 2},
                    "candidate_branch_executed": {
                        "p1_rank2": 2,
                        "p1_rank3": 1,
                        "p2_rank2": 1,
                    },
                    "candidate_correction": {
                        "p1_rank2": {"checked": 1, "fixed": 0, "wrong": 1},
                        "p1_rank3": {"checked": 1, "fixed": 1, "wrong": 0},
                        "p2_rank2": {"checked": 1, "fixed": 1, "wrong": 0},
                    },
                    "forward_kinds": kinds,
                    "outcome_states": states,
                },
            }
        ]
        merged = summarize(rows, 1.5)
        overlap = merged["overlap"]
        self.assertTrue(overlap["outcome_partition_valid"])
        self.assertEqual(overlap["outcome_state_count_sum"], 2)
        before = overlap["outcome_states"]["miss_before_first"]
        self.assertEqual(before["share_of_attempts"], 0.5)
        self.assertEqual(before["current_accept_avg"], 2)
        self.assertEqual(before["next_accept_avg"], 4)
        self.assertEqual(before["next_minus_current_avg"], 2)
        p2 = overlap["outcome_states"]["p2_rank2_fixed"]
        self.assertEqual(p2["count"], 1)
        self.assertIsNone(p2["next_accept_avg"])
        self.assertEqual(overlap["policy_case_rounds"]["three_plus_ranked"], 1)
        self.assertEqual(overlap["candidate_branch_executed"]["p1_rank3"], 1)
        self.assertEqual(overlap["candidate_correction"]["p1_rank3"]["fixed_rate"], 1)
        fused = overlap["forward_kinds"]["multi_fused"]
        self.assertEqual(fused["computed_token_avg"], 128)
        self.assertEqual(fused["computed_token_p50"], 128)
        self.assertEqual(fused["padding_ratio"], 0.179688)
        self.assertEqual(overlap["forward_distribution_decode"]["count"], 2)
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
        self.assertEqual(merged["tokens_per_forward_pass"], 8)
        self.assertEqual(merged["decode_forward_passes"], 1)
        self.assertEqual(merged["prefill_forward_passes"], 1)
        self.assertEqual(merged["total_forward_passes"], 2)
        self.assertEqual(merged["end_to_end_tokens_per_forward_pass"], 4)

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
            wrapper = payload["pytorch_margin_risk_conditional_rank_overlap"]
            self.assertEqual(wrapper["accuracy_status"], "skipped_after_eval_failure")
            self.assertEqual(wrapper["decode"]["request_count"], 1)

    def test_engine_turns_oom_into_failed_stat_and_empty_placeholder(self) -> None:
        engine = MarginRiskConditionalRankOverlapEngine.__new__(MarginRiskConditionalRankOverlapEngine)
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
                "method.margin_risk_conditional_rank_overlap_linearspec.server.overlap_linear_spec_generate",
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

    def test_initial_report_exists_before_any_dataset_finishes(self) -> None:
        content = render(
            Path("/path/that/does/not/exist"),
            DEFAULT_BASELINE_16,
            DEFAULT_BASELINE_32,
        )
        self.assertIn("0/9", content)
        self.assertIn("尚无已完成", content)
        self.assertIn("AIME24", content)
        self.assertIn("九数据集等权", content)
        self.assertIn("MMLU 固定排在默认运行顺序最后", content)
        self.assertIn("15 个互斥状态", content)
        self.assertIn("P1 的 rank-2/rank-3", content)

    def test_report_renders_conditional_rank_states_and_forward_tokens(self) -> None:
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
            states = {
                state: {
                    **outcome(),
                    "share_of_attempts": 0,
                    "current_accept_avg": None,
                    "next_coverage": None,
                    "paired_current_accept_avg": None,
                    "next_accept_avg": None,
                    "next_minus_current_avg": None,
                }
                for state in OUTCOME_STATES
            }
            states["p3_detected_no_candidate"].update(outcome(2, 6, 1, 3, 4))
            states["p3_detected_no_candidate"].update(
                {
                    "share_of_attempts": 1.0,
                    "current_accept_avg": 3.0,
                    "next_coverage": 0.5,
                    "paired_current_accept_avg": 3.0,
                    "next_accept_avg": 4.0,
                    "next_minus_current_avg": 1.0,
                }
            )
            fused = {
                "count": 2,
                "computed_token_avg": 128,
                "computed_token_min": 128,
                "computed_token_p50": 128,
                "computed_token_p90": 128,
                "computed_token_p95": 128,
                "computed_token_p99": 128,
                "computed_token_max": 128,
                "valid_token_avg": 100,
                "padding_token_avg": 28,
                "padding_ratio": 0.21875,
                "rows_avg": 4,
                "query_length_avg": 32,
            }
            payload = {
                "gsm8k": {"pass@1": {"symbolic_correct": 50}},
                "pytorch_margin_risk_conditional_rank_overlap": {
                    "decode": {
                        "tokens_per_forward_pass": 4,
                        "average_forward_passes_per_sample": 2,
                        "model_output_tokens_per_s": 20,
                        "overlap": {
                            "rounds": 2,
                            "prefetch_attempts": 2,
                            "candidate_branches_executed": 6,
                            "continuation_branches_executed": 0,
                            "prefetch_verified_hits": 0,
                            "prefetch_hits": 0,
                            "prefetch_saved_draft_forwards": 0,
                            "crossing_count_rounds": {
                                "0": 0,
                                "1": 0,
                                "2": 0,
                                "3+": 2,
                            },
                            "policy_case_rounds": {
                                "at_most_two_plus_new": 0,
                                "three_plus_ranked": 2,
                            },
                            "fused_row_count": {"2": 0, "3": 0, "4": 2},
                            "candidate_branch_executed": {
                                "p1_rank2": 2,
                                "p1_rank3": 2,
                                "p2_rank2": 2,
                            },
                            "candidate_correction": {
                                "p1_rank2": {
                                    "checked": 0,
                                    "fixed": 0,
                                    "wrong": 0,
                                    "fixed_rate": None,
                                    "wrong_rate": None,
                                },
                                "p1_rank3": {
                                    "checked": 0,
                                    "fixed": 0,
                                    "wrong": 0,
                                    "fixed_rate": None,
                                    "wrong_rate": None,
                                },
                                "p2_rank2": {
                                    "checked": 0,
                                    "fixed": 0,
                                    "wrong": 0,
                                    "fixed_rate": None,
                                    "wrong_rate": None,
                                },
                            },
                            "forward_distribution_all": fused,
                            "forward_distribution_decode": fused,
                            "forward_kinds": {"multi_fused": fused},
                            "outcome_states": states,
                        },
                    }
                },
            }
            (root / "metrics_gsm8k.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )
            content = render(root, DEFAULT_BASELINE_16, DEFAULT_BASELINE_32)
            self.assertNotIn("新质量", content)
            self.assertNotIn("Acc状态", content)
            self.assertIn("P1二中", content)
            self.assertIn("P1三中", content)
            self.assertIn("P2二错", content)
            self.assertIn("P3仅定", content)
            self.assertIn("2/100.00%", content)
            self.assertIn("FwdTok均", content)
            self.assertIn("128", content)


if __name__ == "__main__":
    unittest.main()
