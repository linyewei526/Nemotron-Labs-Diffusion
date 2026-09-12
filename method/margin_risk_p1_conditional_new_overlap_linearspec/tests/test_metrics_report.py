from __future__ import annotations

import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from method.margin_risk_p1_conditional_new_overlap_linearspec.generation import (
    FORWARD_KINDS,
    OUTCOME_STATES,
)
from method.margin_risk_p1_conditional_new_overlap_linearspec.merge_metrics import main, summarize
from method.margin_risk_p1_conditional_new_overlap_linearspec.report import (
    DEFAULT_BASELINE_16,
    DEFAULT_BASELINE_32,
    decode,
    report_datasets,
    render,
)
from method.margin_risk_p1_conditional_new_overlap_linearspec.server import (
    ChatCompletionRequest,
    MarginRiskP1ConditionalNewOverlapEngine,
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
    def test_report_scope_is_dynamic_and_excludes_aime24_mmlu(self) -> None:
        settings = {
            "benchmark": {
                "benchmarks": "gsm8k:1,aime24:1,math-500:1,mmlu:1,gsm8k:1"
            }
        }
        self.assertEqual(report_datasets(settings), ("gsm8k", "math-500"))

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

    def test_merge_keeps_p1_and_omitted_states_and_dense_forward_distribution(self) -> None:
        states = {state: outcome() for state in OUTCOME_STATES}
        states["miss_before_first"] = outcome(1, 2, 1, 2, 4)
        states["p2_omitted_rank2_would_fix"] = outcome(1, 5, 0, 0, 0)
        kinds = {kind: forward_kind() for kind in FORWARD_KINDS}
        kinds["prefill"] = forward_kind(
            count=1, computed=10, valid=10, padding=0, rows=1, query=10
        )
        kinds["multi_fused"] = forward_kind(
            count=2, computed=192, valid=160, padding=32, rows=6, query=64
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
                    "processed_rows": 7,
                    "processed_query_tokens": 202,
                    "valid_query_tokens": 170,
                    "padding_query_tokens": 32,
                    "rounds": 2,
                    "prefetch_attempts": 2,
                    "candidate_branches_executed": 2,
                    "continuation_branches_executed": 2,
                    "crossing_count_rounds": {"0": 0, "1": 1, "2": 1, "3+": 0},
                    "policy_case_rounds": {
                        "c0_new": 0,
                        "c1_p1_new": 1,
                        "c2_p1_new": 1,
                        "c3plus_p1": 0,
                    },
                    "fused_row_count": {"2": 0, "3": 2},
                    "candidate_branch_executed": {"p1_rank2": 2},
                    "candidate_correction": {
                        "p1_rank2": {"checked": 1, "fixed": 0, "wrong": 1},
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
        p2 = overlap["outcome_states"]["p2_omitted_rank2_would_fix"]
        self.assertEqual(p2["count"], 1)
        self.assertIsNone(p2["next_accept_avg"])
        self.assertEqual(overlap["policy_case_rounds"]["c2_p1_new"], 1)
        self.assertEqual(overlap["candidate_branch_executed"]["p1_rank2"], 2)
        self.assertEqual(set(overlap["candidate_correction"]), {"p1_rank2"})
        fused = overlap["forward_kinds"]["multi_fused"]
        self.assertEqual(fused["computed_token_avg"], 96)
        self.assertEqual(fused["computed_token_p50"], 96)
        self.assertEqual(fused["padding_ratio"], 0.166667)
        self.assertEqual(overlap["forward_distribution_decode"]["count"], 2)
        self.assertEqual(overlap["decode_dense_query_token_slots_total"], 192)
        self.assertEqual(overlap["decode_dense_query_token_slots_per_forward"], 96)
        self.assertEqual(overlap["decode_dense_query_token_slots_per_output_token"], 24)
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
            wrapper = payload["pytorch_margin_risk_p1_conditional_new_overlap"]
            self.assertEqual(wrapper["accuracy_status"], "skipped_after_eval_failure")
            self.assertEqual(wrapper["decode"]["request_count"], 1)

    def test_engine_turns_oom_into_failed_stat_and_empty_placeholder(self) -> None:
        engine = MarginRiskP1ConditionalNewOverlapEngine.__new__(MarginRiskP1ConditionalNewOverlapEngine)
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
                "method.margin_risk_p1_conditional_new_overlap_linearspec.server.overlap_linear_spec_generate",
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
        self.assertIn("0/8", content)
        self.assertIn("尚无已完成", content)
        self.assertIn("AIME24", content)
        self.assertIn("八数据集等权", content)
        self.assertIn("AIME24/MMLU", content)
        self.assertIn("14 个互斥状态", content)
        self.assertIn("实际 P1 二选", content)

    def test_report_renders_omitted_p2_p3_states_and_dense_slots(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "Settings.json").write_text(
                json.dumps(
                    {
                        "status": "running",
                        "benchmark": {"benchmarks": "gsm8k:1", "tokens": 64, "temperature": 0},
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
            states["p3_omitted_rank2_would_fix"].update(outcome(2, 6, 1, 3, 4))
            states["p3_omitted_rank2_would_fix"].update(
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
                "computed_token_avg": 96,
                "computed_token_min": 96,
                "computed_token_p50": 96,
                "computed_token_p90": 96,
                "computed_token_p95": 96,
                "computed_token_p99": 96,
                "computed_token_max": 96,
                "valid_token_avg": 80,
                "padding_token_avg": 16,
                "padding_ratio": 0.166667,
                "rows_avg": 3,
                "query_length_avg": 32,
            }
            payload = {
                "gsm8k": {"pass@1": {"symbolic_correct": 50}},
                "pytorch_margin_risk_p1_conditional_new_overlap": {
                    "decode": {
                        "tokens_per_forward_pass": 4,
                        "average_forward_passes_per_sample": 2,
                        "model_output_tokens_per_s": 20,
                        "overlap": {
                            "rounds": 2,
                            "prefetch_attempts": 2,
                            "candidate_branches_executed": 2,
                            "continuation_branches_executed": 2,
                            "prefetch_verified_hits": 0,
                            "prefetch_hits": 0,
                            "prefetch_saved_draft_forwards": 0,
                            "crossing_count_rounds": {
                                "0": 0,
                                "1": 0,
                                "2": 2,
                                "3+": 0,
                            },
                            "policy_case_rounds": {
                                "c0_new": 0,
                                "c1_p1_new": 0,
                                "c2_p1_new": 2,
                                "c3plus_p1": 0,
                            },
                            "fused_row_count": {"2": 0, "3": 2},
                            "candidate_branch_executed": {"p1_rank2": 2},
                            "candidate_correction": {
                                "p1_rank2": {
                                    "checked": 0,
                                    "fixed": 0,
                                    "wrong": 0,
                                    "fixed_rate": None,
                                    "wrong_rate": None,
                                },
                            },
                            "forward_distribution_all": fused,
                            "forward_distribution_decode": fused,
                            "decode_dense_query_token_slots_total": 192,
                            "decode_dense_query_token_slots_per_forward": 96,
                            "decode_dense_query_token_slots_per_output_token": 12,
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
            self.assertIn("P1中", content)
            self.assertIn("P2也会错", content)
            self.assertIn("P3本可修", content)
            self.assertIn("2/100.00%", content)
            self.assertIn("每TokSlot", content)
            self.assertIn("FwdTok均", content)
            self.assertIn("96", content)


if __name__ == "__main__":
    unittest.main()
