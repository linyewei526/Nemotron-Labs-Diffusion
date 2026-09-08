from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np

from observations.sglang_b200_latency_dynamic_block_policy.latency_costs import (
    cold_start_block,
    parse_cost_document,
    snapshot,
)
from observations.sglang_b200_latency_dynamic_block_policy.policy_runtime import (
    choose_action,
)
from observations.sglang_b200_latency_dynamic_block_policy.reporting import init_run
from observations.sglang_b200_latency_dynamic_block_policy.search import (
    IncrementalLambdaEvaluator,
    actions_from_values,
    effective_request_concurrency,
    publish_progress,
    read_committed_trace,
    replay_actions,
    sample_summary,
)
from observations.sglang_b200_latency_dynamic_block_policy.signal_models import (
    FEATURES,
    TraceData,
)


ROOT = Path(__file__).resolve().parents[3]
COST_DOC = ROOT / "configs/NLD_B200_B8_B32_forward_sweep_20260907_zh.md"


def toy_data() -> TraceData:
    return TraceData(
        dataset_names=["gsm8k", "mbpp"],
        dataset=np.asarray([0, 0, 0, 1], dtype=np.uint8),
        request=np.asarray([0, 0, 1, 2], dtype=np.uint32),
        fold=np.zeros(4, dtype=np.uint8),
        round=np.asarray([0, 1, 0, 0], dtype=np.uint32),
        decision=np.asarray([16, 8, 16, 32], dtype=np.uint8),
        accept=np.asarray([[4, 5, 6], [2, 3, 4], [8, 9, 10], [1, 2, 3]], dtype=np.uint8),
        features={},
        quality={},
        request_counts={"gsm8k": 2, "mbpp": 1},
    )


class CoreTests(unittest.TestCase):
    def test_cost_document_is_complete_and_selected_values_match(self) -> None:
        costs = parse_cost_document(COST_DOC)
        self.assertEqual(len(costs), 128)
        self.assertAlmostEqual(costs[2][8], 10.955210)
        self.assertAlmostEqual(costs[128][32], 132.447587)
        selected = snapshot(COST_DOC, [2, 128])["selected_costs_ms"]
        self.assertEqual(set(selected), {"2", "128"})

    def test_tail_concurrency_is_assigned_per_sample_not_per_round(self) -> None:
        data = toy_data()
        # Dataset 0 has two requests under C2 (no tail); dataset 1 has one and
        # therefore uses C1.  Both rounds of request 0 keep C2.
        assigned = effective_request_concurrency(data, 2)
        self.assertEqual(assigned.tolist(), [2, 2, 1])

    def test_sample_then_dataset_macro_is_not_round_or_sample_pooled(self) -> None:
        data = toy_data()
        costs = {
            1: {8: 1.0, 16: 2.0, 32: 4.0},
            2: {8: 2.0, 16: 4.0, 32: 8.0},
        }
        actions = np.asarray([8, 8, 8, 8], dtype=np.uint8)
        result = sample_summary(data, actions, 2, costs)
        # GSM samples: (4+2)/(2+2)=1.5 and 8/2=4 => dataset score 2.75.
        # MBPP sample: 1/1=1 due tail C1. Eight-style outer rule here is
        # two-dataset equal: (2.75+1)/2=1.875.
        self.assertAlmostEqual(result["datasets"]["gsm8k"]["score_token_ms_req"], 2.75)
        self.assertAlmostEqual(result["datasets"]["mbpp"]["score_token_ms_req"], 1.0)
        self.assertAlmostEqual(result["macro"]["score_token_ms_req"], 1.875)

    def test_lambda_is_only_concurrency_specific_size_tendency_knob(self) -> None:
        values = np.asarray([[0, 0, 0], [4, 6, 10]], dtype=float)
        first = np.asarray([True, False])
        costs = {8: 10.0, 16: 12.0, 32: 20.0}
        low = actions_from_values(values, 0.0, first, costs, 32)
        high = actions_from_values(values, 1.0, first, costs, 32)
        self.assertEqual(low.tolist(), [32, 32])
        self.assertEqual(high.tolist(), [32, 8])

    def test_concurrency_specific_cold_start_protocol(self) -> None:
        expected = {2: 32, 4: 32, 8: 16, 16: 16, 32: 16, 64: 8, 128: 8}
        self.assertEqual(
            {concurrency: cold_start_block(concurrency) for concurrency in expected},
            expected,
        )
        with self.assertRaises(ValueError):
            cold_start_block(1)

    def test_frozen_replay_enforces_concurrency_cold_start(self) -> None:
        policy = {
            "concurrency_profiles": {
                str(concurrency): {"cold_start_block": cold_start_block(concurrency)}
                for concurrency in (2, 4, 8, 16, 32, 64, 128)
            }
        }
        for concurrency in (2, 8, 64):
            block = cold_start_block(concurrency)
            data = TraceData(
                dataset_names=["gsm8k"],
                dataset=np.asarray([0], dtype=np.uint8),
                request=np.asarray([0], dtype=np.uint32),
                fold=np.asarray([0], dtype=np.uint8),
                round=np.asarray([0], dtype=np.uint32),
                decision=np.asarray([block], dtype=np.uint8),
                accept=np.asarray([[1, 1, 1]], dtype=np.uint8),
                features={name: np.asarray([np.nan]) for name in FEATURES},
                quality={},
                request_counts={"gsm8k": 1},
            )
            self.assertEqual(replay_actions(data, policy, concurrency)["mismatches"], 0)

    def test_incremental_lambda_scan_matches_full_sample_summary(self) -> None:
        data = toy_data()
        values = np.asarray(
            [[0, 0, 0], [2, 4, 7], [0, 0, 0], [0, 0, 0]], dtype=float
        )
        costs = {
            1: {8: 1.0, 16: 2.0, 32: 4.0},
            2: {8: 2.0, 16: 4.0, 32: 8.0},
        }
        candidates = [{"lambda": value} for value in (0.0, 0.5, 1.0, 2.0)]
        fast = IncrementalLambdaEvaluator(data, values, 2, costs).scan(candidates)
        full = []
        for candidate in candidates:
            actions = actions_from_values(
                values, candidate["lambda"], data.round == 0, costs[2], 32
            )
            full.append(sample_summary(data, actions, 2, costs)["macro"]["score_token_ms_req"])
        np.testing.assert_allclose(fast, full, rtol=0, atol=1e-12)

    def test_runtime_uses_latency_not_block_tokens(self) -> None:
        survival = lambda value, block: {
            "edges": [], "survival": [[value / block] * block],
            "missing_survival": [value / block] * block,
        }
        policy = {
            "policy_type": "b200_latency_scalar_value",
            "signal": {"feature": "a_ma2", "orientation": 1},
            "full_streak_gate": None,
            "value_models": {"all": {
                "8": survival(5, 8), "16": survival(7, 16), "32": survival(10, 32),
            }},
            "concurrency_profiles": {
                "32": {"lambda": 0.5, "costs_ms": {"8": 20, "16": 25, "32": 40}}
            },
        }
        action, scores = choose_action(policy, {"a_ma2": 3}, 32)
        # utilities: -5, -5.5, -10 => B8.
        self.assertEqual(action, 8)
        self.assertAlmostEqual(scores["t32_ms"], 40)

    def test_init_creates_settings_report_and_live_progress(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "run"
            settings = {
                "stage": "all", "benchmarks": "gsm8k:1", "trace_root": "/trace",
                "cost_document": str(COST_DOC), "concurrencies": "2",
                "cv_folds": "2", "signal_bins": "4", "lambda_max": "2",
                "full_gate_grid": "1", "gpu_devices": "0",
                "search_gpu_hold_gb": "1", "search_gpu_hold_chunk_gb": "1",
                "model": "m", "mode": "linearspec_lora", "context_length": "128",
                "tokens": "16", "mem_fraction": "0.5", "gpu_memory_reserve_gb": "0",
                "dtype": "bfloat16", "port": "", "proxy_port": "", "command": "cmd",
            }
            init_run(run, settings)
            self.assertTrue((run / "settings.md").exists())
            self.assertTrue((run / "report.md").exists())
            self.assertTrue((run / "progress.md").exists())
            self.assertEqual(json.loads((run / "run_state.json").read_text())["events"], [])

    def test_validation_progress_does_not_overwrite_search_progress(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            publish_progress(
                None,
                output,
                {"stage": "search_completed", "completed": 33, "total": 33},
            )
            publish_progress(
                None,
                output,
                {"stage": "validation_c2_partial", "completed": 1, "total": 8},
            )
            search = json.loads((output / "progress.json").read_text())
            self.assertEqual(search["stage"], "search_completed")
            self.assertTrue((output / "validation_c2_partial_progress.json").exists())

    def test_committed_validation_does_not_reject_shadow_prefix_diagnostic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            trace_root = Path(temporary)
            record = {
                "event": "sglang_dynamic_block_shadow_round",
                "prompt_fingerprint": "sample-1",
                "request_id": "rid",
                "round_index": 0,
                "decision_block": 16,
                "canonical_replay_match": True,
                "cross_block_common_prefix_match": False,
                "history_before_round": {"history_rounds": 0},
                "branches": {
                    "8": {"accept_length": 3},
                    "16": {"accept_length": 4},
                    "32": {"accept_length": 5},
                },
            }
            (trace_root / "gsm8k.jsonl").write_text(
                json.dumps(record) + "\n", encoding="utf-8"
            )
            args = Namespace(
                trace_root=trace_root,
                max_rows_per_dataset=0,
                split_seed=1,
                cv_folds=2,
                allow_partial_datasets=True,
                concurrency=2,
            )
            data = read_committed_trace(args)
            self.assertEqual(data.size, 1)
            self.assertEqual(data.quality["rows_excluded"], 0)
            self.assertEqual(data.quality["cross_block_mismatch_diagnostic_only"], 1)
            self.assertEqual(set(data.features), set(FEATURES))


if __name__ == "__main__":
    unittest.main()
