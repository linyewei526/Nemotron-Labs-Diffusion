from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parents[1]
OBSERVATIONS = HERE.parent
for path in (str(HERE), str(OBSERVATIONS)):
    if path not in sys.path:
        sys.path.insert(0, path)

from policy_runtime import choose_action  # noqa: E402
from reporting import init_run  # noqa: E402
from search import (  # noqa: E402
    TraceData,
    actions_from_values,
    exact_lambda_candidates,
    indices_weights,
    subset_eight_datasets,
)


def constant_model(block: int, value: int) -> dict:
    return {
        "type": "binned_monotone_survival",
        "block": block,
        "edges": [],
        "survival": [[1.0] * value + [0.0] * (block - value)],
        "missing_survival": [1.0] * value + [0.0] * (block - value),
    }


def policy() -> dict:
    return {
        "policy_type": "dual_action_space_scalar_efficiency",
        "signal": {"feature": "head", "orientation": 1},
        "full_streak_gate": None,
        "value_models": {
            "all": {
                "8": constant_model(8, 6),
                "16": constant_model(16, 10),
                "32": constant_model(32, 12),
            }
        },
        "profiles": {
            "s8": {"cold_start_block": 8, "allowed_blocks": [8, 16, 32], "default_lambda": 0.25},
            "s16": {"cold_start_block": 16, "allowed_blocks": [16, 32], "default_lambda": 0.25},
        },
    }


class RuntimeTest(unittest.TestCase):
    def test_profiles_share_values_but_restrict_actions(self) -> None:
        self.assertEqual(choose_action(policy(), {"head": 1}, "s8", 0.6)[0], 8)
        self.assertEqual(choose_action(policy(), {"head": 1}, "s16", 0.6)[0], 16)
        self.assertEqual(choose_action(policy(), {"head": 1}, "s8", 0.1)[0], 32)
        self.assertEqual(choose_action(policy(), {"head": 1}, "s16", 0.1)[0], 32)

    def test_profile_cold_start_is_applied_offline(self) -> None:
        values = np.asarray([[7, 15, 30], [7, 15, 30]], dtype=float)
        first = np.asarray([True, False])
        self.assertEqual(actions_from_values(values, 0.0, first, "s8").tolist(), [8, 32])
        self.assertEqual(actions_from_values(values, 0.0, first, "s16").tolist(), [16, 32])

    def test_exact_candidates_include_boundaries_and_intervals(self) -> None:
        values = np.asarray([[6, 10, 12], [7, 12, 24]], dtype=float)
        candidates = exact_lambda_candidates(values, np.zeros(2, dtype=bool), "s8", 1.0)
        lambdas = [row["lambda"] for row in candidates]
        self.assertIn(0.25, lambdas)  # (10-6)/(16-8)
        self.assertTrue(any(row["stable_width"] > 0 for row in candidates))


class ProtocolTest(unittest.TestCase):
    def make_data(self) -> TraceData:
        return TraceData(
            dataset_names=["gsm8k", "human-eval", "mmlu"],
            dataset=np.asarray([0, 0, 1, 1, 1, 2, 2], dtype=np.uint8),
            request=np.asarray([0, 1, 2, 2, 3, 4, 5], dtype=np.uint32),
            fold=np.zeros(7, dtype=np.uint8),
            round=np.asarray([0, 0, 0, 1, 0, 0, 0], dtype=np.uint32),
            decision=np.full(7, 16, dtype=np.uint8),
            accept=np.asarray([[4, 5, 6]] * 7, dtype=np.uint8),
            features={"full_streak": np.zeros(7, dtype=np.float32)},
            quality={
                "invalid_row_policy": "exclude",
                "by_dataset": {
                    name: {
                        "rows_original": count,
                        "rows_usable": count,
                        "rows_excluded": 0,
                        "replay_mismatch": 0,
                        "cross_block_mismatch": 0,
                        "excluded_rate": 0,
                    }
                    for name, count in (("gsm8k", 2), ("human-eval", 3), ("mmlu", 2))
                },
            },
            request_counts={"gsm8k": 2, "human-eval": 2, "mmlu": 2},
        )

    def test_mmlu_is_removed_and_dataset_weights_stay_equal(self) -> None:
        data = subset_eight_datasets(self.make_data(), True, 0.05)
        self.assertEqual(data.dataset_names, ["gsm8k", "human-eval"])
        self.assertEqual(data.size, 5)
        weights = indices_weights(data, np.arange(data.size))
        self.assertAlmostEqual(float(weights[data.dataset == 0].sum()), 0.5)
        self.assertAlmostEqual(float(weights[data.dataset == 1].sum()), 0.5)
        self.assertEqual(data.quality["explicitly_ignored_datasets"], ["mmlu"])


class ReportingTest(unittest.TestCase):
    def test_template_is_created_before_search(self) -> None:
        settings = {
            "stage": "all", "benchmarks": "eight", "trace_root": "/trace",
            "policy_lambda_s8": "auto", "policy_lambda_s16": "auto",
            "cv_folds": "5", "signal_bins": "32", "lambda_max": "1",
            "min_dual_improvement": "0.002", "full_gate_grid": "1,2,3",
            "model": "/model", "mode": "linearspec_lora", "lora_mode": "draft_only",
            "gpu_devices": "0", "tp_size": "1", "batch_size": "1",
            "client_concurrency": "1", "search_gpu_hold_gb": "48",
            "search_gpu_hold_chunk_gb": "1", "context_length": "10240",
            "tokens": "8192", "dtype": "bfloat16", "mem_fraction": "0.55",
            "gpu_memory_reserve_gb": "0", "port": "", "proxy_port": "",
            "command": "test",
        }
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            init_run(run_dir, settings)
            self.assertTrue((run_dir / "settings.md").is_file())
            report = (run_dir / "report.md").read_text(encoding="utf-8")
            self.assertIn("双动作空间", report)
            self.assertIn("S8 冻结验证", report)


if __name__ == "__main__":
    unittest.main()

