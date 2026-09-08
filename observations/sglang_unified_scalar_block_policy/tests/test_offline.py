from __future__ import annotations

import json
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
    fit_survival,
    indices_weights,
    pava_increasing,
    predict_survival,
)


def constant_model(block: int, value: int) -> dict:
    return {
        "type": "binned_monotone_survival",
        "block": block,
        "edges": [],
        "survival": [[1.0] * value + [0.0] * (block - value)],
        "missing_survival": [1.0] * value + [0.0] * (block - value),
    }


class PolicyRuntimeTest(unittest.TestCase):
    def test_lambda_is_the_only_size_tendency_knob(self) -> None:
        policy = {
            "signal": {"feature": "a_ma4", "orientation": 1},
            "default_lambda": 0.25,
            "full_streak_gate": None,
            "value_models": {
                "all": {
                    "8": constant_model(8, 6),
                    "16": constant_model(16, 10),
                    "32": constant_model(32, 12),
                }
            },
        }
        actions = [choose_action(policy, {"a_ma4": 5}, lam)[0] for lam in (0.1, 0.25, 0.6)]
        self.assertEqual(actions, [32, 16, 8])

    def test_full_streak_only_selects_a_global_stratum(self) -> None:
        policy = {
            "signal": {"feature": "a_ma4", "orientation": 1},
            "default_lambda": 0.2,
            "full_streak_gate": {"threshold": 2},
            "value_models": {
                "not_full": {str(b): constant_model(b, 3) for b in (8, 16, 32)},
                "full": {
                    "8": constant_model(8, 8),
                    "16": constant_model(16, 14),
                    "32": constant_model(32, 25),
                },
            },
        }
        low, low_scores = choose_action(policy, {"a_ma4": 5, "full_streak": 1})
        high, high_scores = choose_action(policy, {"a_ma4": 5, "full_streak": 2})
        self.assertEqual(low, 8)
        self.assertEqual(high, 32)
        self.assertEqual(low_scores["full_gate"], 0.0)
        self.assertEqual(high_scores["full_gate"], 1.0)


class SearchCoreTest(unittest.TestCase):
    def make_data(self) -> TraceData:
        return TraceData(
            dataset_names=["small", "large"],
            dataset=np.asarray([0, 0, 1, 1, 1, 1], dtype=np.uint8),
            request=np.asarray([0, 1, 2, 2, 3, 3], dtype=np.uint32),
            fold=np.zeros(6, dtype=np.uint8),
            round=np.asarray([1, 1, 1, 2, 1, 2], dtype=np.uint32),
            decision=np.full(6, 16, dtype=np.uint8),
            accept=np.asarray([[2, 2, 2]] * 6, dtype=np.uint8),
            features={"full_streak": np.zeros(6, dtype=np.float32)},
            quality={},
            request_counts={"small": 2, "large": 2},
        )

    def test_hierarchical_weights_make_datasets_equal(self) -> None:
        data = self.make_data()
        indices = np.arange(data.size)
        weights = indices_weights(data, indices)
        self.assertAlmostEqual(float(weights[data.dataset == 0].sum()), 0.5)
        self.assertAlmostEqual(float(weights[data.dataset == 1].sum()), 0.5)
        self.assertAlmostEqual(float(weights[data.request == 2].sum()), 0.25)

    def test_pava_and_survival_are_monotone(self) -> None:
        fitted = pava_increasing(np.asarray([0.8, 0.2, 0.9]), np.ones(3))
        self.assertTrue(np.all(np.diff(fitted) >= -1e-12))
        signal = np.asarray([0, 0, 1, 1, 2, 2], dtype=float)
        accepted = np.asarray([1, 2, 3, 4, 5, 6], dtype=int)
        model = fit_survival(signal, accepted, np.ones(6), block=8, bins=3)
        values = predict_survival(model, np.asarray([0, 1, 2], dtype=float))
        self.assertTrue(np.all(np.diff(values) >= -1e-12))

    def test_actions_never_grow_when_lambda_increases(self) -> None:
        values = np.asarray([[6, 10, 12], [7, 8, 20]], dtype=float)
        first = np.zeros(2, dtype=bool)
        by_lambda = [actions_from_values(values, lam, first) for lam in (0.0, 0.2, 0.8)]
        for before, after in zip(by_lambda, by_lambda[1:]):
            self.assertTrue(np.all(after <= before))


class ReportingTest(unittest.TestCase):
    def test_template_exists_at_initialization(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            settings = {
                "stage": "all", "benchmarks": "nine", "trace_root": "/trace",
                "model": "/model", "mode": "linearspec_lora", "lora_mode": "draft_only",
                "policy_lambda": "auto", "cv_folds": "5", "signal_bins": "32",
                "lambda_grid": "0,0.25", "mmlu_max_samples": "2000",
                "gpu_devices": "0", "tp_size": "1", "batch_size": "1",
                "client_concurrency": "1", "search_gpu_hold_gb": "48",
                "search_gpu_hold_chunk_gb": "1", "context_length": "10240",
                "tokens": "8192", "dtype": "bfloat16", "mem_fraction": "0.55",
                "gpu_memory_reserve_gb": "0", "port": "", "proxy_port": "",
                "command": "test",
            }
            init_run(run_dir, settings)
            self.assertTrue((run_dir / "settings.md").is_file())
            report = (run_dir / "report.md").read_text(encoding="utf-8")
            self.assertIn("实时进度", report)
            self.assertIn("统一标量", report)


if __name__ == "__main__":
    unittest.main()

