from __future__ import annotations

import argparse
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from search import FEATURES, TraceData, search_policy  # noqa: E402


class FullOfflineSearchTest(unittest.TestCase):
    def test_all_candidates_export_one_unified_policy(self) -> None:
        datasets = []
        requests = []
        folds = []
        rounds = []
        decisions = []
        accepts = []
        values = {name: [] for name in FEATURES}
        request_id = 0
        per_dataset = 200
        for dataset_id in range(2):
            for sample in range(per_dataset):
                capability = 2 + (sample % 30)
                for round_index in (0, 1):
                    datasets.append(dataset_id)
                    requests.append(request_id)
                    folds.append(request_id % 5)
                    rounds.append(round_index)
                    decisions.append(16)
                    accepts.append([min(8, capability), min(16, capability), min(32, capability)])
                    for name in FEATURES:
                        if round_index == 0:
                            value = np.nan
                        elif name == "full_streak":
                            value = 2.0 if sample % 2 else 0.0
                        elif name == "nonfull_streak":
                            value = 0.0 if sample % 2 else 2.0
                        else:
                            value = float(capability)
                        values[name].append(value)
                request_id += 1
        data = TraceData(
            dataset_names=["d0", "d1"],
            dataset=np.asarray(datasets, dtype=np.uint8),
            request=np.asarray(requests, dtype=np.uint32),
            fold=np.asarray(folds, dtype=np.uint8),
            round=np.asarray(rounds, dtype=np.uint32),
            decision=np.asarray(decisions, dtype=np.uint8),
            accept=np.asarray(accepts, dtype=np.uint8),
            features={name: np.asarray(items, dtype=np.float32) for name, items in values.items()},
            quality={"rows_original": len(rounds), "rows_usable": len(rounds)},
            request_counts={"d0": per_dataset, "d1": per_dataset},
        )
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            args = argparse.Namespace(
                lambda_grid=[0.0, 0.25, 0.5],
                cv_folds=5,
                signal_bins=8,
                full_gate_grid=[1, 2],
                min_dual_improvement=0.02,
                min_large_precision=0.0,
                max_large_waste=1.0,
                max_loss_vs_l32=32.0,
                min_gain16=2.0,
                min_gain32=4.0,
                default_lambda=None,
                output_dir=output,
                run_dir=None,
            )
            result = search_policy(args, data)
            policy = result["policy"]
            self.assertIn(policy["signal_count"], (1, 2))
            self.assertEqual(policy["blocks"], [8, 16, 32])
            self.assertEqual(set(policy["value_models"]), {"all"} if policy["signal_count"] == 1 else {"not_full", "full"})
            self.assertTrue((output / "policy.json").is_file())
            self.assertTrue((output / "search_results.json").is_file())
            actions = [row["mean_block"] for row in result["pareto"]]
            self.assertTrue(all(after <= before + 1e-12 for before, after in zip(actions, actions[1:])))


if __name__ == "__main__":
    unittest.main()
