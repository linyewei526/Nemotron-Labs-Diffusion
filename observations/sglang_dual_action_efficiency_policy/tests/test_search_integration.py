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


class SearchIntegrationTest(unittest.TestCase):
    def test_shared_signal_exports_two_action_spaces(self) -> None:
        datasets = []
        requests = []
        folds = []
        rounds = []
        decisions = []
        accepts = []
        features = {name: [] for name in FEATURES}
        request_id = 0
        for dataset_id in range(2):
            for sample in range(180):
                capability = 2 + sample % 30
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
                            value = float(sample % 4)
                        elif name == "nonfull_streak":
                            value = float(3 - sample % 4)
                        else:
                            value = float(capability)
                        features[name].append(value)
                request_id += 1
        data = TraceData(
            dataset_names=["d0", "d1"],
            dataset=np.asarray(datasets, dtype=np.uint8),
            request=np.asarray(requests, dtype=np.uint32),
            fold=np.asarray(folds, dtype=np.uint8),
            round=np.asarray(rounds, dtype=np.uint32),
            decision=np.asarray(decisions, dtype=np.uint8),
            accept=np.asarray(accepts, dtype=np.uint8),
            features={name: np.asarray(values, dtype=np.float32) for name, values in features.items()},
            quality={"rows_original": len(rounds), "rows_usable": len(rounds)},
            request_counts={"d0": 180, "d1": 180},
        )
        with tempfile.TemporaryDirectory() as temporary:
            args = argparse.Namespace(
                cv_folds=5,
                signal_bins=8,
                lambda_max=1.0,
                diagnostic_lambda_grid=[0.0, 0.25, 0.5, 1.0],
                full_gate_grid=[1, 2],
                min_dual_improvement=0.002,
                min_gain16=2.0,
                min_gain32=4.0,
                output_dir=Path(temporary),
                run_dir=None,
            )
            result = search_policy(args, data)
            policy = result["policy"]
            self.assertEqual(policy["policy_type"], "dual_action_space_scalar_efficiency")
            self.assertEqual(policy["profiles"]["s8"]["cold_start_block"], 8)
            self.assertEqual(policy["profiles"]["s8"]["allowed_blocks"], [8, 16, 32])
            self.assertEqual(policy["profiles"]["s16"]["cold_start_block"], 16)
            self.assertEqual(policy["profiles"]["s16"]["allowed_blocks"], [16, 32])
            self.assertEqual(policy["signal"], result["policy"]["signal"])
            self.assertTrue((Path(temporary) / "policy.json").is_file())
            self.assertTrue((Path(temporary) / "search_results.json").is_file())


if __name__ == "__main__":
    unittest.main()

