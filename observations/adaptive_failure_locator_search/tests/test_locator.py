#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from observations.adaptive_failure_locator_search.locator_generation import (
    position_diagnostics,
)
from observations.adaptive_failure_locator_search.strategy_search import (
    Candidate,
    candidate_grid,
    metrics,
    predictions_from_score,
    run_search,
    split_requests,
)


def make_round(dataset: str, request_id: str, round_index: int, q: int | None) -> dict:
    confidences = [0.9 - 0.01 * index for index in range(15)]
    if q is not None:
        confidences[q - 1] = 0.2
    prefix_drop = [None]
    local_drop = [None]
    prefix_median = [None]
    for index in range(1, len(confidences)):
        prefix = confidences[:index]
        prefix_drop.append(1.0 - confidences[index] / (sum(prefix) / len(prefix)))
        local_drop.append(1.0 - confidences[index] / confidences[index - 1])
        prefix_median.append(float(np.median(prefix)))
    return {
        "schema_version": 1,
        "event": "linearspec_failure_locator_round",
        "benchmark": dataset,
        "request_id": request_id,
        "round_index": round_index,
        "block_size": 16,
        "analysis_valid": True,
        "eos_hit": False,
        "budget_boundary": False,
        "accept_length": q or 16,
        "mismatch_position": q,
        "position": {
            "selected_confidence": confidences,
            "top1_top2_margin": [max(0.01, value - 0.1) for value in confidences],
            "entropy": [1.0 + 3.0 * (1.0 - value) for value in confidences],
            "prefix_drop_pct": prefix_drop,
            "local_drop_pct": local_drop,
            "prefix_median_before": prefix_median,
        },
    }


class FeatureTests(unittest.TestCase):
    def test_position_one_has_no_prefix_but_has_absolute_features(self) -> None:
        logits = torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 5.0, 1.0, -2.0],
                [0.0, 1.0, 4.0, -2.0],
            ]
        )
        tokens = torch.tensor([3, 1, 2])
        result = position_diagnostics(logits, tokens, mask_token_id=0)
        self.assertIsNone(result["prefix_drop_pct"][0])
        self.assertIsNone(result["local_drop_pct"][0])
        self.assertGreater(result["selected_confidence"][0], 0.9)
        self.assertEqual(len(result["entropy"]), 2)

    def test_strict_threshold_and_first_vs_max(self) -> None:
        score = np.asarray([[0.15, 0.2, 0.9], [0.1, 0.15, 0.14]], dtype=np.float32)
        first = predictions_from_score(score, threshold=0.15, decision="first")
        maximum = predictions_from_score(score, threshold=0.15, decision="max")
        self.assertEqual(first.tolist(), [2, 0])
        self.assertEqual(maximum.tolist(), [3, 0])

    def test_exact_metrics_count_wrong_position_as_fp_and_fn(self) -> None:
        predicted = np.asarray([2, 4, 0, 3], dtype=np.int16)
        labels = np.asarray([2, 3, 1, 0], dtype=np.int16)
        result = metrics(predicted, labels, np.ones(4, dtype=bool))
        self.assertEqual(result["exact_hits"], 1)
        self.assertEqual(result["attempts"], 3)
        self.assertEqual(result["failures"], 3)
        self.assertAlmostEqual(result["precision"], 1 / 3)
        self.assertAlmostEqual(result["recall"], 1 / 3)
        self.assertAlmostEqual(result["full_false_alarm"], 1.0)


class SplitAndSearchTests(unittest.TestCase):
    def test_request_split_never_leaks(self) -> None:
        mapping = split_requests(
            {"gsm8k": {f"r{index}" for index in range(10)}},
            seed=7,
            search_ratio=0.6,
            selection_ratio=0.2,
        )
        self.assertEqual(len(mapping), 10)
        counts = {split: list(mapping.values()).count(split) for split in ("search", "selection", "test")}
        self.assertEqual(counts, {"search": 6, "selection": 2, "test": 2})

    def test_grid_contains_original_and_all_history_windows(self) -> None:
        grid = candidate_grid([1, 2, 4], ["mean"], "compact")
        originals = [candidate for candidate in grid if candidate.original_baseline]
        self.assertEqual(len(originals), 1)
        self.assertEqual(originals[0].threshold, 0.15)
        self.assertEqual({candidate.history_window for candidate in grid if candidate.uses_history}, {1, 2, 4})

    def test_full_search_excludes_aime24_and_writes_incremental_summaries(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            (run_dir / "traces").mkdir()
            for dataset in ("gsm8k", "math-500", "aime24"):
                path = run_dir / "traces" / f"failure_locator_{dataset}.jsonl"
                with path.open("w", encoding="utf-8") as stream:
                    for request_index in range(10):
                        for round_index in range(4):
                            q = None if round_index == 3 else 1 + (request_index + round_index) % 6
                            stream.write(
                                json.dumps(make_round(dataset, f"{dataset}-{request_index}", round_index, q)) + "\n"
                            )
            args = argparse.Namespace(
                run_dir=run_dir,
                trace_dir=run_dir / "traces",
                output_json=run_dir / "analysis" / "strategy_search.json",
                block_size=16,
                history_windows="1,2,4",
                aggregations="mean",
                grid="compact",
                split_seed=11,
                search_ratio=0.6,
                selection_ratio=0.2,
                shortlist=20,
                report_top=5,
                search_max_rounds_per_dataset=1000,
                bootstrap_replicates=5,
                include_boundary_rounds=False,
            )
            payload = run_search(args)
            self.assertEqual(payload["status"], "ok")
            self.assertEqual(payload["winner"]["test"]["macro"]["datasets"], 2)
            self.assertNotIn("aime24", payload["per_dataset_oracle"])
            self.assertTrue((run_dir / "summaries" / "failure_locator_aime24.json").is_file())
            self.assertFalse(payload["selection_contract"]["test_used_for_selection"])


if __name__ == "__main__":
    unittest.main()
