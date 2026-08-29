#!/usr/bin/env python3

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch
from transformers.cache_utils import DynamicCache

from observations.pytorch_linearspec_block_size_shadow.run_manager import (
    atomic_json,
    render_report,
)
from observations.pytorch_linearspec_block_size_shadow.shadow_generation import (
    clone_dynamic_cache,
    decompose_pair,
)
from observations.pytorch_linearspec_block_size_shadow.summarize import summarize_trace


BLOCKS = (4, 8, 16, 32)


def make_round(index: int, accepts: dict[int, int], *, valid: bool = True) -> dict:
    branches = {}
    for size in BLOCKS:
        accept = accepts[size]
        matched = accept - 1
        branches[str(size)] = {
            "accept_length": accept,
            "matched_draft_tokens": matched,
            "accept_rate": accept / size,
            "full_accept": accept == size,
            "zero_draft_match": matched == 0,
            "draft_forward_passes": 1,
            "accepted_confidence_mean": 0.8 - size / 1000,
            "accepted_confidence_min": 0.7,
            "accepted_confidence_last": 0.75,
            "rejected_confidence": None if accept == size else 0.4,
            "rejected_margin": None if accept == size else 0.1,
            "rejected_entropy": None if accept == size else 2.0,
            "position": {
                "selected_confidence": [0.8] * (size - 1),
                "top1_top2_margin": [0.5] * (size - 1),
                "entropy": [1.0] * (size - 1),
                "selected_is_top1": [True] * (size - 1),
                "draft_accepted": [position < matched for position in range(size - 1)],
            },
        }
    pairs = {}
    for small_index, small in enumerate(BLOCKS):
        for large in BLOCKS[small_index + 1 :]:
            payload = decompose_pair(accepts[small], accepts[large], small)
            payload.update(
                {
                    "small_size": small,
                    "large_size": large,
                    "small_ge_large": accepts[small] >= accepts[large],
                    "small_gt_large": accepts[small] > accepts[large],
                    "equal_accept_length": accepts[small] == accepts[large],
                    "draft_agreement": {
                        "agreement_rate": 0.75,
                        "all_common_equal": False,
                        "first_divergence_position": 2,
                    },
                    "verifier_agreement": {"agreement_rate": 0.8},
                }
            )
            pairs[f"{small}_{large}"] = payload
    return {
        "schema_version": 1,
        "event": "linearspec_block_size_shadow_round",
        "request_id": "request-0",
        "round_index": index,
        "block_sizes": list(BLOCKS),
        "analysis_valid": valid,
        "budget_boundary": not valid,
        "any_branch_eos": False,
        "history_before_round": {
            "prev_anchor_a": None if index == 0 else 5.0,
            "a_ma2": None if index == 0 else 5.5,
            "a_ma4": None if index == 0 else 5.25,
            "a_ma8": None if index == 0 else 5.0,
            "a_ewma05": None if index == 0 else 5.2,
            "prev_anchor_conf": None if index == 0 else 0.75,
            "conf_ma2": None if index == 0 else 0.72,
            "conf_ma4": None if index == 0 else 0.7,
            "conf_ma8": None if index == 0 else 0.68,
        },
        "branches": branches,
        "pairs": pairs,
    }


class CoreTests(unittest.TestCase):
    def test_pair_identity_for_all_valid_values(self) -> None:
        for small in BLOCKS:
            for a_small in range(1, small + 1):
                for large in BLOCKS:
                    if large <= small:
                        continue
                    for a_large in range(1, large + 1):
                        row = decompose_pair(a_small, a_large, small)
                        self.assertEqual(row["delta_a"], row["tail"] + row["decay"])

    def test_cache_clone_is_independent(self) -> None:
        keys = torch.arange(24, dtype=torch.float32).reshape(1, 2, 3, 4)
        values = keys + 100
        original = DynamicCache(ddp_cache_data=[(keys, values)])
        cloned = clone_dynamic_cache(original)
        cloned.layers[0].keys.add_(10)
        self.assertTrue(torch.equal(original.layers[0].keys, keys))
        self.assertFalse(torch.equal(original.layers[0].keys, cloned.layers[0].keys))


class SummaryTests(unittest.TestCase):
    def test_all_pairs_and_inclusive_survival_endpoint(self) -> None:
        records = [
            make_round(0, {4: 4, 8: 5, 16: 6, 32: 7}),
            make_round(1, {4: 3, 8: 4, 16: 5, 32: 6}),
            make_round(2, {4: 2, 8: 2, 16: 3, 32: 3}, valid=False),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            trace = Path(temporary) / "trace.jsonl"
            trace.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            summary = summarize_trace(
                trace,
                benchmark="gsm8k",
                benchmark_spec="gsm8k:1",
            )
        self.assertEqual(summary["status"], "ok")
        self.assertEqual(summary["rounds"]["raw"], 3)
        self.assertEqual(summary["rounds"]["analysis"], 2)
        self.assertEqual(set(summary["pairs"]), {"4_8", "4_16", "4_32", "8_16", "8_32", "16_32"})
        endpoint = next(
            row
            for row in summary["survival"]
            if row["pair"] == "4_8" and row["k"] == 5
        )
        self.assertEqual(endpoint["denominator_n2"], 1)
        self.assertEqual(endpoint["numerator_n12"], 0)
        self.assertEqual(endpoint["survival"], 0.0)
        self.assertTrue(endpoint["structural_endpoint"])

    def test_report_macro_excludes_aime24(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            (run_dir / "summaries").mkdir()
            (run_dir / "metrics").mkdir()
            settings = {
                "created_at": "test",
                "benchmarks": "gsm8k:1,aime24:1",
                "benchmark_specs": ["gsm8k:1", "aime24:1"],
                "block_sizes": list(BLOCKS),
            }
            atomic_json(run_dir / "Settings.json", settings)
            (run_dir / "benchmark_status.jsonl").write_text("", encoding="utf-8")
            for name, scale in (("gsm8k", 1), ("aime24", 9)):
                trace = run_dir / f"{name}.jsonl"
                trace.write_text(
                    json.dumps(make_round(0, {4: 4, 8: 5, 16: 6 * scale if scale == 1 else 16, 32: 7 * scale if scale == 1 else 32})) + "\n",
                    encoding="utf-8",
                )
                payload = summarize_trace(
                    trace,
                    benchmark=name,
                    benchmark_spec=f"{name}:1",
                )
                atomic_json(
                    run_dir / "summaries" / f"block_size_shadow_{name}.json",
                    payload,
                )
                atomic_json(
                    run_dir / "metrics" / f"metrics_{name}.json",
                    {name: {"pass@1": {"symbolic_correct": 50 * scale}}},
                )
            render_report(run_dir)
            report = (run_dir / "report.md").read_text(encoding="utf-8")
        self.assertIn("D=1", report)
        self.assertIn("AIME24", report)
        self.assertIn("aime24", report)


if __name__ == "__main__":
    unittest.main()
