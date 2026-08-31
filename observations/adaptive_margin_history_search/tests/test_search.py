#!/usr/bin/env python3
from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import search  # noqa: E402
from reporting import atomic_json, initialize_run  # noqa: E402


def record(
    request_id: str,
    round_index: int,
    q: int,
    margins: list[float],
    dataset: str = "toy",
) -> dict:
    positions = len(margins)
    drops = [None] + [0.2 if index == q - 1 else 0.0 for index in range(1, positions)]
    if q == 1:
        drops[0] = 0.3
    return {
        "event": "linearspec_failure_locator_round",
        "benchmark": dataset,
        "request_id": request_id,
        "round_index": round_index,
        "block_size": positions + 1,
        "mismatch_position": q or None,
        "analysis_valid": True,
        "eos_hit": False,
        "budget_boundary": False,
        "position": {"top1_top2_margin": margins, "prefix_drop_pct": drops},
    }


class SearchTests(unittest.TestCase):
    def test_progress_bar_is_sparse_for_logs_and_can_be_disabled(self) -> None:
        stream = io.StringIO()
        progress = search.ProgressBar("candidate search", 100, unit="candidate", stream=stream)
        for index in range(1, 101):
            progress.update(index, detail="family")
        progress.finish()
        lines = stream.getvalue().splitlines()
        self.assertGreaterEqual(len(lines), 45)
        self.assertLessEqual(len(lines), 55)
        self.assertIn("100.00%", lines[-1])
        self.assertIn("ETA", lines[-1])

        disabled_stream = io.StringIO()
        disabled = search.ProgressBar("disabled", 10, enabled=False, stream=disabled_stream)
        disabled.finish()
        self.assertEqual(disabled_stream.getvalue(), "")

    def test_dataset_macro_is_equal_weight_not_round_weight(self) -> None:
        by_dataset = {
            "small": {"rounds": 1, **{name: 1.0 for name in search.MACRO_FIELDS}},
            "large": {"rounds": 100000, **{name: 0.0 for name in search.MACRO_FIELDS}},
        }
        macro = search.macro_metrics(by_dataset)
        self.assertEqual(macro["datasets"], 2)
        self.assertAlmostEqual(macro["recall"], 0.5)
        self.assertAlmostEqual(macro["correct_fp_round"], 0.5)

    def test_metrics_count_correct_position_false_reports(self) -> None:
        # p=2<q=3 is an early correct-position report; p=1 on q=0 is a full-accept report.
        p = np.asarray([2, 1, 3, 0], dtype=np.int16)
        q = np.asarray([3, 0, 3, 2], dtype=np.int16)
        metric = search.metrics_from_counts(search.count_metrics(p, q, positions=3))
        self.assertEqual(metric["exact_hits"], 1)
        self.assertEqual(metric["correct_false_reports"], 2)
        self.assertAlmostEqual(metric["correct_fp_round"], 0.5)
        self.assertAlmostEqual(metric["correct_fp_report"], 2 / 3)
        self.assertAlmostEqual(metric["position_fpr"], 2 / 8)

    def test_dynamic_missing_history_falls_back_to_half(self) -> None:
        arrays = {
            "features": {(1, "mean", "good_mean"): np.asarray([np.nan, 0.2], dtype=np.float32)},
        }
        item = search.candidate("good_center", 1, "mean", "good_mean", alpha=1.0, offset=0.0)
        threshold, ready = search.dynamic_threshold(
            item,
            arrays,
            {"h1|mean|good_mean": 0.1},
            np.asarray([0, 1]),
        )
        self.assertFalse(bool(ready[0]))
        self.assertEqual(float(threshold[0]), 0.5)
        self.assertTrue(bool(ready[1]))
        self.assertAlmostEqual(float(threshold[1]), 0.6, places=6)

    def test_end_to_end_small_trace(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source" / "traces"
            source.mkdir(parents=True)
            rows = []
            # Ten requests guarantee nonempty request-level search/selection/test.
            for request_index in range(10):
                request_id = f"req-{request_index}"
                rows.append(record(request_id, 0, 2, [0.98, 0.30, 0.95]))
                rows.append(record(request_id, 1, 0, [0.97, 0.92, 0.88]))
            (source / "failure_locator_toy.jsonl").write_text(
                "".join(json.dumps(item) + "\n" for item in rows), encoding="utf-8"
            )
            run_dir = root / "result"
            initialize_run(
                run_dir,
                {
                    "created_at": "test",
                    "trace_mode": "offline",
                    "source_run_dir": str(source.parent),
                    "benchmarks": "toy:1",
                    "block_size": 4,
                    "history_windows": "1",
                    "aggregations": "mean",
                    "grid": "compact",
                    "selection_protocol": "split",
                    "search_ratio": 0.6,
                    "selection_ratio": 0.2,
                    "test_ratio": 0.2,
                },
            )
            args = Namespace(
                run_dir=run_dir,
                trace_dir=source,
                output_json=run_dir / "analysis" / "strategy_search.json",
                benchmarks="toy:1",
                block_size=4,
                history_windows="1",
                aggregations="mean",
                grid="compact",
                selection_protocol="split",
                split_seed=7,
                search_ratio=0.6,
                selection_ratio=0.2,
                shortlist=20,
                report_top=5,
                search_max_rounds_per_dataset=100,
                bootstrap_replicates=5,
                include_boundary_rounds=False,
                no_progress=True,
            )
            payload = search.run_search(args)
            self.assertEqual(payload["analysis_rounds"], 20)
            self.assertFalse(payload["selection_contract"]["test_used_for_selection"])
            self.assertTrue((run_dir / "report.md").is_file())
            self.assertTrue((run_dir / "Settings.json").is_file())
            machine = json.loads((run_dir / "analysis" / "strategy_search.json").read_text())
            self.assertIn("winner", machine)
            self.assertEqual(machine["block_size"], 4)

    def test_full_data_uses_every_request_and_round_for_every_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trace_dir = root / "source" / "traces"
            trace_dir.mkdir(parents=True)
            for dataset, request_count in (("small", 4), ("large", 10)):
                rows = []
                for request_index in range(request_count):
                    request_id = f"{dataset}-req-{request_index}"
                    rows.append(record(request_id, 0, 2, [0.98, 0.30, 0.95], dataset))
                    rows.append(record(request_id, 1, 0, [0.97, 0.92, 0.88], dataset))
                if dataset == "small":
                    boundary = record("small-zero-valid", 0, 0, [0.97, 0.92, 0.88], dataset)
                    boundary["analysis_valid"] = False
                    boundary["eos_hit"] = True
                    rows.append(boundary)
                (trace_dir / f"failure_locator_{dataset}.jsonl").write_text(
                    "".join(json.dumps(item) + "\n" for item in rows), encoding="utf-8"
                )

            run_dir = root / "result"
            initialize_run(
                run_dir,
                {
                    "created_at": "test",
                    "trace_mode": "offline",
                    "source_run_dir": str(trace_dir.parent),
                    "benchmarks": "small:1,large:1",
                    "block_size": 4,
                    "history_windows": "1",
                    "aggregations": "mean",
                    "grid": "compact",
                    "selection_protocol": "full_data",
                    "search_ratio": 0.6,
                    "selection_ratio": 0.2,
                    "test_ratio": 0.2,
                },
            )
            args = Namespace(
                run_dir=run_dir,
                trace_dir=trace_dir,
                output_json=run_dir / "analysis" / "strategy_search.json",
                benchmarks="small:1,large:1",
                block_size=4,
                history_windows="1",
                aggregations="mean",
                grid="compact",
                selection_protocol="full_data",
                split_seed=11,
                search_ratio=0.6,
                selection_ratio=0.2,
                shortlist=20,
                report_top=5,
                search_max_rounds_per_dataset=0,
                bootstrap_replicates=5,
                include_boundary_rounds=False,
                no_progress=True,
            )
            payload = search.run_search(args)
            self.assertEqual(payload["full_data_rounds"], 28)
            self.assertEqual(payload["full_data_requests"], 15)
            self.assertEqual(payload["full_data_evaluable_requests"], 14)
            self.assertEqual(payload["full_data_zero_valid_requests"], 1)
            self.assertEqual(payload["split_round_counts"], {"full_data": 28})
            self.assertTrue(payload["full_data_contract"]["all_candidates_use_all_included_valid_rounds"])
            self.assertTrue(payload["full_data_contract"]["all_source_trace_requests_scanned"])
            self.assertIsNone(payload["search_rounds_after_cap"])
            self.assertEqual(len(payload["all_full_data_candidates"]), payload["candidate_count"])
            self.assertEqual(payload["winner"]["full_data"]["macro"]["datasets"], 2)
            self.assertNotIn("test", payload["winner"])
            for summary in payload["dataset_summaries"].values():
                self.assertEqual(summary["raw_requests"], summary["request_split_counts"]["full_data"])
                self.assertEqual(
                    summary["valid_requests"] + summary["zero_valid_requests"],
                    summary["raw_requests"],
                )
            report = (run_dir / "report.md").read_text(encoding="utf-8")
            self.assertIn("全数据全局最优结果", report)
            self.assertIn("ZeroReq", report)
            self.assertNotIn("最终 test：", report)


if __name__ == "__main__":
    unittest.main()
