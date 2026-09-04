from __future__ import annotations

import json
import fcntl
import io
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

from observations.sglang_dynamic_block_history_signal.reporting import init_run
from observations.sglang_dynamic_block_history_signal.search import (
    ProgressBar,
    actions_for,
    equal_weights,
    filter_trace_rows,
    summarize_actions,
    threshold_grid_macros,
)


class OfflineProtocolTest(unittest.TestCase):
    def test_eval_wrapper_can_make_compact_benchmark_error_fatal_and_keep_runtime(self) -> None:
        project = Path(__file__).resolve().parents[3]
        evaluator = project / "observations/eval_sglang.sh"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fake_pipeline = root / "fake_pipeline.sh"
            fake_pipeline.write_text(
                "#!/bin/bash\n"
                "set -eu\n"
                "mkdir -p \"$SGLANG_FINAL_OUTPUT_DIR\" \"$SGLANG_RUN_DIR\"\n"
                "printf '%s\\n' '{\"status\":\"failed\"}' >\"$SGLANG_FINAL_OUTPUT_DIR/error_gsm8k.json\"\n"
                "printf '%s\\n' 'preserved' >\"$SGLANG_RUN_DIR/server.log\"\n",
                encoding="utf-8",
            )
            fake_pipeline.chmod(0o755)
            output = root / "output"
            env = os.environ.copy()
            env.update(
                {
                    "SGLANG_PIPELINE_OVERRIDE": str(fake_pipeline),
                    "NLD_FAIL_ON_BENCHMARK_ERROR": "1",
                    "NLD_KEEP_FAILED_WORK_DIR": "1",
                }
            )
            completed = subprocess.run(
                [
                    "bash", str(evaluator), "--mode", "linearspec_lora",
                    "--benchmarks", "gsm8k:1", "--tokens", "32",
                    "--context-length", "128", "--output-path", str(output),
                    "--sglang-python", sys.executable,
                    "--eval-python", sys.executable,
                ],
                cwd=project,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 86, completed.stdout + completed.stderr)
            work_dirs = list(output.glob(".eval_*_work_*"))
            self.assertEqual(len(work_dirs), 1)
            self.assertTrue((work_dirs[0] / "sglang_runtime/server.log").is_file())

    def test_dataset_retry_discards_partial_trace_and_completes(self) -> None:
        project = Path(__file__).resolve().parents[3]
        runner = project / "observations/sglang_dynamic_block_history_signal/eval_dynamic_block_history.sh"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fake = root / "fake_eval.sh"
            fake.write_text(
                "#!/bin/bash\n"
                "set -eu\n"
                "counter=${NLD_FAKE_COUNTER:?}\n"
                "selected=${NLD_FAKE_SELECTED_GPU:?}\n"
                "previous=\n"
                "for arg in \"$@\"; do\n"
                "  if [[ $previous == --gpu-devices ]]; then printf '%s\\n' \"$arg\" >\"$selected\"; fi\n"
                "  previous=$arg\n"
                "done\n"
                "n=0\n"
                "[[ ! -f \"$counter\" ]] || n=$(<\"$counter\")\n"
                "n=$((n+1))\n"
                "printf '%s\\n' \"$n\" >\"$counter\"\n"
                "if [[ $n -eq 1 ]]; then\n"
                "  printf '%s\\n' '{\"partial\":true}' >\"$NLD_DYNAMIC_BLOCK_TRACE_FILE\"\n"
                "  exit 86\n"
                "fi\n"
                "printf '%s\\n' '{\"event\":\"sglang_dynamic_block_shadow_round\"}' >\"$NLD_DYNAMIC_BLOCK_TRACE_FILE\"\n",
                encoding="utf-8",
            )
            fake.chmod(0o755)
            counter = root / "attempts.txt"
            selected_gpu = root / "selected_gpu.txt"
            env = os.environ.copy()
            env.update(
                {
                    "NLD_DYNAMIC_EVAL_SGLANG": str(fake),
                    "NLD_OBSERVATION_RESULTS_ROOT": str(root / "results"),
                    "NLD_FAKE_COUNTER": str(counter),
                    "NLD_FAKE_SELECTED_GPU": str(selected_gpu),
                    "NLD_DYNAMIC_GPU_INVENTORY": "7, 60000, 10\n8, 70000, 90",
                }
            )
            completed = subprocess.run(
                [
                    "bash", str(runner), "--stage", "collect",
                    "--benchmarks", "gsm8k:1", "--max-samples", "1",
                    "--dataset-max-attempts", "2",
                    "--dataset-retry-delay-s", "0",
                    "--gpu-devices", "auto",
                    "--auto-gpu-min-free-gb", "48",
                    "--sglang-python", sys.executable,
                    "--eval-python", sys.executable,
                ],
                cwd=project,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            self.assertEqual(counter.read_text(encoding="utf-8").strip(), "2")
            self.assertEqual(selected_gpu.read_text(encoding="utf-8").strip(), "7")
            run_dir = next((root / "results/sglang_dynamic_block_history_signal_results").iterdir())
            trace = (run_dir / "traces/explore/gsm8k.jsonl").read_text(encoding="utf-8")
            self.assertNotIn("partial", trace)
            failed_attempts = list(
                (run_dir / "runtime/attempt_traces/explore/gsm8k").glob("*.jsonl")
            )
            self.assertEqual(len(failed_attempts), 1)
            self.assertIn("partial", failed_attempts[0].read_text(encoding="utf-8"))
            state = json.loads((run_dir / "run_state.json").read_text(encoding="utf-8"))
            statuses = [event["status"] for event in state["events"]]
            self.assertIn("retrying", statuses)
            self.assertEqual(statuses[-1], "completed")

    def test_existing_run_rejects_a_second_writer(self) -> None:
        project = Path(__file__).resolve().parents[3]
        runner = project / "observations/sglang_dynamic_block_history_signal/eval_dynamic_block_history.sh"
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            (run_dir / "runtime").mkdir(parents=True)
            lock_path = run_dir / "runtime/runner.lock"
            with lock_path.open("a+", encoding="utf-8") as lock_file:
                fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
                completed = subprocess.run(
                    [
                        "bash", str(runner), "--stage", "search",
                        "--run-dir", str(run_dir), "--benchmarks", "gsm8k:1",
                        "--allow-partial-search", "--sglang-python", sys.executable,
                        "--eval-python", sys.executable,
                    ],
                    cwd=project,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertEqual(completed.returncode, 89)
                self.assertIn("already owns this run directory", completed.stderr)

    def test_non_tty_progress_is_durable_and_not_duplicated(self) -> None:
        stream = io.StringIO()
        with redirect_stdout(stream):
            progress = ProgressBar("smoke", 10, interactive=False)
            progress.update(5)
            progress.update(5)
            progress.close()
        output = stream.getvalue()
        self.assertIn("5/10", output)
        self.assertEqual(output.count("10/10"), 1)

    def test_hierarchical_weights_make_datasets_equal(self) -> None:
        rows = []
        for dataset, requests, rounds in (("small", 1, 2), ("large", 3, 5)):
            for request in range(requests):
                for round_index in range(rounds):
                    rows.append({"dataset": dataset, "request": str(request), "round": round_index})
        weights = equal_weights(rows)
        for dataset in ("small", "large"):
            total = sum(weight for row, weight in zip(rows, weights) if row["dataset"] == dataset)
            self.assertAlmostEqual(total, 0.5)

    def test_s8_uncertainty_defaults_small_and_first_round_is_l16(self) -> None:
        rows = [{"round": 0}, {"round": 1}, {"round": 2}]
        scores = {"worth16": np.asarray([0.0, 0.2, 0.9]), "worth32": np.asarray([0.0, 0.3, 0.95])}
        actions = actions_for("s8", rows, scores, {"worth16": 0.8, "worth32": 0.9})
        self.assertEqual(actions.tolist(), [16, 8, 32])

    def test_macro_is_dataset_mean_not_pooled_round_mean(self) -> None:
        # Build an intentional 1-vs-9 dataset imbalance.
        rows = [{"dataset": "a", "request": "a", "round": 0, "a8": 8, "a16": 8, "a32": 8}] + [
            {"dataset": "b", "request": "b", "round": i, "a8": 1, "a16": 1, "a32": 1}
            for i in range(9)
        ]
        summary = summarize_actions(rows, np.asarray([8] * len(rows)), "s8", 3, 5)
        self.assertAlmostEqual(summary["macro"]["mean_accept"], 4.5)

    def test_no_promotion_dataset_does_not_dilute_conditional_precision(self) -> None:
        rows = [
            {"dataset": "quiet", "request": "q", "round": 1, "a8": 4, "a16": 4, "a32": 4},
            {"dataset": "promoted", "request": "p", "round": 1, "a8": 4, "a16": 4, "a32": 5},
        ]
        summary = summarize_actions(rows, np.asarray([8, 32]), "s8", 3, 5)
        self.assertAlmostEqual(summary["macro"]["large_precision"], 0.0)
        self.assertAlmostEqual(summary["macro"]["large_waste_rate"], 1.0)

    def test_conditional_constraints_are_dataset_macro_not_event_pooled(self) -> None:
        # Dataset A has ten good promotions; dataset B has one bad promotion.
        # Strict dataset equality therefore gives (1 + 0) / 2, not 10 / 11.
        rows = [
            {
                "dataset": "a", "request": "a", "round": index,
                "a8": 2, "a16": 4, "a32": 10,
            }
            for index in range(10)
        ] + [
            {
                "dataset": "b", "request": "b", "round": index,
                "a8": 2, "a16": 2, "a32": 3,
            }
            for index in range(10)
        ]
        actions = np.asarray([32] * 10 + [32] + [8] * 9)
        summary = summarize_actions(rows, actions, "s8", 3, 5)
        self.assertAlmostEqual(summary["macro"]["large_precision"], 0.5)
        self.assertAlmostEqual(summary["macro"]["large_waste_rate"], 0.5)

    def test_binned_threshold_grid_matches_full_summary(self) -> None:
        rows = [
            {
                "dataset": "a" if index < 3 else "b",
                "request": f"r{index // 2}",
                "round": index % 3,
                "a8": 2 + index,
                "a16": 4 + 2 * index,
                "a32": 7 + 3 * index,
            }
            for index in range(6)
        ]
        scores = {
            "worth16": np.asarray([0.99, 0.4, 0.8, 0.7, 0.95, 0.2]),
            "worth32": np.asarray([0.1, 0.9, 0.6, 0.98, 0.3, 0.8]),
        }
        spec = {"gain16": 3.0, "eff16": 0.25, "gain32": 7.0, "eff32": 0.25}
        thresholds = (0.5, 0.8)
        for selected_thresholds, fast in threshold_grid_macros(
            "s8", rows, scores, thresholds, spec
        ):
            actions = actions_for("s8", rows, scores, selected_thresholds)
            exact = summarize_actions(
                rows,
                actions,
                "s8",
                spec["gain16"],
                spec["gain32"],
                spec["eff16"],
                spec["eff32"],
            )["macro"]
            for name in (
                "mean_block", "mean_accept", "loss_vs_l32", "loss_vs_default",
                "large_rate", "large_precision", "large_waste_rate",
                "l8_rate", "l16_rate", "l32_rate",
            ):
                self.assertAlmostEqual(fast[name], exact[name])

    def test_trace_filter_audits_union_and_reweights_later(self) -> None:
        rows = [
            {"dataset": "a", "request": "x", "replay_match": True, "cross_block_match": True},
            {"dataset": "a", "request": "x", "replay_match": False, "cross_block_match": True},
            {"dataset": "b", "request": "y", "replay_match": False, "cross_block_match": False},
        ]
        usable, quality = filter_trace_rows(rows, "exclude", 0.8)
        self.assertEqual(len(usable), 1)
        self.assertEqual(quality["rows_excluded"], 2)
        self.assertEqual(quality["replay_mismatch"], 2)
        self.assertEqual(quality["cross_block_mismatch"], 1)
        with self.assertRaises(RuntimeError):
            filter_trace_rows(rows, "exclude", 0.5)

    def test_report_template_exists_at_initialization(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            settings = {
                "stage": "collect", "benchmarks": "gsm8k:1", "model": "m",
                "mode": "linearspec_lora", "block_sizes": "8,16,32",
                "physical_block_size": "32", "gpu_devices": "0", "tp_size": "1",
                "batch_size": "1", "client_concurrency": "1", "gpu_memory_reserve_gb": "0",
                "tokens": "32", "context_length": "128", "dtype": "bfloat16",
                "mem_fraction": "0.5", "max_samples": "1", "mmlu_max_samples": "1",
                "seed": "1", "split_seed": "1", "command": "smoke",
            }
            init_run(run_dir, settings)
            self.assertTrue((run_dir / "settings.md").is_file())
            self.assertIn("等待", (run_dir / "report.md").read_text(encoding="utf-8"))
            json.loads((run_dir / "settings.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
