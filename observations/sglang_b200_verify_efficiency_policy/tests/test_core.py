from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

OBSERVATIONS = Path(__file__).resolve().parents[2]
if str(OBSERVATIONS) not in sys.path:
    sys.path.insert(0, str(OBSERVATIONS))

from sglang_b200_verify_efficiency_policy.models import (
    FEATURES,
    SIGNALS,
    TraceData,
    fit_value_models,
    local_oracle_metrics,
    orientations,
    predict_value_models,
    read_trace,
    row_weights,
    sample_summary,
)
from sglang_b200_verify_efficiency_policy.policy_runtime import choose_action
from sglang_b200_verify_efficiency_policy.search import merge_validation_parts


class CoreTests(unittest.TestCase):
    def synthetic_data(self) -> TraceData:
        rows = 80
        request = np.arange(rows, dtype=np.uint32)
        features = {name: np.full(rows, np.nan, dtype=np.float32) for name in FEATURES}
        features["prev_accept"] = np.linspace(1, 16, rows, dtype=np.float32)
        features["prev_block"] = np.resize(np.asarray([8, 16, 32]), rows).astype(np.float32)
        features["prev_full"] = (np.arange(rows) % 2).astype(np.float32)
        features["history_rounds"] = np.ones(rows, dtype=np.float32)
        accept8 = np.clip(np.rint(features["prev_accept"]), 1, 8)
        accept16 = np.clip(np.rint(features["prev_accept"] * 1.25), 1, 16)
        accept32 = np.clip(np.rint(features["prev_accept"] * 1.5), 1, 32)
        return TraceData(
            dataset_names=["gsm8k"],
            dataset=np.zeros(rows, dtype=np.uint8),
            request=request,
            fold=(np.arange(rows) % 2).astype(np.uint8),
            round=np.ones(rows, dtype=np.uint32),
            decision=np.full(rows, 16, dtype=np.uint8),
            accept=np.column_stack([accept8, accept16, accept32]).astype(np.uint8),
            full=np.zeros((rows, 3), dtype=np.uint8),
            features=features,
            quality={},
            request_counts={"gsm8k": rows},
        )

    def test_signal_registry_is_verify_only(self) -> None:
        names = {item["feature"] for item in SIGNALS}
        self.assertFalse(any("draft" in name for name in names))
        self.assertFalse(any("head_conf" == name for name in names))
        self.assertIn("prev_verify_head_conf", names)

    def test_value_fit_and_full_c_summary(self) -> None:
        data = self.synthetic_data()
        spec = next(item for item in SIGNALS if item["name"] == "accept_last")
        signal = orientations(data, spec)
        models = fit_value_models(data, np.arange(data.size), signal, True, 4)
        values = predict_value_models(
            data, np.arange(data.size), signal, models, True
        )
        self.assertEqual(values.shape, (data.size, 3))
        self.assertTrue(np.isfinite(values).all())
        costs = {32: {8: 20.0, 16: 30.0, 32: 50.0}}
        actions = np.full(data.size, 8, dtype=np.uint8)
        result = sample_summary(data, actions, 32, costs)
        self.assertAlmostEqual(result["macro"]["mean_forward_ms"], 20.0)
        self.assertAlmostEqual(
            result["macro"]["pure_forward_token_ms_batch"],
            32 * result["macro"]["score_token_ms_req"],
        )
        oracle = local_oracle_metrics(data, actions, 32, costs)
        self.assertGreaterEqual(oracle["mean_local_regret"], 0.0)

    def test_runtime_ratio_and_fractional(self) -> None:
        model = {
            "edges": [],
            "survival": [[1.0] * 8],
            "missing_survival": [1.0] * 8,
        }
        models = {
            "8": model,
            "16": {**model, "survival": [[1.0] * 11], "missing_survival": [1.0] * 11},
            "32": {**model, "survival": [[1.0] * 12], "missing_survival": [1.0] * 12},
        }
        base = {
            "policy_type": "b200_verify_efficiency",
            "signal": {"feature": "prev_accept", "orientation": 1},
            "secondary_signal": None,
            "value_models": {"all": models},
            "concurrency_profiles": {
                "32": {
                    "cold_start_block": 16,
                    "costs_ms": {"8": 22.489, "16": 28.947, "32": 41.793},
                }
            },
        }
        ratio = {**base, "family": "local_ratio"}
        action, _ = choose_action(ratio, {"history_rounds": 1, "prev_accept": 4}, 32)
        self.assertEqual(action, 16)
        fractional = {
            **base,
            "family": "global_fractional",
            "concurrency_profiles": {
                "32": {**base["concurrency_profiles"]["32"], "rho": 0.5}
            },
        }
        action, _ = choose_action(
            fractional, {"history_rounds": 0, "prev_accept": None}, 32
        )
        self.assertEqual(action, 16)

    def test_reader_rejects_old_draft_only_trace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            branches = {
                str(block): {
                    "block_size": block,
                    "accept_length": 2,
                    "full": False,
                    "model_size": "8b",
                }
                for block in (8, 16, 32)
            }
            record = {
                "event": "sglang_dynamic_block_shadow_round",
                "decision_block": 16,
                "canonical_replay_match": True,
                "cross_block_common_prefix_match": True,
                "prompt_fingerprint": "x",
                "request_id": "x",
                "round_index": 0,
                "history_before_round": {},
                "branches": branches,
            }
            (root / "gsm8k.jsonl").write_text(
                json.dumps(record) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "usable|invalid rows"):
                read_trace(root, 1, 2, "8b", allow_partial=True)

    def test_reader_accepts_fresh_verifier_trace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            branches = {
                str(block): {
                    "block_size": block,
                    "accept_length": min(3, block),
                    "full": False,
                    "model_size": "14b",
                    "verifier_metrics_source": "causal_verify_logits",
                    "verifier_metrics_schema": 2,
                }
                for block in (8, 16, 32)
            }
            record = {
                "event": "sglang_dynamic_block_shadow_round",
                "decision_block": 16,
                "canonical_replay_match": True,
                "cross_block_common_prefix_match": True,
                "prompt_fingerprint": "fresh-request",
                "request_id": "runtime-id",
                "round_index": 1,
                "history_before_round": {
                    "history_rounds": 1,
                    "prev_accept": 3,
                    "prev_block": 16,
                    "prev_full": 0,
                    "prev_verify_head_conf": 0.75,
                },
                "branches": branches,
            }
            (root / "gsm8k.jsonl").write_text(
                json.dumps(record) + "\n", encoding="utf-8"
            )
            data = read_trace(root, 1, 2, "14b", allow_partial=True)
            self.assertEqual(data.size, 1)
            self.assertEqual(data.request_counts, {"gsm8k": 1})
            self.assertAlmostEqual(
                float(data.features["prev_verify_head_conf"][0]), 0.75
            )

    def test_nested_weights_keep_datasets_equal(self) -> None:
        data = self.synthetic_data()
        data.dataset_names = ["gsm8k", "human-eval"]
        data.dataset[:72] = 0
        data.dataset[72:] = 1
        weights = row_weights(data, np.arange(data.size))
        self.assertAlmostEqual(float(weights[data.dataset == 0].sum()), 0.5)
        self.assertAlmostEqual(float(weights[data.dataset == 1].sum()), 0.5)

    def test_full_data_entry_reaches_evaluator_without_max_samples(self) -> None:
        project = OBSERVATIONS.parent
        entry = (
            project
            / "observations/sglang_b200_verify_efficiency_policy/"
            "eval_b200_verify_efficiency.sh"
        )
        cost_document = (
            project / "configs/NLD_B200_B8_B32_forward_sweep_20260907_zh.md"
        )
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            fake_eval = temporary / "fake_eval.sh"
            fake_eval.write_text(
                "#!/bin/bash\n"
                "set -euo pipefail\n"
                "printf '%s\\n' '{\"event\":\"stub\"}' > \"$NLD_DYNAMIC_BLOCK_TRACE_FILE\"\n",
                encoding="utf-8",
            )
            environment = os.environ.copy()
            environment.update(
                {
                    "NLD_OBSERVATION_RESULTS_ROOT": str(temporary / "results"),
                    "NLD_B200_VERIFY_EVAL_SGLANG": str(fake_eval),
                    "PYTHONDONTWRITEBYTECODE": "1",
                }
            )
            completed = subprocess.run(
                [
                    "bash",
                    str(entry),
                    "--stage",
                    "collect",
                    "--model-size",
                    "8b",
                    "--benchmarks",
                    "gsm8k:1",
                    "--concurrencies",
                    "2",
                    "--allow-partial-datasets",
                    "--cost-document",
                    str(cost_document),
                    "--gpu-devices",
                    "0",
                ],
                cwd=project,
                env=environment,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stdout)
            self.assertIn("完成 gsm8k", completed.stdout)

    def test_collection_preserves_benchmark_argument_order(self) -> None:
        project = OBSERVATIONS.parent
        entry = (
            project
            / "observations/sglang_b200_verify_efficiency_policy/"
            "eval_b200_verify_efficiency.sh"
        )
        cost_document = (
            project / "configs/NLD_B200_B8_B32_forward_sweep_20260907_zh.md"
        )
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            fake_eval = temporary / "fake_eval.sh"
            fake_eval.write_text(
                "#!/bin/bash\n"
                "set -euo pipefail\n"
                "printf '%s\\n' '{\"event\":\"stub\"}' > \"$NLD_DYNAMIC_BLOCK_TRACE_FILE\"\n",
                encoding="utf-8",
            )
            environment = os.environ.copy()
            environment.update(
                {
                    "NLD_OBSERVATION_RESULTS_ROOT": str(temporary / "results"),
                    "NLD_B200_VERIFY_EVAL_SGLANG": str(fake_eval),
                    "PYTHONDONTWRITEBYTECODE": "1",
                }
            )
            completed = subprocess.run(
                [
                    "bash",
                    str(entry),
                    "--stage",
                    "collect",
                    "--model-size",
                    "8b",
                    "--benchmarks",
                    "human-eval:1,gsm8k:1",
                    "--concurrencies",
                    "2",
                    "--allow-partial-datasets",
                    "--cost-document",
                    str(cost_document),
                    "--gpu-devices",
                    "0",
                ],
                cwd=project,
                env=environment,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stdout)
            human_done = completed.stdout.index("完成 human-eval")
            gsm_done = completed.stdout.index("完成 gsm8k")
            self.assertLess(human_done, gsm_done, completed.stdout)

    def test_compact_validation_parts_merge_with_equal_dataset_weight(self) -> None:
        def summary_row(samples: int, score: float) -> dict[str, float | int]:
            return {
                "samples": samples,
                "rounds": samples * 3,
                "score_token_ms_req": score,
                "pure_forward_token_ms_batch": 2 * score,
                "decode_tpf": 2 * score,
                "mean_accept": 4 * score,
                "mean_forward_ms": 10.0,
                "l8_rate": 0.5,
                "l16_rate": 0.3,
                "l32_rate": 0.2,
                "baseline_score": 1.0,
            }

        with tempfile.TemporaryDirectory() as directory:
            parts = Path(directory)
            for dataset, samples, score in (
                ("gsm8k", 1000, 1.0),
                ("human-eval", 10, 3.0),
            ):
                dynamic = summary_row(samples, score)
                fixed = {
                    str(block): {
                        "datasets": {dataset: summary_row(samples, 1.0)},
                        "macro": {},
                    }
                    for block in (8, 16, 32)
                }
                payload = {
                    "protocol": {
                        "model_size": "8b",
                        "concurrency": 2,
                        "requests_by_dataset": {dataset: samples},
                        "rows": samples * 3,
                        "policy_family": "local_ratio",
                        "policy_replay_mismatches": 0,
                        "trace_quality": {
                            "rows_original": samples * 3,
                            "rows_usable": samples * 3,
                            "rows_excluded": 0,
                            "by_dataset": {dataset: {}},
                        },
                    },
                    "dynamic_policy": {
                        "datasets": {dataset: dynamic},
                        "macro": {
                            "oracle_agreement": 0.5,
                            "mean_local_regret": 0.1,
                            "p95_local_regret": 0.2,
                            "oracle_mean_efficiency": 2.0,
                        },
                    },
                    "fixed_baselines_same_dynamic_state": fixed,
                    "official_sglang_metrics": {"datasets": {}},
                }
                (parts / f"{dataset}.json").write_text(
                    json.dumps(payload), encoding="utf-8"
                )
            merged = merge_validation_parts(
                parts,
                {"family": "local_ratio"},
                "8b",
                2,
                allow_partial=True,
            )
            macro = merged["dynamic_policy"]["macro"]
            self.assertEqual(macro["datasets"], 2)
            self.assertEqual(macro["samples"], 1010)
            self.assertAlmostEqual(macro["score_token_ms_req"], 2.0)
            self.assertAlmostEqual(macro["relative_gain_vs_designated_fixed"], 1.0)

    def test_low_storage_validation_deletes_trace_and_resumes_from_compact(self) -> None:
        project = OBSERVATIONS.parent
        entry = (
            project
            / "observations/sglang_b200_verify_efficiency_policy/"
            "eval_b200_verify_efficiency.sh"
        )
        cost_document = (
            project / "configs/NLD_B200_B8_B32_forward_sweep_20260907_zh.md"
        )
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            calls = temporary / "eval_calls.txt"
            fake_eval = temporary / "fake_eval.sh"
            fake_eval.write_text(
                "#!/bin/bash\n"
                "set -euo pipefail\n"
                "printf 'call\\n' >> \"$FAKE_EVAL_CALLS\"\n"
                "printf '%s\\n' '{\"event\":\"stub\"}' > \"$NLD_DYNAMIC_BLOCK_TRACE_FILE\"\n",
                encoding="utf-8",
            )
            fake_search = temporary / "fake_search.py"
            fake_search.write_text(
                "import json, pathlib, sys\n"
                "def value(flag): return sys.argv[sys.argv.index(flag)+1]\n"
                "mode=value('--mode')\n"
                "out=pathlib.Path(value('--output-dir'))\n"
                "out.mkdir(parents=True,exist_ok=True)\n"
                "if mode=='validate':\n"
                " p=pathlib.Path(value('--validation-result')); p.parent.mkdir(parents=True,exist_ok=True); dataset=pathlib.Path(value('--trace-root')).stem; c=int(value('--concurrency')); row={'samples':1,'rounds':1,'score_token_ms_req':1.0,'pure_forward_token_ms_batch':1.0,'decode_tpf':1.0,'mean_accept':1.0,'mean_forward_ms':1.0,'l8_rate':1.0,'l16_rate':0.0,'l32_rate':0.0}; fixed={str(b):{'datasets':{dataset:row}} for b in (8,16,32)}; payload={'protocol':{'model_size':'8b','policy_family':'local_ratio','concurrency':c,'policy_replay_mismatches':0},'dynamic_policy':{'datasets':{dataset:row}},'fixed_baselines_same_dynamic_state':fixed}; p.write_text(json.dumps(payload))\n"
                " c=value('--concurrency'); (out/f'validation_local_ratio_c{c}.json').write_text(json.dumps({'merged':True}))\n"
                "elif mode=='merge-validation':\n"
                " c=value('--concurrency'); (out/f'validation_local_ratio_c{c}.json').write_text(json.dumps({'merged':True}))\n",
                encoding="utf-8",
            )
            environment = os.environ.copy()
            environment.update(
                {
                    "NLD_OBSERVATION_RESULTS_ROOT": str(temporary / "results"),
                    "NLD_B200_VERIFY_EVAL_SGLANG": str(fake_eval),
                    "NLD_B200_VERIFY_SEARCH": str(fake_search),
                    "FAKE_EVAL_CALLS": str(calls),
                    "PYTHONDONTWRITEBYTECODE": "1",
                }
            )
            common = [
                "--model-size",
                "8b",
                "--benchmarks",
                "gsm8k:1",
                "--concurrencies",
                "2",
                "--allow-partial-datasets",
                "--cost-document",
                str(cost_document),
                "--gpu-devices",
                "0",
                "--policy-family",
                "local_ratio",
            ]
            collected = subprocess.run(
                ["bash", str(entry), "--stage", "collect", *common],
                cwd=project,
                env=environment,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            self.assertEqual(collected.returncode, 0, collected.stdout)
            run_dir = next(
                (
                    temporary
                    / "results/sglang_b200_verify_efficiency_policy_results"
                ).iterdir()
            )
            legacy_settings_path = run_dir / "settings.json"
            legacy_settings = json.loads(legacy_settings_path.read_text(encoding="utf-8"))
            legacy_settings.pop("validation_trace_retention", None)
            legacy_settings_path.write_text(
                json.dumps(legacy_settings), encoding="utf-8"
            )
            (run_dir / "search/policy_local_ratio.json").write_text(
                json.dumps({"family": "local_ratio"}), encoding="utf-8"
            )
            validation = [
                "bash",
                str(entry),
                "--stage",
                "validate",
                "--run-dir",
                str(run_dir),
                "--validation-trace-retention",
                "delete-after-analysis",
                *common,
            ]
            completed = subprocess.run(
                validation,
                cwd=project,
                env=environment,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stdout)
            raw = run_dir / "traces/validate/local_ratio/c2/gsm8k.jsonl"
            compact = run_dir / "search/validation_parts/local_ratio/c2/gsm8k.json"
            self.assertFalse(raw.exists())
            self.assertTrue(compact.exists())
            first_calls = calls.read_text(encoding="utf-8").count("call")
            resumed = subprocess.run(
                validation,
                cwd=project,
                env=environment,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            self.assertEqual(resumed.returncode, 0, resumed.stdout)
            self.assertEqual(
                calls.read_text(encoding="utf-8").count("call"), first_calls
            )
            settings = json.loads(
                (run_dir / "settings.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                settings["validation_trace_retention"], "delete-after-analysis"
            )


if __name__ == "__main__":
    unittest.main()
