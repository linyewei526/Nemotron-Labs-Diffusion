#!/usr/bin/env python3
"""Small end-to-end tests for the dedicated MT-Bench runner."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import threading
import types
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_mt_bench as mt_bench


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ChatHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
        length = int(self.headers.get("Content-Length", "0"))
        request = json.loads(self.rfile.read(length))
        with self.server.requests_lock:
            self.server.requests.append(request)

        if request["model"] == "judge":
            content = "The response is sound. [[8]]"
            usage = {"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25}
        else:
            turn = 1 if len(request["messages"]) == 2 else 2
            content = f"candidate-turn-{turn}"
            usage = {
                "prompt_tokens": 10 * turn,
                "completion_tokens": 4,
                "total_tokens": 10 * turn + 4,
                "nfe": 6 * turn,
            }

        payload = {
            "model": f"{request['model']}-resolved",
            "choices": [{"message": {"role": "assistant", "content": content}}],
            "usage": usage,
        }
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format: str, *args: object) -> None:
        return


class MTBenchEndToEndTest(unittest.TestCase):
    def test_two_turn_generation_and_single_answer_grading(self) -> None:
        server = ThreadingHTTPServer(("127.0.0.1", 0), ChatHandler)
        server.requests = []
        server.requests_lock = threading.Lock()
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                source_dir = root / "source"
                data_dir = root / "data"
                output_dir = root / "output"
                source_dir.mkdir()

                question_path = source_dir / "question.jsonl"
                reference_path = source_dir / "reference_answer_gpt-4.jsonl"
                prompts_path = source_dir / "judge_prompts.jsonl"
                write_jsonl(
                    question_path,
                    [
                        {
                            "question_id": question_id,
                            "category": "writing",
                            "turns": [f"question-{question_id}", "follow-up"],
                        }
                        for question_id in range(80)
                    ],
                )
                write_jsonl(reference_path, [])
                write_jsonl(
                    prompts_path,
                    [
                        {
                            "name": "single-v1",
                            "system_prompt": "Judge one turn.",
                            "prompt_template": "{question}\n{answer}",
                        },
                        {
                            "name": "single-math-v1",
                            "system_prompt": "Judge one referenced turn.",
                            "prompt_template": "{question}\n{ref_answer_1}\n{answer}",
                        },
                        {
                            "name": "single-v1-multi-turn",
                            "system_prompt": "Judge two turns.",
                            "prompt_template": (
                                "{question_1}\n{question_2}\n{answer_1}\n{answer_2}"
                            ),
                        },
                        {
                            "name": "single-math-v1-multi-turn",
                            "system_prompt": "Judge two referenced turns.",
                            "prompt_template": (
                                "{question_1}\n{question_2}\n{ref_answer_1}\n"
                                "{ref_answer_2}\n{answer_1}\n{answer_2}"
                            ),
                        },
                    ],
                )
                assets = {
                    path.name: {"url": path.as_uri(), "sha256": digest(path)}
                    for path in (question_path, reference_path, prompts_path)
                }
                endpoint = f"http://127.0.0.1:{server.server_port}/v1"
                argv = [
                    "eval_mt_bench.py",
                    "--candidate-server-address",
                    endpoint,
                    "--candidate-model",
                    "candidate",
                    "--candidate-concurrency",
                    "2",
                    "--candidate-generation-algorithm",
                    "ar_native",
                    "--candidate-steps",
                    "32",
                    "--candidate-block-length",
                    "1",
                    "--judge-server-address",
                    endpoint,
                    "--judge-model",
                    "judge",
                    "--judge-concurrency",
                    "2",
                    "--data-dir",
                    str(data_dir),
                    "--output-dir",
                    str(output_dir),
                    "--max-samples",
                    "2",
                ]
                proxy_env = {
                    key: ""
                    for key in (
                        "ALL_PROXY",
                        "HTTPS_PROXY",
                        "HTTP_PROXY",
                        "all_proxy",
                        "https_proxy",
                        "http_proxy",
                    )
                }
                with mock.patch.object(mt_bench, "ASSETS", assets), mock.patch.object(
                    sys, "argv", argv
                ), mock.patch.dict(os.environ, proxy_env):
                    self.assertEqual(mt_bench.main(), 0)

                metrics = json.loads((output_dir / "metrics.json").read_text())
                self.assertEqual(metrics["mt_bench_score"], 8.0)
                self.assertEqual(metrics["mt_bench"]["valid_judgments"], 4)
                self.assertEqual(metrics["generation_usage"]["average_nfe"], 9.0)
                self.assertEqual(metrics["protocol"]["protocol_version"], 2)
                self.assertEqual(
                    metrics["protocol"]["generation"]["system_prompt"],
                    "You are a helpful assistant.",
                )
                self.assertEqual(
                    metrics["protocol"]["generation"]["prompt_preflight"]["status"],
                    "not_requested",
                )
                self.assertRegex(
                    metrics["protocol"]["generation_fingerprint"], r"^[0-9a-f]{64}$"
                )
                self.assertRegex(
                    metrics["protocol"]["judge_fingerprint"], r"^[0-9a-f]{64}$"
                )
                self.assertEqual(
                    metrics["protocol"]["observed_candidate_response_models"],
                    ["candidate-resolved"],
                )
                self.assertEqual(
                    metrics["protocol"]["observed_judge_response_models"],
                    ["judge-resolved"],
                )

                answer_rows = mt_bench.read_jsonl(output_dir / "model_answers.jsonl")
                judgment_rows = mt_bench.read_jsonl(
                    output_dir / "model_judgments.jsonl"
                )
                self.assertEqual(answer_rows[0]["turn_seeds"], [0, 1])
                self.assertEqual(answer_rows[1]["turn_seeds"], [2, 3])
                self.assertTrue(
                    all(row.get("generation_protocol_fingerprint") for row in answer_rows)
                )
                self.assertTrue(
                    all(row.get("judge_protocol_fingerprint") for row in judgment_rows)
                )
                self.assertTrue(
                    all(row.get("judgment_input_sha256") for row in judgment_rows)
                )

                candidate_requests = [
                    request for request in server.requests if request["model"] == "candidate"
                ]
                judge_requests = [
                    request for request in server.requests if request["model"] == "judge"
                ]
                self.assertEqual(len(candidate_requests), 4)
                self.assertEqual(len(judge_requests), 4)
                second_turns = [
                    request for request in candidate_requests if len(request["messages"]) == 4
                ]
                self.assertEqual(len(second_turns), 2)
                for request in second_turns:
                    self.assertEqual(request["messages"][2]["role"], "assistant")
                    self.assertEqual(request["messages"][2]["content"], "candidate-turn-1")
                    self.assertEqual(request["generation_algorithm"], "ar_native")
                    self.assertEqual(request["block_length"], 1)
                    self.assertEqual(
                        request["chat_template_kwargs"],
                        {
                            "enable_thinking": False,
                            "truncate_history_thinking": True,
                        },
                    )
                    self.assertIsInstance(request["seed"], int)

                request_count = len(server.requests)
                resume_argv = argv + ["--resume"]
                with mock.patch.object(mt_bench, "ASSETS", assets), mock.patch.object(
                    sys, "argv", resume_argv
                ), mock.patch.dict(os.environ, proxy_env):
                    self.assertEqual(mt_bench.main(), 0)
                self.assertEqual(len(server.requests), request_count)

                incompatible_argv = resume_argv + [
                    "--system-prompt",
                    "A changed system prompt.",
                ]
                with mock.patch.object(mt_bench, "ASSETS", assets), mock.patch.object(
                    sys, "argv", incompatible_argv
                ), mock.patch.dict(os.environ, proxy_env):
                    with self.assertRaisesRegex(RuntimeError, "Cannot resume"):
                        mt_bench.main()
                self.assertEqual(len(server.requests), request_count)

                xp_dir = Path(__file__).resolve().parents[1]
                decode_path = root / "sglang_decode.jsonl"
                timing_path = root / "sglang_timing.jsonl"
                write_jsonl(
                    decode_path,
                    [{"tokens": 16, "forward_passes": 8, "block_gen_positions": 16}],
                )
                write_jsonl(
                    timing_path,
                    [
                        {
                            "benchmark": "mt-bench",
                            "ok": True,
                            "prompt_tokens": 10,
                            "completion_tokens": 4,
                            "latency_s": 1.0,
                            "start_time_s": 1.0,
                            "end_time_s": 2.0,
                        }
                    ],
                )
                subprocess.run(
                    [
                        sys.executable,
                        str(xp_dir / "sglang_eval/add_sglang_metrics_to_metrics.py"),
                        "--metrics-json",
                        str(output_dir / "metrics.json"),
                        "--eval-results-dir",
                        str(output_dir),
                        "--decode-stats-file",
                        str(decode_path),
                        "--timing-log",
                        str(timing_path),
                        "--benchmark",
                        "mt-bench",
                        "--wall-time-s",
                        "2",
                    ],
                    check=True,
                )

                pytorch_stats_path = root / "pytorch_stats.jsonl"
                write_jsonl(
                    pytorch_stats_path,
                    [
                        {
                            "ok": True,
                            "mode": "ar",
                            "prompt_tokens": 10,
                            "completion_tokens": 4,
                            "raw_generated_tokens": 4,
                            "nfe": 4,
                            "model_time_s": 1.0,
                            "request_time_s": 1.1,
                            "queue_wait_s": 0.1,
                            "finish_reason": "stop",
                            "top_p_requested": 1.0,
                            "top_p_applied": False,
                            "top_k_requested": -1,
                            "top_k_applied": False,
                        }
                    ],
                )
                subprocess.run(
                    [
                        sys.executable,
                        str(
                            xp_dir
                            / "pytorch_nemo_eval/add_pytorch_metrics_to_metrics.py"
                        ),
                        "--metrics-json",
                        str(output_dir / "metrics.json"),
                        "--request-stats-file",
                        str(pytorch_stats_path),
                        "--benchmark",
                        "mt-bench",
                        "--wall-time-s",
                        "2",
                    ],
                    check=True,
                )
                merged_metrics = json.loads((output_dir / "metrics.json").read_text())
                self.assertIn("sglang", merged_metrics)
                self.assertIn("pytorch_native", merged_metrics)
                native_decode = merged_metrics["pytorch_native"]["decode"]
                self.assertEqual(native_decode["top_p_requested"], [1.0])
                self.assertFalse(native_decode["top_p_applied"])
                self.assertEqual(native_decode["top_k_requested"], [-1])
                self.assertFalse(native_decode["top_k_applied"])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_prompt_preflight_hashes_template_and_tokens(self) -> None:
        class FakeTokenizer:
            chat_template = "fake-template-v1"

            def apply_chat_template(self, messages, **kwargs):
                return json.dumps(
                    {"messages": messages, "kwargs": kwargs}, sort_keys=True
                )

            def encode(self, rendered):
                return list(rendered.encode("utf-8"))

        class FakeAutoTokenizer:
            @staticmethod
            def from_pretrained(name, **kwargs):
                self.assertEqual(name, "fake-tokenizer")
                self.assertTrue(kwargs["trust_remote_code"])
                return FakeTokenizer()

        fake_transformers = types.ModuleType("transformers")
        fake_transformers.__version__ = "test-version"
        fake_transformers.AutoTokenizer = FakeAutoTokenizer
        args = argparse_namespace(
            candidate_tokenizer="fake-tokenizer",
            candidate_tokenizer_revision="",
            expected_chat_template_sha256=mt_bench.text_sha256("fake-template-v1"),
            candidate_enable_thinking=False,
            candidate_truncate_history_thinking=True,
            system_prompt="system",
        )
        question = {"turns": ["first", "second"]}
        with mock.patch.dict(sys.modules, {"transformers": fake_transformers}):
            result = mt_bench.prompt_preflight(args, question)
        self.assertEqual(result["status"], "verified_locally")
        self.assertEqual(
            result["chat_template_sha256"],
            mt_bench.text_sha256("fake-template-v1"),
        )
        self.assertGreater(result["turns"]["turn_1"]["prompt_tokens"], 0)
        self.assertNotEqual(
            result["turns"]["turn_1"]["token_ids_sha256"],
            result["turns"]["turn_2"]["token_ids_sha256"],
        )


def argparse_namespace(**kwargs):
    return types.SimpleNamespace(**kwargs)


if __name__ == "__main__":
    unittest.main()
