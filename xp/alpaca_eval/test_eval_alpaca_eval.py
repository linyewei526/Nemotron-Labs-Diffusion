#!/usr/bin/env python3
"""Focused regression tests for the NLD AlpacaEval adapter."""

from __future__ import annotations

import argparse
import importlib.util
import tempfile
import unittest
import zipfile
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("eval_alpaca_eval.py")
SPEC = importlib.util.spec_from_file_location("nld_eval_alpaca_eval", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class AlpacaEvalAdapterTest(unittest.TestCase):
    def test_build_messages_uses_no_implicit_system_prompt(self) -> None:
        self.assertEqual(
            MODULE.build_messages("", "hello"),
            [{"role": "user", "content": "hello"}],
        )
        self.assertEqual(
            MODULE.build_messages("system", "hello"),
            [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "hello"},
            ],
        )

    def test_candidate_extra_records_template_policy(self) -> None:
        args = argparse.Namespace(
            candidate_extra_body_json='{"chat_template_kwargs":{"custom":1}}',
            candidate_enable_thinking=False,
            candidate_truncate_history_thinking=True,
        )
        extra = MODULE.build_candidate_extra(args)
        self.assertEqual(extra["benchmark_name"], "alpaca-eval")
        self.assertEqual(
            extra["chat_template_kwargs"],
            {
                "custom": 1,
                "enable_thinking": False,
                "truncate_history_thinking": True,
            },
        )

    def test_protocol_fingerprint_is_order_independent(self) -> None:
        self.assertEqual(MODULE.fingerprint({"a": 1, "b": 2}), MODULE.fingerprint({"b": 2, "a": 1}))

    def test_strip_thinking(self) -> None:
        self.assertEqual(MODULE.strip_thinking("<think>hidden</think>answer"), "answer")
        self.assertEqual(MODULE.strip_thinking("plain"), "plain")

    def test_safe_wheel_extraction_rejects_path_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            wheel = root / "bad.whl"
            with zipfile.ZipFile(wheel, "w") as archive:
                archive.writestr("../outside.txt", "bad")
            with self.assertRaisesRegex(RuntimeError, "Unsafe wheel member"):
                MODULE.safe_extract_wheel(wheel, root / "site")

    def test_aggregate_usage(self) -> None:
        summary = MODULE.aggregate_usage(
            [
                {
                    "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
                    "response_model": "served-model",
                    "finish_reason": "stop",
                    "elapsed_s": 1.25,
                },
                {
                    "usage": {"prompt_tokens": 7, "completion_tokens": 11, "total_tokens": 18},
                    "response_model": "served-model",
                    "finish_reason": "length",
                    "elapsed_s": 2.0,
                },
            ]
        )
        self.assertEqual(summary["total_tokens"], 23)
        self.assertEqual(summary["response_models"], ["served-model"])
        self.assertEqual(summary["finish_reasons"], {"length": 1, "stop": 1})
        self.assertEqual(summary["request_elapsed_s_sum"], 3.25)


if __name__ == "__main__":
    unittest.main()
