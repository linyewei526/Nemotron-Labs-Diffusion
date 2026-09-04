from __future__ import annotations

import unittest

from xp.pytorch_nemo_eval.add_pytorch_metrics_to_metrics import (
    forward_pass_breakdown,
    summarize,
)
from xp.pytorch_nemo_eval.pytorch_openai_server import NativePyTorchEngine


class DecodeOnlyMetricTests(unittest.TestCase):
    def test_new_linearspec_rows_match_sglang_decode_tpf(self) -> None:
        rows = [
            {
                "ok": True,
                "mode": "linearspec_lora",
                "prompt_tokens": 32,
                "completion_tokens": 123,
                "raw_generated_tokens": 123,
                "nfe": 30,
                "decode_nfe": 30,
                "prefill_nfe": 1,
                "total_nfe": 31,
            },
            {
                "ok": True,
                "mode": "linearspec_lora",
                "prompt_tokens": 32,
                "completion_tokens": 123,
                "raw_generated_tokens": 123,
                "nfe": 30,
                "decode_nfe": 30,
                "prefill_nfe": 1,
                "total_nfe": 31,
            },
        ]

        result = summarize(rows, 1.0)

        self.assertEqual(result["decode_forward_passes"], 60)
        self.assertEqual(result["prefill_forward_passes"], 2)
        self.assertEqual(result["total_forward_passes"], 62)
        self.assertEqual(result["tokens_per_forward_pass"], 4.1)
        self.assertEqual(result["end_to_end_tokens_per_forward_pass"], 3.9677)
        self.assertEqual(result["average_forward_passes_per_sample"], 30)
        self.assertEqual(result["average_total_forward_passes_per_sample"], 31)

    def test_legacy_linearspec_row_drops_one_prefill(self) -> None:
        self.assertEqual(
            forward_pass_breakdown({"mode": "linearspec_base", "nfe": 31}),
            (30.0, 1.0, 31.0),
        )

    def test_ar_and_dlm_keep_existing_model_nfe_convention(self) -> None:
        self.assertEqual(
            forward_pass_breakdown({"mode": "ar", "nfe": 10}),
            (10.0, 0.0, 10.0),
        )
        self.assertEqual(
            forward_pass_breakdown({"mode": "dlm", "nfe": 10}),
            (10.0, 0.0, 10.0),
        )

    def test_server_breakdown_is_mode_aware(self) -> None:
        engine = NativePyTorchEngine.__new__(NativePyTorchEngine)
        engine.mode = "linearspec_lora"
        self.assertEqual(engine._forward_pass_breakdown(31), (30.0, 1.0, 31.0))
        engine.mode = "dlm"
        self.assertEqual(engine._forward_pass_breakdown(31), (31.0, 0.0, 31.0))


if __name__ == "__main__":
    unittest.main()
