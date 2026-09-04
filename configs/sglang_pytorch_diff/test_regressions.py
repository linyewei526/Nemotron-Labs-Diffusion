from __future__ import annotations

import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CUDA_GRAPH_RUNNER = (
    PROJECT_ROOT
    / "sglang_dllm/src/sglang/python/sglang/srt/model_executor/cuda_graph_runner.py"
)
LINEAR_SPEC = (
    PROJECT_ROOT
    / "sglang_dllm/src/sglang/python/sglang/srt/dllm/algorithm/linear_spec.py"
)
SGLANG_PIPELINE = PROJECT_ROOT / "xp/examples/run_sglang_eval_pipeline_gpu_only.sh"


class SGLangPyTorchRegressionTests(unittest.TestCase):
    def test_dllm_capture_selects_draft_before_noncausal_graph(self) -> None:
        text = CUDA_GRAPH_RUNNER.read_text(encoding="utf-8")
        capture_loop = text[text.index("def _capture_one_stream") : text.index("# Trigger CUDA graph capture")]
        pre_draft = capture_loop.index('getattr(self, "_dllm_pre_draft_hook", None)')
        noncausal_capture = capture_loop.index(
            "self.capture_one_batch_size(bs, forward, stream_idx)"
        )
        pre_verify = capture_loop.index('getattr(self, "_dllm_pre_verify_hook", None)')
        causal_capture = capture_loop.index("dllm_causal=True")
        self.assertLess(pre_draft, noncausal_capture)
        self.assertLess(noncausal_capture, pre_verify)
        self.assertLess(pre_verify, causal_capture)

    def test_graph_bake_restores_live_base_weights(self) -> None:
        text = LINEAR_SPEC.read_text(encoding="utf-8")
        bake = text[text.index("gr.init_capture()") : text.index("self._lora_deltas = None")]
        self.assertIn("set_base()", bake)

    def test_all_optional_traces_are_reset_after_server_warmup(self) -> None:
        text = SGLANG_PIPELINE.read_text(encoding="utf-8")
        reset_start = text.index("for startup_trace in")
        benchmark_start = text.index('echo "[4/5] Running benchmark evaluation')
        reset_block = text[reset_start:benchmark_start]
        for name in (
            "SGLANG_CONFIDENCE_TRACE_FILE",
            "SGLANG_LOW_CONFIDENCE_TRACE_FILE",
            "SGLANG_DRAFT_ALIGNMENT_TRACE_FILE",
        ):
            self.assertIn(name, reset_block)
        self.assertIn(': > "$startup_trace"', reset_block)


if __name__ == "__main__":
    unittest.main()
