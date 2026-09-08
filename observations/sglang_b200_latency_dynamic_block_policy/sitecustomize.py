"""Register the B200 latency observation only in explicitly marked processes."""

from __future__ import annotations

import os
import sys


def _install_full_prompt_fingerprint_hook() -> None:
    import hashlib

    import numpy as np
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

    if getattr(ForwardBatch, "_nld_full_prompt_fingerprint_hook", False):
        return
    original = ForwardBatch.init_new.__func__

    def init_new_with_prompt_fingerprint(cls, batch, model_runner):
        result = original(cls, batch, model_runner)
        result.nld_prompt_fingerprints = [
            hashlib.sha256(
                np.asarray(req.origin_input_ids, dtype=np.int64).tobytes()
            ).hexdigest()
            for req in batch.reqs
        ]
        return result

    ForwardBatch.init_new = classmethod(init_new_with_prompt_fingerprint)
    ForwardBatch._nld_full_prompt_fingerprint_hook = True


def _register() -> None:
    if os.environ.get("NLD_B200_LATENCY_POLICY_ENABLE") != "1":
        return
    try:
        from sglang.srt.dllm import algorithm as registry

        _install_full_prompt_fingerprint_hook()
        here = os.path.dirname(os.path.abspath(__file__))
        observations = os.path.dirname(here)
        if observations not in sys.path:
            sys.path.insert(0, observations)
        from sglang_b200_latency_dynamic_block_policy.b200_latency_algorithm import (
            B200LatencyDynamicBlockLinearSpec,
        )

        registry.algo_name_to_cls["LinearSpec"] = B200LatencyDynamicBlockLinearSpec
    except ModuleNotFoundError:
        return
    except Exception as exc:  # pragma: no cover
        print(
            f"[b200-latency-policy sitecustomize] registration failed: {exc!r}",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(86) from exc


_register()

