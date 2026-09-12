"""Process-local registration; shared SGLang files are never modified."""

from __future__ import annotations

import os
import sys


def _install_prompt_fingerprint_hook() -> None:
    import hashlib

    import numpy as np
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

    if getattr(ForwardBatch, "_nld_full_prompt_fingerprint_hook", False):
        return
    original = ForwardBatch.init_new.__func__

    def init_new_with_fingerprint(cls, batch, model_runner):
        result = original(cls, batch, model_runner)
        result.nld_prompt_fingerprints = [
            hashlib.sha256(
                np.asarray(req.origin_input_ids, dtype=np.int64).tobytes()
            ).hexdigest()
            for req in batch.reqs
        ]
        return result

    ForwardBatch.init_new = classmethod(init_new_with_fingerprint)
    ForwardBatch._nld_full_prompt_fingerprint_hook = True


def _register() -> None:
    if os.environ.get("NLD_B200_VERIFY_EFFICIENCY_ENABLE") != "1":
        return
    try:
        from sglang.srt.dllm import algorithm as registry

        _install_prompt_fingerprint_hook()
        here = os.path.dirname(os.path.abspath(__file__))
        observations = os.path.dirname(here)
        if observations not in sys.path:
            sys.path.insert(0, observations)
        from sglang_b200_verify_efficiency_policy.verify_shadow_algorithm import (
            VerifyEfficiencyShadowLinearSpec,
        )

        registry.algo_name_to_cls["LinearSpec"] = VerifyEfficiencyShadowLinearSpec
    except ModuleNotFoundError:
        return
    except Exception as exc:  # pragma: no cover - visible in server log
        print(
            f"[b200-verify-efficiency sitecustomize] registration failed: {exc!r}",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(86) from exc


_register()
