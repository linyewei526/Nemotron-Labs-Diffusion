"""Process-local SGLang algorithm registration for this observation only.

The experiment prepends this directory to ``PYTHONPATH``.  Python imports
``sitecustomize`` during interpreter startup, so SGLang worker processes see
the observation class without changing the shared SGLang source tree.
"""

from __future__ import annotations

import os
import sys


def _install_full_prompt_fingerprint_hook() -> None:
    """Attach a cache-order-independent prompt ID to every ForwardBatch.

    ``ForwardBatch.input_ids`` contains only the uncached prefill suffix when
    radix-prefix caching hits.  Train/selection/test assignment must instead
    use ``Req.origin_input_ids`` so the same sample has the same split across
    collection and validation runs, regardless of request order or cache hits.
    """
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
    if os.environ.get("NLD_DYNAMIC_BLOCK_ENABLE") != "1":
        return
    try:
        from sglang.srt.dllm import algorithm as registry

        _install_full_prompt_fingerprint_hook()

        here = os.path.dirname(os.path.abspath(__file__))
        parent = os.path.dirname(here)
        if parent not in sys.path:
            sys.path.insert(0, parent)
        from sglang_dynamic_block_history_signal.dynamic_shadow_algorithm import (
            DynamicBlockShadowLinearSpec,
        )

        # The generated YAML still says LinearSpec.  Replacement is local to
        # processes carrying NLD_DYNAMIC_BLOCK_ENABLE=1.
        registry.algo_name_to_cls["LinearSpec"] = DynamicBlockShadowLinearSpec
    except ModuleNotFoundError:
        # Some helper processes use the same PYTHONPATH without the SGLang
        # source path.  They do not host a model worker and need no patch.
        return
    except Exception as exc:  # pragma: no cover - visible in server startup log
        print(
            f"[dynamic-block sitecustomize] registration failed: {exc!r}",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(86) from exc


_register()
