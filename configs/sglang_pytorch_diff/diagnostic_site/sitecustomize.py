"""Test-only import hook that invokes LinearSpec's missing pre-draft hook.

This module is loaded only by ``python_with_predraft_hook.sh`` for an isolated
diagnostic server process. It does not modify the checked-out SGLang source.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any


TARGET = "sglang.srt.model_executor.cuda_graph_runner"


def append_log(message: str) -> None:
    log_path = os.environ.get("NLD_DIAG_PREDRAFT_HOOK_LOG")
    if not log_path:
        return
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")


def patch_module(module: ModuleType) -> None:
    runner_class = module.CudaGraphRunner
    original = runner_class.capture_one_batch_size
    if getattr(original, "_nld_diag_predraft_wrapped", False):
        return

    def wrapped(
        self: Any,
        bs: int,
        forward: Any,
        stream_idx: int | None = None,
        dllm_causal: bool = False,
    ) -> Any:
        if bool(getattr(self, "is_dllm", False)) and not dllm_causal:
            hook = getattr(self, "_dllm_pre_draft_hook", None)
            append_log(
                "capture bs={} causal=false hook_present={}".format(
                    bs, hook is not None
                )
            )
            if hook is not None:
                hook()
                append_log("pre_draft_hook_invoked")
        return original(self, bs, forward, stream_idx, dllm_causal)

    wrapped._nld_diag_predraft_wrapped = True  # type: ignore[attr-defined]
    runner_class.capture_one_batch_size = wrapped
    append_log("CudaGraphRunner.capture_one_batch_size patched")


class Loader(importlib.abc.Loader):
    def __init__(self, original: importlib.abc.Loader) -> None:
        self.original = original

    def create_module(self, spec: Any) -> Any:
        create = getattr(self.original, "create_module", None)
        return create(spec) if create is not None else None

    def exec_module(self, module: ModuleType) -> None:
        self.original.exec_module(module)
        patch_module(module)


class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path: Any, target: Any = None) -> Any:
        if fullname != TARGET:
            return None
        try:
            sys.meta_path.remove(self)
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return spec
        spec.loader = Loader(spec.loader)
        return spec


if os.environ.get("NLD_DIAG_APPLY_PREDRAFT_HOOK") == "1":
    sys.meta_path.insert(0, Finder())
    append_log("predraft diagnostic import hook installed")

