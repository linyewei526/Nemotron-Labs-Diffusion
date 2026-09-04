"""Observation-only LinearSpec with dynamic canonical and counterfactual blocks.

The SGLang scheduler allocates ``physical_block_size`` KV slots (normally 32).
For every decode round this class first buckets requests by the action selected
from history.  Inside each bucket it evaluates real variable-length SGLang
ForwardBatch views for L8/L16/L32 from the same committed prefix and identical
request composition.  The selected branch is executed last and is itself the
canonical result, so no numerically different full-batch-vs-sub-batch replay is
needed.  Only that branch updates request history.

This module is registered through the sibling ``sitecustomize.py`` and never
modifies the shared SGLang package.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import math
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from sglang.srt.dllm.algorithm.linear_spec import LinearSpec
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner


logger = logging.getLogger(__name__)


def _parse_sizes(value: str) -> Tuple[int, ...]:
    sizes = tuple(sorted({int(item.strip()) for item in value.split(",") if item.strip()}))
    if not sizes or any(size < 2 for size in sizes):
        raise ValueError(f"invalid NLD_DYNAMIC_BLOCK_SIZES={value!r}")
    return sizes


def _stable_uniform(seed: int, rid: str, round_index: int) -> float:
    raw = f"{seed}|{rid}|{round_index}".encode("utf-8")
    integer = int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")
    return integer / float(1 << 64)


def _weighted_choice(items: Sequence[Tuple[int, float]], u: float) -> int:
    acc = 0.0
    for value, probability in items:
        acc += probability
        if u < acc:
            return value
    return items[-1][0]


def _mean(values: Iterable[Optional[float]]) -> Optional[float]:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return sum(clean) / len(clean) if clean else None


def _slope(values: Sequence[Optional[float]]) -> Optional[float]:
    clean = [(idx, float(value)) for idx, value in enumerate(values) if value is not None]
    if len(clean) < 2:
        return None
    xs = np.asarray([item[0] for item in clean], dtype=np.float64)
    ys = np.asarray([item[1] for item in clean], dtype=np.float64)
    denom = float(((xs - xs.mean()) ** 2).sum())
    return float((((xs - xs.mean()) * (ys - ys.mean())).sum()) / denom) if denom else 0.0


def _dist_rank_zero() -> bool:
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return True
    return torch.distributed.get_rank() == 0


def _feature_value(model: Dict[str, Any], features: Dict[str, Any], index: int) -> float:
    name = model["feature_names"][index]
    mean = float((model.get("means") or [0.0] * len(model["feature_names"]))[index])
    value = features.get(name)
    return mean if value is None or not math.isfinite(float(value)) else float(value)


def _predict_model(model: Dict[str, Any], features: Dict[str, Any]) -> float:
    if model.get("type", "logistic") == "tree":
        node = model["tree"]
        while "value" not in node:
            index = int(node["feature_index"])
            value = _feature_value(model, features, index)
            node = node["left"] if value <= float(node["threshold"]) else node["right"]
        return float(node["value"])
    names = model.get("feature_names") or []
    means = model.get("means") or [0.0] * len(names)
    scales = model.get("scales") or [1.0] * len(names)
    coefficients = model.get("coefficients") or [0.0] * len(names)
    z = float(model.get("intercept", 0.0))
    for name, mean, scale, coefficient in zip(names, means, scales, coefficients):
        value = features.get(name)
        value = float(mean) if value is None or not math.isfinite(float(value)) else float(value)
        scale = float(scale) if float(scale) != 0 else 1.0
        z += float(coefficient) * ((value - float(mean)) / scale)
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-min(z, 60.0)))
    exp_z = math.exp(max(z, -60.0))
    return exp_z / (1.0 + exp_z)


class DynamicBlockShadowLinearSpec(LinearSpec):
    """LinearSpec replacement activated only by this observation's PYTHONPATH."""

    def __init__(self, config: DllmConfig) -> None:
        super().__init__(config)
        self.candidate_sizes = _parse_sizes(
            os.environ.get("NLD_DYNAMIC_BLOCK_SIZES", "8,16,32")
        )
        if max(self.candidate_sizes) > self.block_size:
            raise ValueError(
                f"physical block_size={self.block_size} is smaller than candidates "
                f"{self.candidate_sizes}"
            )
        if 16 not in self.candidate_sizes:
            raise ValueError("candidate sizes must include the required cold-start L16")
        if self.candidate_sizes != (8, 16, 32):
            raise ValueError(
                "this paired experiment requires exactly NLD_DYNAMIC_BLOCK_SIZES=8,16,32"
            )
        self.policy_mode = os.environ.get("NLD_DYNAMIC_BLOCK_POLICY_MODE", "explore")
        self.policy_target = os.environ.get("NLD_DYNAMIC_BLOCK_POLICY_TARGET", "s8")
        self.exploration_seed = int(os.environ.get("NLD_DYNAMIC_BLOCK_SEED", "20260831"))
        self.benchmark = os.environ.get("NLD_DYNAMIC_BLOCK_BENCHMARK", "unknown")
        self.trace_file = Path(
            os.environ.get(
                "NLD_DYNAMIC_BLOCK_TRACE_FILE",
                str(self._confidence_trace_file or "dynamic_block_trace.jsonl"),
            )
        )
        self.trace_file.parent.mkdir(parents=True, exist_ok=True)
        # Prevent the inherited confidence tracer from mixing a second schema
        # into this experiment's trace.  The shared pipeline still knows the
        # same path and therefore clears startup warmup records before eval.
        self._confidence_tracer = None
        self._low_confidence_tracer = None
        self._draft_alignment_tracer = None
        self._history: Dict[str, List[Dict[str, Any]]] = {}
        self._prompt_fingerprints: Dict[str, str] = {}
        self._prompt_tokens: Dict[str, List[int]] = {}
        self._trace_lock = threading.Lock()
        self._debug_cuda_sync = os.environ.get("NLD_DYNAMIC_DEBUG_CUDA_SYNC") == "1"
        self._policy: Optional[Dict[str, Any]] = None
        policy_path = os.environ.get("NLD_DYNAMIC_BLOCK_POLICY_PATH", "")
        if policy_path:
            with open(policy_path, "r", encoding="utf-8") as file:
                payload = json.load(file)
            self._policy = payload.get("policy", payload)

    # ------------------------------------------------------------------
    # Variable-length views over the scheduler's physical Lmax allocation
    # ------------------------------------------------------------------
    def _view(
        self,
        master: ForwardBatch,
        request_indices: Sequence[int],
        size: int,
    ) -> ForwardBatch:
        physical = self.block_size
        device = master.input_ids.device
        if not request_indices:
            raise RuntimeError("dynamic shadow view requires at least one request")
        if any(index < 0 or index >= master.batch_size for index in request_indices):
            raise RuntimeError(
                "dynamic shadow request index is outside master batch: "
                f"indices={list(request_indices)}, batch_size={master.batch_size}"
            )
        expected_tokens = master.batch_size * physical
        actual_shapes = {
            "input_ids": int(master.input_ids.numel()),
            "positions": int(master.positions.numel()),
            "out_cache_loc": int(master.out_cache_loc.numel()),
            "rids": len(master.rids),
            "req_pool_indices": int(master.req_pool_indices.numel()),
            "extend_prefix_lens": int(master.extend_prefix_lens.numel()),
        }
        if (
            actual_shapes["input_ids"] != expected_tokens
            or actual_shapes["positions"] != expected_tokens
            or actual_shapes["out_cache_loc"] != expected_tokens
            or actual_shapes["rids"] != master.batch_size
            or actual_shapes["req_pool_indices"] != master.batch_size
            or actual_shapes["extend_prefix_lens"] != master.batch_size
        ):
            raise RuntimeError(
                "dynamic shadow master ForwardBatch has an unexpected physical layout: "
                f"batch_size={master.batch_size}, physical={physical}, "
                f"expected_tokens={expected_tokens}, shapes={actual_shapes}, "
                f"indices={list(request_indices)}, candidate_size={size}"
            )

        def debug_sync(stage: str) -> None:
            if not self._debug_cuda_sync:
                return
            try:
                torch.cuda.synchronize(device)
            except Exception as exc:
                raise RuntimeError(
                    "dynamic shadow CUDA failure at "
                    f"{stage}; batch_size={master.batch_size}, physical={physical}, "
                    f"indices={list(request_indices)}, candidate_size={size}, "
                    f"shapes={actual_shapes}"
                ) from exc

        debug_sync("view entry (failure originated in the preceding branch)")
        req_gpu = torch.tensor(request_indices, dtype=torch.long, device=device)
        token_idx = torch.cat(
            [
                torch.arange(index * physical, index * physical + size, device=device)
                for index in request_indices
            ]
        )
        prefix = master.extend_prefix_lens.index_select(0, req_gpu)
        debug_sync("extend_prefix_lens selection")
        extend = torch.full((len(request_indices),), size, dtype=torch.int32, device=device)
        seq_lens = prefix.to(master.seq_lens.dtype) + size
        starts = torch.arange(len(request_indices), dtype=torch.int64, device=device) * size
        seq_cpu = seq_lens.detach().cpu()
        prefix_cpu = [int(prefix[i].item()) for i in range(len(request_indices))]
        extend_cpu = [size] * len(request_indices)
        positions = torch.cat(
            [
                torch.arange(
                    int(master.positions[index * physical].item()),
                    int(master.positions[index * physical].item()) + size,
                    dtype=master.positions.dtype,
                    device=device,
                )
                for index in request_indices
            ]
        )
        debug_sync("position construction")
        input_ids = master.input_ids.index_select(0, token_idx).clone()
        debug_sync("input_ids selection")
        req_pool_indices = master.req_pool_indices.index_select(0, req_gpu)
        debug_sync("req_pool_indices selection")
        out_cache_loc = master.out_cache_loc.index_select(0, token_idx)
        debug_sync("out_cache_loc selection")
        kwargs: Dict[str, Any] = {
            "batch_size": len(request_indices),
            "input_ids": input_ids,
            "req_pool_indices": req_pool_indices,
            "seq_lens": seq_lens,
            "out_cache_loc": out_cache_loc,
            "seq_lens_sum": int(seq_lens.sum().item()),
            "seq_lens_cpu": seq_cpu,
            "positions": positions,
            "extend_num_tokens": len(request_indices) * size,
            "extend_seq_lens": extend,
            "extend_prefix_lens": prefix,
            "extend_start_loc": starts,
            "extend_prefix_lens_cpu": prefix_cpu,
            "extend_seq_lens_cpu": extend_cpu,
            "rids": [master.rids[index] for index in request_indices],
            "lora_ids": (
                [master.lora_ids[index] for index in request_indices]
                if master.lora_ids is not None
                else None
            ),
            "num_token_non_padded_cpu": len(request_indices) * size,
            "orig_seq_lens": (
                master.orig_seq_lens.index_select(0, req_gpu)
                if master.orig_seq_lens is not None
                else None
            ),
        }
        if master.num_token_non_padded is not None:
            kwargs["num_token_non_padded"] = torch.tensor(
                len(request_indices) * size,
                dtype=master.num_token_non_padded.dtype,
                device=device,
            )
        return dataclasses.replace(master, **kwargs)

    def _position_metrics(
        self,
        logits: torch.Tensor,
        token_ids: torch.Tensor,
    ) -> Dict[str, List[float]]:
        rows = logits.float().clone()
        rows[:, self.mask_id] = -torch.inf
        log_z = torch.logsumexp(rows, dim=-1)
        selected = rows.gather(1, token_ids.long().unsqueeze(1)).squeeze(1)
        confidence = torch.exp(selected - log_z)
        top_values = torch.topk(rows, k=2, dim=-1).values
        top_probs = torch.exp(top_values - log_z.unsqueeze(1))
        margin = top_probs[:, 0] - top_probs[:, 1]
        probabilities = torch.softmax(rows, dim=-1)
        entropy = -(probabilities * torch.log(probabilities.clamp_min(1e-30))).sum(-1)
        return {
            "confidence": [float(v) for v in confidence.detach().cpu().tolist()],
            "margin": [float(v) for v in margin.detach().cpu().tolist()],
            "margin_risk": [float(v) for v in (1.0 - margin).detach().cpu().tolist()],
            "entropy": [float(v) for v in entropy.detach().cpu().tolist()],
        }

    def _run_branch(
        self,
        model_runner: ModelRunner,
        master: ForwardBatch,
        request_indices: Sequence[int],
        size: int,
    ) -> Tuple[Any, List[Dict[str, Any]], bool]:
        fb = self._view(master, request_indices, size)
        starts: List[int] = []
        for local, rid in enumerate(fb.rids):
            begin = local * size
            end = begin + size
            n_masks = int((fb.input_ids[begin:end] == self.mask_id).sum().item())
            gen_start = size - n_masks
            starts.append(gen_start)
            seed = self._seed_tokens.get(rid)
            if seed is not None and gen_start < size:
                fb.input_ids[begin + gen_start] = int(seed)

        self._load_lora_deltas(model_runner)
        need_swap = self._lora_deltas is not None and not self._graphs_baked
        if need_swap:
            for parameter, delta, _module in self._lora_deltas:
                parameter.data.add_(delta)
        draft_out = model_runner.forward(fb, pp_proxy_tensors=None)
        self._stats_forward_passes += 1
        if need_swap and self._lora_mode == "draft_only":
            for parameter, delta, _module in self._lora_deltas:
                parameter.data.sub_(delta)

        draft_logits_raw = draft_out.logits_output.full_logits
        draft_logits = draft_logits_raw.clone()
        draft_logits[:, self.mask_id] = -1e9
        draft_tokens = torch.argmax(draft_logits, dim=-1)
        mask_positions = fb.input_ids == self.mask_id
        fb.input_ids[mask_positions] = draft_tokens[mask_positions]

        fb.dllm_causal_kv_update = True
        verify_out = model_runner.forward(fb, pp_proxy_tensors=None)
        fb.dllm_causal_kv_update = False
        self._stats_forward_passes += 1
        if need_swap and self._lora_mode == "both":
            for parameter, delta, _module in self._lora_deltas:
                parameter.data.sub_(delta)

        verify_logits = verify_out.logits_output.full_logits.clone()
        verify_logits[:, self.mask_id] = -1e9
        ar_tokens = torch.argmax(verify_logits, dim=-1)
        eos_id = self._get_eos_id(model_runner)
        records: List[Dict[str, Any]] = []

        for local, original_index in enumerate(request_indices):
            begin = local * size
            gen_start = starts[local]
            gen_len = size - gen_start
            offset = begin + gen_start
            if gen_len > 1:
                matches = fb.input_ids[offset + 1 : offset + gen_len] == ar_tokens[
                    offset : offset + gen_len - 1
                ]
                matched = int(matches.cumprod(0).sum().item())
            else:
                matched = 0
            accepted = matched + 1
            output = fb.input_ids[offset : offset + 1]
            if matched:
                output = torch.cat([output, ar_tokens[offset : offset + matched]])
            eos_hit = False
            if eos_id is not None and (output == eos_id).any():
                eos_pos = int((output == eos_id).to(torch.int32).argmax().item()) + 1
                output = output[:eos_pos]
                accepted = eos_pos
                eos_hit = True

            draft_positions = torch.arange(
                offset + 1, offset + gen_len, device=fb.input_ids.device
            )
            if draft_positions.numel():
                position_metrics = self._position_metrics(
                    draft_logits_raw.index_select(0, draft_positions),
                    draft_tokens.index_select(0, draft_positions),
                )
            else:
                position_metrics = {key: [] for key in ("confidence", "margin", "margin_risk", "entropy")}

            accepted_draft = max(0, min(matched, accepted - 1))
            accepted_conf = position_metrics["confidence"][:accepted_draft]
            reject_index = matched if (not eos_hit and matched < max(gen_len - 1, 0)) else None
            next_seed_pos = min(offset + matched, begin + size - 1)
            records.append(
                {
                    "block_size": size,
                    "accept_length": accepted,
                    "matched_draft_tokens": matched,
                    "full": bool(accepted >= gen_len),
                    "eos_hit": eos_hit,
                    "output_token_ids": [int(v) for v in output.detach().cpu().tolist()],
                    "next_seed_token_id": int(ar_tokens[next_seed_pos].item()),
                    "accepted_confidence_mean": _mean(accepted_conf),
                    "accepted_confidence_min": min(accepted_conf) if accepted_conf else None,
                    "accepted_confidence_last": accepted_conf[-1] if accepted_conf else None,
                    "rejected_confidence": (
                        position_metrics["confidence"][reject_index]
                        if reject_index is not None
                        else None
                    ),
                    "rejected_margin": (
                        position_metrics["margin"][reject_index]
                        if reject_index is not None
                        else None
                    ),
                    "rejected_entropy": (
                        position_metrics["entropy"][reject_index]
                        if reject_index is not None
                        else None
                    ),
                    "position": position_metrics,
                    "request_batch_index": int(original_index),
                }
            )

        # Every counterfactual branch reuses the same physical KV slots and
        # FlashInfer metadata objects.  Unlike normal LinearSpec, this
        # observation immediately replans another variable-size branch.  Under
        # high throughput, backend work could still be in flight while shared
        # state was rebound and later surface as a delayed ScatterGather index
        # assert.  This experiment measures acceptance/signal quality rather
        # than wall-clock throughput, so use an explicit correctness barrier.
        torch.cuda.synchronize(fb.input_ids.device)
        return verify_out.logits_output, records, verify_out.can_run_graph

    # ------------------------------------------------------------------
    # History state and pre-outcome decision
    # ------------------------------------------------------------------
    def _history_features(self, rid: str) -> Dict[str, Any]:
        history = self._history.get(rid, [])
        features: Dict[str, Any] = {
            "history_rounds": len(history),
            "current_block": history[-1]["block_size"] if history else 16,
        }
        if not history:
            return features
        previous = history[-1]
        features.update(
            {
                "prev_block": previous["block_size"],
                "prev_accept": previous["accept_length"],
                "prev_accept_ratio": previous["accept_length"] / previous["block_size"],
                "prev_full": float(previous["full"]),
                "prev_head_conf": previous.get("head_conf_mean"),
                "prev_head_margin": previous.get("head_margin_mean"),
                "prev_head_entropy": previous.get("head_entropy_mean"),
                "prev_rejected_conf": previous.get("rejected_confidence"),
                "prev_rejected_margin": previous.get("rejected_margin"),
            }
        )
        for window in (1, 2, 4, 8):
            rows = history[-window:]
            features[f"a_ma{window}"] = _mean(row["accept_length"] for row in rows)
            features[f"ratio_ma{window}"] = _mean(
                row["accept_length"] / row["block_size"] for row in rows
            )
            features[f"full_rate{window}"] = _mean(float(row["full"]) for row in rows)
            features[f"head_conf_ma{window}"] = _mean(row.get("head_conf_mean") for row in rows)
            features[f"head_margin_ma{window}"] = _mean(row.get("head_margin_mean") for row in rows)
            features[f"head_entropy_ma{window}"] = _mean(row.get("head_entropy_mean") for row in rows)
            for threshold in (8, 16):
                known: List[float] = []
                for row in rows:
                    block = row["block_size"]
                    accept = row["accept_length"]
                    if accept > threshold:
                        known.append(1.0)
                    elif block > threshold or accept < block:
                        known.append(0.0)
                features[f"gt{threshold}_known_rate{window}"] = _mean(known)
                features[f"gt{threshold}_known_n{window}"] = len(known)
        features["accept_trend4"] = _slope([row["accept_length"] for row in history[-4:]])
        features["ratio_trend4"] = _slope(
            [row["accept_length"] / row["block_size"] for row in history[-4:]]
        )
        features["full_streak"] = 0
        for row in reversed(history):
            if not row["full"]:
                break
            features["full_streak"] += 1
        features["nonfull_streak"] = 0
        for row in reversed(history):
            if row["full"]:
                break
            features["nonfull_streak"] += 1
        return features

    def _choose_action(self, rid: str, features: Dict[str, Any]) -> Tuple[int, str, Dict[str, float]]:
        round_index = int(features.get("history_rounds", 0))
        if round_index == 0:
            return 16, "cold_start_l16", {}
        if self.policy_mode.startswith("fixed"):
            value = int(self.policy_mode.replace("fixed", ""))
            return value, self.policy_mode, {}
        if self.policy_mode == "explore":
            previous = int(features.get("current_block", 16))
            transitions = {
                8: ((8, 0.55), (16, 0.35), (32, 0.10)),
                16: ((8, 0.25), (16, 0.50), (32, 0.25)),
                32: ((8, 0.10), (16, 0.35), (32, 0.55)),
            }
            action = _weighted_choice(
                transitions.get(previous, transitions[16]),
                _stable_uniform(self.exploration_seed, rid, round_index),
            )
            return action, "stratified_markov_explore", {}
        if self.policy_mode == "frozen":
            if self._policy is None:
                raise RuntimeError("policy_mode=frozen requires NLD_DYNAMIC_BLOCK_POLICY_PATH")
            target = self.policy_target or str(self._policy.get("target", "s8"))
            scores: Dict[str, float] = {}
            models = self._policy.get("models", {})
            thresholds = self._policy.get("thresholds", {})
            if target == "s8":
                scores["worth32"] = _predict_model(models["worth32"], features)
                scores["worth16"] = _predict_model(models["worth16"], features)
                if scores["worth32"] >= float(thresholds["worth32"]):
                    return 32, "frozen_s8_promote32", scores
                if scores["worth16"] >= float(thresholds["worth16"]):
                    return 16, "frozen_s8_promote16", scores
                return 8, "frozen_s8_default", scores
            scores["worth32"] = _predict_model(models["worth32"], features)
            scores["safe8"] = _predict_model(models["safe8"], features)
            if scores["worth32"] >= float(thresholds["worth32"]):
                return 32, "frozen_s16_promote32", scores
            if scores["safe8"] >= float(thresholds["safe8"]):
                return 8, "frozen_s16_downgrade8", scores
            return 16, "frozen_s16_default", scores
        raise ValueError(f"unknown NLD_DYNAMIC_BLOCK_POLICY_MODE={self.policy_mode!r}")

    def _canonical_history_row(self, branch: Dict[str, Any]) -> Dict[str, Any]:
        head = branch["position"]
        prefix = slice(0, min(7, len(head["confidence"])))
        return {
            "block_size": branch["block_size"],
            "accept_length": branch["accept_length"],
            "full": branch["full"],
            "head_conf_mean": _mean(head["confidence"][prefix]),
            "head_margin_mean": _mean(head["margin"][prefix]),
            "head_entropy_mean": _mean(head["entropy"][prefix]),
            "rejected_confidence": branch.get("rejected_confidence"),
            "rejected_margin": branch.get("rejected_margin"),
        }

    def _append_trace(self, record: Dict[str, Any]) -> None:
        if not _dist_rank_zero():
            return
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        with self._trace_lock, self.trace_file.open("a", encoding="utf-8") as file:
            file.write(line + "\n")

    # ------------------------------------------------------------------
    # SGLang entry point
    # ------------------------------------------------------------------
    def run(self, model_runner: ModelRunner, forward_batch: ForwardBatch):
        full_prompt_fingerprints = getattr(
            forward_batch, "nld_prompt_fingerprints", None
        )
        if full_prompt_fingerprints is not None:
            if len(full_prompt_fingerprints) != len(forward_batch.rids):
                raise RuntimeError("ForwardBatch prompt fingerprint count mismatch")
            for rid, fingerprint in zip(
                forward_batch.rids, full_prompt_fingerprints
            ):
                self._prompt_fingerprints[rid] = str(fingerprint)

        mask_index = forward_batch.input_ids == self.mask_id
        if not mask_index.any():
            if forward_batch.input_ids.numel() == 0:
                return LogitsProcessorOutput(next_token_logits=None, full_logits=None), [], False
            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            logits = out.logits_output.next_token_logits
            if logits is not None:
                seeds = logits.clone()
                seeds[:, self.mask_id] = -np.inf
                tokens = torch.argmax(seeds, dim=-1)
                for index, rid in enumerate(forward_batch.rids):
                    self._seed_tokens[rid] = int(tokens[index].item())
            starts = (
                forward_batch.extend_start_loc.detach().cpu().tolist()
                if forward_batch.extend_start_loc is not None
                else [0]
            )
            lengths = (
                forward_batch.extend_seq_lens.detach().cpu().tolist()
                if forward_batch.extend_seq_lens is not None
                else [forward_batch.input_ids.numel()]
            )
            for index, rid in enumerate(forward_batch.rids):
                if rid in self._prompt_fingerprints:
                    continue
                start = int(starts[index])
                length = int(lengths[index])
                values = forward_batch.input_ids[start : start + length].detach().cpu().tolist()
                self._prompt_tokens.setdefault(rid, []).extend(int(value) for value in values)
            return out.logits_output, [], out.can_run_graph

        features_by_rid: Dict[str, Dict[str, Any]] = {}
        decisions: Dict[str, Tuple[int, str, Dict[str, float]]] = {}
        for rid in forward_batch.rids:
            features = self._history_features(rid)
            features_by_rid[rid] = features
            decisions[rid] = self._choose_action(rid, features)

        # Counterfactuals must use the same request composition as the chosen
        # serving bucket.  The previous implementation ran each shadow on the
        # complete scheduler batch and replayed the chosen action on a smaller
        # action group.  Near token ties, that batch-composition change made
        # 2%-6% of chosen replays diverge.  Here every action bucket runs all
        # three candidates on exactly the same indices; its chosen candidate is
        # ordered last and directly becomes the live canonical KV state.
        shadows_by_request: Dict[int, Dict[str, Dict[str, Any]]] = {
            index: {} for index in range(forward_batch.batch_size)
        }
        selected_records: Dict[int, Dict[str, Any]] = {}
        logits_output = None
        can_run_graph = False
        for action in self.candidate_sizes:
            group = [
                index
                for index, rid in enumerate(forward_batch.rids)
                if decisions[rid][0] == action
            ]
            if not group:
                continue
            branch_order = [
                size for size in self.candidate_sizes if size != action
            ] + [action]
            for size in branch_order:
                logits_output, records, can_run_graph = self._run_branch(
                    model_runner, forward_batch, group, size
                )
                for local, original_index in enumerate(group):
                    record = records[local]
                    shadows_by_request[original_index][str(size)] = record
                    if size == action:
                        selected_records[original_index] = record

        next_tokens: List[torch.Tensor] = []
        for index, rid in enumerate(forward_batch.rids):
            action, source, scores = decisions[rid]
            selected = selected_records[index]
            shadow_map = shadows_by_request[index]
            if set(shadow_map) != {str(size) for size in self.candidate_sizes}:
                raise RuntimeError(
                    f"incomplete dynamic shadow branches for request index {index}: "
                    f"{sorted(shadow_map)}"
                )
            # The chosen-last record is shared by the trace and canonical path;
            # equality is guaranteed by construction instead of a second pass.
            replay_match = selected is shadow_map[str(action)]
            common_length = min(
                branch["accept_length"] for branch in shadow_map.values()
            )
            common_prefixes = {
                tuple(branch["output_token_ids"][:common_length])
                for branch in shadow_map.values()
            }
            common_prefix_match = len(common_prefixes) == 1
            output = torch.tensor(
                selected["output_token_ids"],
                dtype=torch.long,
                device=forward_batch.input_ids.device,
            )
            next_tokens.append(output)
            self._seed_tokens[rid] = int(selected["next_seed_token_id"])
            history = self._history.setdefault(rid, [])
            prompt_fingerprint = self._prompt_fingerprints.get(rid)
            if prompt_fingerprint is None:
                # Defensive fallback for non-standard callers that bypass
                # ForwardBatch.init_new.  Normal SGLang serving always uses
                # the full origin-input fingerprint installed above.
                prompt_ids = self._prompt_tokens.get(rid, [])
                prompt_fingerprint = hashlib.sha256(
                    np.asarray(prompt_ids, dtype=np.int64).tobytes()
                ).hexdigest()
            self._append_trace(
                {
                    "schema_version": 2,
                    "event": "sglang_dynamic_block_shadow_round",
                    "created_at_unix": time.time(),
                    "benchmark": self.benchmark,
                    "request_id": str(rid),
                    "prompt_fingerprint": prompt_fingerprint,
                    "round_index": len(history),
                    "policy_mode": self.policy_mode,
                    "policy_target": self.policy_target,
                    "decision_block": action,
                    "decision_source": source,
                    "decision_scores": scores,
                    "history_before_round": features_by_rid[rid],
                    "branches": shadow_map,
                    "canonical_replay_match": replay_match,
                    "canonical_execution": "same_action_bucket_chosen_last",
                    "shadow_batch_scope": "same_action_bucket",
                    "cross_block_common_prefix_match": common_prefix_match,
                    "cross_block_common_prefix_length": common_length,
                    "physical_block_size": self.block_size,
                    "candidate_block_sizes": list(self.candidate_sizes),
                }
            )
            history.append(self._canonical_history_row(selected))
            if selected["eos_hit"]:
                self._seed_tokens.pop(rid, None)
                self._prompt_tokens.pop(rid, None)
                self._prompt_fingerprints.pop(rid, None)
                self._history.pop(rid, None)

        if self._stats_file:
            with open(self._stats_file, "a", encoding="utf-8") as file:
                for index, rid in enumerate(forward_batch.rids):
                    chosen = selected_records[index]
                    file.write(
                        json.dumps(
                            {
                                "forward_passes": 2,
                                "tokens": chosen["accept_length"],
                                "block_gen_positions": chosen["block_size"],
                                "acceptance_rate": chosen["accept_length"] / chosen["block_size"],
                                "observation_shadow_forwards_excluded": True,
                            },
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
        assert logits_output is not None
        return logits_output, next_tokens, can_run_graph
