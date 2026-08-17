#!/usr/bin/env python3
"""Per-round confidence/rank tracing for native PyTorch LinearSpec decoding."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Optional

import torch


def _json_float(value: Optional[float]) -> Optional[float]:
    return None if value is None else float(value)


def _masked_row(row: torch.Tensor, mask_id: int) -> torch.Tensor:
    out = row.detach().float().clone()
    if 0 <= mask_id < out.shape[-1]:
        out[mask_id] = -torch.inf
    return out


def token_confidence(row: torch.Tensor, token_id: int, mask_id: int) -> float:
    """Softmax confidence after excluding MASK from the distribution."""
    if int(token_id) == int(mask_id):
        return 0.0
    clean = _masked_row(row, mask_id)
    log_denom = torch.logsumexp(clean, dim=-1)
    value = torch.exp(clean[int(token_id)] - log_denom)
    return float(value.detach().cpu().item())


def correct_token_rank(row: torch.Tensor, token_id: int, mask_id: int) -> int:
    """One-based competition rank after excluding MASK."""
    clean = _masked_row(row, mask_id)
    target = clean[int(token_id)]
    return int((clean > target).sum().item()) + 1


class NativeLinearSpecConfidenceTracer:
    """Append one JSON record for each native LinearSpec draft/verify round."""

    def __init__(self, trace_file: str | Path) -> None:
        self.trace_file = Path(trace_file).resolve()
        self.trace_file.parent.mkdir(parents=True, exist_ok=True)
        self.trace_file.touch(exist_ok=True)
        self._write_lock = threading.Lock()

    def record_round(
        self,
        *,
        draft_logits_by_position: torch.Tensor,
        draft_tokens: torch.Tensor,
        ar_tokens: torch.Tensor,
        request_id: Any,
        round_index: int,
        generation_offset: int,
        block_length: int,
        matched_draft_tokens: int,
        accepted_draft_tokens: int,
        emitted_tokens: int,
        eos_hit: bool,
        mask_id: int,
        draft_forward_passes: int,
        nfe_after_round: int,
        threshold: float,
        temperature: float,
        mode: str,
    ) -> None:
        if draft_logits_by_position.ndim != 2:
            raise ValueError("draft_logits_by_position must have shape [block, vocab]")
        if draft_tokens.ndim != 1 or ar_tokens.ndim != 1:
            raise ValueError("draft_tokens and ar_tokens must have shape [block]")

        matched = max(0, min(int(matched_draft_tokens), block_length - 1))
        accepted_count = max(0, min(int(accepted_draft_tokens), matched))
        accepted_positions = list(range(1, 1 + accepted_count))
        accepted_generation_positions = [
            int(generation_offset + index) for index in range(accepted_count)
        ]
        accepted_token_ids = [int(draft_tokens[pos].item()) for pos in accepted_positions]
        accepted_confidences = [
            token_confidence(
                draft_logits_by_position[pos],
                int(draft_tokens[pos].item()),
                mask_id,
            )
            for pos in accepted_positions
        ]
        accepted_mean = (
            sum(accepted_confidences) / len(accepted_confidences)
            if accepted_confidences
            else None
        )

        has_rejection = bool((not eos_hit) and matched < block_length - 1)
        rejected_position: Optional[int] = None
        rejected_generation_position: Optional[int] = None
        rejected_draft_token_id: Optional[int] = None
        rejected_correct_token_id: Optional[int] = None
        rejected_draft_confidence: Optional[float] = None
        rejected_correct_token_rank: Optional[int] = None
        if has_rejection:
            rejected_position = matched + 1
            rejected_generation_position = generation_offset + matched
            rejected_draft_token_id = int(draft_tokens[rejected_position].item())
            rejected_correct_token_id = int(ar_tokens[matched].item())
            row = draft_logits_by_position[rejected_position]
            rejected_draft_confidence = token_confidence(
                row, rejected_draft_token_id, mask_id
            )
            rejected_correct_token_rank = correct_token_rank(
                row, rejected_correct_token_id, mask_id
            )

        confidence_drop_abs: Optional[float] = None
        confidence_drop_pct: Optional[float] = None
        if rejected_draft_confidence is not None and accepted_mean is not None:
            confidence_drop_abs = accepted_mean - rejected_draft_confidence
            confidence_drop_pct = (
                confidence_drop_abs / accepted_mean if accepted_mean != 0 else None
            )

        drafted_slice = draft_tokens[1:block_length]
        mask_selected_positions = [
            index + 1
            for index, value in enumerate(drafted_slice.detach().cpu().tolist())
            if int(value) == int(mask_id)
        ]
        record = {
            "schema_version": 2,
            "event": "linearspec_confidence_round",
            "backend": "native_pytorch",
            "created_at_unix": time.time(),
            "request_id": str(request_id),
            "mode": mode,
            "round_index": int(round_index),
            "batch_index": 0,
            "block_size": int(block_length),
            "gen_start": 0,
            "gen_len": int(block_length),
            "generation_offset": int(generation_offset),
            "matched_draft_tokens": matched,
            "emitted_tokens": int(emitted_tokens),
            "accepted_draft_tokens": accepted_count,
            "eos_hit": bool(eos_hit),
            "has_rejection": has_rejection,
            "draft_forward_passes": int(draft_forward_passes),
            "nfe_after_round": int(nfe_after_round),
            "threshold": float(threshold),
            "temperature": float(temperature),
            "accepted_positions": accepted_positions,
            "accepted_generation_positions": accepted_generation_positions,
            "accepted_draft_token_ids": accepted_token_ids,
            "accepted_draft_confidences": accepted_confidences,
            "accepted_draft_confidence_mean": _json_float(accepted_mean),
            "rejected_position": rejected_position,
            "rejected_generation_position": rejected_generation_position,
            "rejected_draft_token_id": rejected_draft_token_id,
            "rejected_correct_token_id": rejected_correct_token_id,
            "rejected_draft_confidence": _json_float(rejected_draft_confidence),
            "rejected_correct_token_rank": rejected_correct_token_rank,
            "confidence_drop_abs": _json_float(confidence_drop_abs),
            "confidence_drop_pct": _json_float(confidence_drop_pct),
            "mask_selected_positions": mask_selected_positions,
        }
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        with self._write_lock:
            with self.trace_file.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
