#!/usr/bin/env python3
"""Self-contained scalar-signal models and trace helpers for this experiment."""

from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from array import array
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


BLOCKS = (8, 16, 32)
EXPECTED_ORDER = (
    "gsm8k", "human-eval", "mbpp", "math-500", "aime25", "gpqa",
    "ifeval", "livecodebench-cpp",
)
EXPECTED_DATASETS = set(EXPECTED_ORDER)
SIGNALS: Tuple[Dict[str, Any], ...] = (
    {"name": "accept_last", "feature": "prev_accept", "orientation": 1, "kind": "accept", "zh": "上一轮接收长度"},
    {"name": "accept_ma2", "feature": "a_ma2", "orientation": 1, "kind": "accept", "zh": "近2轮接收均值"},
    {"name": "accept_ma4", "feature": "a_ma4", "orientation": 1, "kind": "accept", "zh": "近4轮接收均值"},
    {"name": "accept_ma8", "feature": "a_ma8", "orientation": 1, "kind": "accept", "zh": "近8轮接收均值"},
    {"name": "ratio_last", "feature": "prev_accept_ratio", "orientation": 1, "kind": "accept", "zh": "上一轮接收比例"},
    {"name": "ratio_ma2", "feature": "ratio_ma2", "orientation": 1, "kind": "accept", "zh": "近2轮接收比例均值"},
    {"name": "ratio_ma4", "feature": "ratio_ma4", "orientation": 1, "kind": "accept", "zh": "近4轮接收比例均值"},
    {"name": "ratio_ma8", "feature": "ratio_ma8", "orientation": 1, "kind": "accept", "zh": "近8轮接收比例均值"},
    {"name": "full_streak", "feature": "full_streak", "orientation": 1, "kind": "accept", "zh": "连续整块通过轮数"},
    {"name": "nonfull_streak", "feature": "nonfull_streak", "orientation": -1, "kind": "accept", "zh": "连续未整块通过轮数的相反数"},
    {"name": "head_conf_last", "feature": "prev_head_conf", "orientation": 1, "kind": "confidence", "zh": "上一轮头部confidence均值"},
    {"name": "head_conf_ma2", "feature": "head_conf_ma2", "orientation": 1, "kind": "confidence", "zh": "近2轮头部confidence均值"},
    {"name": "head_conf_ma4", "feature": "head_conf_ma4", "orientation": 1, "kind": "confidence", "zh": "近4轮头部confidence均值"},
    {"name": "head_conf_ma8", "feature": "head_conf_ma8", "orientation": 1, "kind": "confidence", "zh": "近8轮头部confidence均值"},
    {"name": "head_margin_last", "feature": "prev_head_margin", "orientation": 1, "kind": "margin", "zh": "上一轮头部margin均值"},
    {"name": "head_margin_ma2", "feature": "head_margin_ma2", "orientation": 1, "kind": "margin", "zh": "近2轮头部margin均值"},
    {"name": "head_margin_ma4", "feature": "head_margin_ma4", "orientation": 1, "kind": "margin", "zh": "近4轮头部margin均值"},
    {"name": "head_margin_ma8", "feature": "head_margin_ma8", "orientation": 1, "kind": "margin", "zh": "近8轮头部margin均值"},
    {"name": "head_entropy_last", "feature": "prev_head_entropy", "orientation": -1, "kind": "entropy", "zh": "上一轮头部entropy的相反数"},
    {"name": "head_entropy_ma2", "feature": "head_entropy_ma2", "orientation": -1, "kind": "entropy", "zh": "近2轮头部entropy均值的相反数"},
    {"name": "head_entropy_ma4", "feature": "head_entropy_ma4", "orientation": -1, "kind": "entropy", "zh": "近4轮头部entropy均值的相反数"},
    {"name": "head_entropy_ma8", "feature": "head_entropy_ma8", "orientation": -1, "kind": "entropy", "zh": "近8轮头部entropy均值的相反数"},
    {"name": "reject_conf_last", "feature": "prev_rejected_conf", "orientation": 1, "kind": "confidence", "zh": "上一轮首个拒绝位confidence"},
    {"name": "reject_margin_last", "feature": "prev_rejected_margin", "orientation": 1, "kind": "margin", "zh": "上一轮首个拒绝位margin"},
)
SIGNAL_BY_NAME = {item["name"]: item for item in SIGNALS}
FEATURES = tuple(dict.fromkeys(item["feature"] for item in SIGNALS))


class ProgressBar:
    def __init__(self, label: str, total: int) -> None:
        self.label = label
        self.total = max(1, int(total))
        self.current = 0
        self.started = time.monotonic()
        self.tty = sys.stderr.isatty()
        self.last_percent = -1
        self.render(True)

    def render(self, force: bool = False) -> None:
        ratio = min(1.0, self.current / self.total)
        percent = int(100 * ratio)
        if not force and not self.tty and percent == self.last_percent:
            return
        elapsed = max(time.monotonic() - self.started, 1e-9)
        rate = self.current / elapsed
        eta = (self.total - self.current) / rate if rate else math.inf
        eta_text = "--:--" if not math.isfinite(eta) else f"{int(eta)//60:02d}:{int(eta)%60:02d}"
        filled = int(30 * ratio)
        text = (
            f"[{self.label}] |{'#' * filled}{'-' * (30-filled)}| "
            f"{self.current}/{self.total} ({100*ratio:5.1f}%) ETA {eta_text}"
        )
        if self.tty:
            sys.stderr.write("\r" + text + "\033[K")
            sys.stderr.flush()
        else:
            print(text, flush=True)
        self.last_percent = percent

    def update(self) -> None:
        self.current += 1
        self.render(self.current >= self.total)

    def close(self) -> None:
        self.current = self.total
        self.render(True)
        if self.tty:
            sys.stderr.write("\n")


def finite(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def fold_for(fingerprint: str, seed: int, folds: int) -> int:
    digest = hashlib.sha256(f"{seed}|{fingerprint}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % folds


@dataclass
class TraceData:
    dataset_names: List[str]
    dataset: np.ndarray
    request: np.ndarray
    fold: np.ndarray
    round: np.ndarray
    decision: np.ndarray
    accept: np.ndarray
    features: Dict[str, np.ndarray]
    quality: Dict[str, Any]
    request_counts: Dict[str, int]

    @property
    def size(self) -> int:
        return int(len(self.round))


def read_shadow_trace(
    root: Path,
    split_seed: int,
    folds: int,
    invalid_policy: str,
    max_invalid_rate: float,
    max_rows_per_dataset: int = 0,
) -> TraceData:
    if invalid_policy not in {"strict", "exclude"}:
        raise ValueError("invalid row policy must be strict or exclude")
    paths = sorted(root.glob("*.jsonl"))
    if not paths:
        raise RuntimeError(f"no trace jsonl files under {root}")
    dataset_names: List[str] = []
    dataset_values = array("B"); request_values = array("I"); fold_values = array("B")
    round_values = array("I"); decision_values = array("B"); accept_values = array("B")
    feature_values = {name: array("f") for name in FEATURES}
    request_ids: Dict[Tuple[str, str], int] = {}; request_folds: Dict[int, int] = {}
    request_seen: Dict[str, set[str]] = defaultdict(set); by_dataset: Dict[str, Dict[str, Any]] = {}
    total_original = total_excluded = replay_bad = cross_bad = 0
    ignored_datasets: List[str] = []
    for path in paths:
        dataset = path.stem
        if dataset in {"aime24", "mmlu"}:
            ignored_datasets.append(dataset)
            print(f"[trace读取] {dataset}: 按实验协议跳过", flush=True)
            continue
        dataset_id = len(dataset_names); dataset_names.append(dataset)
        metrics = {"rows_original": 0, "rows_usable": 0, "rows_excluded": 0, "replay_mismatch": 0, "cross_block_mismatch": 0}
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip(): continue
                record = json.loads(line)
                if record.get("event") != "sglang_dynamic_block_shadow_round": continue
                if max_rows_per_dataset and metrics["rows_original"] >= max_rows_per_dataset: break
                branches = record.get("branches") or {}
                if any(str(block) not in branches for block in BLOCKS):
                    raise RuntimeError(f"{path}:{line_number}: missing L8/L16/L32")
                fingerprint = str(record.get("prompt_fingerprint") or record["request_id"])
                request_seen[dataset].add(fingerprint)
                replay_mismatch = not bool(record.get("canonical_replay_match", False))
                cross_mismatch = not bool(record.get("cross_block_common_prefix_match", True))
                invalid = replay_mismatch or cross_mismatch
                metrics["rows_original"] += 1; total_original += 1
                metrics["replay_mismatch"] += int(replay_mismatch); metrics["cross_block_mismatch"] += int(cross_mismatch)
                replay_bad += int(replay_mismatch); cross_bad += int(cross_mismatch)
                if invalid:
                    metrics["rows_excluded"] += 1; total_excluded += 1; continue
                metrics["rows_usable"] += 1
                key = (dataset, fingerprint); request_id = request_ids.setdefault(key, len(request_ids))
                request_folds.setdefault(request_id, fold_for(fingerprint, split_seed, folds))
                dataset_values.append(dataset_id); request_values.append(request_id); fold_values.append(request_folds[request_id])
                round_values.append(int(record["round_index"])); decision_values.append(int(record["decision_block"]))
                for block in BLOCKS: accept_values.append(int(branches[str(block)]["accept_length"]))
                feature_map = record.get("history_before_round") or {}
                for name in FEATURES: feature_values[name].append(finite(feature_map.get(name)))
        metrics["excluded_rate"] = metrics["rows_excluded"] / max(1, metrics["rows_original"])
        by_dataset[dataset] = metrics
        print(f"[trace读取] {dataset}: 原始={metrics['rows_original']} 可用={metrics['rows_usable']} 排除={metrics['rows_excluded']}", flush=True)
    invalid_rate = total_excluded / max(1, total_original)
    quality = {
        "invalid_row_policy": invalid_policy, "max_invalid_row_rate": max_invalid_rate,
        "rows_original": total_original, "rows_usable": total_original-total_excluded,
        "rows_excluded": total_excluded, "excluded_rate": invalid_rate,
        "replay_mismatch": replay_bad, "cross_block_mismatch": cross_bad,
        "by_dataset": by_dataset,
        "explicitly_ignored_datasets": sorted(set(ignored_datasets)),
    }
    if invalid_rate > max_invalid_rate:
        raise RuntimeError(f"trace exclusion {invalid_rate:.4%} exceeds {max_invalid_rate:.4%}")
    if total_excluded and invalid_policy == "strict":
        raise RuntimeError(f"strict trace audit found {total_excluded} ambiguous rows")
    if total_original == total_excluded: raise RuntimeError("no usable trace rows")
    return TraceData(
        dataset_names=dataset_names, dataset=np.asarray(dataset_values,dtype=np.uint8),
        request=np.asarray(request_values,dtype=np.uint32), fold=np.asarray(fold_values,dtype=np.uint8),
        round=np.asarray(round_values,dtype=np.uint32), decision=np.asarray(decision_values,dtype=np.uint8),
        accept=np.asarray(accept_values,dtype=np.uint8).reshape(-1,3),
        features={name:np.asarray(values,dtype=np.float32) for name,values in feature_values.items()},
        quality=quality, request_counts={name:len(values) for name,values in request_seen.items()},
    )


def subset_eight_datasets(data: TraceData, allow_partial: bool, max_invalid_rate: float) -> TraceData:
    source_names = set(data.dataset_names)
    if "aime24" in source_names: raise RuntimeError("AIME24 must not participate")
    unexpected = source_names - EXPECTED_DATASETS - {"mmlu"}
    if unexpected: raise RuntimeError(f"unexpected datasets in trace root: {sorted(unexpected)}")
    present = [name for name in EXPECTED_ORDER if name in source_names]
    if not allow_partial and set(present) != EXPECTED_DATASETS:
        raise RuntimeError(f"formal search requires eight datasets; missing={sorted(EXPECTED_DATASETS-set(present))}")
    old_to_new = {data.dataset_names.index(name): new for new,name in enumerate(present)}
    indices = np.flatnonzero(np.isin(data.dataset, np.asarray(list(old_to_new),dtype=data.dataset.dtype)))
    if not len(indices): raise RuntimeError("no usable eight-dataset rows")
    remapped = np.asarray([old_to_new[int(value)] for value in data.dataset[indices]],dtype=np.uint8)
    by_dataset = {name:dict((data.quality.get("by_dataset") or {}).get(name,{})) for name in present}
    quality = {
        "invalid_row_policy": data.quality.get("invalid_row_policy"), "max_invalid_row_rate": max_invalid_rate,
        "rows_original": sum(int(row.get("rows_original",0)) for row in by_dataset.values()),
        "rows_usable": sum(int(row.get("rows_usable",0)) for row in by_dataset.values()),
        "rows_excluded": sum(int(row.get("rows_excluded",0)) for row in by_dataset.values()),
        "replay_mismatch": sum(int(row.get("replay_mismatch",0)) for row in by_dataset.values()),
        "cross_block_mismatch": sum(int(row.get("cross_block_mismatch",0)) for row in by_dataset.values()),
        "by_dataset": by_dataset,
        "explicitly_ignored_datasets": sorted(
            set(data.quality.get("explicitly_ignored_datasets") or [])
            | (source_names & {"aime24", "mmlu"})
        ),
    }
    quality["excluded_rate"] = quality["rows_excluded"] / max(1,quality["rows_original"])
    if quality["excluded_rate"] > max_invalid_rate:
        raise RuntimeError(f"eight-dataset exclusion {quality['excluded_rate']:.4%} exceeds {max_invalid_rate:.4%}")
    return TraceData(
        dataset_names=present, dataset=remapped, request=data.request[indices], fold=data.fold[indices],
        round=data.round[indices], decision=data.decision[indices], accept=data.accept[indices],
        features={name:values[indices] for name,values in data.features.items()}, quality=quality,
        request_counts={name:data.request_counts.get(name,0) for name in present},
    )


def indices_weights(data: TraceData, indices: np.ndarray) -> np.ndarray:
    indices = np.asarray(indices,dtype=np.int64); weights = np.zeros(len(indices),dtype=np.float64)
    datasets = np.unique(data.dataset[indices])
    for dataset in datasets:
        local = np.flatnonzero(data.dataset[indices] == dataset)
        requests,inverse,counts = np.unique(data.request[indices[local]],return_inverse=True,return_counts=True)
        weights[local] = 1.0/len(datasets)/len(requests)/counts[inverse]
    return weights


def weighted_quantile_edges(x: np.ndarray, weights: np.ndarray, bins: int) -> np.ndarray:
    order=np.argsort(x,kind="mergesort"); values=x[order]; cumulative=np.cumsum(weights[order])
    targets=cumulative[-1]*np.arange(1,bins)/bins; positions=np.searchsorted(cumulative,targets,side="left")
    edges=np.unique(values[np.minimum(positions,len(values)-1)])
    return edges[edges<values[-1]].astype(np.float64)


def pava_increasing(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    blocks: List[List[float]]=[]
    for index,(value,weight) in enumerate(zip(values,weights)):
        blocks.append([float(index),float(index),float(weight),float(value*weight)])
        while len(blocks)>=2:
            left,right=blocks[-2],blocks[-1]
            if left[3]/left[2] <= right[3]/right[2]+1e-15: break
            blocks[-2:]=[[left[0],right[1],left[2]+right[2],left[3]+right[3]]]
    output=np.empty(len(values),dtype=np.float64)
    for start,end,weight,total in blocks: output[int(start):int(end)+1]=total/weight
    return output


def fit_survival(signal: np.ndarray, accepted: np.ndarray, weights: np.ndarray, block: int, bins: int, shared_edges: Optional[np.ndarray]=None) -> Dict[str,Any]:
    valid=np.isfinite(signal)
    if not valid.any(): raise RuntimeError("candidate signal has no finite training values")
    x=signal[valid].astype(np.float64); y=accepted[valid].astype(np.int16); w=weights[valid].astype(np.float64)
    edges=weighted_quantile_edges(x,w,bins) if shared_edges is None else np.asarray(shared_edges,dtype=np.float64)
    bin_index=np.searchsorted(edges,x,side="left"); bin_count=len(edges)+1
    totals=np.bincount(bin_index,weights=w,minlength=bin_count)
    if np.any(totals<=0): raise RuntimeError("weighted quantile produced empty bin")
    clipped=np.clip(y,0,block)
    histogram=np.bincount(bin_index*(block+1)+clipped,weights=w,minlength=bin_count*(block+1)).reshape(bin_count,block+1)
    raw=np.cumsum(histogram[:,::-1],axis=1)[:,::-1][:,1:]/totals[:,None]
    survival=np.empty((bin_count,block),dtype=np.float64)
    for token in range(block): survival[:,token]=pava_increasing(raw[:,token],totals)
    survival=np.minimum.accumulate(survival,axis=1)
    total_weight=float(weights.sum()); all_y=accepted.astype(np.int16)
    missing=[float(np.sum(weights*(all_y>=token))/total_weight) for token in range(1,block+1)]
    return {"type":"binned_monotone_survival","block":block,"edges":edges.tolist(),"survival":survival.tolist(),"missing_survival":missing}


def predict_survival(model: Mapping[str,Any], signal: np.ndarray) -> np.ndarray:
    signal=np.asarray(signal,dtype=np.float64); survival=np.asarray(model["survival"],dtype=np.float64); missing=np.asarray(model["missing_survival"],dtype=np.float64)
    output=np.full(len(signal),float(missing.sum()),dtype=np.float64); valid=np.isfinite(signal)
    if valid.any():
        bins=np.searchsorted(np.asarray(model["edges"]),signal[valid],side="left"); output[valid]=survival[bins].sum(axis=1)
    return output


def fit_tables(data: TraceData, indices: np.ndarray, signal: np.ndarray, gate: Optional[int], bins: int) -> Dict[str,Dict[str,Any]]:
    strata={"all":np.ones(len(indices),dtype=bool)}
    if gate is not None:
        streak=data.features["full_streak"][indices]; strata={"not_full":streak<gate,"full":streak>=gate}
    result: Dict[str,Dict[str,Any]]={}
    for name,local_mask in strata.items():
        if int(local_mask.sum()) < max(100,bins*4): raise RuntimeError(f"stratum {name} is too small")
        local_indices=indices[local_mask]; local_signal=signal[local_indices]; local_weights=indices_weights(data,local_indices)
        finite_mask=np.isfinite(local_signal)
        if not finite_mask.any(): raise RuntimeError(f"stratum {name} has no finite signal")
        edges=weighted_quantile_edges(local_signal[finite_mask].astype(np.float64),local_weights[finite_mask],bins)
        result[name]={}
        for column,block in enumerate(BLOCKS):
            result[name][str(block)]=fit_survival(local_signal,data.accept[local_indices,column],local_weights,block,bins,edges)
    return result


def predict_tables(data: TraceData, indices: np.ndarray, signal: np.ndarray, tables: Mapping[str,Mapping[str,Any]], gate: Optional[int]) -> np.ndarray:
    output=np.empty((len(indices),3),dtype=np.float64)
    if gate is None:
        for column,block in enumerate(BLOCKS): output[:,column]=predict_survival(tables["all"][str(block)],signal[indices])
        return output
    streak=data.features["full_streak"][indices]
    for stratum,mask in (("not_full",streak<gate),("full",streak>=gate)):
        for column,block in enumerate(BLOCKS): output[mask,column]=predict_survival(tables[stratum][str(block)],signal[indices[mask]])
    return output


def candidate_oof(data: TraceData, spec: Mapping[str,Any], gate: Optional[int], folds: int, bins: int) -> np.ndarray:
    signal=data.features[spec["feature"]].astype(np.float64)*float(spec["orientation"]); historical=data.round>0
    output=np.zeros((data.size,3),dtype=np.float64)
    for fold in range(folds):
        train=np.flatnonzero(historical & (data.fold!=fold)); heldout=np.flatnonzero(historical & (data.fold==fold))
        if not len(train) or not len(heldout): raise RuntimeError(f"empty OOF fold {fold}")
        tables=fit_tables(data,train,signal,gate,bins); output[heldout]=predict_tables(data,heldout,signal,tables,gate)
    return output


def selected_accept(data: TraceData, actions: np.ndarray) -> np.ndarray:
    columns=np.where(actions==8,0,np.where(actions==16,1,2))
    return data.accept[np.arange(data.size),columns].astype(np.float64)


def summarize_transitions(data: TraceData, actions: np.ndarray) -> Dict[str,Any]:
    by_dataset: Dict[str,Any]={}
    for dataset_id,dataset in enumerate(data.dataset_names):
        request_rows: Dict[int,List[int]]=defaultdict(list)
        for index in np.flatnonzero(data.dataset==dataset_id): request_rows[int(data.request[index])].append(int(index))
        matrices=[]; oscillations=[]
        for rows in request_rows.values():
            sequence=actions[sorted(rows,key=lambda index:int(data.round[index]))]
            if len(sequence)>=2:
                matrix=np.zeros((3,3),dtype=np.float64)
                for left,right in zip(sequence[:-1],sequence[1:]): matrix[BLOCKS.index(int(left)),BLOCKS.index(int(right))]+=1
                matrix/=matrix.sum(); matrices.append(matrix)
            if len(sequence)>=3: oscillations.append(float(((sequence[:-2]==sequence[2:])&(sequence[:-2]!=sequence[1:-1])).mean()))
        if matrices:
            mean_matrix=np.mean(matrices,axis=0); by_dataset[dataset]={"matrix":mean_matrix.tolist(),"switch_rate":float(1-np.trace(mean_matrix)),"oscillation_rate":float(np.mean(oscillations)) if oscillations else 0.0,"requests_with_transitions":len(matrices)}
    if by_dataset:
        matrix=np.mean([np.asarray(row["matrix"]) for row in by_dataset.values()],axis=0)
        macro={"matrix":matrix.tolist(),"switch_rate":float(np.mean([row["switch_rate"] for row in by_dataset.values()])),"oscillation_rate":float(np.mean([row["oscillation_rate"] for row in by_dataset.values()]))}
    else: macro=None
    return {"datasets":by_dataset,"macro":macro}


def collect_official_metrics(eval_root: Optional[Path]) -> Dict[str,Any]:
    if eval_root is None or not eval_root.exists(): return {"datasets":{},"macro":None}
    latest: Dict[str,Tuple[float,Dict[str,Any],str]]={}
    for path in eval_root.rglob("sglang_metrics_summary.json"):
        try: payload=json.loads(path.read_text(encoding="utf-8"))
        except (OSError,json.JSONDecodeError): continue
        dataset=str(payload.get("benchmark",""))
        if not dataset: continue
        mtime=path.stat().st_mtime
        if dataset not in latest or mtime>latest[dataset][0]: latest[dataset]=(mtime,payload,str(path))
    results={}
    for dataset,(_mtime,payload,path) in sorted(latest.items()):
        decode=payload.get("decode") or {}; serving=payload.get("serving") or {}
        results[dataset]={"decode_tpf":decode.get("tokens_per_forward_pass"),"mean_tokens_per_block":decode.get("mean_tokens_per_block"),"mean_acceptance_rate":decode.get("mean_acceptance_rate"),"weighted_acceptance_rate":decode.get("weighted_acceptance_rate"),"decode_forward_passes":decode.get("decode_forward_passes"),"decode_tokens":decode.get("decode_tokens"),"wall_output_tokens_per_s_observation_only":serving.get("wall_output_tokens_per_s"),"source":path}
    numeric=("decode_tpf","mean_tokens_per_block","mean_acceptance_rate","weighted_acceptance_rate")
    macro={key:float(np.mean([row[key] for row in results.values() if row.get(key) is not None])) for key in numeric if any(row.get(key) is not None for row in results.values())} if results else None
    return {"datasets":results,"macro":macro}
