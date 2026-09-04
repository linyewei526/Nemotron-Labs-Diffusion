#!/usr/bin/env python3
"""Equal-dataset offline search and frozen-policy validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier


EXPECTED_DATASETS = {
    "gsm8k", "human-eval", "mbpp", "math-500", "aime25", "gpqa",
    "ifeval", "livecodebench-cpp", "mmlu",
}

ACCEPT_FEATURES = [
    "prev_block", "prev_accept", "prev_accept_ratio", "prev_full",
    "a_ma1", "a_ma2", "a_ma4", "a_ma8", "ratio_ma1", "ratio_ma2",
    "ratio_ma4", "ratio_ma8", "full_rate2", "full_rate4", "full_rate8",
    "accept_trend4", "ratio_trend4", "full_streak", "nonfull_streak",
]
CENSOR_FEATURES = ACCEPT_FEATURES + [
    f"gt{threshold}_{kind}{window}"
    for threshold in (8, 16)
    for kind in ("known_rate", "known_n")
    for window in (1, 2, 4, 8)
]
CONF_FEATURES = CENSOR_FEATURES + [
    "prev_head_conf", "prev_head_margin", "prev_head_entropy",
    "prev_rejected_conf", "prev_rejected_margin",
] + [
    f"{name}_ma{window}"
    for name in ("head_conf", "head_margin", "head_entropy")
    for window in (1, 2, 4, 8)
]
FEATURE_GROUPS = {
    "accept": ACCEPT_FEATURES,
    "censor": CENSOR_FEATURES,
    "confidence": CONF_FEATURES,
}


class ProgressBar:
    """TTY progress bar with durable percent lines for redirected logs."""

    def __init__(
        self,
        label: str,
        total: int,
        width: int = 30,
        interactive: Optional[bool] = None,
    ) -> None:
        self.label = label
        self.total = max(int(total), 1)
        self.width = width
        self.current = 0
        self.started = time.monotonic()
        self.last_rendered = 0.0
        self.last_logged_percent = -1
        self.interactive = sys.stderr.isatty() if interactive is None else interactive
        self._render(force=True)

    def _text(self) -> str:
        ratio = min(self.current / self.total, 1.0)
        filled = int(self.width * ratio)
        elapsed = max(time.monotonic() - self.started, 1e-9)
        rate = self.current / elapsed
        eta = (self.total - self.current) / rate if rate > 0 else math.inf
        eta_text = "--:--" if not math.isfinite(eta) else f"{int(eta)//60:02d}:{int(eta)%60:02d}"
        return (
            f"[{self.label}] |{'#' * filled}{'-' * (self.width - filled)}| "
            f"{self.current}/{self.total} ({100 * ratio:5.1f}%) ETA {eta_text}"
        )

    def _render(self, force: bool = False) -> None:
        now = time.monotonic()
        percent = int(100 * self.current / self.total)
        if self.interactive:
            if force or now - self.last_rendered >= 0.2:
                sys.stderr.write("\r" + self._text() + "\033[K")
                sys.stderr.flush()
                self.last_rendered = now
        elif force or percent > self.last_logged_percent:
            print(self._text(), flush=True)
            self.last_logged_percent = percent

    def update(self, amount: int = 1) -> None:
        self.current = min(self.total, self.current + amount)
        self._render(force=self.current == self.total)

    def close(self) -> None:
        if self.current < self.total:
            self.current = self.total
            self._render(force=True)
        if self.interactive:
            sys.stderr.write("\n")
            sys.stderr.flush()


def _finite(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def _split(fingerprint: str, seed: int) -> str:
    digest = hashlib.sha256(f"{seed}|{fingerprint}".encode()).digest()
    value = int.from_bytes(digest[:8], "big") / float(1 << 64)
    if value < 0.70:
        return "train"
    if value < 0.85:
        return "selection"
    return "test"


def read_rows(trace_root: Path, split_seed: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(trace_root.glob("*.jsonl")):
        path_rows = 0
        with path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("event") != "sglang_dynamic_block_shadow_round":
                    continue
                branches = record.get("branches") or {}
                if not all(str(size) in branches for size in (8, 16, 32)):
                    raise ValueError(f"{path}:{line_number}: missing L8/L16/L32 branch")
                fingerprint = str(record.get("prompt_fingerprint") or record["request_id"])
                row = {
                    "dataset": str(record.get("benchmark") or path.stem),
                    "request": fingerprint,
                    "round": int(record["round_index"]),
                    "split": _split(fingerprint, split_seed),
                    "features": record.get("history_before_round") or {},
                    "decision": int(record["decision_block"]),
                    "replay_match": bool(record.get("canonical_replay_match", False)),
                    "cross_block_match": bool(
                        record.get("cross_block_common_prefix_match", True)
                    ),
                    "schema_version": int(record.get("schema_version", 1)),
                }
                for size in (8, 16, 32):
                    row[f"a{size}"] = int(branches[str(size)]["accept_length"])
                rows.append(row)
                path_rows += 1
        print(f"[trace读取] {path.stem}: {path_rows} 轮", flush=True)
    if not rows:
        raise ValueError(f"no dynamic-block records under {trace_root}")
    return rows


def filter_trace_rows(
    rows: Sequence[Dict[str, Any]],
    policy: str,
    max_invalid_rate: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Audit trace integrity and conservatively exclude ambiguous old rows.

    A row is ambiguous when the chosen-size replay did not match its shadow or
    when the three block branches do not share their reported accepted prefix.
    The latter is rare but is excluded as well so current-run continuation does
    not silently train on different output trajectories.  A global rate guard
    prevents a materially broken collection from being laundered by filtering.
    """
    if policy not in {"strict", "exclude"}:
        raise ValueError(f"unknown invalid row policy: {policy}")
    by_dataset: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "rows_original": 0,
            "rows_usable": 0,
            "rows_excluded": 0,
            "replay_mismatch": 0,
            "cross_block_mismatch": 0,
        }
    )
    usable: List[Dict[str, Any]] = []
    replay_bad = cross_bad = union_bad = 0
    for row in rows:
        metric = by_dataset[row["dataset"]]
        metric["rows_original"] += 1
        replay_mismatch = not row["replay_match"]
        cross_mismatch = not row["cross_block_match"]
        invalid = replay_mismatch or cross_mismatch
        replay_bad += int(replay_mismatch)
        cross_bad += int(cross_mismatch)
        union_bad += int(invalid)
        metric["replay_mismatch"] += int(replay_mismatch)
        metric["cross_block_mismatch"] += int(cross_mismatch)
        if invalid:
            metric["rows_excluded"] += 1
        else:
            metric["rows_usable"] += 1
            usable.append(row)
    original = len(rows)
    invalid_rate = union_bad / max(original, 1)
    for metric in by_dataset.values():
        metric["excluded_rate"] = metric["rows_excluded"] / max(
            metric["rows_original"], 1
        )
    quality = {
        "invalid_row_policy": policy,
        "max_invalid_row_rate": max_invalid_rate,
        "rows_original": original,
        "rows_usable": original - union_bad,
        "rows_excluded": union_bad,
        "excluded_rate": invalid_rate,
        "replay_mismatch": replay_bad,
        "cross_block_mismatch": cross_bad,
        "by_dataset": dict(sorted(by_dataset.items())),
    }
    if invalid_rate > max_invalid_rate:
        raise RuntimeError(
            "trace integrity exclusion rate exceeds guard: "
            f"{invalid_rate:.4%} > {max_invalid_rate:.4%}; quality={quality}"
        )
    if union_bad and policy == "strict":
        raise RuntimeError(
            f"trace has {union_bad}/{original} ambiguous rows; quality={quality}"
        )
    selected = list(rows) if policy == "strict" else usable
    if not selected:
        raise RuntimeError("no usable dynamic-block rows after integrity filtering")
    print(
        "[trace质量] "
        f"原始={original} 可用={len(selected)} 排除={union_bad} "
        f"({invalid_rate:.4%}) replay={replay_bad} cross_block={cross_bad}",
        flush=True,
    )
    return selected, quality


def atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def equal_weights(rows: Sequence[Dict[str, Any]]) -> np.ndarray:
    """Dataset equal -> request equal -> rounds within request equal."""
    datasets = sorted({row["dataset"] for row in rows})
    requests: Dict[str, set[str]] = defaultdict(set)
    rounds: Dict[Tuple[str, str], int] = defaultdict(int)
    for row in rows:
        requests[row["dataset"]].add(row["request"])
        rounds[(row["dataset"], row["request"])] += 1
    result = []
    for row in rows:
        result.append(
            1.0
            / len(datasets)
            / len(requests[row["dataset"]])
            / rounds[(row["dataset"], row["request"])]
        )
    return np.asarray(result, dtype=np.float64)


def matrix(rows: Sequence[Dict[str, Any]], names: Sequence[str]) -> np.ndarray:
    return np.asarray(
        [[_finite(row["features"].get(name)) for name in names] for row in rows],
        dtype=np.float64,
    )


def _constant_model(names: Sequence[str], probability: float) -> Dict[str, Any]:
    probability = min(max(float(probability), 1e-6), 1 - 1e-6)
    return {
        "type": "logistic", "feature_names": list(names),
        "means": [0.0] * len(names), "scales": [1.0] * len(names),
        "coefficients": [0.0] * len(names),
        "intercept": math.log(probability / (1 - probability)),
    }


def fit_export(
    family: str,
    names: Sequence[str],
    rows: Sequence[Dict[str, Any]],
    labels: np.ndarray,
    weights: np.ndarray,
    x_values: Optional[np.ndarray] = None,
) -> Tuple[Dict[str, Any], np.ndarray]:
    x = matrix(rows, names) if x_values is None else x_values
    if len(np.unique(labels)) < 2:
        model = _constant_model(names, float(np.average(labels, weights=weights)))
        return model, np.full(len(rows), float(np.average(labels, weights=weights)))
    if family == "logistic":
        imputer = SimpleImputer(strategy="mean", keep_empty_features=True)
        x_imp = imputer.fit_transform(x)
        means = np.asarray(imputer.statistics_, dtype=np.float64)
        scales = np.nanstd(x_imp, axis=0)
        scales[scales < 1e-12] = 1.0
        x_scaled = (x_imp - means) / scales
        estimator = LogisticRegression(C=0.25, max_iter=1000, class_weight=None)
        estimator.fit(x_scaled, labels, sample_weight=weights)
        exported = {
            "type": "logistic", "feature_names": list(names),
            "means": means.tolist(), "scales": scales.tolist(),
            "coefficients": estimator.coef_[0].tolist(),
            "intercept": float(estimator.intercept_[0]),
        }
        return exported, estimator.predict_proba(x_scaled)[:, 1]
    imputer = SimpleImputer(strategy="mean", keep_empty_features=True)
    x_imp = imputer.fit_transform(x)
    estimator = DecisionTreeClassifier(
        max_depth=3, min_samples_leaf=max(20, len(rows) // 200), random_state=0
    )
    estimator.fit(x_imp, labels, sample_weight=weights)

    def export_node(index: int) -> Dict[str, Any]:
        tree = estimator.tree_
        if tree.children_left[index] == tree.children_right[index]:
            counts = tree.value[index][0]
            return {"value": float(counts[1] / max(counts.sum(), 1e-12))}
        return {
            "feature_index": int(tree.feature[index]),
            "threshold": float(tree.threshold[index]),
            "left": export_node(int(tree.children_left[index])),
            "right": export_node(int(tree.children_right[index])),
        }

    exported = {
        "type": "tree", "feature_names": list(names),
        "means": np.asarray(imputer.statistics_, dtype=np.float64).tolist(),
        "tree": export_node(0),
    }
    return exported, estimator.predict_proba(x_imp)[:, 1]


def predict_export(
    model: Dict[str, Any],
    rows: Sequence[Dict[str, Any]],
    x_values: Optional[np.ndarray] = None,
) -> np.ndarray:
    x = matrix(rows, model["feature_names"]) if x_values is None else x_values
    means = np.asarray(model.get("means", np.zeros(x.shape[1])), dtype=np.float64)
    x = np.where(np.isfinite(x), x, means)
    if model["type"] == "logistic":
        scales = np.asarray(model["scales"], dtype=np.float64)
        z = (x - means) / scales @ np.asarray(model["coefficients"]) + float(model["intercept"])
        return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))

    def one(values: np.ndarray) -> float:
        node = model["tree"]
        while "value" not in node:
            node = node["left"] if values[int(node["feature_index"])] <= node["threshold"] else node["right"]
        return float(node["value"])

    return np.asarray([one(values) for values in x])


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.average(values, weights=weights)) if len(values) else math.nan


def actions_for(
    target: str,
    rows: Sequence[Dict[str, Any]],
    scores: Dict[str, np.ndarray],
    thresholds: Dict[str, float],
) -> np.ndarray:
    actions = np.full(len(rows), 8 if target == "s8" else 16, dtype=np.int64)
    if target == "s8":
        actions[scores["worth16"] >= thresholds["worth16"]] = 16
        actions[scores["worth32"] >= thresholds["worth32"]] = 32
    else:
        actions[scores["safe8"] >= thresholds["safe8"]] = 8
        actions[scores["worth32"] >= thresholds["worth32"]] = 32
    # The first round is protocol-fixed L16, regardless of default-small target.
    actions[np.asarray([row["round"] == 0 for row in rows])] = 16
    return actions


def summarize_actions(
    rows: Sequence[Dict[str, Any]], actions: np.ndarray, target: str,
    min_gain16: float, min_gain32: float,
    min_eff16: float = 0.0, min_eff32: float = 0.0,
) -> Dict[str, Any]:
    def compute(
        subset: Sequence[Dict[str, Any]], chosen: np.ndarray, weight: np.ndarray
    ) -> Dict[str, Any]:
        a8 = np.asarray([row["a8"] for row in subset], dtype=float)
        a16 = np.asarray([row["a16"] for row in subset], dtype=float)
        a32 = np.asarray([row["a32"] for row in subset], dtype=float)
        accepted = np.choose((chosen // 8 - 1).clip(0, 3), [a8, a16, a16, a32])
        promote = chosen > (8 if target == "s8" else 16)
        gain16 = a16 - a8
        gain32 = a32 - (a8 if target == "s8" else a16)
        significant = np.where(
            chosen == 16,
            (gain16 >= min_gain16) & (gain16 / 8 >= min_eff16),
            (gain32 >= min_gain32)
            & (gain32 / (24 if target == "s8" else 16) >= min_eff32),
        )
        large_waste = promote & ((accepted - (a8 if target == "s8" else a16)) <= 1)
        downgrade = chosen == 8 if target == "s16" else np.zeros(len(chosen), dtype=bool)
        safe8 = (a16 - a8) <= 0

        def conditional_dataset_macro(
            values: np.ndarray, condition: np.ndarray, empty: float
        ) -> float:
            """Average a conditional rate over datasets, never over pooled events.

            Dataset equal weighting must also hold after conditioning on a
            promotion/downgrade.  Otherwise a dataset that fires the policy
            often can dominate the precision constraint even though ordinary
            means are dataset-equal.  Datasets with no such decision are
            omitted because they provide no conditional observation.
            """
            per_dataset: List[float] = []
            for dataset in sorted({row["dataset"] for row in subset}):
                mask = np.asarray(
                    [row["dataset"] == dataset for row in subset], dtype=bool
                ) & condition
                if mask.any():
                    per_dataset.append(_weighted_mean(values[mask], weight[mask]))
            return float(np.mean(per_dataset)) if per_dataset else empty

        return {
            "rounds": len(subset),
            "requests": len({(item["dataset"], item["request"]) for item in subset}),
            "mean_block": _weighted_mean(chosen.astype(float), weight),
            "mean_accept": _weighted_mean(accepted, weight),
            "tpf_proxy": _weighted_mean(accepted, weight) / max(_weighted_mean(chosen, weight), 1e-12),
            "loss_vs_l32": _weighted_mean(np.maximum(a32 - accepted, 0), weight),
            "loss_vs_default": _weighted_mean(np.maximum((a8 if target == "s8" else a16) - accepted, 0), weight),
            "large_rate": _weighted_mean(promote.astype(float), weight),
            "large_precision": conditional_dataset_macro(
                significant.astype(float), promote, 1.0
            ),
            "large_waste_rate": conditional_dataset_macro(
                large_waste.astype(float), promote, 0.0
            ),
            "downgrade8_rate": _weighted_mean(downgrade.astype(float), weight),
            "downgrade8_safe_rate": conditional_dataset_macro(
                safe8.astype(float), downgrade, 1.0
            ),
            "l8_rate": _weighted_mean((chosen == 8).astype(float), weight),
            "l16_rate": _weighted_mean((chosen == 16).astype(float), weight),
            "l32_rate": _weighted_mean((chosen == 32).astype(float), weight),
        }

    by_dataset: Dict[str, Dict[str, Any]] = {}
    for dataset in sorted({row["dataset"] for row in rows}):
        indices = np.asarray(
            [i for i, row in enumerate(rows) if row["dataset"] == dataset]
        )
        subset = [rows[i] for i in indices]
        by_dataset[dataset] = compute(
            subset, actions[indices], equal_weights(subset)
        )

    # This is not a pooled sample average. ``equal_weights(rows)`` gives every
    # dataset exactly 1/D for ordinary means. Conditional precision/safety is
    # first computed within every dataset that has the decision, then averaged
    # across those datasets; a dataset with zero decisions is neutral.
    macro = compute(rows, actions, equal_weights(rows))
    return {"datasets": by_dataset, "macro": macro, "dataset_count": len(by_dataset)}


def threshold_grid_macros(
    target: str,
    rows: Sequence[Dict[str, Any]],
    scores: Dict[str, np.ndarray],
    thresholds_grid: Sequence[float],
    spec: Dict[str, float],
) -> List[Tuple[Dict[str, float], Dict[str, float]]]:
    """Evaluate a two-threshold grid through 9x9 score-bin sufficient stats.

    The previous implementation rebuilt full per-dataset summaries for every
    candidate.  On the formal ~700k-round trace that repeated Python scans more
    than eighteen thousand times.  Candidate ranking only needs the already
    dataset-equal macro weights, so each score/model pair is first reduced to
    81 cells and all 64 threshold pairs are then exact small-table lookups.
    """
    thresholds = np.asarray(thresholds_grid, dtype=np.float64)
    first_name, second_name = (
        ("worth16", "worth32") if target == "s8" else ("safe8", "worth32")
    )
    first = np.asarray(scores[first_name], dtype=np.float64).copy()
    second = np.asarray(scores[second_name], dtype=np.float64).copy()
    first_round = np.asarray([row["round"] == 0 for row in rows])
    # Force protocol-fixed L16 in the score cells for every threshold pair.
    first[first_round] = 1.0 if target == "s8" else 0.0
    second[first_round] = 0.0
    first_bin = np.searchsorted(thresholds, first, side="right")
    second_bin = np.searchsorted(thresholds, second, side="right")
    side = len(thresholds) + 1
    cells = first_bin * side + second_bin
    cell_count = side * side

    weights = equal_weights(rows)
    a8 = np.asarray([row["a8"] for row in rows], dtype=np.float64)
    a16 = np.asarray([row["a16"] for row in rows], dtype=np.float64)
    a32 = np.asarray([row["a32"] for row in rows], dtype=np.float64)
    default = a8 if target == "s8" else a16
    action_metrics: Dict[int, Dict[str, np.ndarray]] = {}
    datasets = sorted({row["dataset"] for row in rows})
    dataset_masks = {
        dataset: np.asarray(
            [row["dataset"] == dataset for row in rows], dtype=bool
        )
        for dataset in datasets
    }
    conditional_cells: Dict[str, Dict[int, Dict[str, np.ndarray]]] = {
        dataset: {} for dataset in datasets
    }
    for action, accepted in ((8, a8), (16, a16), (32, a32)):
        promote = np.full(len(rows), action > (8 if target == "s8" else 16))
        if action == 16:
            gain = a16 - a8
            significant = (
                (gain >= spec.get("gain16", 0.0))
                & (gain / 8 >= spec.get("eff16", 0.0))
            )
        else:
            gain = a32 - default
            significant = (
                (gain >= spec["gain32"])
                & (gain / (24 if target == "s8" else 16) >= spec.get("eff32", 0.0))
            )
        downgrade = np.full(len(rows), target == "s16" and action == 8)
        values = {
            "weight": np.ones(len(rows)),
            "block": np.full(len(rows), float(action)),
            "accept": accepted,
            "loss32": np.maximum(a32 - accepted, 0),
            "lossdefault": np.maximum(default - accepted, 0),
            "promote": promote.astype(float),
            "significant": (promote & significant).astype(float),
            "waste": (promote & ((accepted - default) <= 1)).astype(float),
            "downgrade": downgrade.astype(float),
            "downgrade_safe": (downgrade & ((a16 - a8) <= 0)).astype(float),
            "l8": np.full(len(rows), action == 8, dtype=float),
            "l16": np.full(len(rows), action == 16, dtype=float),
            "l32": np.full(len(rows), action == 32, dtype=float),
        }
        action_metrics[action] = {
            name: np.bincount(
                cells, weights=weights * value, minlength=cell_count
            )
            for name, value in values.items()
        }
        for dataset, dataset_mask in dataset_masks.items():
            conditional_cells[dataset][action] = {
                name: np.bincount(
                    cells,
                    weights=weights * value * dataset_mask,
                    minlength=cell_count,
                )
                for name, value in values.items()
                if name in {
                    "promote", "significant", "waste",
                    "downgrade", "downgrade_safe",
                }
            }

    cell_first = np.repeat(np.arange(side), side)
    cell_second = np.tile(np.arange(side), side)
    results: List[Tuple[Dict[str, float], Dict[str, float]]] = []
    for first_index, first_threshold in enumerate(thresholds_grid):
        for second_index, second_threshold in enumerate(thresholds_grid):
            if target == "s8":
                chosen = np.where(
                    cell_second > second_index,
                    32,
                    np.where(cell_first > first_index, 16, 8),
                )
            else:
                chosen = np.where(
                    cell_second > second_index,
                    32,
                    np.where(cell_first > first_index, 8, 16),
                )

            def total(name: str) -> float:
                return float(
                    sum(
                        action_metrics[action][name][cell]
                        for cell, action in enumerate(chosen)
                    )
                )

            def conditional_dataset_macro(
                numerator: str, denominator: str, empty: float
            ) -> float:
                values: List[float] = []
                for dataset in datasets:
                    metrics = conditional_cells[dataset]
                    denom = float(
                        sum(
                            metrics[action][denominator][cell]
                            for cell, action in enumerate(chosen)
                        )
                    )
                    if denom <= 0:
                        continue
                    numer = float(
                        sum(
                            metrics[action][numerator][cell]
                            for cell, action in enumerate(chosen)
                        )
                    )
                    values.append(numer / denom)
                return float(np.mean(values)) if values else empty

            total_weight = max(total("weight"), 1e-30)
            promote_weight = total("promote")
            downgrade_weight = total("downgrade")
            mean_block = total("block") / total_weight
            mean_accept = total("accept") / total_weight
            macro = {
                "mean_block": mean_block,
                "mean_accept": mean_accept,
                "tpf_proxy": mean_accept / max(mean_block, 1e-12),
                "loss_vs_l32": total("loss32") / total_weight,
                "loss_vs_default": total("lossdefault") / total_weight,
                "large_rate": promote_weight / total_weight,
                "large_precision": conditional_dataset_macro(
                    "significant", "promote", 1.0
                ),
                "large_waste_rate": conditional_dataset_macro(
                    "waste", "promote", 0.0
                ),
                "downgrade8_rate": downgrade_weight / total_weight,
                "downgrade8_safe_rate": conditional_dataset_macro(
                    "downgrade_safe", "downgrade", 1.0
                ),
                "l8_rate": total("l8") / total_weight,
                "l16_rate": total("l16") / total_weight,
                "l32_rate": total("l32") / total_weight,
            }
            results.append(
                (
                    {first_name: float(first_threshold), second_name: float(second_threshold)},
                    macro,
                )
            )
    return results


def _classification_metrics(labels: np.ndarray, scores: np.ndarray, weights: np.ndarray) -> Dict[str, Any]:
    result: Dict[str, Any] = {"positive_rate": _weighted_mean(labels, weights)}
    if len(np.unique(labels)) == 2:
        result["auroc"] = float(roc_auc_score(labels, scores, sample_weight=weights))
        result["auprc"] = float(average_precision_score(labels, scores, sample_weight=weights))
    else:
        result.update({"auroc": None, "auprc": None})
    return result


def hgb_signal_upper_bounds(
    train: Sequence[Dict[str, Any]],
    test: Sequence[Dict[str, Any]],
    target: str,
    spec: Dict[str, float],
) -> List[Dict[str, Any]]:
    """Non-deployable nonlinear ceiling for each feature group and label.

    The frozen policy remains logistic/shallow-tree.  These boosted models are
    reported only as a signal sufficiency diagnostic: if even their held-out
    ranking is poor, adding policy threshold complexity cannot rescue the
    historical signal.
    """
    train_weights = equal_weights(train)
    # Preserve dataset/request/round ratios while keeping sklearn's regularized
    # objective on the conventional O(n_samples) loss scale.  Passing weights
    # that sum to one would make C=0.25 dominate and collapse logistic models
    # toward an almost constant predictor on the formal trace.
    fit_weights = train_weights * len(train)
    test_weights = equal_weights(test)
    train_labels = labels_for(target, train, spec)
    test_labels = labels_for(target, test, spec)
    results: List[Dict[str, Any]] = []
    total = len(FEATURE_GROUPS) * 2 * len(train_labels)
    progress = ProgressBar(f"{target.upper()} GBDT", total)
    for group, names in FEATURE_GROUPS.items():
        imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        x_train = imputer.fit_transform(matrix(train, names))
        x_test = imputer.transform(matrix(test, names))
        for depth in (3, 6):
            for label_name, labels in train_labels.items():
                if len(np.unique(labels)) < 2:
                    score = np.full(
                        len(test), float(np.average(labels, weights=train_weights))
                    )
                else:
                    model = HistGradientBoostingClassifier(
                        learning_rate=0.06,
                        max_depth=depth,
                        max_iter=120,
                        l2_regularization=1.0,
                        random_state=0,
                    )
                    model.fit(x_train, labels, sample_weight=fit_weights)
                    score = model.predict_proba(x_test)[:, 1]
                metrics = _classification_metrics(
                    test_labels[label_name], score, test_weights
                )
                results.append(
                    {
                        "feature_group": group,
                        "max_depth": depth,
                        "label": label_name,
                        **metrics,
                    }
                )
                progress.update()
    progress.close()
    return results


def _label_specs(target: str) -> List[Dict[str, float]]:
    specs: List[Dict[str, float]] = []
    if target == "s8":
        for gain16 in (2.0, 3.0, 4.0):
            for eff16 in (0.125, 0.25):
                for gain32 in (5.0, 7.0, 9.0):
                    for eff32 in (0.125, 0.25):
                        specs.append({"gain16": gain16, "eff16": eff16, "gain32": gain32, "eff32": eff32})
    else:
        for gain32 in (3.0, 5.0, 7.0):
            for eff32 in (0.125, 0.25):
                for safe8_loss in (0.0, 1.0):
                    specs.append({"gain16": 0.0, "gain32": gain32, "eff32": eff32, "safe8_loss": safe8_loss})
    return specs


def labels_for(target: str, rows: Sequence[Dict[str, Any]], spec: Dict[str, float]) -> Dict[str, np.ndarray]:
    a8 = np.asarray([row["a8"] for row in rows], dtype=float)
    a16 = np.asarray([row["a16"] for row in rows], dtype=float)
    a32 = np.asarray([row["a32"] for row in rows], dtype=float)
    if target == "s8":
        return {
            "worth16": ((a16 - a8 >= spec["gain16"]) & ((a16 - a8) / 8 >= spec["eff16"])).astype(int),
            "worth32": ((a32 - a8 >= spec["gain32"]) & ((a32 - a8) / 24 >= spec["eff32"])).astype(int),
        }
    return {
        "worth32": ((a32 - a16 >= spec["gain32"]) & ((a32 - a16) / 16 >= spec["eff32"])).astype(int),
        "safe8": (a16 - a8 <= spec["safe8_loss"]).astype(int),
    }


def _effective_label_key(target: str, spec: Dict[str, float]) -> Tuple[float, ...]:
    """Collapse specs that induce exactly the same binary training labels."""
    if target == "s8":
        return (
            max(spec["gain16"], 8 * spec["eff16"]),
            max(spec["gain32"], 24 * spec["eff32"]),
        )
    return (
        max(spec["gain32"], 16 * spec["eff32"]),
        spec["safe8_loss"],
    )


def search_target(
    all_rows: Sequence[Dict[str, Any]], target: str, output: Path,
    min_large_precision: float, max_large_waste: float, min_safe8: float,
) -> Dict[str, Any]:
    train = [row for row in all_rows if row["split"] == "train" and row["round"] > 0]
    selection = [row for row in all_rows if row["split"] == "selection"]
    test = [row for row in all_rows if row["split"] == "test"]
    signal_test = [row for row in test if row["round"] > 0]
    if not train or not selection or not test or not signal_test:
        raise RuntimeError(
            f"{target}: train/selection/test split is empty; collect more requests"
        )
    train_weights = equal_weights(train)
    fit_weights = train_weights * len(train)
    thresholds_grid = (0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.98)
    train_matrices = {
        group: matrix(train, names) for group, names in FEATURE_GROUPS.items()
    }
    selection_matrices = {
        group: matrix(selection, names) for group, names in FEATURE_GROUPS.items()
    }
    model_cache: Dict[
        Tuple[str, str, Tuple[float, ...]],
        Tuple[Dict[str, Dict[str, Any]], Dict[str, np.ndarray]],
    ] = {}
    best: Optional[Dict[str, Any]] = None
    searched = 0
    total = len(_label_specs(target)) * len(FEATURE_GROUPS) * 2 * len(thresholds_grid) ** 2
    progress = ProgressBar(f"{target.upper()} 候选", total)
    for spec in _label_specs(target):
        train_labels = labels_for(target, train, spec)
        for group, names in FEATURE_GROUPS.items():
            for family in ("logistic", "tree"):
                models: Dict[str, Dict[str, Any]] = {}
                cache_key = (group, family, _effective_label_key(target, spec))
                cached = model_cache.get(cache_key)
                if cached is None:
                    for label_name, labels in train_labels.items():
                        models[label_name], _ = fit_export(
                            family,
                            names,
                            train,
                            labels,
                            fit_weights,
                            x_values=train_matrices[group],
                        )
                    selection_scores = {
                        name: predict_export(
                            model,
                            selection,
                            x_values=selection_matrices[group],
                        )
                        for name, model in models.items()
                    }
                    model_cache[cache_key] = (models, selection_scores)
                else:
                    models, selection_scores = cached
                for thresholds, macro in threshold_grid_macros(
                    target, selection, selection_scores, thresholds_grid, spec
                ):
                    searched += 1
                    feasible = (
                        macro["large_precision"] >= min_large_precision
                        and macro["large_waste_rate"] <= max_large_waste
                        and (target != "s16" or macro["downgrade8_safe_rate"] >= min_safe8)
                    )
                    utility = macro["mean_accept"] - 0.10 * macro["mean_block"]
                    penalty = (
                        max(0.0, min_large_precision - macro["large_precision"]) * 10
                        + max(0.0, macro["large_waste_rate"] - max_large_waste) * 10
                        + (max(0.0, min_safe8 - macro["downgrade8_safe_rate"]) * 10 if target == "s16" else 0)
                    )
                    rank = (1 if feasible else 0, utility - penalty, -macro["mean_block"])
                    if best is None or rank > best["rank"]:
                        best = {
                            "rank": rank, "spec": spec, "feature_group": group,
                            "family": family, "models": models, "thresholds": thresholds,
                            "selection_macro": macro,
                        }
                    progress.update()
    progress.close()
    assert best is not None
    selection_scores = {
        name: predict_export(
            model,
            selection,
            x_values=selection_matrices[best["feature_group"]],
        )
        for name, model in best["models"].items()
    }
    selection_actions = actions_for(
        target, selection, selection_scores, best["thresholds"]
    )
    selection_summary = summarize_actions(
        selection,
        selection_actions,
        target,
        best["spec"].get("gain16", 0),
        best["spec"]["gain32"],
        best["spec"].get("eff16", 0),
        best["spec"].get("eff32", 0),
    )
    test_matrix = matrix(test, FEATURE_GROUPS[best["feature_group"]])
    test_scores = {
        name: predict_export(model, test, x_values=test_matrix)
        for name, model in best["models"].items()
    }
    test_actions = actions_for(target, test, test_scores, best["thresholds"])
    test_summary = summarize_actions(
        test,
        test_actions,
        target,
        best["spec"].get("gain16", 0),
        best["spec"]["gain32"],
        best["spec"].get("eff16", 0),
        best["spec"].get("eff32", 0),
    )
    signal_scores = {
        name: predict_export(model, signal_test)
        for name, model in best["models"].items()
    }
    test_labels = labels_for(target, signal_test, best["spec"])
    test_weights = equal_weights(signal_test)
    signal_metrics = {
        name: _classification_metrics(test_labels[name], signal_scores[name], test_weights)
        for name in test_labels
    }
    upper_bounds = hgb_signal_upper_bounds(
        train, signal_test, target, best["spec"]
    )
    policy = {
        "schema_version": 1, "target": target,
        "default_block": 8 if target == "s8" else 16,
        "cold_start_block": 16, "models": best["models"],
        "thresholds": best["thresholds"], "label_spec": best["spec"],
        "feature_group": best["feature_group"], "model_family": best["family"],
        "dataset_weighting": "dataset_equal_then_request_equal_then_round_equal",
    }
    atomic_json(output, {"policy": policy})
    return {
        "target": target, "searched_candidates": total,
        "selected": {
            **{
                key: value
                for key, value in best.items()
                if key not in {"models", "rank", "selection_macro"}
            },
            "selection": selection_summary,
        },
        "test": test_summary, "test_signal": signal_metrics,
        "test_signal_upper_bounds": upper_bounds,
        "policy_path": str(output), "policy": policy,
    }


def analyze_validation(
    rows: Sequence[Dict[str, Any]], target: str, spec: Dict[str, float]
) -> Dict[str, Any]:
    def summarize(selected: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        if not selected:
            return {"datasets": {}, "macro": None, "dataset_count": 0}
        actions = np.asarray([row["decision"] for row in selected], dtype=np.int64)
        return summarize_actions(
            selected, actions, target,
            min_gain16=spec.get("gain16", 0.0),
            min_gain32=spec["gain32"],
            min_eff16=spec.get("eff16", 0.0),
            min_eff32=spec.get("eff32", 0.0),
        )

    heldout = [row for row in rows if row["split"] == "test"]
    return {"all_data": summarize(rows), "heldout_test": summarize(heldout)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("search", "validate"), required=True)
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-seed", type=int, default=20260831)
    parser.add_argument("--target", choices=("s8", "s16"))
    parser.add_argument("--min-large-precision", type=float, default=0.80)
    parser.add_argument("--max-large-waste", type=float, default=0.10)
    parser.add_argument("--min-safe8", type=float, default=0.98)
    parser.add_argument(
        "--invalid-row-policy",
        choices=("strict", "exclude"),
        default="strict",
        help="strict rejects any replay/prefix ambiguity; exclude audits and removes it",
    )
    parser.add_argument(
        "--max-invalid-row-rate",
        type=float,
        default=0.05,
        help="fail even in exclude mode when ambiguous rows exceed this fraction",
    )
    parser.add_argument(
        "--allow-partial-datasets",
        action="store_true",
        help="development only: permit search/validation without all nine datasets",
    )
    args = parser.parse_args()
    if not 0.0 <= args.max_invalid_row_rate <= 1.0:
        parser.error("--max-invalid-row-rate must be within [0,1]")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    original_rows = read_rows(args.trace_root, args.split_seed)
    rows, trace_quality = filter_trace_rows(
        original_rows, args.invalid_row_policy, args.max_invalid_row_rate
    )
    datasets = sorted({row["dataset"] for row in rows})
    if "aime24" in datasets:
        raise RuntimeError("AIME24 must not participate in dynamic-block search")
    if not args.allow_partial_datasets and set(datasets) != EXPECTED_DATASETS:
        missing = sorted(EXPECTED_DATASETS - set(datasets))
        extra = sorted(set(datasets) - EXPECTED_DATASETS)
        raise RuntimeError(
            "formal search/validation requires exactly the nine non-AIME24 "
            f"datasets; missing={missing}, extra={extra}. Use "
            "--allow-partial-datasets only for an explicitly developmental run."
        )
    if args.mode == "search":
        payload = {
            "protocol": {
                "datasets": datasets, "dataset_count": len(datasets),
                "rows": len(rows), "requests": len({(row['dataset'], row['request']) for row in rows}),
                "split_seed": args.split_seed,
                "weighting": "dataset equal -> request equal -> request rounds equal",
                "trace_quality": trace_quality,
            },
            "targets": {},
        }
        for target in ("s8", "s16"):
            payload["targets"][target] = search_target(
                rows, target, args.output_dir / f"policy_{target}.json",
                args.min_large_precision, args.max_large_waste, args.min_safe8,
            )
        atomic_json(args.output_dir / "search_results.json", payload)
    else:
        if not args.target:
            parser.error("--target is required for validate mode")
        policy_payload = json.loads(
            (args.output_dir / f"policy_{args.target}.json").read_text(encoding="utf-8")
        )
        policy = policy_payload.get("policy", policy_payload)
        payload = {
            "protocol": {
                "datasets": datasets,
                "dataset_count": len(datasets),
                "rows": len(rows),
                "trace_quality": trace_quality,
            },
            "target": args.target,
            "label_spec": policy["label_spec"],
            **analyze_validation(rows, args.target, policy["label_spec"]),
        }
        atomic_json(args.output_dir / f"validation_{args.target}.json", payload)


if __name__ == "__main__":
    main()
