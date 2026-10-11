"""Evaluation metrics for extreme multi-label category suggestion."""

from __future__ import annotations
from typing import List, Set, Dict, Sequence, Optional
try:
    import numpy as np
except ImportError:
    np = None

def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    if np is not None:
        return float(np.mean(values))
    return float(sum(values) / len(values))


from src.common.types import EvaluationMetrics


def precision_at_k(predicted: Sequence[str], ground_truth: Set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    pred_top_k = set(predicted[:k])
    return len(pred_top_k.intersection(ground_truth)) / float(k)


def recall_at_k(predicted: Sequence[str], ground_truth: Set[str], k: int) -> float:
    if len(ground_truth) == 0:
        return 0.0
    pred_top_k = set(predicted[:k])
    return len(pred_top_k.intersection(ground_truth)) / float(len(ground_truth))


def f1_at_k(precision: float, recall: float) -> float:
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * (precision * recall) / (precision + recall)


def average_precision(predicted: Sequence[str], ground_truth: Set[str]) -> float:
    if len(ground_truth) == 0:
        return 0.0

    hits = 0
    sum_precisions = 0.0
    for rank, item in enumerate(predicted):
        if item in ground_truth:
            hits += 1
            sum_precisions += hits / (rank + 1.0)

    return sum_precisions / float(len(ground_truth))


def compute_metrics(
    predictions: List[Sequence[str]],
    ground_truths: List[Set[str]],
    k_values: Sequence[int] = (1, 5, 10, 20, 50),
    latencies_ms: Optional[Sequence[float]] = None,
) -> EvaluationMetrics:
    """Compute aggregate precision, recall, F1, and mAP across a set of queries."""
    num_queries = len(predictions)
    if num_queries == 0:
        return EvaluationMetrics()

    p_at_k = {k: 0.0 for k in k_values}
    r_at_k = {k: 0.0 for k in k_values}
    f1_dict = {k: 0.0 for k in k_values}
    aps: List[float] = []

    for pred, gt in zip(predictions, ground_truths):
        if not gt:
            continue
        aps.append(average_precision(pred, gt))
        for k in k_values:
            p_at_k[k] += precision_at_k(pred, gt, k)
            r_at_k[k] += recall_at_k(pred, gt, k)

    valid_queries = max(1, len(aps))
    mean_p = {k: val / valid_queries for k, val in p_at_k.items()}
    mean_r = {k: val / valid_queries for k, val in r_at_k.items()}
    for k in k_values:
        f1_dict[k] = f1_at_k(mean_p[k], mean_r[k])

    mean_latency = _mean(latencies_ms) if latencies_ms else 0.0

    return EvaluationMetrics(
        precision_at_k=mean_p,
        recall_at_k=mean_r,
        f1_at_k=f1_dict,
        mAP=_mean(aps) if aps else 0.0,
        mean_latency_ms=mean_latency,
        num_queries=num_queries,
    )
