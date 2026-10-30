from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline

from attritex.data.schema import FEATURES_BY_NAME
from attritex.ml.thresholds import TierCutoffs, assign_risk_tiers

MetricFunction = Callable[[np.ndarray, np.ndarray], float]


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def ranking_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    predictions: np.ndarray | None = None,
) -> dict[str, float]:
    clipped = np.clip(probabilities, 1e-7, 1 - 1e-7)
    binary_preds = predictions if predictions is not None else (probabilities >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "pr_auc": float(average_precision_score(labels, probabilities)),
        "f1": float(f1_score(labels, binary_preds, zero_division=0)),
        "brier": float(brier_score_loss(labels, probabilities)),
        "log_loss": float(log_loss(labels, clipped, labels=[0, 1])),
        "base_rate": float(np.mean(labels)),
    }


def threshold_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    predicted = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    precision = _safe_ratio(tp, tp + fp)
    recall = _safe_ratio(tp, tp + fn)
    return {
        "threshold": float(threshold),
        "precision": precision,
        "recall": recall,
        "f1": _safe_ratio(2 * precision * recall, precision + recall),
        "specificity": _safe_ratio(tn, tn + fp),
        "accuracy": _safe_ratio(tp + tn, len(labels)),
        "flagged_rate": _safe_ratio(tp + fp, len(labels)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def _f1_at(threshold: float) -> MetricFunction:
    return lambda labels, probabilities: threshold_metrics(labels, probabilities, threshold)["f1"]


def bootstrap_intervals(labels: np.ndarray, probabilities: np.ndarray, threshold: float,
                        samples: int, seed: int) -> dict[str, dict[str, float]]:
    metric_functions: dict[str, MetricFunction] = {
        "roc_auc": roc_auc_score,
        "pr_auc": average_precision_score,
        "f1": _f1_at(threshold),
    }
    rng = np.random.default_rng(seed)
    collected: dict[str, list[float]] = {name: [] for name in metric_functions}
    size = len(labels)
    for _ in range(samples):
        indices = rng.integers(0, size, size)
        sample_labels = labels[indices]
        if sample_labels.min() == sample_labels.max():
            continue
        for name, function in metric_functions.items():
            collected[name].append(float(function(sample_labels, probabilities[indices])))
    return {
        name: {
            "lower": float(np.percentile(values, 2.5)),
            "upper": float(np.percentile(values, 97.5)),
            "samples": len(values),
        }
        for name, values in collected.items()
        if values
    }


def _downsample(points: list[dict], limit: int) -> list[dict]:
    if len(points) <= limit:
        return points
    indices = np.unique(np.linspace(0, len(points) - 1, limit).round().astype(int))
    return [points[index] for index in indices]


def roc_points(labels: np.ndarray, probabilities: np.ndarray, limit: int = 120) -> list[dict]:
    false_positive_rate, true_positive_rate, _ = roc_curve(labels, probabilities)
    order = np.argsort(false_positive_rate, kind="stable")
    fpr_sorted = np.clip(np.asarray(false_positive_rate[order], dtype=float), 0.0, 1.0)
    tpr_sorted = np.clip(
        np.maximum.accumulate(np.asarray(true_positive_rate[order], dtype=float)),
        0.0,
        1.0,
    )
    unique_fpr, inverse = np.unique(fpr_sorted, return_inverse=True)
    unique_tpr = np.zeros_like(unique_fpr)
    np.maximum.at(unique_tpr, inverse, tpr_sorted)

    if unique_fpr[0] > 0.0:
        unique_fpr = np.insert(unique_fpr, 0, 0.0)
        unique_tpr = np.insert(unique_tpr, 0, 0.0)
    else:
        unique_fpr[0] = 0.0
        unique_tpr[0] = 0.0

    if unique_fpr[-1] < 1.0:
        unique_fpr = np.append(unique_fpr, 1.0)
        unique_tpr = np.append(unique_tpr, 1.0)
    else:
        unique_fpr[-1] = 1.0
        unique_tpr[-1] = 1.0

    if len(unique_fpr) > limit:
        indices = np.unique(np.linspace(0, len(unique_fpr) - 1, limit).round().astype(int))
        indices[0] = 0
        indices[-1] = len(unique_fpr) - 1
        unique_fpr = unique_fpr[indices]
        unique_tpr = unique_tpr[indices]

    return [
        {"fpr": float(x), "tpr": float(y)}
        for x, y in zip(unique_fpr, unique_tpr, strict=True)
    ]


def precision_recall_points(labels: np.ndarray, probabilities: np.ndarray,
                            limit: int = 120) -> list[dict]:
    precision, recall, _ = precision_recall_curve(labels, probabilities)
    points = [{"recall": float(r), "precision": float(p)}
              for p, r in zip(precision[::-1], recall[::-1], strict=True)]
    return _downsample(points, limit)


def threshold_sweep(labels: np.ndarray, probabilities: np.ndarray) -> list[dict]:
    sweep = []
    for threshold in np.round(np.arange(0.02, 0.99, 0.02), 2):
        metrics = threshold_metrics(labels, probabilities, float(threshold))
        sweep.append({
            "threshold": float(threshold),
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "flagged_rate": metrics["flagged_rate"],
        })
    return sweep


def calibration_points(labels: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> list[dict]:
    edges = np.unique(np.quantile(probabilities, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return [{"predicted": float(probabilities.mean()), "observed": float(labels.mean()),
                 "count": int(len(labels))}]
    bin_ids = np.clip(np.searchsorted(edges, probabilities, side="right") - 1, 0, len(edges) - 2)
    points = []
    for bin_id in range(len(edges) - 1):
        mask = bin_ids == bin_id
        if mask.any():
            points.append({
                "predicted": float(probabilities[mask].mean()),
                "observed": float(labels[mask].mean()),
                "count": int(mask.sum()),
            })
    return points


def cumulative_gains(labels: np.ndarray, probabilities: np.ndarray) -> list[dict]:
    order = np.argsort(-probabilities, kind="stable")
    sorted_labels = labels[order]
    total_positive = sorted_labels.sum()
    gains = []
    for share in np.arange(0.1, 1.01, 0.1):
        cutoff = int(round(share * len(labels)))
        captured = sorted_labels[:cutoff].sum()
        gains.append({
            "population_share": float(round(share, 2)),
            "churners_captured": _safe_ratio(captured, total_positive),
            "lift": _safe_ratio(_safe_ratio(captured, total_positive), share),
        })
    return gains


def score_histogram(labels: np.ndarray, probabilities: np.ndarray, bins: int = 20) -> list[dict]:
    edges = np.linspace(0, 1, bins + 1)
    retained = np.histogram(probabilities[labels == 0], bins=edges)[0]
    churned = np.histogram(probabilities[labels == 1], bins=edges)[0]
    return [
        {"lower": float(edges[i]), "upper": float(edges[i + 1]),
         "retained": int(retained[i]), "churned": int(churned[i])}
        for i in range(bins)
    ]


def tier_summary(labels: np.ndarray, probabilities: np.ndarray, cutoffs: TierCutoffs) -> list[dict]:
    tiers = assign_risk_tiers(probabilities, cutoffs)
    total_positive = max(int(labels.sum()), 1)
    summary = []
    for tier in ("high", "medium", "low"):
        mask = tiers == tier
        summary.append({
            "tier": tier,
            "customers": int(mask.sum()),
            "share_of_customers": _safe_ratio(mask.sum(), len(labels)),
            "observed_churn_rate": _safe_ratio(labels[mask].sum(), mask.sum()),
            "share_of_churners": float(labels[mask].sum() / total_positive),
        })
    return summary


def feature_permutation_importance(pipeline: Pipeline, features: pd.DataFrame,
                                   labels: np.ndarray, repeats: int, seed: int) -> list[dict]:
    result = permutation_importance(
        pipeline, features, labels, scoring="average_precision",
        n_repeats=repeats, random_state=seed, n_jobs=1,
    )
    rows = [
        {
            "feature": name,
            "label": FEATURES_BY_NAME[name].label if name in FEATURES_BY_NAME else name,
            "importance_mean": float(mean),
            "importance_std": float(std),
        }
        for name, mean, std in zip(features.columns, result.importances_mean,
                                   result.importances_std, strict=True)
    ]
    return sorted(rows, key=lambda row: row["importance_mean"], reverse=True)


def evaluate_holdout(labels: np.ndarray, probabilities: np.ndarray, cutoffs: TierCutoffs,
                     bootstrap_samples: int, seed: int) -> dict:
    decision_eval = threshold_metrics(labels, probabilities, cutoffs.high)
    ranking = ranking_metrics(labels, probabilities)
    ranking["f1"] = decision_eval["f1"]
    return {
        "rows": int(len(labels)),
        "positives": int(labels.sum()),
        "ranking": ranking,
        "at_decision_threshold": decision_eval,
        "at_default_threshold": threshold_metrics(labels, probabilities, 0.5),
        "confidence_intervals": bootstrap_intervals(labels, probabilities, cutoffs.high,
                                                    bootstrap_samples, seed),
        "curves": {
            "roc": roc_points(labels, probabilities),
            "precision_recall": precision_recall_points(labels, probabilities),
            "threshold_sweep": threshold_sweep(labels, probabilities),
            "calibration": calibration_points(labels, probabilities),
            "cumulative_gains": cumulative_gains(labels, probabilities),
        },
        "score_histogram": score_histogram(labels, probabilities),
        "tiers": tier_summary(labels, probabilities, cutoffs),
    }
