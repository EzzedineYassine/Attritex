from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import precision_recall_curve


@dataclass(frozen=True)
class TierCutoffs:
    medium: float
    high: float
    medium_recall_target: float

    def to_dict(self) -> dict:
        return asdict(self)


def f1_optimal_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    precision, recall = precision[:-1], recall[:-1]
    denominator = precision + recall
    f1 = np.divide(2 * precision * recall, denominator, out=np.zeros_like(denominator),
                   where=denominator > 0)
    return float(thresholds[int(np.argmax(f1))])


def recall_target_threshold(labels: np.ndarray, probabilities: np.ndarray, target: float) -> float:
    if not 0 < target <= 1:
        raise ValueError("Recall target must be in (0, 1]")
    _, recall, thresholds = precision_recall_curve(labels, probabilities)
    eligible = np.flatnonzero(recall[:-1] >= target)
    if eligible.size == 0:
        return float(thresholds.min())
    return float(thresholds[eligible.max()])


def derive_tier_cutoffs(labels: np.ndarray, probabilities: np.ndarray,
                        medium_recall_target: float) -> TierCutoffs:
    high = f1_optimal_threshold(labels, probabilities)
    medium = min(recall_target_threshold(labels, probabilities, medium_recall_target), high)
    return TierCutoffs(medium=round(medium, 4), high=round(high, 4),
                       medium_recall_target=medium_recall_target)


def assign_risk_tiers(probabilities: np.ndarray, cutoffs: TierCutoffs) -> np.ndarray:
    return np.select(
        [probabilities >= cutoffs.high, probabilities >= cutoffs.medium],
        ["high", "medium"],
        default="low",
    )
