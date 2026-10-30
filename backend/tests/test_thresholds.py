from __future__ import annotations

import numpy as np
import pytest

from attritex.ml.thresholds import (
    TierCutoffs,
    assign_risk_tiers,
    derive_tier_cutoffs,
    f1_optimal_threshold,
    recall_target_threshold,
)


def test_f1_optimal_threshold_selects_valid_threshold():
    labels = np.array([0, 0, 0, 1, 1, 1, 0, 1, 0, 1])
    probs = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9, 0.4, 0.65, 0.35, 0.85])
    threshold = f1_optimal_threshold(labels, probs)
    assert 0.1 <= threshold <= 0.9


def test_recall_target_threshold():
    labels = np.array([0, 0, 1, 1, 1, 1, 0, 1, 0, 1])
    probs = np.array([0.1, 0.2, 0.4, 0.5, 0.6, 0.7, 0.3, 0.8, 0.15, 0.9])
    thresh = recall_target_threshold(labels, probs, 0.8)
    assert thresh <= 0.6

    with pytest.raises(ValueError):
        recall_target_threshold(labels, probs, 1.5)


def test_derive_tier_cutoffs_ordering():
    labels = np.array([0, 0, 0, 1, 1, 1, 0, 1, 0, 1])
    probs = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9, 0.4, 0.65, 0.35, 0.85])
    cutoffs = derive_tier_cutoffs(labels, probs, medium_recall_target=0.85)
    assert cutoffs.medium <= cutoffs.high
    assert cutoffs.medium_recall_target == 0.85


def test_assign_risk_tiers():
    cutoffs = TierCutoffs(medium=0.3, high=0.6, medium_recall_target=0.9)
    probs = np.array([0.1, 0.29, 0.3, 0.55, 0.6, 0.85])
    tiers = assign_risk_tiers(probs, cutoffs)
    assert list(tiers) == ["low", "low", "medium", "medium", "high", "high"]
