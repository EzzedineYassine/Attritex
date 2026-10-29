from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

CandidateRole = Literal["reference", "baseline", "candidate", "comparison"]


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    label: str
    role: CandidateRole
    complexity: int
    eligible: bool
    description: str
    factory: Callable[[int], BaseEstimator]

    def describe(self) -> dict:
        return {
            "name": self.name,
            "label": self.label,
            "role": self.role,
            "complexity": self.complexity,
            "eligible_for_selection": self.eligible,
            "description": self.description,
        }


CANDIDATES: tuple[CandidateSpec, ...] = (
    CandidateSpec(
        name="prior_baseline",
        label="Prior baseline",
        role="reference",
        complexity=0,
        eligible=False,
        description="Predicts the training churn rate for every customer. Any useful model must beat it.",
        factory=lambda seed: DummyClassifier(strategy="prior"),
    ),
    CandidateSpec(
        name="logistic_regression",
        label="Logistic regression",
        role="baseline",
        complexity=1,
        eligible=True,
        description="Regularised linear model on standardised numeric and one-hot categorical features.",
        factory=lambda seed: LogisticRegression(C=1.0, max_iter=2000),
    ),
    CandidateSpec(
        name="logistic_regression_balanced",
        label="Logistic regression (class-weighted)",
        role="comparison",
        complexity=1,
        eligible=False,
        description=(
            "Same model with balanced class weights. Kept for comparison only: reweighting "
            "inflates predicted probabilities, and ChurnSense reports probabilities to users."
        ),
        factory=lambda seed: LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced"),
    ),
    CandidateSpec(
        name="hist_gradient_boosting",
        label="Histogram gradient boosting",
        role="candidate",
        complexity=2,
        eligible=True,
        description="Shallow boosted trees with L2 regularisation and a conservative learning rate.",
        factory=lambda seed: HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=250, max_leaf_nodes=15, min_samples_leaf=40,
            l2_regularization=1.0, early_stopping=False, random_state=seed,
        ),
    ),
    CandidateSpec(
        name="random_forest",
        label="Random forest",
        role="candidate",
        complexity=3,
        eligible=True,
        description="Bagged decision trees with a minimum leaf size to limit overfitting.",
        factory=lambda seed: RandomForestClassifier(
            n_estimators=300, min_samples_leaf=10, max_features="sqrt", n_jobs=-1,
            random_state=seed,
        ),
    ),
)

CANDIDATES_BY_NAME = {candidate.name: candidate for candidate in CANDIDATES}
