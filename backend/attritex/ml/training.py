from __future__ import annotations

import logging
import math
import platform
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

from attritex import __version__
from attritex.data.loading import read_customer_csv, read_provenance
from attritex.data.schema import (
    EXCLUDED_FEATURES,
    ID_COLUMN,
    MODEL_FEATURES,
    TARGET_COLUMN,
    model_feature_names,
)
from attritex.data.validation import ValidationReport, validate_customers
from attritex.ml.artifacts import ARTIFACT_SCHEMA_VERSION, save_bundle
from attritex.ml.candidates import CANDIDATES, CandidateSpec
from attritex.ml.evaluation import (
    evaluate_holdout,
    feature_permutation_importance,
    ranking_metrics,
)
from attritex.ml.explanations import build_reference_profile
from attritex.ml.preprocessing import build_model_pipeline, encoded_feature_names
from attritex.ml.thresholds import TierCutoffs, derive_tier_cutoffs

logger = logging.getLogger(__name__)

CV_SCORING = {
    "pr_auc": "average_precision",
    "roc_auc": "roc_auc",
    "f1": "f1",
    "brier": "neg_brier_score",
    "log_loss": "neg_log_loss",
}
NEGATED_SCORES = {"brier", "log_loss"}


class TrainingDataError(ValueError):
    pass


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 42
    test_size: float = 0.2
    cv_folds: int = 5
    medium_recall_target: float = 0.9
    permutation_repeats: int = 10
    bootstrap_samples: int = 1000
    max_invalid_fraction: float = 0.02

    def to_dict(self) -> dict:
        return {**asdict(self), "selection_metric": "pr_auc", "selection_rule": "one_standard_error"}


@dataclass(frozen=True)
class PreparedData:
    features: pd.DataFrame
    labels: np.ndarray
    quality: dict


@dataclass(frozen=True)
class CandidateResult:
    spec: CandidateSpec
    pipeline: Pipeline
    cv_scores: dict[str, dict[str, float]]
    test_ranking: dict[str, float]

    def to_dict(self) -> dict:
        return {**self.spec.describe(), "cv": self.cv_scores, "test": self.test_ranking}


@dataclass(frozen=True)
class TrainingResult:
    metadata: dict
    evaluation: dict
    artifact_dir: Path


def prepare_training_data(raw: pd.DataFrame, config: TrainingConfig) -> PreparedData:
    report: ValidationReport = validate_customers(raw, require_target=True)
    invalid_rows = len(report.invalid_row_numbers)
    if report.input_rows == 0:
        raise TrainingDataError("Training data is empty.")
    invalid_fraction = invalid_rows / report.input_rows
    if invalid_fraction > config.max_invalid_fraction:
        examples = "; ".join(f"row {i.row_number} {i.column}: {i.message}" for i in report.issues[:5])
        raise TrainingDataError(
            f"{invalid_rows} of {report.input_rows} rows failed validation "
            f"({invalid_fraction:.1%} > {config.max_invalid_fraction:.0%}). Examples: {examples}"
        )

    frame = report.frame
    duplicates_removed = 0
    if ID_COLUMN in frame.columns:
        identified = frame[ID_COLUMN] != ""
        duplicate_mask = identified & frame.duplicated(subset=[ID_COLUMN], keep="first")
        duplicates_removed = int(duplicate_mask.sum())
        frame = frame[~duplicate_mask]

    labels = frame[TARGET_COLUMN].to_numpy(dtype=int)
    positives = int(labels.sum())
    minimum_per_class = config.cv_folds * 4
    if positives < minimum_per_class or len(labels) - positives < minimum_per_class:
        raise TrainingDataError(
            f"Need at least {minimum_per_class} churned and retained customers; "
            f"found {positives} churned and {len(labels) - positives} retained."
        )

    features = frame[model_feature_names()].reset_index(drop=True)
    missing_counts = {name: int(count) for name, count in features.isna().sum().items() if count}
    quality = {
        "input_rows": report.input_rows,
        "invalid_rows_dropped": invalid_rows,
        "duplicate_ids_dropped": duplicates_removed,
        "modelled_rows": int(len(labels)),
        "churned": positives,
        "retained": int(len(labels) - positives),
        "churn_rate": float(labels.mean()),
        "missing_values_imputed": missing_counts,
        "validation_issue_examples": [issue.to_dict() for issue in report.issues[:10]],
    }
    return PreparedData(features=features, labels=labels, quality=quality)


def _summarise_cv(raw_scores: dict) -> dict[str, dict[str, float]]:
    summary = {}
    for name in CV_SCORING:
        values = np.asarray(raw_scores[f"test_{name}"], dtype=float)
        if name in NEGATED_SCORES:
            values = -values
        summary[name] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "folds": [float(value) for value in values],
        }
    return summary


def evaluate_candidates(train_x: pd.DataFrame, train_y: np.ndarray, test_x: pd.DataFrame,
                        test_y: np.ndarray, config: TrainingConfig) -> list[CandidateResult]:
    folds = StratifiedKFold(n_splits=config.cv_folds, shuffle=True, random_state=config.seed)
    results = []
    for spec in CANDIDATES:
        logger.info("Cross-validating %s", spec.name)
        pipeline = build_model_pipeline(spec.factory(config.seed))
        raw_scores = cross_validate(pipeline, train_x, train_y, cv=folds, scoring=CV_SCORING)
        pipeline.fit(train_x, train_y)
        test_probabilities = pipeline.predict_proba(test_x)[:, 1]
        test_predictions = pipeline.predict(test_x)
        results.append(CandidateResult(
            spec=spec,
            pipeline=pipeline,
            cv_scores=_summarise_cv(raw_scores),
            test_ranking=ranking_metrics(test_y, test_probabilities, test_predictions),
        ))
    return results


def select_candidate(results: list[CandidateResult], folds: int) -> tuple[CandidateResult, dict]:
    eligible = [result for result in results if result.spec.eligible]
    if not eligible:
        raise TrainingDataError("No candidate is eligible for selection.")
    best = max(eligible, key=lambda result: result.cv_scores["pr_auc"]["mean"])
    best_mean = best.cv_scores["pr_auc"]["mean"]
    standard_error = best.cv_scores["pr_auc"]["std"] / math.sqrt(folds)
    floor = best_mean - standard_error
    within = [result for result in eligible if result.cv_scores["pr_auc"]["mean"] >= floor]
    chosen = min(within, key=lambda result: (result.spec.complexity,
                                             -result.cv_scores["pr_auc"]["mean"]))
    rationale = {
        "metric": "pr_auc",
        "best_candidate": best.spec.name,
        "best_cv_mean": best_mean,
        "best_cv_standard_error": standard_error,
        "acceptance_floor": floor,
        "candidates_within_one_se": [result.spec.name for result in within],
        "selected": chosen.spec.name,
        "explanation": (
            f"{chosen.spec.label} was selected because it is the simplest eligible model whose "
            f"mean cross-validated PR-AUC is within one standard error of the best "
            f"({best.spec.label}, {best_mean:.3f} ± {standard_error:.3f})."
        ),
    }
    return chosen, rationale


def _build_baseline_comparison(results: list[CandidateResult], chosen: CandidateResult,
                               holdout: dict) -> dict:
    by_name = {result.spec.name: result for result in results}
    dummy = by_name.get("dummy_stratified") or by_name.get("prior_baseline")
    dummy_metrics = {
        "name": dummy.spec.name if dummy else "dummy_stratified",
        "label": dummy.spec.label if dummy else "Dummy Classifier (Stratified)",
        "roc_auc": float(dummy.test_ranking["roc_auc"]) if dummy else 0.5,
        "pr_auc": float(dummy.test_ranking["pr_auc"]) if dummy else float(holdout["ranking"]["base_rate"]),
        "f1": float(dummy.test_ranking["f1"]) if dummy else 0.0,
    }
    production_metrics = {
        "name": chosen.spec.name,
        "label": chosen.spec.label,
        "roc_auc": float(holdout["ranking"]["roc_auc"]),
        "pr_auc": float(holdout["ranking"]["pr_auc"]),
        "f1": float(holdout["at_decision_threshold"]["f1"]),
    }
    return {
        "baseline": dummy_metrics,
        "production": production_metrics,
        "lift": {
            "roc_auc_delta": round(production_metrics["roc_auc"] - dummy_metrics["roc_auc"], 4),
            "pr_auc_delta": round(production_metrics["pr_auc"] - dummy_metrics["pr_auc"], 4),
            "f1_delta": round(production_metrics["f1"] - dummy_metrics["f1"], 4),
        },
    }


def _model_version(created_at: datetime, model_name: str, data_sha: str) -> str:
    return f"{created_at:%Y%m%d.%H%M%S}-{model_name}-{data_sha[:8]}"


def _build_metadata(*, created_at: datetime, chosen: CandidateResult, config: TrainingConfig,
                    provenance: dict, quality: dict, split: dict, cutoffs: TierCutoffs,
                    rationale: dict, holdout: dict, reference_profile: dict) -> dict:
    return {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "model_version": _model_version(created_at, chosen.spec.name, provenance["sha256"]),
        "created_at": created_at.isoformat(),
        "model": chosen.spec.describe(),
        "selection": rationale,
        "features": {
            "model_inputs": model_feature_names(),
            "specs": [spec.to_dict() for spec in MODEL_FEATURES],
            "excluded": [{"name": name, "reason": reason} for name, reason in EXCLUDED_FEATURES.items()],
            "encoded": encoded_feature_names(chosen.pipeline),
        },
        "training_config": config.to_dict(),
        "dataset": {"provenance": provenance, "quality": quality, "split": split},
        "risk_tiers": {
            "cutoffs": cutoffs.to_dict(),
            "derivation": (
                "Cutoffs are derived from out-of-fold predictions on the training split only. "
                "High risk starts at the F1-optimal threshold (the decision threshold). Medium "
                f"risk starts at the highest threshold that still recalls "
                f"{cutoffs.medium_recall_target:.0%} of churners."
            ),
            "holdout_summary": holdout["tiers"],
        },
        "headline_metrics": {
            "roc_auc": holdout["ranking"]["roc_auc"],
            "pr_auc": holdout["ranking"]["pr_auc"],
            "brier": holdout["ranking"]["brier"],
            "precision": holdout["at_decision_threshold"]["precision"],
            "recall": holdout["at_decision_threshold"]["recall"],
            "f1": holdout["at_decision_threshold"]["f1"],
            "decision_threshold": cutoffs.high,
            "evaluated_on": "holdout test split",
        },
        "reference_profile": reference_profile,
        "environment": {
            "attritex": __version__,
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }


def train_from_frame(raw: pd.DataFrame, provenance: dict, artifact_dir: Path,
                     config: TrainingConfig | None = None) -> TrainingResult:
    config = config or TrainingConfig()
    prepared = prepare_training_data(raw, config)
    train_x, test_x, train_y, test_y = train_test_split(
        prepared.features, prepared.labels, test_size=config.test_size,
        stratify=prepared.labels, random_state=config.seed,
    )
    split = {
        "strategy": "stratified holdout",
        "train_rows": int(len(train_y)),
        "test_rows": int(len(test_y)),
        "train_churn_rate": float(train_y.mean()),
        "test_churn_rate": float(test_y.mean()),
    }

    results = evaluate_candidates(train_x, train_y, test_x, test_y, config)
    chosen, rationale = select_candidate(results, config.cv_folds)
    logger.info("Selected %s", chosen.spec.name)

    folds = StratifiedKFold(n_splits=config.cv_folds, shuffle=True, random_state=config.seed)
    out_of_fold = cross_val_predict(
        build_model_pipeline(chosen.spec.factory(config.seed)), train_x, train_y,
        cv=folds, method="predict_proba",
    )[:, 1]
    cutoffs = derive_tier_cutoffs(train_y, out_of_fold, config.medium_recall_target)

    test_probabilities = chosen.pipeline.predict_proba(test_x)[:, 1]
    holdout = evaluate_holdout(test_y, test_probabilities, cutoffs,
                               config.bootstrap_samples, config.seed)
    logger.info("Computing permutation importance")
    importance = feature_permutation_importance(chosen.pipeline, test_x, test_y,
                                                config.permutation_repeats, config.seed)

    created_at = datetime.now(UTC).replace(microsecond=0)
    metadata = _build_metadata(
        created_at=created_at, chosen=chosen, config=config, provenance=provenance,
        quality=prepared.quality, split=split, cutoffs=cutoffs, rationale=rationale,
        holdout=holdout, reference_profile=build_reference_profile(train_x),
    )
    candidates_payload = []
    for result in results:
        payload = result.to_dict()
        if result.spec.name == chosen.spec.name:
            payload["test"] = {**payload["test"], "f1": float(holdout["at_decision_threshold"]["f1"])}
        candidates_payload.append(payload)

    evaluation = {
        "model_version": metadata["model_version"],
        "synthetic_data": bool(provenance.get("synthetic")),
        "candidates": candidates_payload,
        "baseline_comparison": _build_baseline_comparison(results, chosen, holdout),
        "selection": rationale,
        "holdout": holdout,
        "out_of_fold": {
            "description": "Selected model, out-of-fold predictions on the training split.",
            "ranking": ranking_metrics(train_y, out_of_fold),
        },
        "permutation_importance": {
            "scoring": "average_precision",
            "repeats": config.permutation_repeats,
            "evaluated_on": "holdout test split",
            "features": importance,
        },
    }
    save_bundle(artifact_dir, chosen.pipeline, metadata, evaluation)
    return TrainingResult(metadata=metadata, evaluation=evaluation, artifact_dir=artifact_dir)


def train_from_file(data_path: Path, artifact_dir: Path,
                    config: TrainingConfig | None = None) -> TrainingResult:
    provenance = read_provenance(data_path).to_dict()
    raw = read_customer_csv(data_path)
    return train_from_frame(raw, provenance, artifact_dir, config)
