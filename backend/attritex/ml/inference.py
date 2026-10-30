from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from attritex.data.schema import FEATURES_BY_NAME, ID_COLUMN, model_feature_names
from attritex.data.validation import RowIssue, validate_customers
from attritex.ml.artifacts import ModelBundle
from attritex.ml.explanations import explain_customer
from attritex.ml.thresholds import assign_risk_tiers

RISK_TIER_LABELS = {"high": "High risk", "medium": "Medium risk", "low": "Low risk"}
HISTOGRAM_BINS = 20
DEFAULT_BASELINE_CHURN_RATE = 0.2654


class InputValidationError(ValueError):
    def __init__(self, issues: list[RowIssue]):
        super().__init__("; ".join(f"{issue.column}: {issue.message}" for issue in issues))
        self.issues = issues


@dataclass
class BatchScoringResult:
    rows: list[dict]
    summary: dict
    output_frame: pd.DataFrame
    issues: list[RowIssue] = field(default_factory=list)


def _probabilities(bundle: ModelBundle, features: pd.DataFrame) -> np.ndarray:
    return bundle.pipeline.predict_proba(features[model_feature_names()])[:, 1]


def _tier_context(bundle: ModelBundle, tier: str) -> dict | None:
    for entry in bundle.metadata["risk_tiers"].get("holdout_summary", []):
        if entry["tier"] == tier:
            return entry
    return None


def model_limitations(bundle: ModelBundle) -> list[str]:
    dataset = bundle.metadata["dataset"]
    provenance = dataset["provenance"]
    rows = dataset["quality"]["modelled_rows"]
    limitations = [
        "The probability is a statistical estimate learned from historical customers. "
        "It is not a guarantee that this customer will or will not leave.",
        "Local attribution describes relative probability shift; it does not establish individual causation.",
        f"The model was trained on {rows:,} customers from '{provenance['name']}'. Accuracy on "
        "other businesses, regions, or later time periods has not been verified.",
    ]
    if provenance.get("synthetic"):
        limitations.insert(0, "This model was trained on a synthetic development fixture. "
                              "Its outputs are for software testing only.")
    return limitations


def score_customer(bundle: ModelBundle, record: dict[str, Any]) -> dict:
    report = validate_customers(pd.DataFrame([record]))
    if report.issues:
        raise InputValidationError(report.issues)
    features = report.frame[model_feature_names()]
    probability = float(_probabilities(bundle, features)[0])
    cutoffs = bundle.cutoffs
    tier = str(assign_risk_tiers(np.array([probability]), cutoffs)[0])
    feature_record = features.iloc[0].to_dict()
    monthly_charges = float(feature_record.get("MonthlyCharges") or 0.0)
    expected_revenue_at_risk = round(monthly_charges * 12.0 * probability, 2)
    factors = explain_customer(bundle.pipeline, feature_record, bundle.reference_profile)
    top_increasing = [item for item in factors if item["effect"] > 0][:3]
    top_reducing = [item for item in factors if item["effect"] < 0][:2]
    imputed = report.imputed_fields.get(1, [])
    warnings = list(bundle.warnings)
    if imputed:
        labels = ", ".join(FEATURES_BY_NAME[name].label for name in imputed)
        warnings.append(f"Not provided and estimated from training data: {labels}.")
    return {
        "customer_id": record.get(ID_COLUMN) or None,
        "churn_probability": probability,
        "expected_revenue_at_risk": expected_revenue_at_risk,
        "risk_tier": tier,
        "risk_tier_label": RISK_TIER_LABELS[tier],
        "flagged": probability >= cutoffs.high,
        "decision_threshold": cutoffs.high,
        "tier_cutoffs": cutoffs.to_dict(),
        "tier_context": _tier_context(bundle, tier),
        "factors": factors,
        "local_attribution": {
            "top_risk_increasing": top_increasing,
            "top_risk_reducing": top_reducing,
        },
        "imputed_fields": imputed,
        "warnings": warnings,
        "limitations": model_limitations(bundle),
        "model_version": bundle.version,
    }


def _issue_text(issues: list[RowIssue]) -> str:
    return "; ".join(f"{issue.column}: {issue.message}" for issue in issues)


def _batch_summary(
    probabilities: np.ndarray,
    monthly_charges: np.ndarray,
    tiers: np.ndarray,
    input_rows: int,
    invalid_rows: int,
    threshold: float,
    baseline_churn_rate: float,
) -> dict:
    edges = np.linspace(0, 1, HISTOGRAM_BINS + 1)
    counts = np.histogram(probabilities, bins=edges)[0] if probabilities.size else np.zeros(HISTOGRAM_BINS)
    flagged_count = int((probabilities >= threshold).sum()) if probabilities.size else 0
    scored_count = int(probabilities.size)
    predicted_churn_rate = float(probabilities.mean()) if scored_count else 0.0
    total_arr_at_risk = (
        round(float(np.sum(monthly_charges * 12.0 * probabilities)), 2) if scored_count else 0.0
    )
    return {
        "input_rows": input_rows,
        "scored_rows": scored_count,
        "invalid_rows": invalid_rows,
        "flagged_rows": flagged_count,
        "mean_probability": float(probabilities.mean()) if scored_count else None,
        "predicted_churn_rate": predicted_churn_rate,
        "baseline_churn_rate": baseline_churn_rate,
        "expected_churners": float(probabilities.sum()),
        "total_arr_at_risk": total_arr_at_risk,
        "tier_counts": {tier: int((tiers == tier).sum()) for tier in ("high", "medium", "low")},
        "histogram": [
            {"lower": float(edges[i]), "upper": float(edges[i + 1]), "count": int(counts[i])}
            for i in range(HISTOGRAM_BINS)
        ],
    }


def score_batch(bundle: ModelBundle, raw: pd.DataFrame) -> BatchScoringResult:
    report = validate_customers(raw)
    cutoffs = bundle.cutoffs
    valid = report.frame
    probabilities = _probabilities(bundle, valid) if len(valid) else np.array([], dtype=float)
    monthly_charges = (
        valid["MonthlyCharges"].to_numpy(dtype=float) if len(valid) else np.array([], dtype=float)
    )
    tiers = assign_risk_tiers(probabilities, cutoffs) if len(valid) else np.array([], dtype=str)

    issues_by_row: dict[int, list[RowIssue]] = {}
    for issue in report.issues:
        issues_by_row.setdefault(issue.row_number, []).append(issue)
    scored_by_row = {
        int(position) + 1: (
            float(probability),
            str(tier),
            round(float(monthly) * 12.0 * float(probability), 2),
        )
        for position, probability, tier, monthly in zip(
            valid.index, probabilities, tiers, monthly_charges, strict=True
        )
    }

    source = raw.reset_index(drop=True)
    identifiers = source[ID_COLUMN].astype(str).str.strip() if ID_COLUMN in source.columns else None
    rows = []
    for row_number in range(1, len(source) + 1):
        customer_id = identifiers.iloc[row_number - 1] if identifiers is not None else ""
        scored = scored_by_row.get(row_number)
        rows.append({
            "row": row_number,
            "customer_id": customer_id or None,
            "status": "scored" if scored else "invalid",
            "churn_probability": scored[0] if scored else None,
            "expected_revenue_at_risk": scored[2] if scored else None,
            "risk_tier": scored[1] if scored else None,
            "flagged": bool(scored and scored[0] >= cutoffs.high),
            "issues": _issue_text(issues_by_row.get(row_number, [])),
            "imputed_fields": report.imputed_fields.get(row_number, []),
        })

    output = source.copy()
    output["row"] = [row["row"] for row in rows]
    output["status"] = [row["status"] for row in rows]
    output["churn_probability"] = [
        round(row["churn_probability"], 4) if row["churn_probability"] is not None else None
        for row in rows
    ]
    output["expected_revenue_at_risk"] = [row["expected_revenue_at_risk"] for row in rows]
    output["risk_tier"] = [row["risk_tier"] or "" for row in rows]
    output["flagged_at_threshold"] = [row["flagged"] for row in rows]
    output["validation_issues"] = [row["issues"] for row in rows]
    output["imputed_fields"] = [";".join(row["imputed_fields"]) for row in rows]
    output["model_version"] = bundle.version

    baseline_rate = float(
        bundle.metadata.get("dataset", {})
        .get("quality", {})
        .get("churn_rate", DEFAULT_BASELINE_CHURN_RATE)
    )
    summary = _batch_summary(
        probabilities,
        monthly_charges,
        tiers,
        len(source),
        len(report.invalid_row_numbers),
        cutoffs.high,
        baseline_rate,
    )
    return BatchScoringResult(rows=rows, summary=summary, output_frame=output, issues=report.issues)
