from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from attritex.data.schema import (
    FEATURES_BY_NAME,
    INTERNET_ADDON_FEATURES,
    MODEL_FEATURES,
    model_feature_names,
)

Record = dict[str, Any]


def _mode(series: pd.Series, fallback: str) -> str:
    modes = series.dropna().mode()
    return str(modes.iloc[0]) if not modes.empty else fallback


def build_reference_profile(features: pd.DataFrame) -> Record:
    profile: Record = {}
    for spec in MODEL_FEATURES:
        column = features[spec.name]
        if spec.kind == "numeric":
            median = float(column.median())
            profile[spec.name] = round(median) if spec.integer else round(median, 2)
        else:
            profile[spec.name] = _mode(column, spec.allowed_values[0])
    internet_users = features[features["InternetService"].isin(["DSL", "Fiber optic"])]
    for addon in INTERNET_ADDON_FEATURES:
        profile[addon] = _mode(internet_users[addon], "No")
    phone_users = features[features["PhoneService"] == "Yes"]
    profile["MultipleLines"] = _mode(phone_users["MultipleLines"], "No")
    return profile


def _is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and np.isnan(value))


def format_feature_value(feature: str, value: Any) -> str:
    if _is_missing(value):
        return "Not provided"
    spec = FEATURES_BY_NAME[feature]
    if spec.unit == "USD":
        return f"${float(value):,.2f}"
    if spec.unit == "months":
        months = int(value)
        return f"{months} month" if months == 1 else f"{months} months"
    return str(value)


def _match_parent_feature(encoded_name: str, feature_names: list[str]) -> str | None:
    if encoded_name in feature_names:
        return encoded_name
    for name in sorted(feature_names, key=len, reverse=True):
        if encoded_name.startswith(f"{name}_"):
            return name
    return None


def _sigmoid(value: float) -> float:
    clipped = float(np.clip(value, -30.0, 30.0))
    return float(1.0 / (1.0 + np.exp(-clipped)))


def _linear_contributions(
    pipeline: Pipeline,
    instance_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    feature_names: list[str],
) -> dict[str, float]:
    preprocess = pipeline.named_steps["preprocess"]
    estimator = pipeline.named_steps["model"]
    x_instance = np.asarray(preprocess.transform(instance_df), dtype=float)[0]
    x_reference = np.asarray(preprocess.transform(reference_df), dtype=float)[0]
    encoded_names = [str(name) for name in preprocess.get_feature_names_out()]

    if hasattr(estimator, "coef_"):
        coefs = np.asarray(estimator.coef_[0], dtype=float)
        raw_contributions = coefs * (x_instance - x_reference)
        aggregated: dict[str, float] = dict.fromkeys(feature_names, 0.0)
        for encoded_name, contribution in zip(encoded_names, raw_contributions, strict=True):
            parent = _match_parent_feature(encoded_name, feature_names)
            if parent is not None:
                aggregated[parent] += float(contribution)
        return aggregated

    base_prob = float(pipeline.predict_proba(instance_df)[0, 1])
    base_logit = float(np.log(np.clip(base_prob, 1e-6, 1 - 1e-6) / (1 - np.clip(base_prob, 1e-6, 1 - 1e-6))))
    fallback: dict[str, float] = {}
    for feature in feature_names:
        modified = instance_df.copy()
        modified[feature] = reference_df[feature].iloc[0]
        alt_prob = float(pipeline.predict_proba(modified)[0, 1])
        alt_logit = float(np.log(np.clip(alt_prob, 1e-6, 1 - 1e-6) / (1 - np.clip(alt_prob, 1e-6, 1 - 1e-6))))
        fallback[feature] = base_logit - alt_logit
    return fallback


def explain_customer(pipeline: Pipeline, record: Record, reference: Record) -> list[dict]:
    feature_names = model_feature_names()
    instance_df = pd.DataFrame([record])[feature_names]
    reference_df = pd.DataFrame([reference])[feature_names]

    baseline_probability = float(pipeline.predict_proba(instance_df)[0, 1])
    clipped_prob = float(np.clip(baseline_probability, 1e-6, 1.0 - 1e-6))
    baseline_logit = float(np.log(clipped_prob / (1.0 - clipped_prob)))

    contributions = _linear_contributions(pipeline, instance_df, reference_df, feature_names)

    items: list[dict] = []
    for feature_name, logit_shift in contributions.items():
        if abs(logit_shift) < 1e-6:
            continue
        probability_if_reference = _sigmoid(baseline_logit - logit_shift)
        effect = baseline_probability - probability_if_reference
        items.append({
            "feature": feature_name,
            "label": FEATURES_BY_NAME[feature_name].label,
            "customer_value": format_feature_value(feature_name, record.get(feature_name)),
            "reference_value": format_feature_value(feature_name, reference.get(feature_name)),
            "probability_if_reference": float(probability_if_reference),
            "effect": float(effect),
            "attribution_score": float(logit_shift),
            "direction": "risk_increasing" if logit_shift > 0 else "risk_reducing",
        })

    increasing = sorted(
        (item for item in items if item["attribution_score"] > 0),
        key=lambda item: item["attribution_score"],
        reverse=True,
    )[:3]
    reducing = sorted(
        (item for item in items if item["attribution_score"] < 0),
        key=lambda item: item["attribution_score"],
    )[:2]

    return increasing + reducing
