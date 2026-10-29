from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from churnsense.data.schema import (
    FEATURES_BY_NAME,
    INTERNET_ADDON_FEATURES,
    MODEL_FEATURES,
    NO_INTERNET,
    NO_PHONE,
    model_feature_names,
)

Record = dict[str, Any]
Counterfactual = Callable[[Record, Record], Record | None]
MINIMUM_REPORTED_EFFECT = 0.005


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


def _replace(record: Record, **changes: Any) -> Record | None:
    if all(record.get(name) == value for name, value in changes.items()):
        return None
    return {**record, **changes}


def _single(feature: str) -> Counterfactual:
    return lambda record, reference: _replace(record, **{feature: reference[feature]})


def _tenure(record: Record, reference: Record) -> Record | None:
    tenure = reference["tenure"]
    if record["tenure"] == tenure:
        return None
    return {**record, "tenure": tenure, "TotalCharges": round(tenure * record["MonthlyCharges"], 2)}


def _monthly_charges(record: Record, reference: Record) -> Record | None:
    monthly = reference["MonthlyCharges"]
    if record["MonthlyCharges"] == monthly:
        return None
    return {**record, "MonthlyCharges": monthly,
            "TotalCharges": round(record["tenure"] * monthly, 2)}


def _internet_service(record: Record, reference: Record) -> Record | None:
    service = reference["InternetService"]
    if service == record["InternetService"]:
        return None
    if service == "No":
        addons = dict.fromkeys(INTERNET_ADDON_FEATURES, NO_INTERNET)
    elif record["InternetService"] == "No":
        addons = {addon: reference[addon] for addon in INTERNET_ADDON_FEATURES}
    else:
        addons = {}
    return {**record, "InternetService": service, **addons}


def _internet_addon(feature: str) -> Counterfactual:
    def counterfactual(record: Record, reference: Record) -> Record | None:
        if record[feature] == NO_INTERNET:
            return None
        return _replace(record, **{feature: reference[feature]})
    return counterfactual


def _phone_service(record: Record, reference: Record) -> Record | None:
    service = reference["PhoneService"]
    if service == record["PhoneService"]:
        return None
    lines = NO_PHONE if service == "No" else reference["MultipleLines"]
    return {**record, "PhoneService": service, "MultipleLines": lines}


def _multiple_lines(record: Record, reference: Record) -> Record | None:
    if record["MultipleLines"] == NO_PHONE:
        return None
    return _replace(record, MultipleLines=reference["MultipleLines"])


@dataclass(frozen=True)
class Factor:
    key: str
    features: tuple[str, ...]
    counterfactual: Counterfactual

    @property
    def label(self) -> str:
        return FEATURES_BY_NAME[self.key].label


FACTORS: tuple[Factor, ...] = (
    Factor("Contract", ("Contract",), _single("Contract")),
    Factor("tenure", ("tenure", "MonthlyCharges"), _tenure),
    Factor("MonthlyCharges", ("tenure", "MonthlyCharges"), _monthly_charges),
    Factor("PaymentMethod", ("PaymentMethod",), _single("PaymentMethod")),
    Factor("PaperlessBilling", ("PaperlessBilling",), _single("PaperlessBilling")),
    Factor("InternetService", ("InternetService",), _internet_service),
    *(Factor(addon, (addon,), _internet_addon(addon)) for addon in INTERNET_ADDON_FEATURES),
    Factor("PhoneService", ("PhoneService",), _phone_service),
    Factor("MultipleLines", ("MultipleLines",), _multiple_lines),
    Factor("SeniorCitizen", ("SeniorCitizen",), _single("SeniorCitizen")),
    Factor("Partner", ("Partner",), _single("Partner")),
    Factor("Dependents", ("Dependents",), _single("Dependents")),
)


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


def explain_customer(pipeline: Pipeline, record: Record, reference: Record) -> list[dict]:
    feature_names = model_feature_names()
    baseline_probability = float(pipeline.predict_proba(pd.DataFrame([record])[feature_names])[0, 1])

    applicable: list[tuple[Factor, Record]] = []
    for factor in FACTORS:
        if any(_is_missing(record.get(name)) for name in factor.features):
            continue
        alternative = factor.counterfactual(record, reference)
        if alternative is not None:
            applicable.append((factor, alternative))
    if not applicable:
        return []

    alternatives = pd.DataFrame([alternative for _, alternative in applicable])[feature_names]
    alternative_probabilities = pipeline.predict_proba(alternatives)[:, 1]

    explanations = []
    for (factor, _), alternative_probability in zip(applicable, alternative_probabilities,
                                                    strict=True):
        effect = baseline_probability - float(alternative_probability)
        if abs(effect) < MINIMUM_REPORTED_EFFECT:
            continue
        explanations.append({
            "feature": factor.key,
            "label": factor.label,
            "customer_value": format_feature_value(factor.key, record[factor.key]),
            "reference_value": format_feature_value(factor.key, reference[factor.key]),
            "probability_if_reference": float(alternative_probability),
            "effect": effect,
        })
    return sorted(explanations, key=lambda item: abs(item["effect"]), reverse=True)
