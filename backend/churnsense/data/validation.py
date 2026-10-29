from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from churnsense.data.schema import (
    ID_COLUMN,
    INTERNET_ADDON_FEATURES,
    MODEL_FEATURES,
    NO_INTERNET,
    NO_PHONE,
    TARGET_COLUMN,
    FeatureSpec,
    categorical_feature_names,
    model_feature_names,
)

BOOLEAN_ALIASES = {"0": "No", "1": "Yes", "false": "No", "true": "Yes", "no": "No", "yes": "Yes"}
BOOLEAN_STYLE_FEATURES = {"SeniorCitizen"}


class SchemaError(ValueError):
    def __init__(self, message: str, missing_columns: list[str] | None = None):
        super().__init__(message)
        self.missing_columns = missing_columns or []


@dataclass(frozen=True)
class RowIssue:
    row_number: int
    column: str
    message: str

    def to_dict(self) -> dict:
        return {"row": self.row_number, "column": self.column, "message": self.message}


@dataclass
class ValidationReport:
    frame: pd.DataFrame
    issues: list[RowIssue] = field(default_factory=list)
    imputed_fields: dict[int, list[str]] = field(default_factory=dict)
    input_rows: int = 0

    @property
    def invalid_row_numbers(self) -> list[int]:
        return sorted({issue.row_number for issue in self.issues})

    @property
    def valid_rows(self) -> int:
        return len(self.frame)


def require_columns(frame: pd.DataFrame, require_target: bool) -> None:
    required = model_feature_names() + ([TARGET_COLUMN] if require_target else [])
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise SchemaError(f"Missing required columns: {', '.join(missing)}", missing)


def _normalise_text(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().fillna("")


def _normalise_categorical(series: pd.Series, spec: FeatureSpec) -> pd.Series:
    text = _normalise_text(series)
    if spec.name in BOOLEAN_STYLE_FEATURES:
        text = text.map(lambda value: BOOLEAN_ALIASES.get(value.lower(), value))
    return text


def _categorical_issues(values: pd.Series, spec: FeatureSpec) -> list[tuple[int, str]]:
    allowed = set(spec.allowed_values)
    invalid = values[(values != "") & ~values.isin(allowed)]
    expected = ", ".join(spec.allowed_values)
    return [(position, f"'{value}' is not one of: {expected}") for position, value in invalid.items()]


def _numeric_issues(raw: pd.Series, numeric: pd.Series, spec: FeatureSpec) -> list[tuple[int, str]]:
    issues: list[tuple[int, str]] = []
    unparseable = numeric.isna() & (raw != "")
    issues.extend((position, f"'{raw[position]}' is not a number") for position in raw[unparseable].index)
    if spec.minimum is not None:
        too_low = numeric < spec.minimum
        issues.extend((p, f"must be at least {spec.minimum:g}") for p in numeric[too_low].index)
    if spec.maximum is not None:
        too_high = numeric > spec.maximum
        issues.extend((p, f"must be at most {spec.maximum:g}") for p in numeric[too_high].index)
    if spec.integer:
        fractional = numeric.notna() & (numeric % 1 != 0)
        issues.extend((p, "must be a whole number") for p in numeric[fractional].index)
    return issues


def _consistency_issues(frame: pd.DataFrame) -> list[tuple[int, str, str]]:
    issues: list[tuple[int, str, str]] = []
    no_internet = frame["InternetService"] == "No"
    has_internet = frame["InternetService"].isin(["DSL", "Fiber optic"])
    for addon in INTERNET_ADDON_FEATURES:
        values = frame[addon]
        mismatch_without = no_internet & (values != "") & (values != NO_INTERNET)
        mismatch_with = has_internet & (values == NO_INTERNET)
        for position in frame.index[mismatch_without]:
            issues.append((position, addon, f"must be '{NO_INTERNET}' when InternetService is 'No'"))
        for position in frame.index[mismatch_with]:
            issues.append((position, addon, f"cannot be '{NO_INTERNET}' when internet is active"))
    no_phone = frame["PhoneService"] == "No"
    has_phone = frame["PhoneService"] == "Yes"
    lines = frame["MultipleLines"]
    for position in frame.index[no_phone & (lines != "") & (lines != NO_PHONE)]:
        issues.append((position, "MultipleLines", f"must be '{NO_PHONE}' when PhoneService is 'No'"))
    for position in frame.index[has_phone & (lines == NO_PHONE)]:
        issues.append((position, "MultipleLines", f"cannot be '{NO_PHONE}' when PhoneService is 'Yes'"))
    return issues


def _normalise_target(series: pd.Series) -> tuple[pd.Series, list[tuple[int, str]]]:
    text = _normalise_text(series).map(lambda value: BOOLEAN_ALIASES.get(value.lower(), value))
    invalid = text[~text.isin(["Yes", "No"])]
    issues = [(position, f"'{value}' must be Yes or No") for position, value in invalid.items()]
    return (text == "Yes").astype(int), issues


def validate_customers(frame: pd.DataFrame, require_target: bool = False) -> ValidationReport:
    require_columns(frame, require_target)
    source = frame.reset_index(drop=True)
    clean = pd.DataFrame(index=source.index)
    raw_issues: list[tuple[int, str, str]] = []

    for spec in MODEL_FEATURES:
        if spec.kind == "categorical":
            values = _normalise_categorical(source[spec.name], spec)
            raw_issues.extend((p, spec.name, m) for p, m in _categorical_issues(values, spec))
            clean[spec.name] = values.replace("", np.nan).astype(object)
        else:
            raw = _normalise_text(source[spec.name])
            numeric = pd.to_numeric(raw.replace("", np.nan), errors="coerce").astype(float)
            raw_issues.extend((p, spec.name, m) for p, m in _numeric_issues(raw, numeric, spec))
            clean[spec.name] = numeric

    categorical_view = clean[categorical_feature_names()].fillna("")
    raw_issues.extend(_consistency_issues(categorical_view))

    if require_target:
        target, target_issues = _normalise_target(source[TARGET_COLUMN])
        raw_issues.extend((p, TARGET_COLUMN, m) for p, m in target_issues)
        clean[TARGET_COLUMN] = target

    if ID_COLUMN in source.columns:
        clean.insert(0, ID_COLUMN, _normalise_text(source[ID_COLUMN]))

    issues = sorted(
        (RowIssue(position + 1, column, message) for position, column, message in raw_issues),
        key=lambda issue: (issue.row_number, issue.column),
    )
    invalid_positions = {issue.row_number - 1 for issue in issues}
    valid = clean.drop(index=sorted(invalid_positions))

    feature_block = valid[model_feature_names()]
    imputed = {
        int(position) + 1: [name for name in feature_block.columns if pd.isna(row[name])]
        for position, row in feature_block[feature_block.isna().any(axis=1)].iterrows()
    }
    return ValidationReport(frame=valid, issues=issues, imputed_fields=imputed, input_rows=len(source))
