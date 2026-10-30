from __future__ import annotations

import pandas as pd
import pytest

from attritex.data.validation import SchemaError, validate_customers


def valid_record():
    return {
        "customerID": "CUST-001",
        "gender": "Female",
        "SeniorCitizen": "0",
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": "12",
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "Yes",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": "45.50",
        "TotalCharges": "546.00",
    }


def test_validation_passes_valid_record():
    df = pd.DataFrame([valid_record()])
    report = validate_customers(df)
    assert len(report.issues) == 0
    assert report.valid_rows == 1
    assert report.frame["tenure"].iloc[0] == 12.0


def test_missing_required_column_raises_schema_error():
    df = pd.DataFrame([valid_record()]).drop(columns=["Contract"])
    with pytest.raises(SchemaError) as exc_info:
        validate_customers(df)
    assert "Contract" in exc_info.value.missing_columns


def test_invalid_categorical_generates_issue():
    rec = valid_record()
    rec["Contract"] = "Lifetime"
    df = pd.DataFrame([rec])
    report = validate_customers(df)
    assert len(report.issues) == 1
    assert report.issues[0].column == "Contract"
    assert report.valid_rows == 0


def test_out_of_bounds_numeric_generates_issue():
    rec = valid_record()
    rec["tenure"] = "-5"
    df = pd.DataFrame([rec])
    report = validate_customers(df)
    assert any(i.column == "tenure" for i in report.issues)


def test_non_integer_tenure_generates_issue():
    rec = valid_record()
    rec["tenure"] = "12.7"
    df = pd.DataFrame([rec])
    report = validate_customers(df)
    assert any(i.column == "tenure" and "whole number" in i.message for i in report.issues)


def test_consistency_service_violations():
    rec = valid_record()
    rec["InternetService"] = "No"
    rec["OnlineSecurity"] = "Yes"
    df = pd.DataFrame([rec])
    report = validate_customers(df)
    assert any(i.column == "OnlineSecurity" for i in report.issues)


def test_phone_consistency_violation():
    rec = valid_record()
    rec["PhoneService"] = "No"
    rec["MultipleLines"] = "Yes"
    df = pd.DataFrame([rec])
    report = validate_customers(df)
    assert any(i.column == "MultipleLines" for i in report.issues)


def test_blank_total_charges_marked_for_imputation():
    rec = valid_record()
    rec["TotalCharges"] = ""
    df = pd.DataFrame([rec])
    report = validate_customers(df)
    assert len(report.issues) == 0
    assert 1 in report.imputed_fields
    assert "TotalCharges" in report.imputed_fields[1]


def test_blank_total_charges_zero_tenure():
    rec = valid_record()
    rec["tenure"] = "0"
    rec["TotalCharges"] = ""
    df = pd.DataFrame([rec])
    report = validate_customers(df)
    assert len(report.issues) == 0
    assert 1 in report.imputed_fields
    assert "TotalCharges" in report.imputed_fields[1]


def test_target_normalisation():
    rec = valid_record()
    rec["Churn"] = "Yes"
    df = pd.DataFrame([rec])
    report = validate_customers(df, require_target=True)
    assert len(report.issues) == 0
    assert report.frame["Churn"].iloc[0] == 1
