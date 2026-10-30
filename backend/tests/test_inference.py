from __future__ import annotations

import pandas as pd
import pytest

from attritex.ml.inference import (
    InputValidationError,
    score_batch,
    score_customer,
)


def sample_customer_dict():
    return {
        "customerID": "TEST-1",
        "gender": "Female",
        "SeniorCitizen": "0",
        "Partner": "No",
        "Dependents": "No",
        "tenure": 2,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 85.0,
        "TotalCharges": 170.0,
    }


def test_score_customer_success(bundle):
    result = score_customer(bundle, sample_customer_dict())
    assert 0.0 <= result["churn_probability"] <= 1.0
    assert result["risk_tier"] in ("low", "medium", "high")
    assert isinstance(result["factors"], list)
    assert len(result["limitations"]) > 0


def test_score_customer_invalid_input_raises(bundle):
    record = sample_customer_dict()
    record["Contract"] = "InvalidContract"
    with pytest.raises(InputValidationError):
        score_customer(bundle, record)


def test_score_batch_success(bundle):
    raw = pd.DataFrame([
        sample_customer_dict(),
        {**sample_customer_dict(), "customerID": "TEST-2", "Contract": "BadContract"},
    ])
    result = score_batch(bundle, raw)
    assert result.summary["input_rows"] == 2
    assert result.summary["scored_rows"] == 1
    assert result.summary["invalid_rows"] == 1
    assert len(result.rows) == 2
    assert result.rows[0]["status"] == "scored"
    assert result.rows[1]["status"] == "invalid"
