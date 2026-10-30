from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from attritex.data.schema import model_feature_names
from attritex.ml.preprocessing import (
    TotalChargesImputer,
    build_column_transformer,
    build_model_pipeline,
    encoded_feature_names,
)


def test_total_charges_imputer_derives_value_when_missing():
    imputer = TotalChargesImputer()
    df = pd.DataFrame({
        "tenure": [10.0, 5.0],
        "MonthlyCharges": [20.0, 50.0],
        "TotalCharges": [np.nan, 200.0],
    })
    imputer.fit(df)
    transformed = imputer.transform(df)
    assert transformed["TotalCharges"].iloc[0] == 200.0
    assert transformed["TotalCharges"].iloc[1] == 200.0


def test_total_charges_imputer_zero_tenure_imputation():
    imputer = TotalChargesImputer()
    df = pd.DataFrame({
        "tenure": [0.0],
        "MonthlyCharges": [65.0],
        "TotalCharges": [np.nan],
    })
    imputer.fit(df)
    transformed = imputer.transform(df)
    assert transformed["TotalCharges"].iloc[0] == 0.0


def test_column_transformer_encodes_numeric_and_categorical():
    transformer = build_column_transformer()
    names = model_feature_names()
    sample = pd.DataFrame([{
        "SeniorCitizen": "0",
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12.0,
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
        "MonthlyCharges": 45.0,
        "TotalCharges": 540.0,
    }])[names]

    transformer.fit(sample)
    result = transformer.transform(sample)
    assert result.shape[0] == 1
    assert result.shape[1] > 0


def test_build_model_pipeline_trains_and_predicts():
    pipeline = build_model_pipeline(LogisticRegression())
    df = pd.DataFrame([
        {
            "SeniorCitizen": "0", "Partner": "Yes", "Dependents": "No", "tenure": 12.0,
            "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "DSL",
            "OnlineSecurity": "Yes", "OnlineBackup": "No", "DeviceProtection": "No",
            "TechSupport": "Yes", "StreamingTV": "No", "StreamingMovies": "No",
            "Contract": "Month-to-month", "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check", "MonthlyCharges": 45.0, "TotalCharges": 540.0,
        },
        {
            "SeniorCitizen": "1", "Partner": "No", "Dependents": "No", "tenure": 2.0,
            "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic",
            "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No",
            "TechSupport": "No", "StreamingTV": "Yes", "StreamingMovies": "Yes",
            "Contract": "Month-to-month", "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check", "MonthlyCharges": 90.0, "TotalCharges": 180.0,
        },
    ])
    y = np.array([0, 1])
    pipeline.fit(df, y)
    probs = pipeline.predict_proba(df)
    assert probs.shape == (2, 2)
    assert len(encoded_feature_names(pipeline)) > 0
