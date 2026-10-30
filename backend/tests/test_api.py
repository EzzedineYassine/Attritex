from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient

from attritex.api.app import create_app
from attritex.config import Settings


def sample_customer_payload():
    return {
        "customerID": "TEST-1",
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


def test_health_endpoint_ok(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert data["model_version"] is not None


def test_health_endpoint_degraded_when_artifacts_missing(tmp_path):
    settings = Settings(
        data_dir=tmp_path / "data",
        artifact_dir=tmp_path / "empty_artifacts",
        cors_origins=("*",),
        max_upload_bytes=1024,
        max_batch_rows=10,
    )
    empty_client = TestClient(create_app(settings))
    response = empty_client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "degraded"
    assert data["model_loaded"] is False


def test_model_info_endpoint(client):
    response = client.get("/api/model")
    assert response.status_code == 200
    data = response.json()
    assert "model_version" in data
    assert "headline_metrics" in data
    assert len(data["features"]) > 0


def test_model_evaluation_endpoint_roc_and_baseline(client):
    response = client.get("/api/model/evaluation")
    assert response.status_code == 200
    data = response.json()
    assert "candidates" in data
    assert "holdout" in data
    assert "permutation_importance" in data
    assert "baseline_comparison" in data
    baseline = data["baseline_comparison"]["baseline"]
    assert baseline["label"] == "Dummy Classifier (Stratified)"
    assert "roc_auc" in baseline
    assert "pr_auc" in baseline
    assert "f1" in baseline

    roc = data["holdout"]["curves"]["roc"]
    assert roc[0] == {"fpr": 0.0, "tpr": 0.0}
    assert roc[-1] == {"fpr": 1.0, "tpr": 1.0}
    fprs = [point["fpr"] for point in roc]
    assert fprs == sorted(fprs)
    assert len(fprs) == len(set(fprs))


def test_predict_single_customer_success_and_business_metrics(client):
    payload = sample_customer_payload()
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert 0.0 <= data["churn_probability"] <= 1.0
    expected_arr = round(payload["MonthlyCharges"] * 12.0 * data["churn_probability"], 2)
    assert data["expected_revenue_at_risk"] == pytest.approx(expected_arr, abs=0.02)
    assert data["risk_tier"] in ("low", "medium", "high")
    assert isinstance(data["factors"], list)
    assert "local_attribution" in data
    inc = data["local_attribution"]["top_risk_increasing"]
    red = data["local_attribution"]["top_risk_reducing"]
    assert len(inc) <= 3
    assert len(red) <= 2
    assert all(item["effect"] > 0 for item in inc)
    assert all(item["effect"] < 0 for item in red)


def test_predict_empty_payload_returns_422(client):
    response = client.post("/api/predict", json={})
    assert response.status_code == 422


def test_predict_negative_tenure_returns_422(client):
    payload = sample_customer_payload()
    payload["tenure"] = -1
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 422


def test_predict_excessive_tenure_returns_422(client):
    payload = sample_customer_payload()
    payload["tenure"] = 121
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 422


def test_predict_negative_monthly_charges_returns_422(client):
    payload = sample_customer_payload()
    payload["MonthlyCharges"] = -10.0
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 422


def test_predict_negative_total_charges_returns_422(client):
    payload = sample_customer_payload()
    payload["TotalCharges"] = -5.0
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 422


def test_predict_malformed_types_returns_422(client):
    payload = sample_customer_payload()
    payload["tenure"] = "invalid_string"
    assert client.post("/api/predict", json=payload).status_code == 422

    payload2 = sample_customer_payload()
    payload2["MonthlyCharges"] = "not_a_float"
    assert client.post("/api/predict", json=payload2).status_code == 422

    payload3 = sample_customer_payload()
    payload3["Partner"] = ["Invalid"]
    assert client.post("/api/predict", json=payload3).status_code == 422

    payload4 = sample_customer_payload()
    payload4["Contract"] = "Infinite"
    assert client.post("/api/predict", json=payload4).status_code == 422


def test_predict_extra_field_rejected(client):
    payload = sample_customer_payload()
    payload["unauthorized_property"] = 999
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 422


def test_predict_zero_tenure_imputation_logic(client):
    payload = sample_customer_payload()
    payload["tenure"] = 0
    payload["TotalCharges"] = None
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert 0.0 <= data["churn_probability"] <= 1.0
    assert data["expected_revenue_at_risk"] == pytest.approx(
        round(payload["MonthlyCharges"] * 12.0 * data["churn_probability"], 2),
        abs=0.02,
    )
    assert "TotalCharges" in data["imputed_fields"]


def test_predict_batch_json(client):
    csv_content = (
        "customerID,gender,SeniorCitizen,Partner,Dependents,tenure,PhoneService,MultipleLines,"
        "InternetService,OnlineSecurity,OnlineBackup,DeviceProtection,TechSupport,StreamingTV,"
        "StreamingMovies,Contract,PaperlessBilling,PaymentMethod,MonthlyCharges,TotalCharges\n"
        "CUST1,Female,0,No,No,2,Yes,No,Fiber optic,No,No,No,No,Yes,Yes,Month-to-month,Yes,Electronic check,85.0,170.0\n"
    )
    files = {"file": ("customers.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")}
    response = client.post("/api/predict/batch", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["summary"]["input_rows"] == 1
    assert data["summary"]["scored_rows"] == 1
    assert data["summary"]["total_arr_at_risk"] > 0
    assert "predicted_churn_rate" in data["summary"]
    assert len(data["rows"]) == 1


def test_predict_batch_csv_download(client):
    csv_content = (
        "customerID,gender,SeniorCitizen,Partner,Dependents,tenure,PhoneService,MultipleLines,"
        "InternetService,OnlineSecurity,OnlineBackup,DeviceProtection,TechSupport,StreamingTV,"
        "StreamingMovies,Contract,PaperlessBilling,PaymentMethod,MonthlyCharges,TotalCharges\n"
        "CUST1,Female,0,No,No,2,Yes,No,Fiber optic,No,No,No,No,Yes,Yes,Month-to-month,Yes,Electronic check,85.0,170.0\n"
    )
    files = {"file": ("customers.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")}
    response = client.post("/api/predict/batch?format=csv", files=files)
    assert response.status_code == 200
    assert "text/csv" in response.headers["content-type"]
    assert "churn_probability" in response.text
    assert "expected_revenue_at_risk" in response.text


def test_batch_missing_required_columns_returns_422(client):
    csv_content = (
        "customerID,gender,SeniorCitizen,Partner,Dependents,tenure\n"
        "CUST1,Female,0,No,No,2\n"
    )
    files = {"file": ("incomplete.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")}
    response = client.post("/api/predict/batch", files=files)
    assert response.status_code == 422
    data = response.json()
    assert data["error"]["code"] == "invalid_schema"


def test_batch_empty_file_returns_422(client):
    files = {"file": ("empty.csv", io.BytesIO(b""), "text/csv")}
    response = client.post("/api/predict/batch", files=files)
    assert response.status_code == 422


def test_batch_header_only_returns_422(client):
    csv_content = (
        "customerID,gender,SeniorCitizen,Partner,Dependents,tenure,PhoneService,MultipleLines,"
        "InternetService,OnlineSecurity,OnlineBackup,DeviceProtection,TechSupport,StreamingTV,"
        "StreamingMovies,Contract,PaperlessBilling,PaymentMethod,MonthlyCharges,TotalCharges\n"
    )
    files = {"file": ("header_only.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")}
    response = client.post("/api/predict/batch", files=files)
    assert response.status_code == 422


def test_batch_invalid_extension_returns_415(client):
    files = {"file": ("data.txt", io.BytesIO(b"hello"), "text/plain")}
    response = client.post("/api/predict/batch", files=files)
    assert response.status_code == 415


def test_batch_template_endpoint(client):
    response = client.get("/api/predict/batch/template")
    assert response.status_code == 200
    assert "text/csv" in response.headers["content-type"]
    assert "customerID" in response.text
    assert "MonthlyCharges" in response.text
    assert "attritex_batch_template.csv" in response.headers["content-disposition"]
