from __future__ import annotations

import io

from churnsense.api.app import create_app
from churnsense.config import Settings


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
    from fastapi.testclient import TestClient

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


def test_model_evaluation_endpoint(client):
    response = client.get("/api/model/evaluation")
    assert response.status_code == 200
    data = response.json()
    assert "candidates" in data
    assert "holdout" in data
    assert "permutation_importance" in data


def test_predict_single_customer_success(client):
    response = client.post("/api/predict", json=sample_customer_payload())
    assert response.status_code == 200
    data = response.json()
    assert 0.0 <= data["churn_probability"] <= 1.0
    assert data["risk_tier"] in ("low", "medium", "high")
    assert isinstance(data["factors"], list)


def test_predict_single_customer_validation_error(client):
    payload = sample_customer_payload()
    payload["Contract"] = "InvalidContract"
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert "error" in data


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


def test_batch_template_endpoint(client):
    response = client.get("/api/predict/batch/template")
    assert response.status_code == 200
    assert "text/csv" in response.headers["content-type"]
    assert "customerID" in response.text
    assert "MonthlyCharges" in response.text
