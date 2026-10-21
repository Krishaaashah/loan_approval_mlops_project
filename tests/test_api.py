import io

import numpy as np
from fastapi.testclient import TestClient

import main
from prediction_model import predict


class FakeModel:
    def predict(self, data):
        return np.ones(len(data), dtype=int)


def test_health_endpoint():
    response = TestClient(main.app).get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_prediction_endpoint_uses_local_model(monkeypatch):
    monkeypatch.setattr(predict, "get_model", lambda: FakeModel())
    payload = {
        "Gender": "Male",
        "Married": "Yes",
        "Dependents": "0",
        "Education": "Graduate",
        "Self_Employed": "No",
        "ApplicantIncome": 5000,
        "CoapplicantIncome": 1000,
        "LoanAmount": 120,
        "Loan_Amount_Term": 360,
        "Credit_History": 1,
        "Property_Area": "Urban",
    }

    response = TestClient(main.app).post("/prediction_api", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "Approved"
    assert body["decision"] == "Approved"
    assert 0 <= body["approval_probability"] <= 1
    assert body["confidence"] in {"Low", "Medium", "High"}


def test_batch_prediction_does_not_require_s3_in_local_mode(monkeypatch):
    monkeypatch.setattr(predict, "get_model", lambda: FakeModel())
    csv_data = (
        "Gender,Married,Dependents,Education,Self_Employed,"
        "ApplicantIncome,CoapplicantIncome,LoanAmount,Loan_Amount_Term,"
        "Credit_History,Property_Area\n"
        "Male,Yes,0,Graduate,No,5000,1000,120,360,1,Urban\n"
    )

    response = TestClient(main.app).post(
        "/batch_prediction",
        files={"file": ("input.csv", io.BytesIO(csv_data.encode()), "text/csv")},
    )

    assert response.status_code == 200
    assert "Prediction" in response.text
