from __future__ import annotations

import json

import pandas as pd
from fastapi.testclient import TestClient

import api.main as api_main
from api.database import build_engine as make_engine
from dashboard.app import validate_csv_schema


def test_api_prediction_health_batch_and_persisted_history(
    predictor, sample_features, tmp_path, monkeypatch
):
    database_path = tmp_path / "predictions.sqlite3"
    monkeypatch.setattr(
        api_main,
        "build_engine",
        lambda: make_engine(f"sqlite:///{database_path.as_posix()}"),
    )
    monkeypatch.setattr(api_main, "ChurnPredictor", lambda: predictor)

    with TestClient(api_main.app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["model_ready"] is True

        single = client.post(
            "/predict",
            json={"features": sample_features, "customer_label": "synthetic-a"},
        )
        assert single.status_code == 200
        assert 0 <= single.json()["probability"] <= 1
        assert single.json()["history_saved"] is True

        batch = client.post(
            "/predict/batch",
            json={
                "records": [
                    {"features": sample_features},
                    {"features": {**sample_features, "Contract": "new category"}},
                ]
            },
        )
        assert batch.status_code == 200
        assert batch.json()["count"] == len(batch.json()["results"]) == 2

        invalid = client.post("/predict", json={"features": {"Tenure Months": 4}})
        assert invalid.status_code == 422
        invalid_batch = client.post(
            "/predict/batch",
            json={
                "records": [
                    {"features": sample_features},
                    {"features": {"Tenure Months": 4}},
                ]
            },
        )
        assert invalid_batch.status_code == 422
        assert "Record 1" in invalid_batch.json()["detail"]

        page = client.get("/history?limit=2&offset=0")
        assert page.status_code == 200
        assert page.json()["total"] == 3
        assert len(page.json()["items"]) == 2
        detail_id = page.json()["items"][0]["id"]
        assert client.get(f"/history/{detail_id}").status_code == 200
        assert client.get("/analytics/summary").json()["total_predictions"] == 3


def test_api_readiness_and_feature_schema(predictor, tmp_path, monkeypatch):
    monkeypatch.setattr(
        api_main,
        "build_engine",
        lambda: make_engine(f"sqlite:///{(tmp_path / 'health.sqlite3').as_posix()}"),
    )
    monkeypatch.setattr(api_main, "ChurnPredictor", lambda: predictor)
    profile_path = tmp_path / "artifacts" / "eda" / "data_profile.json"
    profile_path.parent.mkdir(parents=True)
    profile_path.write_text(
        json.dumps(
            {
                "shape": {"rows": 5},
                "target_counts": {"No": 3, "Yes": 2},
                "descriptive_observations": ["Synthetic example."],
                "segment_churn": {},
            }
        )
    )
    monkeypatch.setattr(api_main, "PROJECT_ROOT", tmp_path)
    with TestClient(api_main.app) as client:
        assert client.get("/model/info").status_code == 200
        response = client.get("/features")
        assert response.status_code == 200
        assert response.json()["target_column"] == "Churn Label"
        assert client.get("/metrics").json()["held_out_test"]["roc_auc"] is not None
        assert client.get("/analytics/training").json()["shape"]["rows"] == 5


def test_predictions_remain_available_when_history_is_disabled(
    predictor, sample_features, tmp_path, monkeypatch
):
    monkeypatch.setattr(
        api_main,
        "build_engine",
        lambda: make_engine(f"sqlite:///{(tmp_path / 'offline.sqlite3').as_posix()}"),
    )
    monkeypatch.setattr(api_main, "ChurnPredictor", lambda: predictor)
    with TestClient(api_main.app) as client:
        client.app.state.history_ready = False
        response = client.post("/predict", json={"features": sample_features})
        assert response.status_code == 200
        assert response.json()["history_saved"] is False
        assert "unavailable" in response.json()["history_warning"]


def test_dashboard_csv_schema_utility():
    valid, errors = validate_csv_schema(
        pd.DataFrame({"tenure": [1], "contract": ["monthly"]}),
        ["tenure", "contract"],
    )
    assert valid and not errors
    valid, errors = validate_csv_schema(
        pd.DataFrame({"tenure": [1], "extra": ["value"]}),
        ["tenure", "contract"],
    )
    assert not valid
    assert any("Missing" in error for error in errors)
    assert any("Unexpected" in error for error in errors)
