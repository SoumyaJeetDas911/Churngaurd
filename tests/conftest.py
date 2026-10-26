from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

import src.train as train_module
from src.predict import ChurnPredictor


@pytest.fixture
def synthetic_frame() -> pd.DataFrame:
    rng = np.random.default_rng(2026)
    size = 180
    tenure = rng.integers(1, 73, size=size)
    monthly = rng.uniform(20, 120, size=size).round(2)
    contract = rng.choice(
        ["Month-to-month", "One year", "Two year"], size=size, p=[0.55, 0.25, 0.20]
    )
    support = rng.choice(["Yes", "No"], size=size)
    churn = ((tenure < 20) & (contract == "Month-to-month")) | (monthly > 96)
    return pd.DataFrame(
        {
            "CustomerID": [f"TEST-{index:04d}" for index in range(size)],
            "Count": 1,
            "Tenure Months": tenure,
            "Contract": contract,
            "Monthly Charges": monthly,
            "Total Charges": [
                "" if index % 37 == 0 else f"{tenure[index] * monthly[index]:.2f}"
                for index in range(size)
            ],
            "Tech Support": support,
            "Churn Label": np.where(churn, "Yes", "No"),
            "Churn Value": churn.astype(int),
            "Churn Score": np.where(churn, 90, 10),
            "Churn Reason": np.where(churn, "test post-outcome", None),
            "CLTV": rng.integers(1000, 7000, size=size),
            "City": "Synthetic City",
        }
    )


@pytest.fixture
def model_artifact(tmp_path, monkeypatch, synthetic_frame):
    csv_path = tmp_path / "synthetic.csv"
    synthetic_frame.to_csv(csv_path, index=False)
    artifact_path = tmp_path / "models" / "synthetic.joblib"
    monkeypatch.setattr(train_module, "PROJECT_ROOT", tmp_path)
    result = train_module.train(
        csv_path=csv_path,
        artifact_path=artifact_path,
        estimators={
            "logistic_regression": LogisticRegression(
                max_iter=500, class_weight="balanced", random_state=5
            )
        },
        seed=5,
    )
    return artifact_path, result


@pytest.fixture
def predictor(model_artifact) -> ChurnPredictor:
    return ChurnPredictor(model_artifact[0])


@pytest.fixture
def sample_features() -> dict:
    return {
        "Tenure Months": 12,
        "Contract": "Month-to-month",
        "Monthly Charges": 89.0,
        "Total Charges": 1068.0,
        "Tech Support": "No",
    }
