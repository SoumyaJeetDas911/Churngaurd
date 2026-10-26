from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.data_validation import validate_and_prepare
from src.evaluate import evaluate_probabilities
from src.preprocessing import build_pipeline


def test_data_validation_converts_total_charges_and_excludes_leakage(synthetic_frame):
    features, target, metadata = validate_and_prepare(synthetic_frame)

    assert target.name == "churn"
    assert set(target.unique()) == {0, 1}
    assert pd.api.types.is_numeric_dtype(features["Total Charges"])
    assert features["Total Charges"].isna().sum() > 0
    assert not {"CustomerID", "Churn Value", "Churn Score", "Churn Reason", "CLTV"} & set(
        features.columns
    )
    assert metadata["target_column"] == "Churn Label"
    assert metadata["excluded_columns"]["CustomerID"] == "unique customer identifier"


def test_data_validation_rejects_missing_or_invalid_target():
    with pytest.raises(ValueError, match="Missing churn target"):
        validate_and_prepare(pd.DataFrame({"tenure": [1, 2]}))
    with pytest.raises(ValueError, match="Unrecognized values"):
        validate_and_prepare(
            pd.DataFrame({"Churn": ["yes", "maybe"], "feature": [1, 2]})
        )


def test_canonical_customer_id_spelling_is_excluded():
    frame = pd.DataFrame(
        {
            "customerID": ["id-1", "id-2", "id-3", "id-4"],
            "tenure": [1, 2, 3, 4],
            "Churn": ["yes", "no", "yes", "no"],
        }
    )
    features, _, metadata = validate_and_prepare(frame)
    assert "customerID" not in features
    assert metadata["excluded_columns"]["customerID"] == "unique customer identifier"


def test_pipeline_imputes_encodes_and_ignores_unknown_categories():
    frame = pd.DataFrame(
        {"tenure": [1, 5, 10, None], "contract": ["monthly", "yearly", "monthly", "yearly"]}
    )
    target = np.array([1, 0, 1, 0])
    pipeline = build_pipeline(
        frame, LogisticRegression(max_iter=200, random_state=1)
    )
    pipeline.fit(frame, target)
    transformed = pipeline.named_steps["preprocessor"].transform(
        pd.DataFrame({"tenure": [None], "contract": ["future plan"]})
    )
    assert transformed.shape[0] == 1
    assert pipeline.predict_proba(
        pd.DataFrame({"tenure": [None], "contract": ["future plan"]})
    ).shape == (1, 2)
    names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    assert any("contract" in name for name in names)


def test_training_writes_consistent_artifact_metadata(model_artifact):
    artifact_path, result = model_artifact
    metadata = json.loads((artifact_path.parent / "metadata.json").read_text())
    metrics = json.loads((artifact_path.parent / "metrics.json").read_text())

    assert artifact_path.is_file()
    assert result["selected_model"] == "logistic_regression"
    assert metadata["model_type"] == result["selected_model"]
    assert metadata["decision_threshold"] == 0.5
    assert metadata["preprocessing"]["fit_split"] == "training only"
    assert "scikit-learn" in metadata["library_versions"]
    assert set(metadata["feature_schema"]) == {
        "Tenure Months",
        "Contract",
        "Monthly Charges",
        "Total Charges",
        "Tech Support",
    }
    assert metrics["selected_model"] == metadata["model_type"]
    matrix = metrics["held_out_test"]["confusion_matrix"]
    assert len(matrix) == 2 and all(len(row) == 2 for row in matrix)
    assert sum(map(sum, matrix)) == metadata["split_rows"]["test"]
    assert 0 <= metrics["held_out_test"]["roc_auc"] <= 1


def test_evaluation_handles_single_class_without_auc():
    metrics = evaluate_probabilities(np.array([0, 0]), np.array([0.1, 0.9]), 0.5)
    assert metrics["roc_auc"] is None
    assert metrics["pr_auc"] is None
    assert metrics["confusion_matrix"] == [[1, 1], [0, 0]]
