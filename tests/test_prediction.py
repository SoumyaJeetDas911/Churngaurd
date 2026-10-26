from __future__ import annotations

import pytest

from src.predict import PredictionValidationError, risk_category


def test_single_and_batch_predictions_return_valid_probabilities(predictor, sample_features):
    result = predictor.predict(sample_features, include_explanation=False)
    batch = predictor.predict_batch(
        [sample_features, {**sample_features, "Contract": "Unseen contract"}]
    )

    assert 0 <= result["probability"] <= 1
    assert result["predicted_class"] in {"Yes", "No"}
    assert result["risk_category"] in {"low", "medium", "high"}
    assert len(batch) == 2
    assert all(0 <= item["probability"] <= 1 for item in batch)


def test_prediction_rejects_missing_extra_and_non_numeric_features(
    predictor, sample_features
):
    incomplete = dict(sample_features)
    incomplete.pop("Contract")
    with pytest.raises(PredictionValidationError, match="missing features"):
        predictor.predict(incomplete, include_explanation=False)
    with pytest.raises(PredictionValidationError, match="unknown features"):
        predictor.predict({**sample_features, "surprise": 1}, include_explanation=False)
    with pytest.raises(PredictionValidationError, match="must be numeric"):
        predictor.predict(
            {**sample_features, "Tenure Months": "not-a-number"},
            include_explanation=False,
        )
    with pytest.raises(PredictionValidationError, match="excluded fields"):
        predictor.predict(
            {**sample_features, "Churn Label": "Yes"}, include_explanation=False
        )


def test_probability_risk_bands_and_retention_suggestions(predictor, sample_features):
    assert risk_category(0.29) == "low"
    assert risk_category(0.3) == "medium"
    assert risk_category(0.6) == "high"
    with pytest.raises(ValueError, match="between 0 and 1"):
        risk_category(1.1)
    suggestions = predictor.predict(sample_features, include_explanation=False)[
        "retention_suggestions"
    ]
    assert any("contract" in item.lower() for item in suggestions)


def test_shap_explanation_has_encoded_names_and_caveat(predictor, sample_features):
    result = predictor.predict(sample_features)
    assert result["explanation"] is not None
    assert result["explanation"]["method"] == "SHAP"
    assert result["explanation"]["caveat"]
    assert all(
        item["feature"].startswith(("numeric__", "categorical__"))
        for item in result["explanation"]["features"]
    )
